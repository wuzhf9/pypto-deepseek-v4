# Packed BF16 Expert Cache 详细实现计划

> 当前实现已经收敛为唯一正式 packed cache 格式。下文 Stage 0–6 中的 V1/V2 名称仅保留为迁移过程和
> 性能基线记录，不代表当前代码仍提供旧格式接口。磁盘 manifest 的 `version: 2` 只用于 schema 校验。

## 1. 目标与范围

本计划把 `packed_bf16_expert_cache_plan.md` 细化为可逐阶段实现和验证的修改清单。

当前目标：

1. 新增 expert cache format v2，每层只保存三个完整 packed BF16 tensor；
2. decode 优先接入 lazy slice，只读取 6 个 selected experts；
3. prefill 先接入 packed clone，删除逐 expert 二次 pack；
4. 验证后再接入 mmap direct H2D，删除完整 Host clone；
5. format v2 完成后重新 profile，再决定是否继续 Host prefetch。

本次不做：

- 不修改 MoE kernel、TensorSpec 或模型数值语义；
- 不实现 FP4/FP8 device staging；
- 不实现 prefill unique-expert dispatch；
- 不实现异步 H2D；
- 不修改 WorkerBackend、DeviceBufferPool 或 state 生命周期；
- 不在同一变更中删除 format v1 回退。

## 2. 当前接口边界

当前生产调用保持不变：

```text
Runner
  → WeightLoader.get_layer_moe_routed_pack()       prefill full pack
  → WeightLoader.get_layer_moe_selected_experts() decode six experts
  → HostStagingTensor
  → WorkerBackend
```

Format detection、safetensors handle、mmap/slice 和 v1/v2 分支全部限制在 WeightLoader/expert-cache helper
内部。Runner 和 WorkerBackend 不得感知 cache version。

现有公开 CLI 保持：

```text
--expert-cache-dir <path>
```

不新增 serving CLI 参数；切换 format 只通过指定新的 cache directory 完成。

## 3. 文件变更总表

### 3.1 新增文件

| 文件 | 内容 |
|---|---|
| `serving/expert_cache.py` | format 常量、manifest/file 检测、v1/v2 reader、lazy selected slice 和 handle 生命周期 |
| `serving/convert_packed_expert_cache.py` | 独立生成 format v2 三 packed tensor cache，不影响现有 v1 converter |
| `tests/test_expert_cache.py` | 小 tensor 的 v1/v2 reader、lazy slice、错误格式和 close 测试 |
| `tests/test_convert_packed_expert_cache.py` | format v2 converter、manifest、partial layer 和 atomic output 测试 |

### 3.2 修改文件

| 文件 | 修改内容 |
|---|---|
| `serving/weight_loader.py` | 委托 expert cache reader；decode 走 slice；prefill 走 packed full view/clone；新增 profile |
| `tests/test_weight_loader.py` | 增加 v2 integration、selected 顺序、v1 fallback 和完整 clone 禁止测试 |

### 3.3 明确不修改

```text
serving/runner.py
serving/runtime_types.py
serving/backends/base.py
serving/backends/worker_backend.py
serving/backends/device_pool.py
serving/backends/worker_state_store.py
serving/state.py
models/*
serving/run_model.py
serving/generate.py
serving/convert_expert_cache.py
```

`HostStagingTensor` 和 `StagingKind.PREFILL_ROUTED/DECODE_SELECTED` 已足够表达新路径。

## 4. `serving/expert_cache.py`

### 4.1 常量

集中定义 converter 和 loader 共用的格式常量，禁止两处复制字符串：

```python
EXPERT_CACHE_FORMAT = "dsv4_bf16_layer_experts"
EXPERT_CACHE_V1 = 1
EXPERT_CACHE_V2 = 2

PACKED_W1 = "routed_w1_t"
PACKED_W2 = "routed_w2_t"
PACKED_W3 = "routed_w3_t"
PACKED_KEYS = (PACKED_W1, PACKED_W2, PACKED_W3)
```

提供统一文件名 helper：

```python
def layer_expert_cache_filename(layer_id: int) -> str:
    return f"layer_{layer_id:03d}_experts.safetensors"
```

### 4.2 Manifest 数据

建议增加 immutable 描述：

```python
@dataclass(frozen=True)
class ExpertCacheManifest:
    version: int
    n_layers: int
    n_routed_experts: int
    dim: int
    moe_inter_dim: int
    dtype: str
    layers: Mapping[int, str]
```

加载规则：

1. `expert_cache_dir is None`：reader disabled；
2. 有 manifest：严格校验 format、version、模型维度和 dtype；
3. 无 manifest：允许 legacy v1 文件按 key 检测，保持现有测试和旧 cache 可用；
4. 无 manifest 但发现 v2 packed keys：允许读取，但记录为 unmanifested v2，测试要求 shape/dtype 全校验；
5. manifest 与文件 keys/version 冲突：明确失败，不静默回退 checkpoint。

不允许把错误或不完整的 v2 文件解释成 v1。

### 4.3 Layer file info cache

新增：

```python
@dataclass(frozen=True)
class LayerExpertCacheInfo:
    layer_id: int
    path: Path
    version: int
    keys: frozenset[str]
```

Reader 内部缓存：

```python
self._handles: dict[Path, Any]
self._layer_info: dict[int, LayerExpertCacheInfo]
```

避免当前 `_load_cached_expert()` 每加载一个专家都重新执行 `set(handle.keys())`。

### 4.4 Format detection

`_inspect_layer(layer_id)`：

```text
文件不存在                  → None，WeightLoader 回退 checkpoint
三个 packed keys 全部存在    → v2
只存在合法 per-expert keys    → v1
packed keys 部分存在          → ValueError
packed 与 per-expert 混合      → ValueError
manifest 声明完整层但文件缺失  → FileNotFoundError
```

v2 必须精确校验：

```text
routed_w1_t [n_experts, dim, moe_inter_dim] BF16
routed_w2_t [n_experts, moe_inter_dim, dim] BF16
routed_w3_t [n_experts, dim, moe_inter_dim] BF16
```

验证 shape/dtype 只读取 tensor metadata/view，不 clone 全 tensor。

### 4.5 V1 reader

把当前 `_load_cached_expert()` 的 v1 逻辑迁入 helper：

```python
def load_v1_expert(layer_id, expert_id, *, device) -> tuple[Tensor, Tensor, Tensor]: ...
```

保持现有语义：

- CPU 返回独立 contiguous clone；
- 非 CPU 返回 `.to(device).contiguous()`；
- 缺少该 expert 的任一 key 时视为 cache corruption，不能静默回 checkpoint；
- 完整 layer 文件不存在时才允许 WeightLoader 回退 checkpoint。

这一点比当前“缺 key 返回 None”更严格，避免损坏 cache 被部分掩盖。

### 4.6 V2 selected slice

核心接口：

```python
def copy_selected_into(
    layer_id: int,
    expert_ids: Sequence[int],
    *,
    out_w1: torch.Tensor,
    out_w2: torch.Tensor,
    out_w3: torch.Tensor,
) -> bool:
    """Return False only when the layer cache file does not exist."""
```

实现顺序：

1. `_inspect_layer()`；
2. v1：逐 expert tensor copy，保持兼容；
3. v2：取得三个 `get_slice()` object；
4. 按原始 `expert_ids` 顺序创建 18 个 selected views；
5. 在写 output 前完成所有 selected view 的 shape/dtype 校验；
6. 依次 copy 到 `out_w1/out_w2/out_w3` 对应 slot；
7. 返回 True。

必须保持 duplicate/任意顺序语义；不能为 I/O 排序后忘记恢复 slot 顺序。未来若按 expert ID 排序读取，
需要显式维护 `(expert_id, original_slot)`。

禁止：

```python
handle.get_tensor("routed_w1_t").clone()
```

decode v2 路径只能 slice selected expert。

### 4.7 V2 single expert

为了保持 `WeightLoader.get_moe_routed_expert()` 语义，提供：

```python
def load_expert(layer_id, expert_id, *, device) -> tuple[Tensor, Tensor, Tensor] | None: ...
```

v2 CPU 路径从三个 lazy slice 取得单 expert，再 clone 48 MiB 为独立 contiguous tensor；该接口主要用于
测试、converter fallback 和现有内部调用。生产 selected decode 应直接调用 `copy_selected_into()`，避免
创建 6 组中间 clone。

### 4.8 V2 prefill full pack

提供两个显式阶段接口，不用 bool 参数隐藏大内存行为：

```python
def load_packed_clone(layer_id, *, device) -> tuple[Tensor, Tensor, Tensor] | None: ...

def load_packed_mmap(layer_id) -> tuple[Tensor, Tensor, Tensor] | None: ...
```

规则：

- `load_packed_clone()`：三个完整 tensor 各 clone 一次；Phase 3 使用；
- `load_packed_mmap()`：返回只读 mmap-backed full views；Phase 4 使用；
- v1 对两者返回 None，让 WeightLoader 走当前逐 expert pack fallback；
- mmap 接口只允许 CPU；
- 两个接口都必须校验 exact shape、BF16 和 contiguous；
- 不把 mmap tensor 缓存成 backend-lifetime 普通 layout weight。

Phase 3 和 Phase 4 不同时保留两条生产分支。Phase 4 验证通过后，生产 v2 prefill 只走 mmap；packed clone
可以保留为内部诊断 helper或删除，避免运行时配置分叉。

### 4.9 Close

```python
def close(self) -> None:
    # idempotent
    # close every safe_open handle
    # clear layer info and handle maps
```

Reader 不拥有 WorkerBackend、DeviceTensor 或 selected output tensors。

## 5. `serving/convert_packed_expert_cache.py`

新增独立 v2 converter。现有 `serving/convert_expert_cache.py` 保持代码、CLI 和 format v1 输出完全不变，
继续作为已验证的 v1 cache 生成工具。v2 完整验证前，不在旧 converter 中增加 version 分支。

### 5.1 Converter 输出收敛到 v2

修改：

```python
FORMAT/VERSION/filename helper → 从 serving.expert_cache import
```

新 converter 只输出 format v2。Loader 负责读取 v1/v2，两个 converter 各自只维护一种输出格式。

### 5.2 CLI

保留：

```text
--checkpoint
--weight-index
--output
--layers
--overwrite
--profile
```

新 converter 不提供 `--experts`。Format v2 的 layer file 必须包含全部 256 experts；partial expert cache
不能补零或伪装成完整 packed cache。现有 v1 converter 的 `--experts` 行为保持不变。

### 5.3 单层构建 helper

抽出可用 small config 单测的函数：

```python
def build_packed_layer(
    loader: DeepSeekV4WeightLoader,
    layer_id: int,
    *,
    config: DeepSeekV4FlashConfig,
) -> dict[str, torch.Tensor]: ...
```

实现：

1. 分配三个 exact-shape BF16 CPU tensor；
2. 循环 expert_id；
3. 加载/反量化/transpose 单 expert；
4. copy 到三个 pack 的对应 first-dimension slot；
5. 每个 expert 后释放 checkpoint routed layout；
6. 返回三个 packed tensors；
7. 最后统一 shape/dtype/contiguous 校验。

严禁多层并行转换；单层 pack 已约 12 GiB。

### 5.4 原子写入

每层：

```text
build full layer
  → save_file(temp_path)
  → reopen temp and validate three keys/shape/dtype
  → os.replace(temp_path, final_path)
  → update manifest atomically
```

异常时删除本次 temp file，不修改已有 final file 和 manifest entry。

Manifest 写入也使用同目录 temp + replace，防止中途中断留下半个 JSON。

### 5.5 Existing output policy

- 输出目录无 manifest 且非空：拒绝，避免混入未知文件；
- manifest version=1：拒绝原地混写，要求新目录；
- manifest version=2 且 source/config 一致：允许 skip/overwrite；
- source checkpoint 或维度不一致：拒绝；
- `--overwrite` 只覆盖明确选中的 layer 文件。

推荐远端使用新目录，例如：

```text
/data/wuzhifeng/dsv4_bf16_packed_expert_cache
```

不要直接覆盖当前 `/data/wuzhifeng/dsv4_bf16_expert_cache`，直到 v2 完整验证通过。

## 6. `serving/weight_loader.py`

### 6.1 初始化

新增：

```python
self._expert_cache = ExpertCacheReader(
    self.expert_cache_dir,
    config=self.config,
    profile_callback=self._record_profile,
)
```

具体命名可以是 `LayerExpertCacheReader`。当 `expert_cache_dir=None` 时 reader disabled，不访问 manifest。

Expert cache handles 从 WeightLoader 通用 `_file_handles` 分离给 reader；checkpoint handles 继续由原 map
管理。这样 close ownership 清晰，也便于测试 decode 只调用 slice。

### 6.2 `get_moe_routed_expert()`

目标顺序：

```text
validate ids/device
  → expert_cache.load_expert()
  → 命中返回
  → layer cache file 不存在才回退 checkpoint
```

删除/替换当前 `_load_cached_expert()`，不要保留两套 cache detection。

### 6.3 `get_layer_moe_selected_experts()`

保持 output allocation 和 `HostStagingTensor` 不变。

新流程：

```python
ids = normalize(...)
allocate selected_w1/w2/w3
if self._expert_cache.copy_selected_into(...):
    record profile
else:
    current checkpoint fallback loop
return HostStagingTensor(..., DECODE_SELECTED)
```

注意：

- v2 命中时不能再调用 `get_moe_routed_expert()` 六次；
- selected output 仍是普通 contiguous Host tensor；
- slot 名称仍为 `w1_t/w2_t/w3_t`；
- selected ID 原顺序不变；
- H2D bytes 仍为 288 MiB/层。

### 6.4 `get_layer_moe_routed_pack()` Phase 3

优先：

```python
packed = self._expert_cache.load_packed_clone(layer_id, device=target)
if packed is not None:
    wrap three tensors as PREFILL_ROUTED
    return
```

v1 或无 layer file 时继续当前 256-expert loop。

保留当前 `release_each_expert` 参数和 Runner 调用不变，避免把无关 API 清理混入性能改造；该参数当前无效，
可在后续独立重构中删除。

### 6.5 `get_layer_moe_routed_pack()` Phase 4

真实 ChipWorker mmap H2D smoke 通过后：

```python
packed = self._expert_cache.load_packed_mmap(layer_id)
```

直接用 mmap tensors 构造 HostStagingTensor。不得调用：

```text
tensor.clone()
tensor.contiguous()  # 若已经 contiguous，避免潜在 copy
tensor.to(cpu)
```

校验失败直接报错，不退回 clone 或 v1；否则性能行为会静默变化。

### 6.6 Profile

保留现有兼容指标，并增加：

```text
expert_cache.v1.expert_load
expert_cache.v2.inspect
expert_cache.v2.selected_slice_copy
expert_cache.v2.prefill_clone
expert_cache.v2.prefill_mmap_view
```

`selected_experts.build` 继续覆盖完整 selected output build，便于 v1/v2 A/B。

Prefill mmap 后，原 `layer.values.routed_pack` 可能接近 metadata 时间，而 page fault/I/O 转移到
`layer.materialize`。必须结合 `prefill.total` 判断，不把计时项迁移误认为收益。

### 6.7 Release/close

```python
def close(self):
    self._expert_cache.close()
    self.release()
```

或者在 `release(name=None)` 内调用 reader close，但必须保证：

- `release_prefix()` 不关闭 expert cache handles；
- `close()` 幂等；
- Runner close 顺序仍先 backend 再 WeightLoader，避免 mmap 在同步 H2D 之前关闭；
- 异常路径也关闭 reader。

## 7. `tests/test_expert_cache.py`

使用 `_small_config()` 等价的小尺寸配置，例如：

```text
n_experts=3
activated=2
dim=4
inter=3
```

测试矩阵：

### 7.1 Detection/manifest

- manifest v2 + 三 packed keys → v2；
- 无 manifest + 三 packed keys → v2；
- 无 manifest + per-expert keys → legacy v1；
- partial packed keys → error；
- mixed v1/v2 keys → error；
- manifest config mismatch → error；
- manifest layer file missing → error；
- 整个 layer 不在 cache → return None/fallback signal。

### 7.2 Lazy selected path

- selected `[2, 0]` 输出 slot 顺序正确；
- duplicate `[1, 1]` 正确；
- 只取得 selected slices；
- 不调用 full packed clone；
- 输出 exact shape/dtype/contiguous；
- 任何 slice 校验失败时，不产生半写 output；
- v1/v2 selected 逐元素一致。

可用 fake handle 包装/spy `get_slice` 和 `get_tensor` 调用，明确断言 decode v2 不调用 full
`get_tensor().clone()`。

### 7.3 Full pack

- packed clone 返回独立 storage；
- packed mmap 返回 safetensors-backed view；
- 两者逐元素一致；
- mmap exact shape/dtype/contiguous；
- v1 full pack helper返回 None，由 WeightLoader fallback。

### 7.4 Close

- close 关闭每个 handle 一次；
- repeated close harmless；
- close 后访问明确失败或重新打开策略必须固定，不允许悬空 view 被 reader 继续返回。

## 8. `tests/test_convert_packed_expert_cache.py`

为了不依赖 FLASH 大 shape，直接测试 `build_packed_layer(..., config=small_config)`：

- 两/三个 experts 正确写入第一维；
- w1/w2/w3 shape/dtype/contiguous；
- v2 manifest 内容；
- partial experts 被拒绝；
- existing v1 output dir 被拒绝；
- v2 matching dir 支持 skip；
- overwrite 只修改选中 layer；
- save 后 reopen 校验；
- save/validation failure 不覆盖 final；
- manifest 原子更新；
- temp file 在失败时清理。

不在单测中创建真实 12 GiB tensor。

## 9. `tests/test_weight_loader.py`

保留现有 `test_loader_uses_layer_expert_cache_for_expert_selected_and_pack()` 作为 v1 回归，并新增 v2：

```text
test_loader_uses_v2_packed_cache_for_prefill
test_loader_uses_v2_lazy_slices_for_selected_decode
test_loader_v2_selected_preserves_id_order
test_loader_v2_single_expert_matches_slice
test_loader_rejects_corrupt_v2_cache
test_loader_closes_expert_cache_handles
```

测试还需断言：

- v2 selected 不增加普通 `_layout_cache_bytes`；
- v2 full pack 仍返回 `PREFILL_ROUTED`；
- v2 selected 仍返回 `DECODE_SELECTED`；
- slot 名称不变；
- checkpoint 中放入不同的负值，确认 v2 cache 命中而不是静默回 checkpoint；
- layer cache 文件不存在时仍能正确回 checkpoint；
- cache 文件存在但损坏时必须失败，不允许回 checkpoint。

## 10. 分阶段实施顺序

### Stage 0：冻结基线与约束（已完成）

1. 保留现有 43 层 S=1/S=1024 profile 数据；
2. 记录当前 v1 decode selected load/build 基线；
3. 确认新 cache 使用独立目录，当前 v1 cache 不覆盖；
4. 确认远端磁盘能临时容纳验证层或完整 v2 cache。

完成门槛：无代码变化，只明确 A/B 口径和输出目录。

#### Stage 0 实际结果

Stage 0 基于 commit `cc82384`、当前 format v1 cache 和无 L2 swimlane 配置完成。本阶段未修改生产代码。

Prefill 基线直接复用 `prefill_routed_expert_prefetch_plan.md` 已采集的数据，不重复占用 NPU：

| 指标 | 43 层 S=1 | 43 层 S=1024 |
|---|---:|---:|
| prefill total | 222.737 s | 290.728 s |
| routed Host pack 总时间 | 150.291 s | 163.133 s |
| routed Host pack 平均/层 | 3.495 s | 3.794 s |
| expert cache load 总时间 | 74.839 s | 81.438 s |
| materialize/H2D 上界总时间 | 19.301 s | 19.628 s |

对应远端任务：

```text
S=1 pass:
task_20260712_001446_282807929561

S=1024 first run 507018:
task_20260712_002235_293845326

S=1024 exact rerun pass, output=(1, 1024, 4, 4096), BF16, finite:
task_20260712_002905_30137829220
```

V1 decode selected 基线使用 5 层、S=1、3 decode steps、no-head、profile、无 L2 swimlane：

```text
task_20260712_011243_35155832997
exit=0
```

15 个 layer-step 样本：

| 指标 | 全部 15 样本 | Warm step 2–3（10 样本） |
|---|---:|---:|
| selected experts build 平均 | 98.954 ms/层 | 95.680 ms/层 |
| selected experts build min–max | 65.788–139.141 ms/层 | — |
| 6 experts cache load 平均 | 50.790 ms/层 | 48.684 ms/层 |
| selected post materialize 平均 | 11.771 ms/层 | 12.411 ms/层 |
| 5 层 decode total | cold 35.299 s | 680.387 ms/step |

Cold decode total 包含 selected kernel 首次编译/runtime 初始化，不作为 v1/v2 selected cache 性能比较值。
Stage 3 重点比较 warm `selected_experts.build`、`expert_cache.load`、post materialize 和相同 5 层 decode
total。

远端容量检查：

```text
/data filesystem size       7,634,757,242,880 bytes
available                   1,774,167,101,440 bytes = 1.614 TiB
current v1 cache              554,054,041,635 bytes = 516.003 GiB
planned v2 full cache       about the same 516 GiB
estimated remaining        about 1.110 TiB
```

独立 v2 目录当前不存在，Stage 2/3 固定使用：

```text
/data/wuzhifeng/dsv4_bf16_packed_expert_cache
```

现有 v1 目录保持只读基线，不覆盖、不原地转换：

```text
/data/wuzhifeng/dsv4_bf16_expert_cache
```

Stage 0 完成判断：

- prefill 基线已冻结；
- v1 decode selected 基线已冻结；
- v1/v2 目录边界已确定；
- 磁盘可同时容纳完整 v1/v2 cache，并保留约 1.11 TiB 余量；
- 可以进入 Stage 1。

### Stage 1：公共格式与 v1 reader 迁移（已完成）

5. 新增 `serving/expert_cache.py` 常量、manifest 和 file detection；
6. 把当前 v1 cache load 迁入 reader；
7. WeightLoader 委托 reader，但生产行为仍完全是 v1；
8. 增加 v1/detection/close 单测。

完成门槛：完整本地测试通过，远端当前 v1 cache 的 1 层 prefill/decode 不变。

#### Stage 1 实际结果

实际修改：

```text
new serving/expert_cache.py
modify serving/weight_loader.py
new tests/test_expert_cache.py
modify tests/test_weight_loader.py
```

`ExpertCacheReader` 已建立以下边界：

- 公共 format/version/packed-key/file-name 常量；
- manifest v1/v2 metadata 校验；
- legacy 无 manifest v1 文件检测；
- layer file format 和 keys 缓存；
- v1 单 expert clone/materialize；
- `expert_cache.load` 原 profile 名称保持不变；
- expert-cache safe_open handles 独立 ownership 和幂等 close；
- 完整 v2、partial v2 和 mixed v1/v2 keys 检测；
- v2 tensor loading 暂不实现，符合 Stage 1 范围。

WeightLoader 当前生产 selected/prefill 流程仍通过原有循环消费 v1 single expert；Runner、Backend、models、
runtime types 和 converter 均未修改。

本地验证：

```text
python -m compileall -q serving tests       pass
pytest -q tests/test_expert_cache.py tests/test_weight_loader.py
25 passed
pytest -q
284 passed
git diff --check                           pass
```

远端使用当前 v1 cache 执行单层 S=1、no-head、prefill + 3 decode、profile、无 L2 swimlane：

```text
task_20260712_012332_364687514276
prefill output=(1, 1, 4, 4096), BF16, finite
decode step 1/2/3 output=(1, 1, 4, 4096), BF16, finite
warm decode total=175.689 ms, 141.070 ms
expert_cache.load=1265.289 ms/256 experts during prefill
expert_cache.load=30.732–44.298 ms/6 experts during decode
selected_experts.build=66.275–107.442 ms/层
device lock released
exit=0
```

Stage 1 完成判断：v1 数值、profile、fallback 和资源生命周期未回归，可以进入 Stage 2 独立 packed
converter 实现。

### Stage 2：Format v2 converter（已完成）

9. 新增 `convert_packed_expert_cache.py` 输出三个 packed tensors；
10. 新 converter 不提供 partial `--experts`，旧 converter 保持不变；
11. 实现 temp save、reopen validation 和 atomic replace；
12. 新增 converter 单测；
13. 本地 small config 生成/读取 round trip。

完成门槛：小 tensor v2 与 v1 逐元素一致；不接入 serving production v2。

#### Stage 2 实际结果

实际新增：

```text
serving/convert_packed_expert_cache.py
tests/test_convert_packed_expert_cache.py
```

现有 `serving/convert_expert_cache.py` 未修改，继续只生成 format v1。

新 packed converter 已实现：

- 复用 `serving.expert_cache` 的 format v2、packed keys 和 layer filename 常量；
- `--layers` 支持单层、逗号列表和 inclusive range；
- 不提供 `--experts`，每个输出 layer 必须包含全部 experts；
- 单层三个 exact-shape contiguous BF16 packed tensors；
- manifest v2 的 source/config/layout/layer metadata；
- 非空无 manifest 目录拒绝；
- v1/mismatched manifest 目录拒绝；
- matching v2 layer 支持 skip，`--overwrite` 支持明确覆盖；
- layer safetensors temp save、reopen validation、atomic replace；
- manifest temp write、atomic replace；
- 写入失败不覆盖已有 final layer，temp file 自动清理；
- 多层始终串行转换。

本地 small-config round trip 覆盖 2 layers、2 experts、dim=4、inter=3，验证 packed 第一维 expert 顺序、
w1/w2/w3 transpose layout、manifest metadata 和 safetensors metadata。

验证结果：

```text
python -m compileall -q serving tests                         pass
pytest -q tests/test_convert_packed_expert_cache.py \
  tests/test_expert_cache.py tests/test_weight_loader.py
32 passed
pytest -q                                                     291 passed
git diff --check                                              pass
python -m serving.convert_packed_expert_cache --help          pass
```

Stage 2 按计划不生成远端真实 12 GiB layer，也不让 WeightLoader 读取 v2 tensor。远端 layer 0 转换、decode
lazy slice、RSS 和 selected H2D 验证属于 Stage 3，避免在 reader 尚不能消费 v2 时提前创建生产实验 cache。

Stage 2 完成判断：独立 converter 和可回退输出边界已建立，可以进入 Stage 3 decode lazy slice。

### Stage 3：Decode lazy slice 优先接入（已完成）

14. Reader 实现 v2 `copy_selected_into()`；
15. WeightLoader selected decode 优先使用 v2 slice；
16. 增加 spy 测试，证明不 clone full packed tensor；
17. 转换远端 layer 0 到独立 v2 目录；
18. 运行固定 selected IDs 和单层连续 decode；
19. 记录 Host RSS、selected build 和 H2D bytes。

完成门槛：

- v1/v2 selected tensor 逐元素一致；
- decode H2D 仍约 288 MiB/层；
- RSS 不出现接近 12 GiB 的单层跳升；
- decode selected load/build 无明显回退；
- 单层连续 decode exit=0。

未满足时停止，不接入 prefill v2。

#### Stage 3 实际结果

实际修改：

```text
modify serving/expert_cache.py
modify serving/weight_loader.py
modify tests/test_expert_cache.py
modify tests/test_weight_loader.py
```

实现内容：

- v2 三 packed tensor 使用 `get_slice().get_shape()/get_dtype()` 做 metadata 校验，不读取完整 tensor；
- `load_expert()` 支持单 expert lazy slice clone，供兼容接口和 Stage 3 prefill fallback；
- 新增 `copy_selected_into()`，一次取得三个 PySafeSlice，只构造 selected expert views；
- selected IDs 的顺序和 duplicate 语义保持；
- 写 output 前完成全部 output/slice shape、dtype、contiguous 校验，避免半写；
- v2 selected profile 为 `expert_cache.v2.selected_slice_copy`；
- WeightLoader 命中 v2 selected path 时不再调用 6 次 `load_expert()`；
- v1 文件和无 cache layer 继续走原 per-expert/checkpoint fallback。

本地验证：

```text
Stage 1–3 定向测试                         38 passed
完整 pytest                               297 passed
python -m compileall -q serving tests      pass
git diff --check                           pass
```

远端 layer 0 通过独立 CPU converter 生成：

```text
path=/data/wuzhifeng/dsv4_bf16_packed_expert_cache/layer_000_experts.safetensors
logical_bytes=12,884,901,888
directory_bytes=12,884,907,007
experts=256
build=32.719 s
save + reopen validation=9.764 s
manifest format=dsv4_bf16_layer_experts version=2 layers=[0]
```

Converter 是 CPU-only 工具。首次通过 `task-submit` 调用时，wrapper 自动追加 `--device 3`，argparse 在
转换前退出，未写 layer 文件；随后改为在 `source set_env.sh` 后直接运行 module。后续 converter 命令不再
通过 task-submit，只有 NPU model validation 使用 task-submit。

Host-only v1/v2 固定 IDs `[0,1,2,3,4,5]` 对比：

```text
v1/v2 w1/w2/w3 selected tensors exact equal
selected shapes:
  w1 [6, 4096, 2048]
  w2 [6, 2048, 4096]
  w3 [6, 4096, 2048]
RSS before=1,081,753,600 bytes
RSS after =1,415,122,944 bytes
RSS delta =  333,369,344 bytes ≈ 318 MiB
v2 selected_slice_copy=65.556 ms
selected_experts.build=66.002 ms
```

RSS 增量接近 288 MiB selected output 加少量 file-backed pages/runtime overhead，没有接近 12 GiB 的整包
跳升。

远端 Worker 单层 S=1、no-head、prefill + 3 decode、profile、无 L2 swimlane：

```text
task_20260712_014224_391050320877
prefill output=(1, 1, 4, 4096), BF16, finite
decode 1/2/3 output=(1, 1, 4, 4096), BF16, finite
selected_slice_copy=147.587, 100.315, 40.384 ms
selected_experts.build=147.736, 100.428, 40.535 ms
selected post materialize=10.363, 8.633, 9.255 ms
decode total=10026.024 ms cold, 129.830 ms warm, 71.601 ms warm
device lock released
exit=0
```

三块 selected tensor 的实际 logical bytes：

```text
3 × 6 × 4096 × 2048 × BF16 = 301,989,888 bytes = 288 MiB
```

WorkerBackend 对三块 `DECODE_SELECTED` staging 各执行一次 exact-size H2D，因此 decode selected H2D 保持
301,989,888 bytes/层，没有读取或上传完整 12 GiB pack。

与 Stage 0 v1 基线方向性比较：

```text
v1 warm selected build (5-layer aggregate)   95.680 ms/层
v2 layer-0 warm selected build               100.428, 40.535 ms/层
v1 layer-0 warm decode total                 175.689, 141.070 ms
v2 layer-0 warm decode total                 129.830, 71.601 ms
```

单层/单 session 样本不作为稳定性能结论，但确认 v2 decode 没有明显回退，并具有减少 per-expert clone 管理
开销的方向性收益。

Stage 3 临时 prefill fallback 会对 256 experts 逐个执行 v2 slice clone，再拼 full pack：

```text
v2 expert_slice=3768.104 ms/256
v2 prefill total=20.104 s
Stage 1 v1 prefill total=14.152 s
```

这不是 Stage 4 packed prefill 实现，且相对 v1 明显回退。因此当前 v2 目录只完成 decode lazy-slice 验证，
不能作为生产 prefill cache；Stage 4 必须接入三个完整 packed tensor clone，删除 256-slice fallback 后才能
继续扩大转换层数。

Stage 3 完成判断：selected 精度、RSS、288 MiB H2D、Worker dispatch 和 close 均通过，可以进入 Stage 4；
Stage 4 完成前保持生产命令使用 v1 cache。

### Stage 4：Prefill packed clone（已完成）

20. Reader 实现 `load_packed_clone()`；
21. WeightLoader v2 prefill 使用三个 full clone；
22. 本地验证 v1/v2 full pack 逐元素一致；
23. 远端 layer 0 单层 prefill；
24. 转换 layer 0–4，运行 5 层四 variant；
25. profile routed pack、materialize、prefill total 和 RSS。

完成门槛：

- 精度/finite 通过；
- 768-key/per-expert pack loop 不再出现在 v2 profile；
- routed Host build 明显低于 v1；
- decode Stage 3 结果保持不变。

#### Stage 4 实际结果

实际修改：

```text
modify serving/expert_cache.py
modify serving/weight_loader.py
modify tests/test_expert_cache.py
modify tests/test_weight_loader.py
```

实现内容：

- Reader 新增 `load_packed_clone()`，v2 layer 一次取得三个完整 packed tensor 并 clone；
- profile 名为 `expert_cache.v2.packed_clone`；
- layer 缺失或为 v1 时返回 `None`，WeightLoader 保持原 per-expert/checkpoint fallback；
- WeightLoader prefill 命中 v2 后直接包装三块 clone 为 `PREFILL_ROUTED` staging；
- v2 prefill 不再分配空 pack、循环 256 experts、逐 slot `copy_`；
- selected decode 继续使用 Stage 3 的 lazy slice 路径。

本地验证：

```text
Stage 1–4 Reader/WeightLoader 定向测试       35 passed
完整 pytest                                301 passed
python -m compileall -q serving tests       pass
git diff --check                            pass
```

测试覆盖 v2 full clone 的 shape/dtype/contiguous/profile、v1 fallback、v1/v2 full pack 逐元素一致，以及
WeightLoader spy 确认 v2 routed pack 不调用 `load_expert()`。

远端真实 layer 0 host-only 全量对比：

```text
v1/v2 256 experts × w1/w2/w3 exact equal
packed shapes:
  w1 [256, 4096, 2048]
  w2 [256, 2048, 4096]
  w3 [256, 4096, 2048]
profile events=[expert_cache.v2.packed_clone]
RSS before=1,052,217,344 bytes
RSS after packed clone=25,838,964,736 bytes
RSS delta=24,786,747,392 bytes ≈ 23.084 GiB
```

单层 logical tensor 是 12 GiB；full clone 时 safetensors mmap pages 和匿名 clone 同时计入 RSS，因此峰值约
为两份 packed tensor。该结果符合 Stage 4 clone 语义，但也是 Stage 5 mmap direct H2D 需要解决的主要问题。
后续逐 expert 读取 v1 做 exact compare 会继续触碰 v1 mmap pages，因此 compare 尾部 RSS 不作为生产路径峰值。

远端 Worker 单层 S=1、no-head、prefill + 3 decode、profile、无 L2 swimlane：

```text
task_20260712_015020_402121724339
prefill output=(1, 1, 4, 4096), BF16, finite
decode 1/2/3 output=(1, 1, 4, 4096), BF16, finite
layer.values.routed_pack=295.638 ms
expert_cache.v2.packed_clone=294.313 ms/1
layer.materialize=948.629 ms
prefill.total=13.261 s (包含首次 kernel compile)
warm decode.total=414.571, 162.935 ms
device lock released
exit=0
```

Profile 不再出现 Stage 3 临时 fallback 的 `expert_cache.v2.expert_slice=.../256`。Decode 仍出现
`expert_cache.v2.selected_slice_copy`，确认 Stage 3 路径保持。

远端继续串行转换 layer 1–4；每层 logical bytes 都是 12,884,901,888，build 为 33.962–40.014 s，save +
reopen validation 为 7.851–9.527 s。v2 目录 manifest 已包含 layer 0–4。

五层四 variant Worker 验证：

```text
task_20260712_020009_414805812331
layer 0 swa_hash routed_pack=223.354 ms packed_clone=222.842 ms/1
layer 1 swa_hash routed_pack=242.825 ms packed_clone=242.466 ms/1
layer 2 csa_hash routed_pack=273.016 ms packed_clone=272.472 ms/1
layer 3 hca_topk routed_pack=234.141 ms packed_clone=233.631 ms/1
layer 4 csa_topk routed_pack=245.980 ms packed_clone=245.624 ms/1
prefill output=(1, 1, 4, 4096), BF16, finite
decode output=(1, 1, 4, 4096), BF16, finite
device lock released
exit=0
```

五层 v2 routed Host pack 平均为 243.863 ms/层。相对 Stage 0 冻结的 v1 43 层 S=1 平均 3.495 s/层，
方向性下降约 93.0%；H2D/materialize 仍约 316 ms/层（layer 0 首次初始化为 931 ms），符合 Stage 4 只优化
Host build、不改变完整 packed H2D bytes 的设计。

Stage 4 完成判断：精度/finite、四 variant、单层连续 decode、五层 dispatch/close 均通过；每层只有一次
`packed_clone`，768-key/per-expert pack loop 已从 v2 prefill profile 消失，routed Host build 明显降低，
decode Stage 3 路径保持。可以进入 Stage 5 mmap direct H2D；在 Stage 5 证明稳定前，Stage 4 clone 是 v2
prefill 的正确性基线。

### Stage 5：Prefill mmap direct H2D（已评估，未接入生产）

26. 新增独立真实 ChipWorker smoke：mmap full tensor直接作为 `copy_to()` source；
27. 验证 copy 后 device tensor 与 mmap Host tensor 一致；
28. WeightLoader v2 prefill 从 clone 切换到 mmap full view；
29. 远端重复 1 层和 5 层；
30. 比较 values/materialize/total/RSS，确认只是计时迁移还是端到端收益。

完成门槛：

- mmap direct H2D 数值正确；
- handle 生命周期无 use-after-close；
- Host anonymous RSS 明显下降；
- prefill total 不回退；
- 异常/close 正常释放。

如果 direct mmap H2D 不稳定或没有端到端收益，保留 packed clone 作为生产 v2，不强行进入 Stage 6。

#### Stage 5 实际结果

实验期间修改并在性能门槛失败后回退：

```text
temporary modify serving/expert_cache.py
temporary modify serving/weight_loader.py
temporary modify tests/test_expert_cache.py
temporary modify tests/test_weight_loader.py
```

实验 Reader 新增 `load_packed_mmap()`，直接返回 safetensors CPU mmap tensors，严格校验完整 shape、
BF16、CPU 和 contiguous，不执行 `clone()`、`.to(cpu)` 或额外 `.contiguous()`。Profile 名称为
`expert_cache.v2.prefill_mmap_view`。WeightLoader 曾在实验分支切换到该接口完成真实单层和五层验证；由于性能
门槛未通过，最终生产路径恢复为 Stage 4 `load_packed_clone()`，并移除无调用方的实验接口与测试，不在生产
代码保留不可达分支。未来 PyPTO/runtime 改善 Host mmap DMA 后可根据本节数据重新实现和评估。

本地验证：

```text
Stage 1–4 Reader/WeightLoader 定向测试       35 passed
完整 pytest                                301 passed
python -m compileall -q serving tests       pass
git diff --check                            pass
```

独立真实 ChipWorker smoke：

```text
task_20260712_020654_5136026837
routed_w1_t shape=[256, 4096, 2048]
BF16 bytes=4,294,967,296
mmap tensor 直接作为 ChipWorker.copy_to() source
copy_to 返回后关闭 handle、删除 source，再执行 D2H
首尾共 2,048 samples exact equal
device lock released
exit=0
```

该结果确认当前 `copy_to()` 对完整 mmap source 数值正确且同步；Runner 的正常生命周期更保守，始终保持
ExpertCacheReader handle 到 backend 完成并关闭之后。

实验性 mmap 生产路径单层 S=1、no-head、prefill + 3 decode、profile、无 L2 swimlane：

```text
task_20260712_020918_8748022728
prefill/decode output BF16 finite
routed_pack=0.704 ms
prefill_mmap_view=0.301 ms/1
materialize=1,746.954 ms
routed_pack + materialize=1,747.658 ms
prefill.total=12.328 s（含首次 kernel compile）
max RSS=15,507,660 KiB ≈ 14.789 GiB
exit=0
```

Stage 4 单层 clone 对应 `routed_pack + materialize=1,244.267 ms`。mmap 虽把 values 降到 metadata 时间，
但 page fault/DMA 转移到 materialize 后，暴露时间反而增加 503.391 ms，约 40.5%。Stage 4 host-only packed
clone 后 RSS 约 24.064 GiB，mmap 单层 Worker 最大 RSS 约 14.789 GiB；采集方式不完全相同，但方向上确认
去除了约一份匿名 12 GiB clone 的峰值压力。

实验性 mmap 五层四 variant：

```text
task_20260712_021112_10951327665
layer routed_pack=0.548–0.770 ms
layer materialize=699.734–1,681.069 ms
five-layer routed_pack + materialize=5,505.175 ms
prefill.total=45.690 s
prefill/decode output BF16 finite
max RSS=67,816,972 KiB ≈ 64.675 GiB
device lock released
exit=0
```

Stage 4 clone 五层对应：

```text
routed_pack + materialize=3,420.925 ms
prefill.total=41.875 s
```

因此 mmap 五层暴露的 Host/H2D 时间回退 2,084.250 ms，约 61.0%；完整 prefill 回退约 9.1%。五个保持
打开的 layer mmap 所触碰的 file-backed pages 也会累计到 RSS，mmap 并不能让完整 43 层文件页不占 Host
resident set。

按预设门槛，WeightLoader 已恢复 Stage 4 clone。恢复后远端单层最终确认：

```text
task_20260712_021431_14930932005
packed_clone=227.570 ms/1
routed_pack=227.997 ms
materialize=997.866 ms
routed_pack + materialize=1,225.863 ms
prefill/decode output BF16 finite
device lock released
exit=0
```

Stage 5 完成判断：mmap direct H2D 的数值、同步语义、handle 生命周期、单层/五层四 variant 和 RSS 均已
验证；虽然降低匿名 Host 峰值，但端到端性能门槛未通过。因此不把 mmap 接入生产，保留 packed clone 作为
v2 prefill 正确且更快的实现。Stage 6 若继续完整转换和验证，应使用当前 clone 生产路径。

### Stage 6：完整 43 层转换与验证（已完成）

31. 串行转换完整 43 层到新目录；
32. 校验 manifest、43 files、总 bytes 和每层 keys；
33. 运行 43 层 S=1、S=13；
34. 运行 43 层 S=1024，多轮处理偶发 507018；
35. 运行带 head generate prefill + decode；
36. 对比 v1/v2 decode selected 性能；
37. 采集完整 prefill profile 和 Host RSS。

完成门槛：完整精度/finite、资源关闭和性能目标通过。

#### Stage 6 实际结果

本地最终回归：

```text
python -m compileall -q serving tests       pass
pytest -q                                   301 passed
git diff --check                            pass
```

远端从已有 layer 0–4 继续串行转换 layer 5–42，未并行构建。转换进程 exit=0；新增层单层 build 为
31.764–60.163 s，save + reopen validation 为 7.776–19.835 s，每层 logical bytes 都是
12,884,901,888。独立 metadata 校验结果：

```text
[STAGE6_CACHE] PASS
manifest layers=0–42
files=43
total_bytes=554,050,796,320 bytes = 516.000014 GiB
keys/layer=3
shape/dtype=expected/BF16
temp_files=0
```

43 层 S=1、no-head、profile、无 L2 swimlane：

```text
task_20260712_030300_63703418429
output=(1, 1, 4, 4096), BF16, finite
prefill.total=117.360 s
routed_pack=15.746 s/43 layers, 366.182 ms/层
packed_clone=15.727 s/43 layers
materialize=27.393 s/43 layers
max RSS=575,415,256 KiB = 548.759 GiB
exit=0
```

相对 Stage 0 V1 S=1 基线：prefill 222.737 s → 117.360 s，方向性提升 47.3%；routed Host pack
150.291 s → 15.746 s，下降 89.5%。不同 session 的首次编译和 page-cache 状态可能影响总时间，但 Host pack
收益与每层单次 packed clone profile 一致。

43 层非 tile 对齐 S=13：

```text
task_20260712_030617_67090811155
output=(1, 13, 4, 4096), BF16, finite
prefill.total=122.873 s
routed_pack=15.308 s/43 layers
materialize=31.480 s/43 layers
max RSS=575,779,352 KiB = 549.106 GiB
exit=0
```

43 层 S=1024 首次运行即通过，未出现偶发 507018：

```text
task_20260712_031200_72525616419
output=(1, 1024, 4, 4096), BF16, finite
prefill.total=170.903 s
routed_pack=15.714 s/43 layers
materialize=29.670 s/43 layers
kernel=92.586 s/43 layers
max RSS=575,932,592 KiB = 549.252 GiB
exit=0
```

相对 Stage 0 V1 S=1024 prefill 290.728 s，方向性提升 41.2%。

完整 43 层带 head generate：

```text
task_20260712_031635_76333115876
prompt_tokens=5
generated_tokens=2
prefill.total=137.694 s, run_head=True
43-layer decode.total=47.005 s, run_head=True
selected build=2,131.954 ms/43 layers, 49.580 ms/层
max RSS=576,104,044 KiB = 549.416 GiB
exit=0
```

生成文本为空是两个生成 token 被 tokenizer 解码为空白/特殊内容；token count、head、sampling 和完整 decode
链均实际执行且无异常。

完整 43 层 V1/V2 decode A/B 使用相同 S=1、3 decode steps、no-head、profile、无 L2 swimlane配置：

```text
V1 task_20260712_032231_81578313191
V2 task_20260712_033628_92489621739
```

Warm step 2–3：

| 指标 | V1 | V2 | V2 变化 |
|---|---:|---:|---:|
| selected build 平均/层 | 83.784 ms | 44.402 ms | -47.0% |
| 43 层 decode step 2 | 8.981 s | 7.022 s | -21.8% |
| 43 层 decode step 3 | 8.927 s | 7.018 s | -21.4% |
| warm decode 平均 | 8.954 s | 7.020 s | -21.6% |

V1 的 6-expert cache load 为 41.520 ms/层；V2 selected slice copy 为 44.296 ms/层，但 V2 删除了后续六个
独立 expert tensor 的 Python 拼装/复制管理，总 selected build 仍明显降低。两边 prefill/decode 输出均为有限
BF16，设备锁均正常释放。

43 层实测最大 RSS 稳定在 548.759–549.416 GiB，确认当前保留全部 expert handles、触碰 516 GiB
file-backed pages、保留 14.672 GiB 固定 Host layouts，并持有当前层 12 GiB packed clone 的生命周期模型。
file-backed pages 可由 Linux 回收，但当前无压力运行中基本全部计入 RSS。

Stage 6 完成判断：43 层 cache 完整性、S=1、S=13、S=1024、四 variant、head、generate、decode selected、
资源释放和 profile/RSS 均通过；S=1024 未复现 507018。V2 prefill Host pack 和 warm decode 均较 V1 明显
改善，可以进入 Stage 7，基于新 profile 重新评估 prefetch，而不是沿用 V1 Host load 假设。

### 最终格式收敛（已完成）

- 删除 per-expert 旧格式 reader、manifest 字段、converter 和测试；
- 正式 converter 收敛为 `serving/convert_expert_cache.py`；
- Reader 只保留 `load_routed_pack()` 与 `copy_selected_into()`；
- 指定 cache 目录时强制要求当前 manifest schema，未声明层才允许回退 checkpoint；
- profile 收敛为 `expert_cache.routed_pack` 和 `expert_cache.selected_slice_copy`；
- manifest 保留 `version: 2` 以直接消费已完成验证的 516 GiB cache，不重新转换数据；
- 本地 299 tests 通过；远端 43 层 metadata、单层连续 decode、五层四 variant 通过。

### Stage 7：重新评估 prefetch（已完成）

38. 用最终 packed cache 数据更新 `prefill_routed_expert_prefetch_plan.md`；
39. 重新计算 Host load、materialize、kernel overlap；
40. 判断剩余 Host exposed time 是否值得实现 prefetch；
41. 记录最终实现决策。

实际结果：S=1/S=1024 cold 理论上限分别为 13.12%/9.01%；考虑 Host clone 与 H2D 的内存带宽竞争，
现实收益预计约为 4%–9%/2%–6%。Depth=1 会增加约 12 GiB Host anonymous peak，使完整模型最坏 RSS
从约 549 GiB 上升到约 561 GiB。综合收益、内存和线程/ownership复杂度，当前不实现 Host prefetch；只有
业务明确需要继续争取约 5% TTFT 时才作为独立优化重启。

## 11. 远端命令顺序

以下只定义形态，实际 v2 目录在实施时确认。

### 11.1 单层转换

```bash
python -m serving.convert_expert_cache \
  --checkpoint /data/wuzhifeng/dsv4_ckpt \
  --output /data/wuzhifeng/dsv4_bf16_packed_expert_cache \
  --layers 0 \
  --profile
```

### 11.2 单层验证

```bash
python serving/run_model.py -p a2a3 -d {} \
  --checkpoint /data/wuzhifeng/dsv4_ckpt \
  --expert-cache-dir /data/wuzhifeng/dsv4_bf16_packed_expert_cache \
  --max-layers 1 --no-head -s 1 --decode-steps 3 --profile
```

### 11.3 五层验证

```bash
python serving/run_model.py -p a2a3 -d {} \
  --checkpoint /data/wuzhifeng/dsv4_ckpt \
  --expert-cache-dir /data/wuzhifeng/dsv4_bf16_packed_expert_cache \
  --max-layers 5 --no-head -s 1 --decode-steps 1 --profile
```

### 11.4 完整验证

```bash
python serving/run_model.py -p a2a3 -d {} \
  --checkpoint /data/wuzhifeng/dsv4_ckpt \
  --expert-cache-dir /data/wuzhifeng/dsv4_bf16_packed_expert_cache \
  --max-layers 43 --no-head -s 1024 --decode-steps 0 --profile
```

Converter 是 CPU-only 工具，在 `source set_env.sh && cd dsv4` 后直接执行，不使用 task-submit。Model/NPU
验证通过 `task-submit --device auto --max-time 0 --run "..."` 执行。功能验证不开启 L2 swimlane；性能
A/B 也固定使用相同无-swimlane配置，除非另建 kernel 性能实验。

## 12. Profile 验收指标

### Decode

```text
expert_cache.v2.selected_slice_copy
selected_experts.build
layer.selected_decode.post_materialize
STAGING_SELECTED H2D bytes
Host RSS delta
```

目标：

- selected Host/H2D 保持约 288 MiB/层；
- 不出现 12 GiB RSS delta；
- selected build 不比 v1 稳定回退超过 10%；
- 多 decode step 无 file-handle 或 allocation 增长。

### Prefill packed clone

```text
expert_cache.v2.prefill_clone
layer.values.routed_pack
layer.materialize
prefill.total
Host RSS peak
```

目标：删除约 81.7 秒/43 层的旧二次 pack work 是理论上界，不设为硬门槛；先以 routed Host build 至少
下降 30%、完整 prefill 明显下降作为第一轮验收线。

### Prefill mmap

```text
expert_cache.v2.prefill_mmap_view
layer.values.routed_pack
layer.materialize
prefill.total
anonymous RSS / file RSS / page cache
```

目标：

- anonymous Host pack 峰值下降约 12 GiB；
- 即使 values 时间转移到 materialize，prefill total 不能回退；
- 完整 wall time 相对 packed clone 有稳定收益才采用 mmap 生产路径。

## 13. 回退点

每个阶段保持可独立回退：

```text
Stage 1 失败 → 恢复 WeightLoader 内置 v1 reader
Stage 2 失败 → 删除/修复独立 v2 converter，当前 v1 converter 和 serving 不受影响
Stage 3 失败 → v2 不进入 decode，停止后续阶段
Stage 4 失败 → v2 不进入 prefill，保留 decode 实验结果但不设为生产目录
Stage 5 失败 → 保留 packed clone，不使用 mmap direct H2D
Stage 6 失败 → serving 继续指定当前 v1 cache directory
```

不要在验证过程中覆盖或删除当前 v1 cache。新目录通过完整验收并切换后，旧 cache 清理需要用户明确确认。

## 14. 最终验收标准

- Packed converter 只产生完整、原子、可校验的 format v2 layer files；
- Decode v2 只读取 selected six experts，不 clone full pack；
- Decode Host/H2D 仍约 288 MiB/层；
- Prefill 不再循环 256 experts 拼装 full pack；
- Prefill H2D 仍约 12 GiB/层，数值与 v1 一致；
- mmap 仅在端到端收益和生命周期验证通过后启用；
- 43 层 S=1/S=13/S=1024 和 generate prefill/decode 通过；
- close 后所有 safe_open handles 和 Worker owned allocations 释放；
- Runner、Backend、models 和 state 无 cache-format 分支；
- format v2 新 profile 完成，并据此重新决定是否实现 Host prefetch。

现有 `convert_expert_cache.py` 的删除不属于本计划完成条件。只有 v2 完整 43 层转换、prefill/decode 精度、
性能和回退验证全部通过后，才单独提出删除 v1 converter 和旧格式 reader 的清理方案。
