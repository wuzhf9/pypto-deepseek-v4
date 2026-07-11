# WorkerBackend 与 Device-resident 最新实现方案

## 1. 结论

当前代码已经完成 WorkerBackend 的架构前置工作：

- `DeepSeekV4Runner` 只接收注入的 `Backend`，不 import 或判断具体 backend。
- `Backend` 已覆盖 materialize、dispatch、Host 控制数据读取、最终输出导出和 state 生命周期。
- `DeepSeekV4StatePlan` 只提供 immutable aux 与 mutable state schema。
- `DirectStateStore` 已用 current/next 双缓冲验证 state binding 与 commit 语义。
- Embedding kernel 已直接输出 `[B, S, HC_MULT, HIDDEN]`，Runner 不再做 Host expand。
- 普通 non-routed runtime layout 已固定保存在 WeightLoader Host cache 中。

因此下一步不需要再次重构模型编排。实现重点是：

1. 给固定权重和 routed staging 补充稳定的逻辑类型。
2. 实现长期存活的 `ChipWorker`、DeviceTensor pool 和 Worker state store。
3. 让 WorkerBackend 按现有 Backend contract 完成 device materialize 与 dispatch。
4. 只保留 selected indices 和最终输出的必要 D2H。

完整 43 层、带 head、`max_seq_len=4096` 的静态峰值预算约为 27.33 GiB，满足 64 GB 显存。
详细计算见 `device_resident_memory_plan.md`。

## 2. 当前实测基线

最新 DirectBackend profile 使用：

```text
seq_len=1
max_layers=5
no_head
decode_steps=3
```

第 2、3 个 decode 均为 compile-cache hit：

| 指标 | Warm 平均 |
|---|---:|
| 5 层 decode total | 1678 ms |
| Embedding | 144 ms |
| 5 层 block | 1512 ms |
| 单层 block | 302 ms |
| pre-MoE materialize | 17.7 ms/层 |
| pre-MoE kernel | 104.5 ms/层 |
| selected expert load/build | 84.7 ms/层 |
| post-MoE kernel | 94.1 ms/层 |

Device resident 不会消除 selected expert Host load/build 和 selected pack H2D。预计完整模型 warm decode
收益为 10%–20%，乐观上限约 25%。第一版验收应以实测 H2D/D2H bytes 和 warm token latency 为准，
不能只用理论带宽判断。

## 3. 当前代码边界

### 3.1 已完成

```text
serving/backends/base.py
    Backend protocol + KernelCase + KernelBindings + step lifecycle

serving/backends/direct_backend.py
    Host materialize + direct dispatch + output boundary

serving/backends/direct_state_store.py
    Host current/next state pairs

serving/backends/factory.py
    concrete backend composition root；direct/worker 均可构造

serving/backends/worker_backend.py
    Worker 多层 prefill/decode/head + fixed/upload/output/staging device 生命周期

serving/runtime_types.py
    RuntimeWeight / HostStagingTensor / StepContext 公共类型

serving/runner.py
    backend-neutral model orchestration

serving/state.py
    DeepSeekV4StatePlan + LayerStateSchema + immutable aux cache

serving/run_model.py / serving/generate.py
    concrete backend construction and injection
```

### 3.2 WorkerBackend 的下一实现阻塞

公共 runtime value contract 已完成：固定 layout 返回 `RuntimeWeight`，full routed/selected pack 返回
`HostStagingTensor`，DirectBackend 会在 materialize 时无损解包。因此 Worker 已可区分：

- 固定 non-routed 权重；
- input IDs 与 immutable aux；
- full routed expert pack；
- selected expert pack；
- 前一个 kernel 的 DeviceTensor output。

WorkerBackend 的多层 prefill、selected decode 和 head 已经完成：24-byte indices D2H、selected expert
staging、pre-MoE→post-MoE intermediate residency、跨层 hidden 消费和连续 state swap 均已打通；
ratio 0/4/128、hash/top-k 以及 CSA/HCA/SWA 路径已在远端覆盖。下一阻塞是完整 43 层、长 prompt、显存
峰值及 Direct/Worker warm decode 性能 A/B 验证。完整 43 层的短序列和 S=13 已通过；S=4096 当前先被
既有 embedding/block kernel 的 runtime work-count 边界阻塞，尚未进入 device-resident 显存峰值阶段。

## 4. 目标依赖关系

```text
run_model.py / generate.py
        │ create_backend()
        ▼
DeepSeekV4Runner ────────────────┐
        │ Backend protocol       │ WeightLoader / StatePlan
        ▼                        │
DirectBackend                    │
WorkerBackend                    │
  ├── ChipWorker                 │
  ├── DeviceBufferPool           │
  ├── WorkerStateStore           │
  ├── fixed weight cache         │
  └── active-step upload cache   │
```

禁止依赖：

```text
runner.py → WorkerBackend / ChipWorker / DeviceTensor / DeviceBufferPool
state.py  → WorkerBackend / ChipWorker / DeviceTensor
weight_loader.py → ChipWorker / DeviceTensor
worker_backend.py → runner.py
```

## 5. 文件改动总表

### 5.1 新增文件

| 文件 | 内容 |
|---|---|
| `serving/runtime_types.py` | 固定权重、Host staging、step context 等 backend-neutral 类型 |
| `serving/backends/device_pool.py` | DeviceTensor allocation、复用、copy、统计和释放 |
| `serving/backends/worker_state_store.py` | Device current/next state pairs |
| `serving/backends/worker_backend.py` | ChipWorker resident backend |
| `tests/test_runtime_types.py` | runtime value 类型与 key 测试 |
| `tests/test_device_pool.py` | FakeChipWorker allocation/copy/reuse/close 测试 |
| `tests/test_worker_backend.py` | Worker materialize、dispatch、state、D2H 和失败清理测试 |

### 5.2 修改文件

| 文件 | 修改内容 |
|---|---|
| `serving/weight_loader.py` | 固定 getter 返回 `RuntimeWeight`；routed pack 返回 `HostStagingTensor` |
| `serving/backends/base.py` | 增加 step lifecycle 和显式 `KernelBindings` |
| `serving/backends/direct_backend.py` | unwrap 新 value 类型；实现 step no-op；适配 `KernelBindings` |
| `serving/backends/factory.py` | `worker` 分支创建 `WorkerBackend` |
| `serving/runner.py` | 只增加 backend-neutral `begin_step/end_step`，不增加 worker 分支 |
| `serving/profiler.py` | 输出 H2D/D2H、resident bytes、pool peak、cache hit/miss |
| `serving/run_model.py` | 透传 Worker runtime 配置与 `--enable-l2-swimlane` |
| `serving/generate.py` | 同步 Worker runtime 配置 |
| `tests/test_weight_loader.py` | fixed/staging 类型、key、Host cache identity 测试 |
| `tests/test_backend.py` | Direct contract 与新 KernelBindings/step contract 回归 |
| `tests/test_generate.py` | Worker 参数到 factory 的透传测试 |
| `tests/test_run_model.py` | Worker CLI composition 测试 |

### 5.3 第一版不修改

- `models/embedding.py`
- `models/block.py`
- `models/split_block.py`
- `models/head.py`
- `serving/state.py`

State schema 和四维 embedding output 已满足 WorkerBackend 需求。不得为 Worker 再增加第二套 Runner
或第二套 StatePlan。

## 6. `serving/runtime_types.py`

### 6.1 固定权重

```python
@dataclass(frozen=True)
class RuntimeWeightKey:
    name: str
    dtype: torch.dtype
    layout: str
    layout_version: int = 1
    padding_profile: str | None = None


@dataclass(frozen=True)
class RuntimeWeight:
    key: RuntimeWeightKey
    host_tensor: torch.Tensor
```

Key 不包含 device id。每个 WorkerBackend 实例只绑定一个 device，device scope 由实例隔离。

### 6.2 Routed staging

```python
class StagingKind(Enum):
    PREFILL_ROUTED = "prefill_routed"
    DECODE_SELECTED = "decode_selected"


@dataclass(frozen=True)
class HostStagingTensor:
    host_tensor: torch.Tensor
    kind: StagingKind
    slot: str
```

`slot` 使用 kernel-facing 语义，例如 `w1_t/w2_t/w3_t`。Pool key 为：

```text
(kind, slot, shape, dtype)
```

不能把 routed/selected pack 包装成 RuntimeWeight，否则会误进入固定 device cache。

### 6.3 Step context

```python
class StepKind(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class StepContext:
    kind: StepKind
    seq_len: int
    start_pos: int
```

Worker 用 step context 管理 input IDs 和 immutable aux 的短期 DeviceTensor cache。

## 7. `serving/weight_loader.py`

### 7.1 公开稳定身份

- 删除私有 `_RuntimeWeightKey`。
- 从 `runtime_types.py` 导入 `RuntimeWeightKey/RuntimeWeight`。
- `_layout_cache` 改为：

```python
dict[tuple[RuntimeWeightKey, str], RuntimeWeight]
```

- `_get_runtime_weight()` 返回 `RuntimeWeight`。
- Host cache hit 返回同一个 RuntimeWeight entry。
- `layout_cache_bytes` 按 `entry.host_tensor` 计算。

### 7.2 固定 getter

以下结构化 getter 的字段改为 `RuntimeWeight`：

- embedding/head；
- layer HC；
- attention common；
- compressor/indexer；
- gate、FFN norm；
- shared experts。

Runner 保留 wrapper，不得提前取 `.host_tensor`。

### 7.3 Routed expert

`get_moe_routed_expert()` 是 Host pack 的内部构建步骤，继续返回裸 Host tensor。若内部 transpose helper
返回 RuntimeWeight，必须立即 unwrap，且保持 `cache=False`。

最终 pack getter 返回：

```text
get_layer_moe_routed_pack()
  → HostStagingTensor(PREFILL_ROUTED, w1_t/w2_t/w3_t)

get_layer_moe_selected_experts()
  → HostStagingTensor(DECODE_SELECTED, w1_t/w2_t/w3_t)
```

普通 Host layout cache 仍固定缓存全部 non-routed 权重，不增加 LRU 或容量参数。

## 8. `serving/backends/base.py`

### 8.1 显式调用绑定

新增：

```python
@dataclass
class KernelBindings:
    tensors: Mapping[str, Any]
```

Direct 可只保存 tensors。Worker 在内部附加本次 dispatch 的 scratch、staging、new outputs 和 consumed
intermediates，避免依赖全局 `_pending_call` 隐式状态。

接口改为：

```python
def materialize(specs, values) -> KernelBindings: ...
def run(case, specs, bindings: KernelBindings) -> dict[str, Any]: ...
```

### 8.2 Step lifecycle

```python
def begin_step(context: StepContext) -> None: ...
def end_step() -> None: ...
```

Runner 必须用 `try/finally` 保证 end_step。Direct 不分配 step 资源，但会验证 begin/end 配对；Worker 在
end_step 释放 active raw upload cache。该接口是 backend-neutral，不得检查 `is_resident`。

## 9. `serving/backends/device_pool.py`

远端当前 API 已确认：

```python
ChipWorker(config)
worker.alloc_tensor(shape, dtype, init=host_tensor) -> DeviceTensor
worker.free_tensor(device_tensor)
worker.copy_to(device.data_ptr, host.data_ptr(), nbytes)
worker.copy_from(host.data_ptr(), device.data_ptr, nbytes)
worker.run(compiled, *args, config=config)
worker.close()
```

`DeviceTensor` 提供 `data_ptr`、`shape`、`dtype`、`nbytes`。

### 9.1 Allocation 分类

```text
FIXED_WEIGHT       直到 backend.close
STATE              直到 backend.close
ACTIVE_UPLOAD      当前 prefill/decode step
INTERMEDIATE       直到被下一个 kernel 消费或导出
SCRATCH            当前 dispatch
STAGING_ROUTED     prefill block dispatch，可复用 3 × 4 GiB buffer
STAGING_SELECTED   decode post dispatch，可复用 288 MiB buffer
```

所有 allocation 登记 owner、shape、dtype、nbytes、category 和 in-use 状态。

### 9.2 Pool 行为

- Fixed/state 使用显式 persistent allocation。
- Scratch/intermediate 按 `(shape, dtype)` 复用。
- Routed/selected staging 按 `(kind, slot, shape, dtype)` 复用，并通过 `copy_to()` 覆盖。
- Prefill 完成后默认释放 12 GiB full-routed staging；decode selected staging 可保留到 close。
- `close()` 必须幂等，并对重复 free、跨 worker handle 和仍在使用的 allocation 报错。

### 9.3 初始化语义

当前 Direct `TensorSpec.create_tensor()` 对缺失 tensor 默认初始化为零。Worker 第一版必须保持一致：

- correctness-first：新分配或复用 scratch 时上传对应初始 Host tensor；
- 只有证明 kernel 完整覆盖的 buffer 才可标记 `UNINITIALIZED_FULL_WRITE`；
- 不得为了性能默认跳过清零。

## 10. `serving/backends/worker_state_store.py`（已实现）

结构与 DirectStateStore 对称：

```python
@dataclass
class WorkerStatePair:
    spec: StateTensorSpec
    current: DeviceTensor
    next: DeviceTensor
```

`prepare()`：

- current 使用 `StateTensorSpec.create_tensor()` 生成 Host 初值并上传；
- next 只分配同 shape/dtype buffer；
- 每层每个 state 独立分配两份。

`commit()`：

- 验证 kernel output 是预绑定的 next handle；
- 全部验证成功后统一交换 current/next；
- kernel 失败时 Runner 不调用 commit。

State store 只依赖 allocator protocol，不 import Runner。

## 11. `serving/backends/worker_backend.py`（单层 prefill/decode 已实现）

### 11.1 初始化

Runtime import 放在构造函数内，保证 Host-only 单测不要求本地安装 PyPTO runtime：

```python
run_config = RunConfig(platform=platform, device_id=device_id, **runtime_cfg)
self.worker = worker_factory(run_config)  # production 为 ChipWorker
self.pool = DeviceBufferPool(self.worker)
self.state_store = WorkerStateStore(self.pool)
```

一个 backend session 只创建一个 ChipWorker，所有 DeviceTensor 都属于该 worker。

### 11.2 Compile cache

继续使用当前 Direct key：

```text
case.name
all TensorSpec shapes
all TensorSpec dtypes
```

编译可继续使用 Host dummy tensor。执行统一使用：

```python
self.worker.run(compiled, *ordered_args, config=self.run_config)
```

### 11.3 Materialize 决策

| Value 类型 | Worker 行为 |
|---|---|
| `RuntimeWeight` | fixed cache hit 或 `alloc_tensor(init=host)` |
| `HostStagingTensor` | acquire staging buffer + `copy_to()` |
| owned `DeviceTensor` | shape/dtype 校验后直接绑定 |
| raw `torch.Tensor` | 当前 step upload cache；用于 input IDs/aux |
| missing scratch/output | 从 pool acquire device buffer |

Raw `torch.Tensor` cache key 使用当前 step 内的对象身份、data_ptr、shape 和 dtype。`end_step()` 前 Host
对象仍被 StatePlan/Runner 持有；step 结束即清理，避免 Python id 重用造成跨 step 命中。

### 11.4 Dispatch 后释放

`run()` 成功后：

- scratch 立即归还 pool；
- routed/selected staging 标记为空闲；
- 被本次 kernel 消费的 intermediate 归还 pool；
- 新 output 保留并返回；
- persistent fixed/state/active upload 不动。

失败时还要回收本次新 output，且 state 不 commit。

### 11.5 D2H

```python
host = torch.empty(shape, dtype=dtype)
worker.copy_from(host.data_ptr(), device.data_ptr, device.nbytes)
```

- `read_control(indices)`：只复制 `[1, 1, 6] int32`，即 24 bytes，并释放 indices intermediate。
- `export_output()`：复制最终 logits/hidden，并释放最终 intermediate。
- `export_debug_tensor()`：复制但不释放，仅在 verbose 模式使用。

### 11.6 Close 顺序

```text
拒绝新 dispatch
→ active uploads
→ intermediate/scratch/staging free lists
→ state current/next
→ fixed weights
→ compile cache
→ ChipWorker.close()
```

## 12. `serving/runner.py`

只增加 step lifecycle：

```python
self.backend.begin_step(
    StepContext(kind=StepKind.PREFILL, seq_len=seq_len, start_pos=0)
)
try:
    ... existing orchestration unchanged ...
finally:
    self.backend.end_step()
```

Decode 同理。不得加入：

```python
if backend == "worker"
if backend.is_resident
isinstance(tensor, DeviceTensor)
```

WeightLoader getter 返回的 RuntimeWeight/HostStagingTensor 直接进入 values。Selected indices 继续通过
现有 `backend.read_control()` 读取。

## 13. Factory、CLI 与 profiler

### Factory

`factory.py` 的 worker 分支延迟 import `WorkerBackend` 并创建实例。Direct 分支保持不变。

### CLI

`run_model.py` 和 `generate.py` 增加/透传：

```text
--enable-l2-swimlane
--keep-prefill-routed-staging   默认 false
```

两者只形成 runtime_cfg，不感知 Worker 内部 pool。

### Profiler

Backend profile 至少记录：

```text
h2d.bytes / h2d.ms
d2h.bytes / d2h.ms
alloc.count / free.count / reuse.count
fixed.hit / fixed.miss / fixed.bytes
state.bytes
active_upload.bytes
intermediate.peak_bytes
staging.routed.bytes
staging.selected.bytes
device.peak_owned_bytes
```

Direct 对 device 指标返回零，保持统一输出格式。

## 14. Prefill 数据流

```text
begin_step(PREFILL)
→ input_ids active upload
→ embedding RuntimeWeight fixed hit/miss
→ embedding output DeviceTensor
→ each layer:
     fixed RuntimeWeight cache
     immutable aux active upload cache
     full routed HostStagingTensor → 12 GiB staging
     state next DeviceTensor outputs
     scratch/intermediate pool
     worker.run
     state commit/swap
     previous hidden recycled
→ optional head on device
→ final D2H
→ release full-routed staging（默认）
→ end_step
```

## 15. Decode 数据流

```text
begin_step(DECODE)
→ input_ids + aux active uploads
→ embedding output device resident
→ each layer pre-MoE:
     fixed weights + current/next state
     worker.run
     state commit
     indices 24-byte D2H
→ Host selected expert build（不可消除）
→ selected pack copy to 288 MiB staging
→ post-MoE consumes pre outputs on device
→ previous intermediates recycled
→ optional head on device
→ final D2H
→ end_step
```

## 16. 实现顺序

### Step 1：Runtime value contract（已完成）

已新增 `runtime_types.py`，并修改 WeightLoader、DirectBackend 和测试。该阶段只改变类型，未引入
DeviceTensor。

验收结果：本地 247 项测试通过；远端 max-layers 5 + head、1 次 decode 通过，Host cache 语义与输出
保持不变。

### Step 2：KernelBindings 与 step lifecycle（已完成）

已修改 base、DirectBackend、Runner 和 FakeBackend tests。Direct begin/end 不分配资源，但会检查重复
begin、无活动 step 的 end，并在 close 时清理活动状态。

验收结果：本地 251 项测试通过；异常路径会调用 end_step，begin 失败不会误调用 end；远端
max-layers 5 + head、3 次连续 decode 通过。Runner 仍无 concrete backend import，且不读取
`KernelBindings.tensors`。

### Step 3：DeviceBufferPool + FakeChipWorker（已完成）

已实现 `serving/backends/device_pool.py` 和纯本地 FakeChipWorker 测试，覆盖
alloc/free/copy/reuse/bytes/close。Pool 将 fixed/state persistent allocation 与其余 reusable allocation
分开，并按 `(category, reuse_key, shape, dtype)` 精确复用。

验收结果：11 项 pool 定向测试及本地 262 项完整测试通过；重复 release/free、跨 pool lease、shape/dtype
copy mismatch、活动 reusable free、close 全量回收均已覆盖。真实 ChipWorker 小 tensor
alloc/init/copy_to/copy_from/reuse/free smoke 通过，任务 exit=0。

### Step 4：WorkerStateStore（已完成）

已基于现有 LayerStateSchema 实现 device current/next。current 按 `StateTensorSpec.init_value` 上传，next
仅分配；commit 会先验证该层全部 output handle，再原子交换所有 pair。

验收结果：10 项定向测试与本地 272 项完整测试通过，覆盖 ratio 0/4/128、24 个真实 schema buffer、
多次 swap、失败不交换、重复 prepare、幂等 close 和只释放 state ownership。真实 ChipWorker current
初始化、两次 next 写入/commit/handle 交换和关闭 smoke 通过，任务 exit=0。

### Step 5：WorkerBackend embedding vertical slice（已完成）

已实现 compile、fixed RuntimeWeight、raw input upload、device output、final D2H，并把 factory 的 worker
分支和 `--enable-l2-swimlane` CLI 透传接通。远端运行：

```text
--backend worker --max-layers 0 --no-head
```

验收结果：WorkerBackend/入口定向测试及本地 277 项完整测试通过。远端开启 L2 swimlane 后：

- seq_len=1 embedding prefill 通过；
- 同一 session 的 prefill + 2 次 decode 全部通过，覆盖 fixed/compile/pool 复用；
- seq_len=13 非 tile 对齐动态形状通过；
- 所有输出 shape/dtype 正确且 finite，任务 exit=0，并成功生成 swimlane 数据。

### Step 6：单层 prefill（已完成）

已加入 full-routed staging、scratch/intermediate 消费和完整 state output mapping。同一 prefill step 内
staging 按 `(kind, slot, shape, dtype)` 复用，默认在 step 结束后释放；可通过
`--keep-prefill-routed-staging` 保留为空闲 pool allocation。

验收结果：本地 279 项完整测试通过；远端开启 L2 swimlane 的 seq_len=1 和 seq_len=13 单层
SWA/hash prefill 均通过，输出 finite。相同 input IDs 下 Direct/Worker 输出逐元素完全一致：
`equal=True, max_abs_diff=0.0`。所有远端任务 exit=0，并成功生成 embedding/block swimlane 数据。

### Step 7：单层 selected decode（已完成）

通用 Worker 生命周期已经覆盖 state input/output、indices D2H、selected staging 和 pre→post
intermediate residency，无需新增 selected-decode 专用分支。已补充定向测试，验证 control D2H bytes、
selected staging 跨 step 复用、pre outputs 消费释放和 final output 边界。

验收结果：本地 280 项完整测试通过；远端开启 L2 swimlane 的单层 prefill + 3 次 decode 全部通过，
pre/post-MoE 输出 finite，state 连续交换正常。固定 token 输入下 Direct/Worker 的 prefill 和三个 decode
output 均逐元素完全一致：每一步 `equal=True, max_abs_diff=0.0`，任务 exit=0。

### Step 8：多层三种 ratio（已完成）

已运行 max-layers 4，覆盖 ratio 0/4/128、SWA/CSA/HCA、hash/top-k、跨层 hidden 和两次连续 decode；
随后运行 max-layers 5 + head，补充 CSA/top-k 组合和 head 的 prefill/decode 路径。

验收结果：远端均开启 L2 swimlane。4 层 no-head 的 prefill 与两次 decode 输出 shape/dtype 正确且
finite；固定 token 输入下，与 DirectBackend 的三步输出均逐元素完全一致：每一步
`equal=True, max_abs_diff=0.0`。5 层带 head 的 prefill 与一次 decode 均输出 `(1, 129280)`、
`torch.float32` 且 finite。两个任务均 exit=0。

### Step 9：完整模型与性能（进行中）

完整 43 层、head/no-head、长 prompt、多步 decode。交替 Direct/Worker A/B，报告 cold upload、warm
decode、H2D/D2H bytes 和 device peak。

当前功能验收结果（均未开启 swimlane，使用 `--max-time 0`）：

- 43 层 no-head、S=1 的 prefill + 1 次 decode 通过，输出 BF16、finite，任务 exit=0；
- 43 层带 head、S=1 的 prefill + 1 次 decode 通过，输出 `(1, 129280)`、FP32、finite，任务 exit=0；
- 43 层 no-head、S=13 的 prefill + `start_pos=13` decode 通过，输出 BF16、finite，任务 exit=0；
- 以上任务 close 后均正常释放设备锁，未出现 device allocation/OOM 错误。

长序列阻塞：43 层 S=4096 在 embedding 阶段报
`aclrtSynchronizeStreamWithTimeout (AICPU) failed: 507018`，尚未进入 block 或显存峰值。最小化验证显示：

- Worker 与 Direct 的 embedding-only S=4096 均复现相同错误，排除 Worker device-resident 专属问题；
- embedding-only S=1008 通过、S=1024 失败；当前 embedding 使用
  `pl.spmd(tokens * H_BLOCKS)`，两者分别为 32256/32768 个 work item，阈值与 32768 精确重合；
- 单层 S=1008 的 embedding 通过，但首个 SWA block 同样报 507018，说明 block 也需要长序列任务切分；
- 因此前不能用该失败推断 64 GB 显存不足。需先重构 embedding 和 block 的长序列 work partition，再重跑
  S=4096 device peak。

性能 A/B、H2D/D2H bytes 和实际 device peak 尚未采集。

### Step 10：默认切换与 Direct 删除

Worker 满足完成标准后才改默认值。观察稳定后删除 DirectBackend；Runner 无需修改。

## 17. 测试矩阵

### 本地

- RuntimeWeight key/layout/version/padding 隔离。
- Routed getter 永不返回 RuntimeWeight。
- Direct unwrap 后输出与当前完全一致。
- FakeChipWorker 混合 Host/Device 参数顺序。
- Pool persistent/reusable/staging 分类。
- State failure 不 swap。
- run failure 无 allocation 泄漏。
- begin/end step 异常安全。
- fixed weight 第二次 materialize 为 hit 且 H2D bytes 不增长。

### 远端

```text
0 layer embedding
1 layer prefill
1 layer + 3 decode
4 layers + 2 decode
5 layers + head
43 layers + head/no-head
S=13 dynamic shape
S=4096 memory peak
```

所有用例同时比较 Direct/Worker shape、dtype、finite 和数值容忍。

## 18. 风险与约束

- DeviceTensor 只能交给创建它的 ChipWorker，禁止跨 backend/session 复用。
- Pool 不能仅按 shape/dtype 让同一 buffer 同时绑定多个可写参数。
- Routed 12 GiB staging 必须复用或在 prefill 后释放，禁止逐层累计。
- Host selected expert build 仍约 85 ms/层，不属于 fixed-resident 收益。
- Runtime `run_ms` 当前混合计算和隐式 copy；必须用 Worker 实测确认收益。
- Scratch 初始化语义未验证前不能跳过 zero/init copy。
- Host layout cache 与 fixed device cache 都不使用 LRU；显存预算依赖 routed 永不进入 fixed cache。

## 19. 完成标准

- Runner 和 StatePlan 不 import Worker/DeviceTensor。
- 固定 non-routed 权重只上传一次，warm decode fixed H2D bytes 为零。
- Mutable state、hidden、pre→post intermediate 全程留在 NPU。
- 每层只发生 24-byte indices D2H，以及最终 API output D2H。
- Prefill device peak 不超过 30 GiB 静态目标，完整 runtime 不超过 64 GB。
- Worker close 后 owned allocation 为零。
- 完整 43 层数值与 Direct 一致。
- Warm decode 端到端收益稳定，目标区间 10%–20%。
