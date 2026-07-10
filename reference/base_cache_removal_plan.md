# 删除 `_base_cache` 的完整重构方案

## 1. 目标与结论

当前仓库的生产执行路径中，不存在同一个 checkpoint tensor 生成多个 runtime layout 的需求。
`_base_cache` 只在首次生成 transpose layout 时短暂保存原布局 tensor，随后在当前层结束时释放，
没有产生有效复用。

本次重构目标是删除 `_base_cache` 及其完整生命周期，只保留最终 kernel-facing
`_layout_cache`。

当前数据流：

```text
checkpoint
→ 反量化/转换原布局
→ _base_cache
→ transpose/padding
→ _layout_cache
→ kernel
→ 每层释放 _base_cache
```

目标数据流：

```text
checkpoint
→ 反量化/转换临时 tensor
→ transpose/padding
→ _layout_cache
→ 原布局临时 tensor 释放
→ kernel
```

Identity tensor 的目标路径：

```text
checkpoint
→ dtype/contiguous
→ _layout_cache
```

Routed expert 继续走独立临时路径：

```text
expert cache/checkpoint
→ 最终 expert layout
→ selected/routed pack
→ kernel
```

Routed expert 继续使用 `cache=False`，不进入普通 layout cache。Shared experts 是每层、每个 token
都必用的固定权重，继续进入普通 layout cache。

普通 layout cache 不再提供 LRU 或容量参数。它以惰性方式缓存当前 runner 实际访问到的全部
非 routed 不可变权重，并在 `runner.close()` 时统一释放。

## 2. 当前 layout 唯一性

每类 checkpoint 参数在生产路径中只对应一种最终布局：

| 参数类型 | 唯一 layout |
|---|---|
| embedding、norm、scale、bias、APE、sink、tid2eid | `identity` |
| attention/compressor/indexer/gate/shared/routed linear | `linear_t` |
| layer HC function | `hc_t` |
| head HC function | `hc_head_padded_t` |
| head HC base | `hc_head_base_padded` |

没有生产调用让同一个规范化参数名同时进入：

```text
identity + linear_t
linear_t + hc_t
BF16 linear_t + FP32 linear_t
不同 padding profile
```

`get_layer_compressor_ratio128()` 与 `get_layer_compressor_ratio4_attention()` 即使访问相同命名
模式，也使用相同 dtype 和 layout。Hash gate 与 top-k gate 的 `gate.weight` 也都使用 BF16
`linear_t`。

测试中存在同一参数分别生成 BF16 和 FP32 `linear_t` 的用例，但这是 cache key 隔离测试，不是
runner、converter 或其他生产工具的实际需求。

### 2.1 固定 layout cache 工作集

以下数字根据实际 `bf16_lowvram_cache/weight_index.json` 的 shape，以及当前 kernel-facing dtype
计算：

| 分类 | Entries | Bytes | GiB |
|---|---:|---:|---:|
| Embedding | 1 | 1,059,061,760 | 0.986 |
| Head、final norm、head HC | 5 | 2,119,180,356 | 1.974 |
| 43层 HC | 258 | 135,275,592 | 0.126 |
| 43层 attention common | 387 | 9,198,604,032 | 8.567 |
| Gate 与 FFN norm | 129 | 99,878,912 | 0.093 |
| Shared experts | 129 | 2,164,260,864 | 2.016 |
| Compressor 与 indexer | 290 | 977,225,984 | 0.910 |
| 合计 | 1,199 | 15,753,487,500 | 14.672 |

`--no-head` 时不会加载 head 相关 layout，固定工作集为：

```text
13,634,307,144 bytes
= 12.698 GiB
```

仅 routed experts 被排除。全部43层 routed expert 的 BF16 权重总量为516 GiB，不适合进入固定
layout cache。当前远程服务器可用 host 内存约1,871.7 GiB，单 runner 的14.672 GiB固定工作集
约占0.784%，因此无需为普通 layout cache 保留容量淘汰机制。

## 3. `weight_loader.py` 数据结构调整

删除：

```python
self.max_cache_bytes
self._base_cache
self._base_cache_bytes
self.max_layout_cache_bytes
```

保留：

```python
self._layout_cache
self._layout_cache_bytes
self._file_handles
```

删除以下方法：

```python
_insert_base_cache()
_evict_base_if_needed()
_release_base_keys()
_evict_layout_if_needed()

release_base()
release_base_prefix()
```

删除以下 profile 事件：

```text
cache.base.hit
cache.base.miss
cache.base.evict
```

保留：

```text
cache.layout.hit
cache.layout.miss
```

`_layout_cache` 从 `OrderedDict` 改为普通 `dict`。Cache hit 直接返回 tensor，不再通过
`pop()`/重新插入维护 MRU 顺序。

## 4. Source tensor 加载接口

当前 `get_tensor()` 同时承担加载和 base cache 职责。本次不保留仓外 API 兼容，直接删除公开的
`get_tensor()`，并用只负责加载和转换的私有函数替代：

```python
def _load_tensor(
    self,
    name: str,
    *,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    dequantize: bool = True,
) -> torch.Tensor:
    ...
```

行为：

```text
读取 safetensors
→ 判断 FP8/FP4/plain
→ 必要时加载 scale
→ 反量化
→ dtype 转换
→ target device
→ contiguous
→ 返回
```

该函数不再查询或写入内存 tensor cache。

同样直接删除公开的 `get_linear_weight()`，并改为私有加载函数：

```python
def _load_linear_weight(
    self,
    name: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    tensor = self._load_tensor(
        name,
        dtype=dtype,
        device=device,
        dequantize=True,
    )
    if not tensor.is_floating_point():
        raise TypeError(...)
    return tensor
```

`DeepSeekV4WeightLoader` 内部实现中的 source load 调用一次性迁移：

```text
get_tensor(...)          → _load_tensor(...)
get_linear_weight(...)   → _load_linear_weight(...)
```

类外调用和测试不应改为依赖新的私有方法，而应迁移到 `get_runtime_tensor()`、
`get_linear_t()` 或结构化 `get_*_weights()` 接口。低层反量化函数本身继续通过独立函数单测覆盖。

最终类接口中不再存在：

```python
get_tensor()
get_linear_weight()
```

也不保留 deprecated wrapper、兼容别名或忽略 `cache` 参数的空实现。对 runtime weight 的公开
访问统一通过 `get_runtime_tensor()`、`get_linear_t()` 和各类结构化 `get_*_weights()` 接口完成。

## 5. Transpose layout 构建

当前 `_get_transposed_weight()` 使用：

```python
tensor = self.get_linear_weight(
    name,
    dtype=dtype,
    device=target,
    cache=cache,
)
out = tensor.t().contiguous()
```

改为：

```python
def build() -> torch.Tensor:
    tensor = self._load_linear_weight(
        name,
        dtype=dtype,
        device=target,
    )
    start = time.perf_counter()
    out = tensor.t().contiguous()
    self._record_profile(f"transpose.{layout}", start)
    return out
```

继续通过最终 layout cache 返回：

```python
return self.get_runtime_weight(
    key,
    build=build,
    device=target,
    cache=cache,
)
```

语义为：

- Layout hit：不执行 `build()`。
- Layout miss：加载一次临时原 tensor 并生成 layout。
- Layout 写入后，临时原 tensor 失去引用并释放。
- `cache=False`：不查询、不写入 layout cache，用于 routed experts。

## 6. Identity layout 构建

`get_runtime_tensor()` 改为直接使用 `_load_tensor()`：

```python
def get_runtime_tensor(...):
    key = RuntimeWeightKey(name, dtype, layout)
    return self.get_runtime_weight(
        key,
        build=lambda: self._load_tensor(
            name,
            dtype=dtype,
            device=target,
            dequantize=True,
        ),
        device=target,
        cache=cache,
    )
```

不再需要：

```python
get_tensor(..., cache=False)
```

Embedding、norm、scale、bias、APE、sink、tid2eid 等继续直接进入 layout cache。

## 7. Head padding 构建

以下方法改用 `_load_tensor()`：

```python
_build_head_hc_fn_t()
_build_head_hc_base()
```

例如：

```python
hc_fn = self._load_tensor(
    "hc_head_fn",
    dtype=torch.float32,
    device=device,
)
```

生成 padded layout 后只缓存：

```text
hc_head_padded_t
hc_head_base_padded
```

原始 `hc_head_fn` 和 `hc_head_base` 不保留。

## 8. Release 接口收敛

删除 base cache 后，release 只管理 layout cache。

建议收敛为：

```python
def release(self, name: str | None = None) -> None:
    if name is None:
        self._layout_cache.clear()
        self._layout_cache_bytes = 0
        self._close_file_handles()
        return

    self._release_layout_keys(
        key for key in self._layout_cache
        if key[0].name == name
    )


def release_prefix(self, prefix: str) -> None:
    self._release_layout_keys(
        key for key in self._layout_cache
        if key[0].name.startswith(prefix)
    )


def close(self) -> None:
    self.release()
```

可以删除：

```python
release_layout()
release_layout_prefix()
```

因为只剩一种 cache 后，它们与 `release()` / `release_prefix()` 重复。保留通用接口对现有 converter
改动更小。

## 9. Runner 清理

当前 prefill 和 decode 每层结束都会执行：

```python
self._release_layer_weights(layer_id)
```

其内部只调用：

```python
release_base_prefix(...)
```

删除 base cache 后，应删除：

```python
DeepSeekV4Runner._release_layer_weights()
```

并删除 prefill 中：

```python
with self.profiler.timer("layer.release", ...):
    self._release_layer_weights(layer_id)
```

以及 decode 中对应代码。

循环收敛为：

```python
for layer_id in range(self.max_layers):
    hidden = self._run_prefill_block(...)
```

和：

```python
for layer_id in range(self.max_layers):
    hidden = self._run_decode_block(...)
```

运行时临时 tensor 由 Python 局部引用生命周期释放；固定 layout cache 在 runner close 时统一释放。
`layer.release` profile 事件也随之删除。

## 10. 构造参数与 helper 清理

从 `DeepSeekV4WeightLoader.__init__()` 删除：

```python
max_cache_bytes: int = 0
max_layout_cache_bytes: int = 0
```

从 `load_weight_loader_from_checkpoint()` 删除两个同名参数，并从以下入口删除
`max_layout_cache_bytes` 的透传：

- `DeepSeekV4Runner.__init__()`。
- `serving/runner.py` 的 `--max-layout-cache-bytes`。
- `serving/generate.py` 的 `--max-layout-cache-bytes`。
- `_create_runner()` 的对应参数传递。

直接删除含义模糊的：

```python
cache_bytes
```

统一使用：

```python
layout_cache_bytes
```

不保留 `cache_bytes` 兼容属性。仓内代码和测试全部迁移到 `layout_cache_bytes`。

## 11. Profiler 调整

当前 profiler 输出：

```text
base_cache_bytes=...
layout_cache_bytes=...
```

改为只输出：

```text
layout_cache_bytes=...
```

首次单层 prefill 预期：

```text
cache.layout.miss=21
transpose.linear_t=9
transpose.hc_t=2
layout_cache_bytes=1,331,668,440
```

Decode pre-MoE 预期：

```text
cache.layout.hit=18
```

Decode post-MoE 预期：

```text
cache.layout.hit=3
expert_cache.load=6
selected_experts.build=1
```

不应再出现任何 `cache.base.*` 或 `base_cache_bytes`。
普通 layout cache 也不再输出 `cache.layout.evict`。

## 12. Routed expert 路径

保持以下行为不变：

```python
get_linear_t(..., cache=False)
```

用于没有 BF16 expert cache 时的原始专家转换。

有 BF16 expert cache 时：

```text
handle.get_tensor()
→ clone().contiguous()
→ selected/routed pack
```

普通 `_layout_cache` 不缓存 routed experts，避免把43层 × 256专家全部留在 host 内存。后续 routed
expert 应单独实现容量受限的 expert LRU。

Shared experts 不属于该排除范围。以下调用继续使用默认 `cache=True`：

```python
get_layer_moe_shared(layer_id)
```

43层 shared experts 的最终 BF16 layout 合计2,164,260,864 bytes（2.016 GiB），每个 token 都会
使用，固定缓存比重复加载更合适。

## 13. 测试迁移

删除或重写 base cache 测试：

- 删除 `base_cache_bytes` 断言。
- 删除 `_base_cache` 内部结构断言。
- 删除 `release_base_prefix()` 测试。
- 删除 base LRU 测试。
- 删除 `max_cache_bytes` 构造参数测试。
- 删除 layout LRU/eviction 测试。
- 删除 `max_layout_cache_bytes` 参数和 CLI 转发测试。

保留并加强以下测试。

### 13.1 Identity layout 复用

```python
first = loader.get_runtime_tensor(name, dtype=...)
second = loader.get_runtime_tensor(name, dtype=...)
assert second is first
```

### 13.2 Transpose 只执行一次

```python
first = loader.get_linear_t(name)
second = loader.get_linear_t(name)

assert second is first
assert profile["transpose.linear_t"].count == 1
assert profile["cache.layout.miss"].count == 1
assert profile["cache.layout.hit"].count == 1
```

### 13.3 `cache=False`

```python
first = loader.get_linear_t(name, cache=False)
second = loader.get_linear_t(name, cache=False)

assert first is not second
assert loader.layout_cache_bytes == 0
```

### 13.4 固定 cache 范围

验证：

- 多次访问普通不可变权重始终返回同一 cache entry。
- Shared expert 重复访问命中固定 layout cache。
- Routed expert 的 `cache=False` 路径不增加 `layout_cache_bytes`。
- 不存在 layout eviction profile 或容量参数。

### 13.5 Exact/prefix release

验证：

```python
release(name)
release_prefix("layers.0.")
release()
close()
```

均正确更新 `layout_cache_bytes`。

### 13.6 文件句柄复用

通过两个 `cache=False` runtime load 验证同一 safetensors 文件只打开一次；`close()` 后
`_file_handles` 为空。

### 13.7 Head layout

验证 padded tensor：

- 首次生成。
- 第二次对象 identity 命中。
- Padding 内容正确。
- 不保留原始 head HC tensor。

## 14. 本地验证

执行：

```bash
python -m compileall serving tests
pytest -q tests/test_weight_loader.py tests/test_generate.py
pytest -q
git diff --check
```

预期全量测试不少于当前的 `222 passed`。

## 15. 远程验证

使用：

```bash
task-submit --device auto --run "python serving/runner.py \
  --checkpoint /data/wuzhifeng/dsv4_ckpt \
  --expert-cache-dir /data/wuzhifeng/dsv4_bf16_expert_cache \
  -p a2a3 -d {} -s 1 \
  --max-layers 1 \
  --no-head \
  --decode-steps 1 \
  --profile"
```

验收条件：

- Prefill 和 decode 输出 finite。
- 任务 `exit=0`。
- Prefill：21 个 layout miss。
- Decode pre：18 个 layout hit。
- Decode post：3 个 shared layout hit。
- Decode 不出现 common weight raw load、dequant 或 transpose。
- 不再输出 base cache 指标。
- Layout cache bytes 与重构前一致。
- 首次 prefill 峰值 host 内存理论减少约 `269,484,032` bytes。

## 16. 风险与接口边界

### 多 runner 内存线性增长

普通 layout cache 不再淘汰，单个完整 runner（包含 head 和 shared experts）的固定工作集约为
14.672 GiB。多个 runner 不共享 host tensor，因此内存随实例数线性增长：

```text
1 runner  ≈ 14.7 GiB
8 runners ≈ 117 GiB
```

当前远程服务器可用内存约1,871.7 GiB，可以承受该规模。若未来改成大量 runner 实例，应优先
实现进程内共享不可变 layout，而不是重新给普通权重增加 LRU。

### Prefill 临时 routed pack

完整 prefill 会为当前层临时构造约12 GiB routed expert pack。最后一层 prefill 附近，固定
`--no-head` layout cache 与 routed pack 的权重相关峰值约为24.7 GiB；head 在 block 完成后才
加载，通常不与 routed pack 完整重叠。该峰值对当前服务器可接受。

### Source loader API 为有意破坏性重构

本次明确不提供仓外 API 兼容，以下接口直接删除：

```text
get_tensor()
get_linear_weight()
cache_bytes
max_cache_bytes
max_layout_cache_bytes
```

仓内生产调用、converter 和测试在同一次修改中完成迁移。完成后使用 `rg` 确认不存在上述方法或
属性的残留引用；safetensors handle 自身的 `handle.get_tensor()` 不属于
`DeepSeekV4WeightLoader` API，继续保留。

### Runtime key 保持完整

虽然当前没有多-layout 需求，仍保留 `RuntimeWeightKey` 中的：

```text
name
dtype
layout
layout_version
padding_profile
```

它们用于 cache 正确性、未来 layout 规则升级，以及后续 device-resident cache 复用同一身份。

## 17. 重构完成后的结构

```text
WeightLoader
├── 固定 non-routed runtime layout cache
│   └── 包含 shared experts
├── safetensors file-handle cache
└── routed experts 临时加载/独立 expert LRU

Runner
├── compiled program cache
├── persistent state
└── 不再执行 per-layer weight release
```

本次重构只减少 host 中间缓存并简化生命周期，不改变数值计算、kernel 接口或 H2D 行为。
