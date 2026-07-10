# Runtime-layout Host Cache 与 Device-resident Backend 设计

## 1. 目标与结论

`runtime-layout host cache` 和 `device-resident backend` 在功能上可以独立实现、独立开关和独立
验证，但建议共享同一套权重身份、最终布局和生命周期接口，并按以下顺序落地：

```text
checkpoint
  ↓
基础 tensor cache
  ↓
runtime-layout host cache      可独立实现
  ↓ 首次上传 / device cache miss
device-resident cache          可独立启用，但消费最终 runtime layout
  ↓
compiled kernel
```

先实现 host runtime-layout cache，可以在不改 kernel 和 direct backend 行为的情况下验证收益；
随后 device-resident backend 复用相同的最终 layout，避免再次实现 dtype 转换、transpose、padding
和 cache key。

两者不是强依赖关系。device backend 可以在 miss 时临时生成 layout 后上传，但不建议形成独立的
layout 转换路径。

## 2. 当前代码边界

当前 `DeepSeekV4WeightLoader` 已有基础 CPU tensor cache。它缓存的是已经读取、反量化、dtype
转换并 contiguous 后的原布局 tensor，而不是最终 kernel runtime layout。

`get_linear_t()` 每次仍执行：

```python
tensor = self.get_linear_weight(...)
out = tensor.t().contiguous()
```

HC transpose、head padding 等路径也会重复生成最终布局。

`DeepSeekV4Runner` 每层结束后调用：

```python
self.weight_loader.release_prefix(f"layers.{layer_id}.")
```

因此 layer 权重的基础 cache 不能跨 decode token 命中。embedding、head 等不带 layer prefix 的
权重可以继续保留。

当前 `_DirectBackend` 只有 compiled-program cache。它把 runner 组装的 host `torch.Tensor` 直接
传给 compiled kernel，没有不可变权重注册、device cache 或持久 device allocation。

## 3. PyPTO runtime 能力

远程 Ascend 环境中的 PyPTO runtime 已提供：

- `pypto.runtime.DeviceTensor`
- `pypto.runtime.ChipWorker`
- `ChipWorker.alloc_tensor(..., init=host_tensor)`
- `ChipWorker.free_tensor(...)`
- `ChipWorker.copy_to()` / `copy_from()`
- compiled program 的 host `torch.Tensor` 与 `DeviceTensor` 混合传参
- dispatch 前的 shape、dtype 检查
- `DeviceTensor` 参数跳过调用时 H2D/D2H

`DeviceTensor` 的 device pointer 属于创建它的 worker 地址空间，生命周期由调用方管理。因此
device-resident backend 必须持有长期存活的 `ChipWorker`，并在 `close()` 中释放全部 device
allocation 后关闭 worker。

现有 kernel tensor shape 和 dtype 接口不需要因此修改。

## 4. 统一 runtime weight key

host layout cache 和 device cache 应使用同一个稳定 key。不能使用 `tensor.data_ptr()`，因为 tensor
重新读取后地址会变化，也不能只使用 kernel 参数名，例如不同 layer 都存在 `wq_a_t`。

建议定义：

```python
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RuntimeWeightKey:
    name: str
    dtype: torch.dtype
    layout: str
    layout_version: int = 1
    padding_profile: str | None = None
```

字段含义：

- `name`：规范化 checkpoint parameter name，包含 layer/expert 身份。
- `dtype`：最终 kernel dtype。
- `layout`：例如 `identity`、`linear_t`、`hc_t`、`hc_head_padded_t`。
- `layout_version`：layout 规则变化时主动失效旧 cache。
- `padding_profile`：同一参数存在多个 padding 方案时区分最终布局。

## 5. Runtime-layout host cache

### 5.1 固定 host layout cache

最终方案不保留 `_base_cache`。Source tensor 只在 layout miss 时临时存在，transpose、padding
完成后仅保留可直接传给 kernel 的最终 contiguous tensor：

```text
checkpoint
→ 临时反量化/dtype 转换 tensor
→ transpose/padding
→ 固定 _layout_cache
```

普通 host layout cache 不提供 LRU、eviction 或容量参数。它惰性缓存 runner 实际访问到的全部
非 routed 不可变权重，包括：

- embedding 与 head。
- 43层 attention、HC、gate、norm。
- compressor 与 indexer。
- shared experts。

仅 routed experts、selected expert pack 和 full routed pack 不进入该固定 cache。完整普通权重
工作集包含 shared experts 和 head 时为15,753,487,500 bytes（14.672 GiB）；`--no-head` 时为
13,634,307,144 bytes（12.698 GiB）。

### 5.2 建议接口

```python
def get_runtime_weight(
    self,
    key: RuntimeWeightKey,
    *,
    build: Callable[[], torch.Tensor],
) -> torch.Tensor:
    ...
```

或者保持现有便捷方法，在其内部构造 key：

```python
def get_linear_t(
    self,
    name: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
    cache: bool = True,
) -> torch.Tensor:
    key = RuntimeWeightKey(name, dtype, "linear_t")
    return self.get_runtime_weight(key, build=...)
```

需要逐步接入：

- 普通 linear transpose。
- layer HC transpose。
- head HC transpose 和 padding。
- 其他明确的 reshape、padding、pack runtime layout。

### 5.3 Release 策略

Runner 不再执行 per-layer weight release。普通 layout 在首次访问后保留到 `runner.close()`，并
由 `weight_loader.close()` 统一释放。

仍保留 `release(name)` 和 `release_prefix(prefix)` 作为显式管理接口，但 serving 主循环不调用。
Host layout 即使已经上传到 device 也继续保留，用作 direct backend、device miss 恢复和正确性
基线；当前服务器的 host 内存足以容纳该固定工作集。

## 6. Backend 接口扩展

当前 backend 只有 `run()` 和 `close()`，不足以表达不可变权重注册。建议增加：

```python
TensorArg = torch.Tensor | DeviceTensor


def make_weight_resident(
    self,
    key: RuntimeWeightKey,
    host_tensor: torch.Tensor,
) -> TensorArg:
    ...
```

两种 backend 的语义分别为：

```python
class _DirectBackend:
    def make_weight_resident(self, key, host_tensor):
        return host_tensor


class _ResidentBackend:
    def make_weight_resident(self, key, host_tensor):
        # device cache hit: 返回原 DeviceTensor
        # device cache miss: alloc_tensor(init=host_tensor)，登记 bytes/LRU
        ...
```

backend 还应提供：

- device cache bytes/capacity。
- hit、miss、upload、eviction 统计。
- 显式 eviction。
- `close()` 中的统一释放。

## 7. Resident backend 生命周期

resident backend 持有单个长期存活的 `ChipWorker`：

```python
self.worker = ChipWorker(run_config)
```

编译后的程序通过同一个 worker 执行：

```python
self.worker.run(compiled, *ordered_args, config=run_config)
```

device cache miss 时：

```python
device_tensor = self.worker.alloc_tensor(
    host_tensor.shape,
    host_tensor.dtype,
    init=host_tensor,
)
```

eviction 或 close 时：

```python
self.worker.free_tensor(device_tensor)
```

关闭顺序必须是：

```text
停止新 dispatch
→ 释放 device cache 中的 DeviceTensor
→ 清理 compiled registration/cache
→ 关闭 ChipWorker
```

## 8. Runner 权重身份传递

runner 当前把权重放入普通字典后只保留 kernel 参数名，例如 `wq_a_t`。backend 无法由该名字判断
其所属 layer 和 checkpoint parameter，因此 resident 转换必须发生在权重身份丢失之前。

推荐流程：

```python
host = weight_loader.get_linear_t(parameter_name)
key = RuntimeWeightKey(parameter_name, host.dtype, "linear_t")
runtime_arg = backend.make_weight_resident(key, host)
values[spec_name] = runtime_arg
```

不建议让 backend 在 `run()` 中仅根据 `spec.name` 自动猜测权重身份。

长期可以引入包含 `key` 与 `host_tensor` 的轻量 `RuntimeWeight` 对象，但第一版可以在 runner 的
权重组装辅助函数中集中处理，以减少对现有 weight dataclass 的改动。

## 9. 分阶段实现

### V1：Host runtime-layout cache

- 引入 `RuntimeWeightKey`。
- 删除 `_base_cache`，source tensor 仅作为 layout 构建临时值。
- 使用普通 `dict` 实现固定 `_layout_cache`，不设置容量或 LRU。
- 接入 `get_linear_t()`、HC transpose、head padding。
- Identity 权重和 shared experts 同样进入固定 layout cache。
- Routed experts 保持 `cache=False`，不进入固定 layout cache。
- direct backend 行为保持不变。
- 增加 cache hit/miss/bytes profile。
- 删除 runner 的 per-layer weight release。

V1 可独立验证，不需要远程 device allocation 接口。

### V2：只常驻不可变权重

- 实现持有长期 `ChipWorker` 的 resident backend。
- 实现 `make_weight_resident()` 和 device LRU。
- 只把不可变 common weight 替换成 `DeviceTensor`。
- hidden、input IDs、state、scratch、output 继续使用 host tensor。
- 利用 PyPTO 的 host/device 混合传参减少改造范围。

第一批常驻候选：

- attention common weights。
- compressor/indexer weights。
- HC weights。
- gate weights。
- shared experts。
- embedding。
- LM head。

### V3：State 与中间 tensor 常驻

- state device allocation。
- device output 与原地 state 更新。
- pre-MoE 到 post-MoE 的中间 tensor 保持在 device。
- scratch buffer 池化。
- 只对最终需要 host 消费的数据显式 D2H。

### V4：Routed expert device LRU

- `(layer_id, expert_id, layout_version)` 作为 expert device key。
- selected expert cache hit 直接绑定 device tensor。
- miss 时从最终 host layout 或离线 runtime-layout 文件上传。
- 根据真实 routing 命中率设置容量和淘汰策略。

## 10. 验证方案

### 10.1 V1 验证

- weight loader 单测：key 隔离、hit、miss、prefix release、close。
- 同一 `get_linear_t()` 重复调用返回同一 cache entry。
- layout tensor shape、dtype、contiguous 与原实现一致。
- 多 decode token profile 中 transpose 次数由每 token 降为首次一次。
- Shared experts 命中固定 cache，routed experts 不增加普通 layout cache bytes。
- 完整43层加 head 的 host cache bytes 预期约为14.672 GiB。

### 10.2 V2 正确性验证

- direct 与 resident backend 对相同 checkpoint、输入和 state 输出一致。
- 单层 prefill。
- 单层 selected decode pre/post。
- 多层、多 decode step。
- cache hit 后连续运行，确认没有 stale pointer。
- eviction 后重新上传并保持正确性。
- runner close 后无 device allocation 泄漏。

### 10.3 V2 性能验证

- 同设备、交替 A/B 测试 direct 与 resident backend。
- 开启 `--enable-l2-swimlane`。
- 分开记录 cold start、首次 upload 和 warm decode。
- 至少比较：
  - total token latency。
  - H2D/D2H bytes 和时间。
  - device cache hit/miss/eviction。
  - host layout hit/miss。
  - worker registration/dispatch 开销。

只有 warm decode 稳定降低且显存、host 内存受控时，才将 resident backend 作为默认候选。

## 11. 实现原则

- direct backend 始终保留，作为正确性和性能基线。
- 普通 host layout cache 固定保留全部 non-routed 权重，不提供 LRU 或容量参数。
- Shared experts 属于固定 cache；routed experts 使用独立 expert cache/LRU。
- Device-resident backend 仍可独立开关，并单独管理 device 容量。
- cache identity 使用语义 key，不使用 host pointer。
- device pointer 不跨 worker 使用。
- Device cache 和 routed expert LRU 的容量必须显式可配置并可观测。
- 第一版只处理不可变权重，不同时改造 state 和输出生命周期。
- Host layout 在 device 命中后仍保留到 runner close。
