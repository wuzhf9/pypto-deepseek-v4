# Runtime Value Contract

[`serving/runtime_types.py`](../../serving/runtime_types.py) 定义 runner、weight loader 与 device runtime 之间共享的不可变描述符，用类型明确固定权重、专家 staging、Kernel 入口和整模 step 的不同语义。

## 模块定位

这些类型不执行拷贝、分配或 Kernel。Weight loader 创建 `RuntimeWeight` 与 `HostStagingTensor`；runner 创建 `KernelCase` 与 `StepContext`；[`DeviceRuntime`](../../serving/device_runtime.py) 根据具体类型选择 materialize 和生命周期策略。

使用这些描述符的目的，是避免仅凭普通 `torch.Tensor` 推断“是否常驻、是否 staging、使用哪个复用槽位”。

## 类型总览

| 类型 | 创建者 | 消费者 | 表达的语义 |
|---|---|---|---|
| `KernelCase` | runner | device runtime | Kernel 名、JIT entrypoint 和 spec builder |
| `RuntimeWeightKey` | weight loader | loader cache、device runtime | 最终 Kernel-facing 权重 layout 的稳定身份 |
| `RuntimeWeight` | weight loader | device runtime | 可固定常驻的 Host 权重 layout |
| `StagingKind` | weight loader | device runtime | prefill routed 或 decode selected staging 类别 |
| `HostStagingTensor` | weight loader | device runtime | 临时 Host tensor 与语义 staging slot |
| `StepKind` | runner | device runtime | prefill 或 decode |
| `StepContext` | runner | device runtime | 当前整模 step 的 kind、长度和位置 |

## KernelCase

```text
KernelCase(name, fn, spec_builder)
```

- `name` 参与 runtime compile cache key 和 profile 输出。
- `fn` 提供 `.compile(*dummy_args, config=...)`。
- `spec_builder` 由 runner 用来按 `seq_len` 或 `start_pos` 构造 `TensorSpec`；runtime 接收的 `specs` 已经构造完成。

`KernelCase` 是 frozen dataclass，但不验证函数签名。Runner 必须保证 `fn` 与 specs 的参数顺序一致。

## RuntimeWeightKey 与 RuntimeWeight

`RuntimeWeightKey` 包含：

| 字段 | 作用 |
|---|---|
| `name` | normalized checkpoint 参数名 |
| `dtype` | 最终 runtime dtype |
| `layout` | `identity`、`linear_t`、HC padding 等 layout 名 |
| `layout_version` | 同名 layout 规则发生变化时区分版本，默认 `1` |
| `padding_profile` | 区分 padding 宽度等附加配置 |

`RuntimeWeight(key, host_tensor)` 保留 Host tensor 原始对象身份。Weight loader 用 `(key, target_device)` 缓存 Host layout；DeviceRuntime 只用 `key` 缓存对应 fixed NPU allocation。

`RuntimeWeight` 使用 `eq=False`，因此两个包装相同内容的实例不会按 tensor 值比较。稳定复用依据是显式 key，而不是 dataclass value equality。

## HostStagingTensor

```text
HostStagingTensor(host_tensor, kind, slot)
```

| `kind` | Device allocation category | 当前用途 |
|---|---|---|
| `PREFILL_ROUTED` | `STAGING_ROUTED` | 一层全部 routed experts |
| `DECODE_SELECTED` | `STAGING_SELECTED` | 当前 token 选中的 routed experts |

`slot` 当前使用 `w1_t`、`w2_t`、`w3_t`，参与 device pool exact-match reuse key。相同 slot 仍必须同时匹配 category、shape 和 dtype 才能复用同一 allocation。

该类型同样保留 Host tensor 身份且 `eq=False`。它不表示磁盘 cache，也不表示权重已经位于 NPU。

## StepContext

```text
StepContext(kind, seq_len, start_pos)
```

Runner 当前创建：

| Path | Kind | `seq_len` | `start_pos` |
|---|---|---|---|
| Prefill | `PREFILL` | 实际 prompt 长度 | `0` |
| Decode | `DECODE` | `1` | 当前 token 位置 |

`StepContext` 本身不校验数值；runner 在调用 `begin_step()` 之前完成 input shape 和 position 校验。DeviceRuntime 用它建立不可嵌套的 active-step 生命周期。

## 数据与所有权

| 描述符 | 持有的对象 | 是否拥有释放责任 |
|---|---|---|
| `RuntimeWeight` | Host tensor 引用 | 否；weight loader cache 管理引用 |
| `HostStagingTensor` | Host tensor 引用 | 否；当前调用栈管理引用 |
| `KernelCase` | 函数与 builder 引用 | 否 |
| `StepContext` | 标量 metadata | 否 |

描述符均不拥有 DeviceLease。DeviceRuntime materialize 后创建的设备 allocation 由 runtime 和 device pool 管理。

## 约束与错误处理

- 类型是 frozen dataclass 或 enum，构造后不能重绑定字段。
- `RuntimeWeightKey` 的 dtype 必须与对应 Kernel `TensorSpec` 一致，否则 runtime 拒绝 materialize。
- Staging tensor 必须是 CPU、exact shape、exact dtype；staging 路径不自动 cast。
- `KernelCase.name` 必须稳定，否则相同 Kernel 无法命中 compile cache。
- `StepContext` 不能代替 runner 输入校验。

## 验证方法

### Host 侧契约测试

```bash
pytest -q tests/serving/test_runtime_types.py
```

测试验证 frozen 行为、layout metadata 对 key equality 的影响、Host tensor identity、staging kind/slot 和 step fields。

### Materialize 集成

```bash
pytest -q tests/serving/test_device_runtime.py tests/serving/test_weight_loader.py
```

该组合验证 weight loader 产生的 descriptors 被 runtime 映射到 fixed weight 或 staging，而不是按普通 Host tensor 处理。

## 代码索引

| 内容 | 位置 |
|---|---|
| 类型定义 | [`serving/runtime_types.py`](../../serving/runtime_types.py) |
| 生产者 | [`serving/weight_loader.py`](../../serving/weight_loader.py)、[`serving/runner.py`](../../serving/runner.py) |
| 消费者 | [`serving/device_runtime.py`](../../serving/device_runtime.py) |
| 测试 | [`tests/serving/test_runtime_types.py`](../../tests/serving/test_runtime_types.py) |

## 相关文档

- [`Serving 总览`](README.md)
- [`Weight Loader`](07_weight_loader.md)
- [`Device Memory`](09_device_memory.md)
- [`Device Runtime`](10_device_runtime.md)
