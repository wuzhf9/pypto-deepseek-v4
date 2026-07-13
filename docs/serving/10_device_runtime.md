# Device Runtime

[`DeviceRuntime`](../../serving/device_runtime.py) 通过一个长生命周期 `ChipWorker` 执行 Serving Kernel，并把 `TensorSpec`、runtime values、DeviceBufferPool 和 DeviceStateStore 组合成明确的 step 生命周期。

## 模块定位

入口脚本创建 runtime 并注入 runner。Runner 决定执行哪个 Kernel、准备哪些 values；runtime 负责：

- 创建 `RunConfig` 和单个 `ChipWorker`；
- 把 Host/runtime values materialize 为 DeviceTensor bindings；
- 编译并缓存 Kernel；
- 调用 worker；
- 管理 fixed weights、staging、intermediate、scratch 和 state；
- 在控制量或 public 输出边界执行 D2H；
- step 和 runtime 结束时清理资源。

Runtime 不加载 checkpoint，也不实现模型数学。

## 构造接口

| 参数 | 作用 |
|---|---|
| `platform` | 传给 PyPTO `RunConfig` |
| `device_id` | 单卡设备编号 |
| `runtime_cfg` | 展开为额外 `RunConfig` 字段，例如 `enable_l2_swimlane` |
| `keep_prefill_routed_staging` | 是否在 step 结束后保留 prefill routed staging allocations |
| `worker_factory` | 测试注入；默认 `pypto.runtime.ChipWorker` |
| `run_config_factory` | 测试注入；默认 `pypto.runtime.RunConfig` |

构造后 runtime 立即拥有 worker、device pool 和空 state store，但 state 必须由 runner 通过 `prepare_state()` 单独准备。

## Step 状态机

```text
begin_step(StepContext)
        │
        ├── materialize(specs, values)
        ├── run(case, specs, bindings)
        ├── ...更多 kernel dispatch...
        ├── read_control() / export_output()
        └── end_step()
```

同一时间只能有一个 active step。未 begin 时不能 materialize、run 或 D2H owned tensor；active step 未结束时不能再次 begin。`end_step()` 归还剩余 step leases，并清空 active Host upload identity cache。

## Materialize 类型分派

| Value | 处理方式 | Allocation |
|---|---|---|
| `RuntimeWeight` | 按 key 查找或上传 fixed weight | Persistent `FIXED_WEIGHT` |
| `HostStagingTensor` | exact shape/dtype 上传到语义 slot | Reusable `STAGING_ROUTED/SELECTED` |
| Host `torch.Tensor` | 允许转换到 spec dtype，step 内按 identity 复用上传 | Reusable `ACTIVE_UPLOAD` |
| DeviceTensor-compatible value | 校验 shape/dtype/data pointer 后直接绑定 | 不新分配 |
| 缺省 output | 分配并预绑定 output | Reusable `INTERMEDIATE` |
| 缺省 scratch | 按 spec 创建初值 | Reusable `SCRATCH` |

预绑定 state output 属于 DeviceStateStore 的 persistent tensor，runtime 把它加入 `output_tensors`，但不创建 lease 或取得所有权。

### Fixed weight

首次遇到某个 `RuntimeWeightKey` 时，runtime 校验 Host tensor exact shape/dtype，分配 persistent NPU buffer 并 H2D。后续相同 key 复用 DeviceLease，同时再次校验当前 `TensorSpec`。Fixed weights 保留到 runtime close。

### Active Host upload

普通 Host tensor 的 step-local key 为：

```text
(id(tensor), tensor.data_ptr(), spec.shape, spec.dtype)
```

State plan 在同一 step 跨层返回相同 aux tensor 对象时可命中该缓存，避免重复 H2D。`end_step()` 清空 key，并把相关 allocations 归还 pool；allocation 本身可在后续 step 通过 free list 复用。

### Expert staging

Staging category 由 `StagingKind` 决定，reuse key 使用 semantic slot。Kernel run 完成后 staging lease归还 pool。

默认情况下，prefill step 结束时进一步真正 free 本次 `STAGING_ROUTED` allocations；启用 `keep_prefill_routed_staging` 后保留这些 idle allocations，供后续相同 category/slot/shape/dtype staging 复用。该选项不保存专家权重内容，下一次 acquire 仍会 H2D 覆盖。

`STAGING_SELECTED` 在每层 post-MoE 后归还 free list，并自然跨 decode step exact-match 复用 allocation。

## Kernel 编译与执行

Compile cache key 为：

```text
(KernelCase.name, all spec shapes, all spec dtypes)
```

Cache miss 时 runtime 为每个 spec 创建 Host dummy tensor，调用 `case.fn.compile(..., config=run_config)`；cache hit 直接复用 compiled object。Runtime 保存最近一次 `last_compile_seconds`、`last_run_seconds` 和 `last_compile_cache_hit` 供 profiler 读取。

`KernelBindings` 按 specs 顺序保存 DeviceTensor，并记录 dispatch 完成后应释放的 scratch、transient staging 和已消费 intermediates。Binding 只能 run 一次；worker 调用即使失败，`finally` 仍释放这些 leases。

## Intermediate 消费

Runtime 分配的 output intermediate 保持 in-use，并可直接作为后续 Kernel input。Materialize 识别这种输入并把 lease 记为 `consumed_leases`；后续 Kernel run 结束后才归还它。

这使 embedding → Block、Block → Block、decode pre-MoE → post-MoE 的大 tensor 保持在 NPU，同时在最后消费者完成后及时进入复用池。

## D2H 边界

| 接口 | 行为 |
|---|---|
| `read_control()` | 复制 owned tensor 到 Host，并归还 step-owned lease；用于 decode indices |
| `export_output()` | 复制并返回 contiguous Host tensor，同时归还 step-owned lease；用于 public 输出 |
| `export_debug_tensor()` | 复制到 Host但不释放；用于逐层诊断 |

三个接口只接受当前 runtime 拥有的 tensor，并要求 active step。State store 预绑定 tensor不通过这些 public export 路径释放。

## State 集成

`prepare_state()`、`state_inputs()`、`state_outputs()` 和 `commit_state()` 直接委托 DeviceStateStore。Runtime close 时先关闭 state store，再关闭整个 pool，避免 state lease 被 pool 提前整体 free。

## 关闭顺序

`close()` 可重复调用：

1. 若仍有 active step，执行 step cleanup。
2. 关闭 state store。
3. 关闭 device pool，free 所有 active/idle allocations。
4. 清空 fixed-weight、ownership、prefill staging 和 compile metadata。
5. 标记 closed。
6. 关闭 `ChipWorker`。

关闭后任何新 step 都被拒绝。

## 约束与错误处理

- Runtime values 和 `TensorSpec` 的 shape/dtype 必须一致。
- `RuntimeWeight` 与 staging 不允许隐式 dtype cast；普通 Host tensor允许按 spec cast。
- `run()` 只接受 `KernelBindings`，且每个 bindings 只能消费一次。
- 只有 runtime-owned DeviceTensor 可以 D2H export。
- Step cleanup 会尝试释放全部剩余 leases，并在完成清理后重新抛出首个释放错误。
- 同一 DeviceTensor identity 不能同时归属于两个不同 leases。

## 性能与可观测性

`pool_stats` 暴露 allocation、reuse、H2D/D2H、current/active/peak bytes 和 category bytes。Profile runtime timer 读取最近一次 compile/run 字段。

区分三种 warm 行为：

- Compile cache hit：不再 JIT compile。
- Fixed weight hit：不再为该权重 H2D。
- Buffer reuse：不再 alloc，但有 `init` 的 staging/active upload仍会 H2D。

它们是不同机制，不能只用一个“cache hit”描述。

## 验证方法

### Host 侧 runtime 测试

```bash
pytest -q tests/serving/test_device_runtime.py
```

测试使用 fake ChipWorker 验证 fixed residency、compile cache、active upload、prefill staging 开关、prebound state output、intermediate 消费、selected decode D2H、single-use bindings 和 close。

### NPU 集成验证

```bash
python smoke_model.py \
  --checkpoint /path/to/checkpoint \
  --expert-cache-dir /path/to/expert-cache \
  --seq-len 1 \
  --max-layers 1 \
  --decode-steps 1
```

该方法验证真实 `RunConfig`、`ChipWorker`、compile、alloc/copy、prefill 和 decode dispatch。性能验证可另加 `--profile`；本文不记录一次性时间结果。

## 代码索引

| 内容 | 位置 |
|---|---|
| Runtime | [`serving/device_runtime.py`](../../serving/device_runtime.py) |
| Device pool | [`serving/device_pool.py`](../../serving/device_pool.py) |
| State store | [`serving/device_state_store.py`](../../serving/device_state_store.py) |
| Runner | [`serving/runner.py`](../../serving/runner.py) |
| Runtime 测试 | [`tests/serving/test_device_runtime.py`](../../tests/serving/test_device_runtime.py) |

## 相关文档

- [`Serving 总览`](README.md)
- [`Runtime Values`](08_runtime_values.md)
- [`Device Memory`](09_device_memory.md)
- [`整模 Runner`](05_runner.md)
- [`Profiling`](11_profiling.md)
