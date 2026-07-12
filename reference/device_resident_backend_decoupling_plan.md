# WorkerBackend 与 Runner 解耦约束

> 状态：已完成。`DirectBackend` 与 `DirectStateStore` 已删除，WorkerBackend 是唯一生产 backend；
> 文中保留的 Direct 数值/性能数据仅作为历史验收证据。

## 1. 当前状态

Direct backend 的结构性解耦及物理删除已经完成，当前代码基线是：

```text
serving/generate.py / serving/run_model.py
                │
                ▼
        DeepSeekV4Runner              模型流程编排
                │
                ▼
       serving.backends.Backend       唯一执行边界
                │
                ▼
           WorkerBackend              唯一生产实现
                │
                ├── WorkerStateStore
                └── DeviceBufferPool / ChipWorker

DeepSeekV4StatePlan                   state schema + immutable aux 生成
DeepSeekV4WeightLoader                Host runtime layout + routed pack
```

`DeepSeekV4Runner` 接收注入的 backend；`DeepSeekV4StatePlan` 也不拥有 mutable runtime state。
WorkerBackend 已在既有 Backend 边界内完成实现，不存在第二套 WorkerRunner。

当前 `factory.py` 的 `backend="worker"` 已替换为真实 WorkerBackend 构造逻辑。Worker 已完成 embedding、
单层 full-routed prefill/selected decode，以及多 ratio、多层和 head 验证；4 层 prefill + 两次 decode
与 Direct 逐元素完全一致，5 层带 head 的 prefill/decode 也已通过。当前 43 层 head/no-head、S=1 及
43 层 S=13 已通过。首次 43 层带 head warm decode A/B 中，Worker 为 4,307.312 ms，Direct 为
14,435.375 ms，Worker 加速 3.3514 倍；Worker pool peak 约 26.76 GiB。embedding 的 32768-task
边界已通过 token-block SPMD 修复，standalone S=4096、Worker 5 层和完整 43 层 S=1024 均已通过。
完整 43 层 S=1024 曾有一轮偶发 507018，但相同命令复测 exit=0，暂不视为稳定的长序列阻塞。性能
稳定性仍需独立 session 复测。

Direct 删除实际只影响 factory/CLI、Direct 实现及其专属测试，没有修改 Runner、模型编排、StatePlan、
WeightLoader 或模型 kernel 的语义。

完整接口和实施顺序以 `device_resident_implementation_plan.md` 为准；本文专门规定依赖边界和可删除性。

## 2. 目标依赖图

```text
generate / run_model
        │
        ├── create_backend(name, options)
        │            │
        │            └── WorkerBackend ── WorkerStateStore
        │                                  │
        │                                  └── DeviceBufferPool ── ChipWorker
        │
        └── DeepSeekV4Runner ── Backend protocol
                    │
                    ├── DeepSeekV4WeightLoader
                    └── DeepSeekV4StatePlan
```

允许的依赖方向：

```text
runner           → Backend protocol
factory          → WorkerBackend
worker backend   → Backend protocol + WorkerStateStore + DeviceBufferPool
state stores     → DeepSeekV4StatePlan / LayerStateSchema
backends         → public runtime value descriptors
weight loader    → public runtime value descriptors
```

禁止的依赖方向：

```text
WorkerBackend → serving.runner
Runner        → ChipWorker / DeviceTensor / DeviceBufferPool
WorkerBackend → DirectBackend / DirectStateStore
WeightLoader  → ChipWorker / DeviceTensor
models/*      → serving.backends
```

## 3. Runner 保留的职责

`DeepSeekV4Runner` 只处理 backend 无关的模型语义：

- 校验 prefill/decode 输入和位置。
- 根据 layer spec 选择 embedding、block、split block 和 head kernel。
- 组织 prefill、selected decode pre/post-MoE 的执行顺序。
- 从 WeightLoader 取得逻辑权重和 routed expert Host pack。
- 在 pre-MoE 后通过 `read_control()` 取得 selected indices。
- 通过 `export_output()` 返回 public Host output。
- 记录模型阶段级 profiler 信息。

Runner 不处理：

- Host/NPU TensorSpec 的实际分配。
- runtime weight 是否上传或命中 device cache。
- mutable state 的存储、current/next 交换和释放。
- hidden、scratch、output 和 staging 的复用。
- H2D/D2H 的实现。
- ChipWorker 的构造、运行和关闭。
- device allocation 的统计。

Runner 中不得新增：

```python
if backend_name == "worker":
if backend.is_resident:
isinstance(value, DeviceTensor)
```

它只能在统一 step 边界调用：

```python
backend.begin_step(step_context)
try:
    # existing model orchestration
finally:
    backend.end_step()
```

`begin_step/end_step` 用于 Worker 回收 active-step 资源。

## 4. Backend 统一能力

当前 `serving/backends/base.py` 的 `Backend` 是唯一协议，以下两个通用扩展已经完成：

- `materialize()` 返回显式 `KernelBindings`，不暴露具体 tensor 类型给 Runner。
- 增加 `begin_step()` / `end_step()` 生命周期钩子。

统一能力包括：

| 能力 | WorkerBackend |
|---|---|
| logical value materialize | owned/borrowed device handle |
| fixed weights | first-miss upload，永久 resident |
| mutable state | WorkerStateStore 双缓冲 |
| scratch/output | DeviceBufferPool |
| routed experts | HostStagingTensor → staging slot |
| kernel dispatch | `ChipWorker.run()` |
| control read | 小规模 D2H |
| public output | 最终 D2H |
| close | 分类释放后关闭 ChipWorker |

`KernelBindings` 由 backend 创建并消费。Runner 只把它从 `materialize()` 传给 `run()`，不得读取其内部
handle 或 ownership metadata。

## 5. 公共 runtime value 层

已新增 `serving/runtime_types.py`，公共语义类型不再放在 Direct、Worker 或 WeightLoader 私有命名空间：

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

class StagingKind(Enum):
    PREFILL_ROUTED = auto()
    DECODE_SELECTED = auto()

@dataclass(frozen=True)
class HostStagingTensor:
    host_tensor: torch.Tensor
    kind: StagingKind
    slot: str
```

规则：

- 普通固定 layout 只能用 `RuntimeWeight`。
- routed/selected pack 只能用 `HostStagingTensor`。
- 普通用户输入和 aux 可以继续是 raw Host tensor，由 WorkerBackend 作为 active-step upload 处理。
- `DeviceTensor` 只在 WorkerBackend 内部或作为明确 borrowed input 出现，不进入 WeightLoader。

这使 device-cache 策略由类型表达，不依赖参数名和 shape 猜测。

## 6. WorkerBackend 的内部边界

### 6.1 `worker_backend.py`

该文件已完成单层 prefill/decode 垂直切片，最终唯一负责：

- lazy import/创建长期存活的 `ChipWorker`。
- 沿用 DirectBackend 的 compile key 语义。
- 把 logical values 转换成 kernel bindings。
- 调用 `ChipWorker.run(compiled, *args, config=...)`。
- 实现 control/output/debug 的 D2H。
- 协调 pool、state store、compiled cache 和关闭顺序。

它不得 import Runner，也不得调用 DirectBackend 复用实现。两者共享逻辑应下沉到 `base.py` 或无设备
依赖的 helper。

### 6.2 `device_pool.py`

该文件已实现，负责 Worker 拥有的 device allocation 和 copy 统计：

- fixed weights
- active uploads
- hidden/intermediate
- scratch/output
- prefill routed staging
- decode selected staging

使用当前已确认的 PyPTO API：

```python
worker.alloc_tensor(shape, dtype, init=host_tensor, worker_id=0)
worker.free_tensor(device_tensor, worker_id=0)
worker.copy_to(dst_dev_ptr, src_host_ptr, nbytes, worker_id=0)
worker.copy_from(dst_host_ptr, src_dev_ptr, nbytes, worker_id=0)
```

pool 必须区分 owned 与 borrowed handle；只释放 owned allocation。

### 6.3 `worker_state_store.py`

该文件已实现，只负责：

- 根据 `LayerStateSchema` 分配 current/next device tensor。
- 返回当前 input 和下一组 output binding。
- kernel 成功后交换 handle。
- kernel 失败时保持 current 不变。
- 关闭时释放两组 state。

它不得自行生成 RoPE/top-k，也不得知道模型层循环。

## 7. DirectBackend 历史验收基线

Direct 曾用于验证公共 Backend contract、state 双缓冲以及 Worker 数值结果。相关逐元素对齐和 A/B 数据
保留为历史证据；生产代码不再提供 Direct 构造、CLI 选择或 Host dispatch 路径。

## 8. Factory 和 CLI 边界

`serving/backends/factory.py` 是唯一 concrete backend 选择点：

```python
backend = create_backend(
    name=args.backend,
    platform=args.platform,
    enable_l2_swimlane=args.enable_l2_swimlane,
    keep_prefill_routed_staging=args.keep_prefill_routed_staging,
)
runner = DeepSeekV4Runner(..., backend=backend)
```

`runner.py` 不解析 backend 名称。`run_model.py` 和 `generate.py` 只负责把 CLI/config 交给 factory。

Worker-only 选项：

- `--backend worker`
- `--enable-l2-swimlane`
- `--keep-prefill-routed-staging`

`--backend` 暂时保留为 worker-only 兼容参数；`--backend direct` 由 argparse 明确拒绝。

## 9. 文件变更边界

新增：

```text
serving/runtime_types.py
serving/backends/device_pool.py
serving/backends/worker_state_store.py
serving/backends/worker_backend.py
tests/test_runtime_types.py
tests/test_device_pool.py
tests/test_worker_backend.py
```

修改：

```text
serving/weight_loader.py
serving/backends/base.py
serving/backends/direct_backend.py
serving/backends/factory.py
serving/runner.py
serving/profiler.py
serving/run_model.py
serving/generate.py
对应现有测试文件
```

第一版不修改：

```text
models/embedding.py
models/block.py
models/split_block.py
models/head.py
serving/state.py
```

`serving/state.py` 已经是 backend-neutral plan/schema；只有发现 schema 无法表达实际 state 时才扩展，不为
DeviceTensor 增加分支。

## 10. DirectBackend 删除完成记录

以下删除准入条件均已满足：

1. Runner 和模型代码只 import `Backend`/公共 runtime types，不 import DirectBackend。
2. factory 是唯一 DirectBackend 构造点。
3. Worker 的单测使用 fake worker，不依赖 DirectBackend。
4. embedding、单层 prefill、selected decode、5 层、43 层及多 step 数值通过。
5. state commit/rollback、staging 复用、fixed cache hit 和幂等 close 均有测试。
6. 远程 NPU 验证覆盖非 tile 对齐序列和 ratio-4/ratio-128 层。
7. Worker warm decode 性能和 device 峰值达到实现方案中的验收线。

实际删除范围为：

```text
delete serving/backends/direct_backend.py
delete serving/backends/direct_state_store.py
delete Direct-only tests
remove direct branch/default from serving/backends/factory.py
remove direct CLI choice
```

Runner、WeightLoader、StatePlan 和模型 kernel 均未发生结构性修改。

## 11. 实施顺序

本文不另建一套阶段计划。严格执行 `device_resident_implementation_plan.md` 的十步顺序：先建立公共
runtime value 和 Backend 生命周期契约，再实现 fake-worker 可测的 pool/state，随后按 embedding、
单层 prefill、selected decode、多层/全模型顺序扩展。这样每一步都保持 Direct 可运行，并把 Worker
专属复杂度限制在 backend package 内。
