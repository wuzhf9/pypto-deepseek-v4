# Device Resident Backend 解耦方案

## 1. 结论

Device resident 应实现为独立的 `WorkerBackend`，并与 `serving/runner.py` 在实现、tensor 类型和资源
生命周期上完全解耦。

这里的“完全解耦”不是复制一套 `WorkerRunner`。模型层循环、kernel 选择、prefill/decode 流程以及
selected-expert 控制流仍只保留一份，由 `DeepSeekV4Runner` 负责；Direct 和 Worker 通过统一的
backend 接口提供不同的 tensor 存储与执行机制。

目标是：Worker 验证完成后，删除 Direct 只需要删除 Direct 实现、相关测试和 CLI 选项，不再修改
Runner、WeightLoader、State schema 或模型编排代码。

本方案是 `device_resident_implementation_plan.md` 中 backend/runner 边界的进一步收敛。后续实现时，
如两份文档存在冲突，以本文的解耦约束为准。

## 2. 目标依赖关系

```text
serving/generate.py
        │
        ▼
DeepSeekV4Runner                 模型编排
        │
        ▼
ExecutionBackend protocol       唯一 backend 边界
        │
        ├── DirectBackend       Host tensor + 直接 runtime 调用
        │
        └── WorkerBackend       ChipWorker + NPU resident
```

依赖方向必须满足：

```text
runner         → backend protocol
backend factory→ direct / worker concrete backend
direct         → backend protocol + torch/runtime
worker         → backend protocol + ChipWorker/device pool
worker         → state schema、runtime weight descriptor
direct         → state schema、runtime weight descriptor
```

禁止出现：

```text
worker backend → runner.py
runner.py      → ChipWorker / DeviceTensor / DeviceBufferPool
direct backend → worker backend
worker backend → direct backend
```

## 3. Runner 的职责

`DeepSeekV4Runner` 只负责模型语义和执行顺序：

- 校验 prefill/decode 输入。
- 根据 layer spec 选择 block kernel。
- 组织 embedding、layer、selected decode pre/post MoE、head 的调用顺序。
- 调用 WeightLoader 获取逻辑权重或 routed expert host pack。
- 在 selected decode pre kernel 后请求读取 `indices`。
- 维护 profiler 的模型阶段信息。
- 通过 backend 导出最终 public API 输出。

Runner 不再负责：

- Host 或 NPU tensor 分配。
- TensorSpec 的具体 materialize 策略。
- mutable state 的实际存储和交换。
- 固定权重是否上传、复用和释放。
- aux tensor 是否上传、复用和释放。
- hidden/scratch/output 的 buffer 复用。
- H2D/D2H 实现。
- ChipWorker 创建、运行或关闭。
- 对 runtime tensor 调用任何 PyTorch tensor 方法。

Runner 中不得出现以下判断：

```python
if backend == "worker":
if backend.is_resident:
if isinstance(tensor, DeviceTensor):
```

也不得直接执行：

```python
tensor.contiguous()
tensor.to(...)
tensor.float()
torch.isfinite(tensor)
```

上面这些操作只能用于 Runner 明确持有的用户输入或已经由 backend 导出的 Host tensor。

## 4. Backend 的职责

两种 backend 必须实现相同的逻辑能力：

- 根据模型 state schema 初始化执行 session。
- 将 logical values 绑定到 TensorSpec。
- 编译并执行 kernel。
- 管理 mutable state 的 current/next buffer。
- 提交成功 kernel 的 state 输出。
- 读取少量 Host 控制数据。
- 将 public output 导出为 Host `torch.Tensor`。
- 提供统一的 profile counters。
- 幂等释放所有 backend-owned 资源。

两种 backend 的差异只存在于接口内部：

| 能力 | DirectBackend | WorkerBackend |
|---|---|---|
| Tensor 存储 | Host `torch.Tensor` | Worker-owned device tensor |
| 固定权重 | WeightLoader Host cache | Host cache + NPU resident cache |
| mutable state | Host current/next | NPU current/next |
| aux | Host cache | active NPU aux cache |
| scratch/output | Host allocate | device buffer pool |
| kernel 执行 | compiled callable | `ChipWorker.run()` |
| control read | 直接返回/复制 Host tensor | 小规模 D2H |
| public output | Host tensor | 最终 D2H |

## 5. 文件结构

将原计划中的单一 `serving/backend.py` 改为 backend package：

```text
serving/backends/
├── __init__.py
├── base.py
├── direct_backend.py
├── worker_backend.py
├── factory.py
└── device_pool.py
```

具体实现文件使用 `*_backend.py`，使文件名与 `DirectBackend`、`WorkerBackend` 类名直接对应；
`factory.py` 位于 `serving/backends/` 包内，完整路径已经表达 backend 语义，因此不使用重复的
`backend_factory.py`。

### 5.1 `serving/backends/base.py`

包含 backend 无关定义：

- `KernelCase`
- `TensorHandle` opaque protocol/type alias
- `ExecutionBackend` protocol
- `BackendSessionConfig`
- 通用错误类型
- 通用 profile counter 数据结构

该文件不得 import ChipWorker，也不得依赖具体 Direct/Worker 实现。

### 5.2 `serving/backends/direct_backend.py`

从当前 `runner.py` 移入 `_DirectBackend`，并实现完整 `ExecutionBackend`：

- 保持当前 compile cache key 和运行配置。
- Host TensorSpec materialize。
- Host mutable state current/next 管理。
- RuntimeWeight unwrap。
- control read 和 output export。
- close 后清除 compile cache 和 state store。

Direct 的数值、public output 类型和 kernel 行为必须保持当前基线。

### 5.3 `serving/backends/worker_backend.py`

实现完整独立的 `WorkerBackend`：

- 创建唯一、长期存活的 `ChipWorker`。
- 创建并持有 `DeviceBufferPool`。
- 管理 fixed weight resident cache。
- 管理 NPU mutable state current/next。
- 管理 active aux cache。
- 管理 hidden ping-pong、scratch/output 和 routed staging。
- 执行必要 H2D/D2H。
- 使用 `ChipWorker.run()` 执行 compiled kernel。
- 按分类显式释放 device tensor，最后关闭 ChipWorker。

该模块不得 import `serving.runner`。

### 5.4 `serving/backends/device_pool.py`

仅为 WorkerBackend 服务：

- fixed allocations
- state allocations
- active aux allocations
- hidden ping-pong
- reusable scratch/output
- prefill routed staging
- decode selected-expert staging
- allocation bytes 和 copy bytes 统计
- 幂等 close 与泄漏检查

DirectBackend 不得依赖该模块。

### 5.5 `serving/backends/factory.py`

负责 concrete backend 的选择和构造：

```python
def create_backend(
    name: str,
    *,
    platform: str,
    device_id: int,
    runtime_cfg: dict[str, Any] | None,
) -> ExecutionBackend:
    ...
```

Runner 不直接 import `DirectBackend` 或 `WorkerBackend`。如需更强的测试隔离，Runner 构造函数直接
接收 `ExecutionBackend` 实例，由 CLI/factory 在外部完成选择。

推荐的最终构造方式：

```python
backend = create_backend(...)
runner = DeepSeekV4Runner(..., backend=backend)
```

测试可以直接注入 FakeBackend，不需要 monkeypatch ChipWorker。

## 6. ExecutionBackend 接口

第一版建议接口如下，具体参数可在实现时根据 TensorSpec 调整：

```python
class ExecutionBackend(Protocol):
    def prepare(
        self,
        *,
        weight_loader: DeepSeekV4WeightLoader,
        state_template: DeepSeekV4StatePlan,
    ) -> None: ...

    def materialize(
        self,
        case: KernelCase,
        specs: list[TensorSpec],
        values: Mapping[str, LogicalTensorValue],
        *,
        scope: ExecutionScope,
    ) -> KernelBindings: ...

    def run(
        self,
        case: KernelCase,
        specs: list[TensorSpec],
        bindings: KernelBindings,
    ) -> KernelOutputs: ...

    def commit_state(
        self,
        layer_id: int,
        outputs: KernelOutputs,
    ) -> None: ...

    def read_control(self, tensor: TensorHandle) -> torch.Tensor: ...

    def export_output(self, tensor: TensorHandle) -> torch.Tensor: ...

    def release_scope(self, scope: ExecutionScope) -> None: ...

    def close(self) -> None: ...
```

### 6.1 Opaque tensor handle

Runner 不应约束 backend tensor 的具体类型：

```python
TensorHandle = TypeVar("TensorHandle")
```

或定义只暴露 shape/dtype metadata 的 protocol。Runner 不应依赖 `.device`、`.data_ptr()` 或 PyTorch
view 方法。

### 6.2 Logical values

`values` 中允许的逻辑值应集中定义：

```text
RuntimeWeight          固定模型权重，带稳定 cache key
HostDynamicTensor      input_ids、routed/selected pack、动态 aux
TensorHandle           上一个 kernel 的 backend output
StateInputRef          当前 layer state input 引用
StateOutputRef         当前 layer state output 引用
```

Runner 只组装逻辑值；DirectBackend 将其解析为 Host tensor，WorkerBackend 将其解析为 NPU tensor。

### 6.3 State commit

Kernel 成功后 Runner 调用 `commit_state(layer_id, outputs)`。两种 backend 内部完成：

- Direct：Host current/next 引用交换。
- Worker：device current/next handle 交换。

Kernel 抛错时 Runner 不调用 commit，current state 保持有效。

### 6.4 Control read

selected decode 中只通过：

```python
indices_host = backend.read_control(pre_outputs["indices"])
```

获取 expert indices。Direct 直接返回 Host tensor；Worker 执行约 24-byte D2H。Runner 不知道具体
传输方式。

### 6.5 Public output

`prefill()` 和 `decode()` 返回前统一调用：

```python
return backend.export_output(final_output)
```

因此 Direct 和 Worker 对外都保持 Host `torch.Tensor` API。

## 7. State 解耦

`serving/state.py` 继续保持 runtime 无关，只定义：

- `LayerSpec`
- mutable state schema
- state shape/dtype
- state 初始 Host value
- state input/output name mapping
- RoPE/top-k Host cache 和 aux builder

`state.py` 不得 import：

- `ChipWorker`
- device tensor 类型
- `WorkerBackend`
- `DeviceBufferPool`

Host/NPU state 的实际分配由各 backend 在 `prepare()` 中完成。Runner 不再固定创建一套 active Host
state 后再为 Worker 额外创建一套 device state。

如果需要共享逻辑，可增加 runtime-neutral 的：

```text
StateSchema
StateTensorSpec
StateIOPair
```

但不要增加由 Runner 判断类型的 `HostStateStore | ResidentStateStore` 联合分支。

## 8. WeightLoader 解耦

`serving/weight_loader.py` 保持 Host-only：

- 读取 checkpoint。
- 解量化和生成 kernel runtime layout。
- 缓存除 routed experts 外的 Host runtime layout。
- 返回带稳定 identity 的 `RuntimeWeight`。
- routed experts 继续按需构建 Host pack。

WeightLoader 不负责：

- 创建 ChipWorker。
- 上传权重到 NPU。
- 保存 device handle。
- 释放 device allocation。

DirectBackend 使用 `RuntimeWeight.host_tensor`；WorkerBackend 使用其 key 建立 NPU resident cache。这样
移除 DirectBackend 不需要修改 WeightLoader。

## 9. Embedding 路径

Runner 不应出现 direct/worker embedding 分支。

首选方案是 Direct 和 Worker 统一使用输出形状为：

```text
[B, S, HC_MULT, HIDDEN]
```

的 embedding kernel，使后续 layer 输入完全一致。DirectBackend 也执行该 kernel，因此 Runner 无需在
Host 上执行：

```python
out.unsqueeze(2).expand(...).contiguous()
```

如果短期内必须保留旧 kernel，则将两个 kernel variant 封装在 `KernelCase` 中，由 backend 解析，
不能在 Runner 中检查 backend 类型。统一 kernel 仍是最终收敛目标。

## 10. Routed Expert 边界

Routed expert 不进入 fixed device cache，但接口不应使用强模型语义方法：

```python
stage_prefill_routed(...)
stage_decode_selected(...)
```

更通用的接口是由 logical value 携带 staging policy：

```python
HostDynamicTensor(
    tensor=pack,
    reuse_key=("routed", layer_id, mode),
    lifetime=ExecutionLifetime.LAYER,
)
```

DirectBackend 直接使用其 Host tensor；WorkerBackend 按 lifetime 选择或复用 staging buffer 并 H2D。
这样 backend protocol 不依赖 DeepSeek MoE 参数名称。

## 11. Profiler 解耦

Backend 暴露统一 counters/snapshot：

```text
compile_seconds
run_seconds
compile_cache_hit
h2d_bytes / h2d_seconds
d2h_bytes / d2h_seconds
resident_weight_bytes
state_bytes
scratch_peak_bytes
staging_peak_bytes
```

`serving/profiler.py` 只读取该统一 snapshot，不判断 Direct 或 Worker 类型。Direct 中 device/copy 指标
为零。

## 12. Runner 重构后的主流程

### 12.1 Prefill

```text
Runner 获取逻辑 embedding weight/input
→ backend materialize + run embedding
→ 对每层：
     Runner 选择 kernel 并组装 logical values
     backend materialize
     backend run
     backend commit_state
→ 可选 head
→ backend export_output
```

### 12.2 Decode

```text
embedding
→ 每层 selected decode pre：
     backend materialize + run
     backend commit_state
     backend read_control(indices)
→ WeightLoader 根据 Host indices 构建 selected expert pack
→ selected decode post：
     backend materialize + run
→ head
→ backend export_output
```

整个流程中 Runner 不知道中间 tensor 位于 Host 还是 NPU。

## 13. 实现顺序

### Step 1：建立 backend package 和协议

- 新增 `serving/backends/base.py`。
- 新增 `serving/backends/__init__.py`。
- 定义 `KernelCase`、opaque handle、scope 和 backend protocol。
- 暂不改变数值路径。

### Step 2：迁移 DirectBackend

- 新增 `serving/backends/direct_backend.py`。
- 将当前 `_DirectBackend` 从 `runner.py` 原样迁出。
- 补齐统一 materialize/state/export 接口。
- Direct 单元测试和当前远端基线必须通过。

### Step 3：Runner 只依赖协议

- 删除 Runner 内 backend concrete class。
- 将 Host `_materialize_specs()` 下沉到 DirectBackend。
- Runner 改用 opaque output 和 `export_output()`。
- 删除 Runner 对中间 output 的 `.contiguous()`/finite check；调试检查走 backend export/debug API。
- 加 FakeBackend 测试，确认 Runner 不依赖 torch/device tensor 方法。

### Step 4：统一 state schema

- `state.py` 提供 state schema/IO mapping/aux provider。
- DirectBackend 接管 active Host state。
- Runner 仅调用 state binding 和 commit 接口。
- Direct 全量回归。

### Step 5：统一 embedding 输出

- 增加或切换到四维 HC embedding kernel。
- Direct 先验证数值一致。
- Runner 删除 Host expand。

### Step 6：实现 WorkerBackend 基础设施

- 新增 `device_pool.py` 和 `worker_backend.py`。
- 注入 FakeChipWorker 测试 allocation/copy/run/free/close。
- 实现 compile cache 和通用 materialize。

### Step 7：逐类启用 device resident

按以下顺序：

1. 固定权重。
2. mutable state。
3. active aux。
4. hidden ping-pong。
5. scratch/output/intermediate。
6. routed/selected staging。
7. control D2H 和最终 output D2H。

每一步只修改 WorkerBackend 或其私有模块；除非发现 protocol 缺口，不再加入 Runner 的 worker 分支。

### Step 8：完整验证

- Direct current baseline。
- Direct/Worker 单层数值一致。
- max-layer 5 + head 一致。
- 完整 43 层 prefill/decode 一致。
- 多 decode step state swap 正确。
- 第二轮固定权重无重复 H2D。
- 默认执行中间 tensor 无隐式 D2H。

### Step 9：切换默认 backend

- Worker 验证完成后将 CLI 默认改为 `worker`。
- 保留一段时间的显式 `--backend direct` 回归入口。
- 观察完整模型稳定性和性能。

### Step 10：删除 DirectBackend

满足本文第 15 节条件后：

- 删除 `serving/backends/direct_backend.py`。
- 删除 Direct 专属测试和 profile baseline。
- factory 删除 `direct` 分支。
- CLI 删除 `direct` choice，并让 Worker 成为唯一 backend。
- 删除 Direct 专属依赖/import。

不得因此修改 Runner 模型流程。

## 14. 测试要求

### 14.1 Backend contract tests

同一组 contract tests 分别运行于：

- DirectBackend
- WorkerBackend + FakeChipWorker

覆盖：

- prepare/close 幂等性。
- materialize shape/dtype 校验。
- kernel 参数顺序。
- compile cache。
- state commit/失败不 commit。
- control read。
- output export。
- scope release。

### 14.2 Runner backend-neutral test

FakeBackend 返回不具备 `.contiguous()`、`.to()`、`.float()` 的 opaque handle。完整跑通 Runner 单层
编排，以此防止 Runner 重新依赖具体 tensor 类型。

### 14.3 Direct regression

解耦阶段必须保持：

- 当前 unit tests 全部通过。
- 当前远端 Direct 输出一致。
- 当前 Direct 的 total test time 不发生无法解释的明显回退。

### 14.4 Worker lifecycle

验证：

- 一个 Runner session 只创建一个 ChipWorker。
- 固定权重只上传一次。
- decode step 不重复上传固定 state/weight。
- kernel 失败时 state 不 swap。
- close 后所有显式 allocation 已释放。

## 15. 删除 DirectBackend 的准入条件

只有同时满足以下条件才删除 Direct：

- WorkerBackend 完整 43 层 prefill/decode 数值通过。
- 带 head 和 `--no-head` 都通过。
- 多 decode step state 更新正确。
- resident bytes 峰值满足 64 GB 预算。
- 无中间 hidden/state 的默认 D2H。
- 固定权重 warm run 的 H2D bytes 为零。
- routed/selected staging 生命周期正确。
- Worker close 无 device allocation 泄漏。
- Worker 性能稳定优于或不劣于 Direct 的目标基线。
- Runner backend-neutral contract test 通过。

## 16. 完成后的删除范围

正确完成本次解耦后，移除 DirectBackend 的变更范围应严格限制为：

```text
delete serving/backends/direct_backend.py
edit   serving/backends/factory.py
edit   serving/backends/__init__.py
edit   serving/generate.py              # CLI choice/default
delete/edit Direct 专属测试
edit   reference 文档
```

以下文件不应因为删除 Direct 而变化：

```text
serving/runner.py
serving/weight_loader.py
serving/state.py
models/block.py
models/split_block.py
models/moe.py
```

如果删除 Direct 时仍需在 Runner 中清理大量 `if worker` 分支，说明本次 backend 解耦没有完成。
