# Device Runtime 收敛重构方案

## 1. 目标与前提

当前生产路径只有一个基于 PyPTO `ChipWorker` 的 device-resident 执行实现，并且后续明确不再新增其他类型
backend。本次重构的目标是删除已经失去作用的多 backend 抽象，把公开概念统一收敛为
`DeviceRuntime`，同时继续保持模型编排、设备执行和资源管理之间的职责边界。

重构后的核心关系为：

```text
DeepSeekV4Runner
    │ 模型流程、权重 values、kernel case、prefill/decode 编排
    ▼
DeviceRuntime
    │ ChipWorker、compile/run、H2D/D2H、step 生命周期、device state
    ├── DeviceBufferPool
    └── DeviceStateStore
```

本次重构遵循以下前提：

- 删除 `Backend` Protocol 和 backend factory；
- 删除 worker-only 的 `--backend` CLI 参数；
- 不保留 `WorkerBackend`、`create_backend()`、旧 import 路径或 `--backend worker` 的兼容别名；
- `DeepSeekV4Runner` 仍通过构造参数注入 runtime，不把 `DeviceRuntime` 的职责合并进 Runner；
- `ChipWorker` 是 PyPTO 的真实类型，`DeviceRuntime` 内部的 `_worker` 成员和相关 `worker_id` 参数保持原名；
- 不改变 kernel、TensorSpec、权重格式、数值语义、显存生命周期和 profile event 名称。

## 2. 最终目录结构

重构完成后，删除 `serving/backends/` 目录，相关模块直接放在 `serving/` 下：

```text
serving/
├── device_pool.py
├── device_runtime.py
├── device_state_store.py
├── generate.py
├── profiler.py
├── run_model.py
├── runner.py
├── runtime_types.py
├── state.py
└── weight_loader.py
```

对应测试文件收敛为：

```text
tests/
├── test_device_pool.py
├── test_device_runtime.py
├── test_device_state_store.py
├── test_generate.py
├── test_run_model.py
├── test_runner.py
└── test_runtime_types.py
```

## 3. 名称映射

| 当前名称 | 最终名称 | 说明 |
|---|---|---|
| `BackendName` | 删除 | 单一实现不再需要名称枚举 |
| `Backend` | 删除 | 不再维护多实现 Protocol |
| `create_backend()` | 删除 | 入口直接构造 `DeviceRuntime` |
| `WorkerBackend` | `DeviceRuntime` | 唯一 device-resident 执行 runtime |
| `WorkerKernelBindings` | `KernelBindings` | 唯一 materialize/run bindings 类型 |
| `WorkerStatePair` | `DeviceStatePair` | device state 双 buffer pair |
| `WorkerStateStore` | `DeviceStateStore` | device-resident state 管理器 |
| `backend` 参数/局部变量 | `runtime` | Runner 和 composition 层统一命名 |
| `self.backend` | `self.runtime` | Runner 成员统一命名 |
| `backend_timer()` | `runtime_timer()` | profiler API 与新概念一致 |
| `_record_backend()` | `_record_runtime()` | profiler 内部方法一致 |

以下名称不修改：

- `ChipWorker`、`_worker`、`worker_factory` 和 `worker_id`：它们对应真实 PyPTO API；
- `DeviceBufferPool`、`DeviceLease`、`AllocationCategory`：当前名称已经准确；
- `RuntimeWeight`、`HostStagingTensor`、`StepContext`：仍是 runtime 输入和生命周期描述；
- profile event 名称，例如 `layer.kernel`、`prefill.total` 和 `decode.total`。

## 4. 删除、移动和新增文件

### 4.1 删除文件

| 文件 | 删除内容与原因 |
|---|---|
| `serving/backends/base.py` | 删除 `BackendName`、`Backend`、通用 `KernelBindings`；`KernelCase` 先迁移到 `runtime_types.py` |
| `serving/backends/factory.py` | 删除只有 `worker` 分支的 `create_backend()` |
| `serving/backends/__init__.py` | 删除已经不存在的 backends package |
| `serving/backends/worker_backend.py` | 内容迁移并重命名到 `serving/device_runtime.py` 后删除 |
| `serving/backends/worker_state_store.py` | 内容迁移并重命名到 `serving/device_state_store.py` 后删除 |
| `serving/backends/device_pool.py` | 内容迁移到 `serving/device_pool.py` 后删除 |
| `tests/test_worker_backend.py` | 测试迁移到 `tests/test_device_runtime.py` 后删除 |
| `tests/test_worker_state_store.py` | 测试迁移到 `tests/test_device_state_store.py` 后删除 |
| `tests/test_backend.py` | Runner 编排测试迁移到 `tests/test_runner.py`；factory 测试删除后移除旧文件 |

`__pycache__` 不属于源码，不纳入补丁；本地运行测试后由环境自行生成，也不得提交。

### 4.2 新增文件

| 文件 | 来源与内容 |
|---|---|
| `serving/device_runtime.py` | 从 `worker_backend.py` 迁移，公开 `DeviceRuntime` 和 `KernelBindings` |
| `serving/device_state_store.py` | 从 `worker_state_store.py` 迁移，公开 `DeviceStatePair` 和 `DeviceStateStore` |
| `serving/device_pool.py` | 从原 backends package 平移，更新 import 和 docstring |
| `tests/test_device_runtime.py` | 原 WorkerBackend 单元测试按新 API/名称迁移 |
| `tests/test_device_state_store.py` | 原 WorkerStateStore 单元测试按新名称迁移 |
| `tests/test_runner.py` | 保留原 `test_backend.py` 中真正验证 Runner/runtime 边界的测试 |

## 5. 各文件具体修改

### 5.1 `serving/runtime_types.py`

1. 从 `base.py` 迁入 `KernelCase`：

   ```python
   @dataclass(frozen=True)
   class KernelCase:
       name: str
       fn: Any
       spec_builder: Any
   ```

2. 增加所需的 `Any` import，并把 `KernelCase` 加入 `__all__`；
3. 将模块描述从 backend-neutral 调整为 serving runtime descriptors；
4. 更新 `HostStagingTensor`、`StepContext` 等 docstring 中不再准确的 backend 表述；
5. 不在该文件中放置 `KernelBindings`：bindings 包含 `DeviceLease`，属于具体 `DeviceRuntime` 的内部 dispatch
   类型。

### 5.2 `serving/device_pool.py`

从 `serving/backends/device_pool.py` 平移实现，只做概念和 import 收敛：

1. 保持所有 allocation、reuse、copy、accounting 行为不变；
2. 将 `WorkerBackend-owned` 改为 `DeviceRuntime-owned`；
3. 将“owning WorkerBackend”改为“owning DeviceRuntime”；
4. 保留 `_worker`、`worker_id`、`alloc_tensor()`、`copy_to()`、`copy_from()` 和 `free_tensor()` 等真实
   ChipWorker 相关命名；
5. 不修改 allocation category、reuse key、释放顺序或统计口径。

### 5.3 `serving/device_state_store.py`

从 `worker_state_store.py` 迁移并完成以下修改：

1. import 改为 `serving.device_pool`；
2. `WorkerStatePair` 改为 `DeviceStatePair`；
3. `WorkerStateStore` 改为 `DeviceStateStore`；
4. `_layers` 和 `_layer()` 的类型标注改为 `DeviceStatePair`；
5. 异常字符串中的 `WorkerStateStore` 改为 `DeviceStateStore`；
6. 更新 `__all__`；
7. state current/next 双 buffer、prepare/commit/close 行为保持不变。

### 5.4 `serving/device_runtime.py`

从 `worker_backend.py` 迁移，是本次重构的核心文件：

1. import 调整为：
   - `KernelCase` 从 `serving.runtime_types` 导入；
   - pool 类型从 `serving.device_pool` 导入；
   - state store 从 `serving.device_state_store` 导入；
2. 删除对 `Backend` 和通用 `KernelBindings` 的依赖；
3. 将 `WorkerKernelBindings` 改为独立的 `KernelBindings` dataclass，字段保持：
   - `tensors`；
   - `scratch_leases`；
   - `transient_leases`；
   - `consumed_leases`；
   - `output_tensors`；
   - `consumed`；
4. 将 `WorkerBackend` 改为 `DeviceRuntime`；
5. `_state_store` 构造改为 `DeviceStateStore(self._pool)`；
6. `materialize()` 返回 `KernelBindings`；
7. `run()` 参数直接标注为 `KernelBindings`，保留重复消费检查，但不再需要“必须是
   WorkerKernelBindings”的跨实现检查；如仍保留防御性类型检查，错误消息必须使用 `DeviceRuntime` 和
   `KernelBindings`；
8. 将错误消息中的 `WorkerBackend` 改为 `DeviceRuntime`；
9. 更新类和模块 docstring、`__all__`；
10. 保持以下行为完全不变：
    - `ChipWorker`/`RunConfig` 延迟导入；
    - compile cache key；
    - fixed weights 常驻；
    - routed/selected staging 生命周期；
    - active upload cache；
    - step lease 归还顺序；
    - `--keep-prefill-routed-staging` 对应的构造参数和语义；
    - export/read_control/close 行为；
    - pool profile/accounting。

### 5.5 `serving/runner.py`

1. import `DeviceRuntime` 和新的 `KernelCase`；
2. 构造参数由 `backend: Backend` 改为 `runtime: DeviceRuntime`；
3. `self.backend` 全部改为 `self.runtime`；
4. 以下调用只改接收者名称，不改时序：
   - `prepare_state()`；
   - `begin_step()` / `end_step()`；
   - `materialize()` / `run()`；
   - `state_inputs()` / `state_outputs()` / `commit_state()`；
   - `read_control()`；
   - `export_output()` / `export_debug_tensor()`；
   - `close()`；
5. `profiler.backend_timer()` 改为 `profiler.runtime_timer()`；
6. 模块和类 docstring 中的 backend-owned/backend-neutral 改为 runtime-owned/device runtime；
7. 保持 runtime 构造注入，不允许 Runner 内部自行创建 `DeviceRuntime`；
8. Python 测试仍可注入实现相同方法的 fake runtime，不新增运行时类型强制。

### 5.6 `serving/profiler.py`

1. `backend_timer()` 改为 `runtime_timer()`；
2. 参数 `backend` 改为 `runtime`；
3. `_record_backend()` 改为 `_record_runtime()`；
4. `last_compile_seconds`、`last_run_seconds` 和 `last_compile_cache_hit` 的读取逻辑保持不变；
5. 所有打印出的 profile event 名称和字段保持不变，确保旧、新 profile 可以直接 A/B。

### 5.7 `serving/generate.py`

1. 删除 `create_backend` import，改为 import `DeviceRuntime`；
2. 从 argparse 删除 `--backend`；
3. `_create_runner()` 中直接构造 `DeviceRuntime`；
4. 局部变量 `backend` 改为 `runtime`；
5. Runner 构造参数由 `backend=backend` 改为 `runtime=runtime`；
6. Runner 初始化失败时仍立即 `runtime.close()`，保持异常资源清理；
7. `platform`、`device`、`enable_l2_swimlane`、`keep_prefill_routed_staging` 的透传保持不变；
8. prompt、tokenizer、generation 和输出逻辑不修改。

这是有意的 CLI breaking change：

```bash
# 旧命令
python serving/generate.py ... --backend worker

# 新命令
python serving/generate.py ...
```

重构后传入任何 `--backend` 都由 argparse 报 unknown argument，不再特判 `direct`。

### 5.8 `serving/run_model.py`

与 `generate.py` 使用相同的 composition 方式：

1. 模块描述改为 device-runtime-injected runner smoke entrypoint；
2. 删除 factory import 和 `--backend` 参数；
3. 直接构造 `DeviceRuntime`；
4. 局部变量和 Runner 参数统一改为 `runtime`；
5. Runner 初始化异常时仍关闭 runtime；
6. 其余 smoke、prefill/decode 和精度检查逻辑保持不变。

## 6. 测试修改

### 6.1 `tests/test_runtime_types.py`

- import 新的 `KernelCase`；
- 增加最小构造测试，确认 name、fn 和 spec_builder 原样保存；
- 其余 RuntimeWeight、HostStagingTensor 和 StepContext 测试不变。

### 6.2 `tests/test_device_pool.py`

- import 路径从 `serving.backends.device_pool` 改为 `serving.device_pool`；
- pool 行为测试本身不修改；
- 测试名称中如有 backend/worker 概念，改为 device runtime ownership 表述。

### 6.3 `tests/test_device_state_store.py`

- 从 `test_worker_state_store.py` 迁移；
- import 改为 `DeviceStateStore` 和新的 device pool 路径；
- 局部变量和断言中的 `WorkerStateStore` 改为 `DeviceStateStore`；
- prepare、双 buffer swap、非法输出、重复 prepare 和 close 测试全部保留。

### 6.4 `tests/test_device_runtime.py`

- 从 `test_worker_backend.py` 迁移；
- import 改为 `DeviceRuntime`、`KernelBindings`、新 pool 路径和 `runtime_types.KernelCase`；
- helper `_backend()` 改为 `_runtime()`；
- 测试变量 `backend` 改为 `runtime`；
- `WorkerKernelBindings` 断言改为 `KernelBindings`；
- `FakeChipWorker` 保持原名，因为它模拟的是真实 PyPTO 类型；
- materialize、compile cache、state、selected staging、routed staging、step cleanup、close、pool stats 和异常
  回收测试全部保留；
- 不新增旧 `WorkerBackend` import 兼容测试。

### 6.5 `tests/test_runner.py`

从当前 `tests/test_backend.py` 迁移真正属于 Runner/runtime 边界的测试：

- opaque device output 在层间传递；
- 只在公共输出边界 export；
- prefill/decode step begin/end；
- 执行异常时 end_step；
- begin_step 失败时不错误调用 end_step；
- selected indices 通过 runtime `read_control()` 读取；
- state inputs/outputs 由 runtime 注入。

同时完成：

- `_DelegatingBackend` 改为 `_FakeRuntime`；
- fake bindings 使用测试私有 dataclass/简单对象，或使用新的 `KernelBindings`；
- `runner.backend` 改为 `runner.runtime`；
- 删除 `create_backend()` 创建/reject unknown/reject direct 的全部 factory 测试。

### 6.6 `tests/test_generate.py`

1. 删除显式 `--backend worker` 成功测试；
2. 删除 `--backend direct` choices 拒绝测试；
3. parse args 结果不再断言 `args.backend`；
4. 用 monkeypatch `generate.DeviceRuntime` 的 fake constructor 替代 monkeypatch `create_backend`；
5. 断言 platform/device/runtime_cfg/keep-prefill 参数仍正确传入；
6. 断言 Runner 收到 `runtime=...`，且没有收到 platform/device 等 runtime 构造参数；
7. Runner 初始化失败时断言 fake runtime 被关闭；
8. 从测试 `SimpleNamespace` 中删除 `backend="worker"`；
9. prompt-file、tokenizer、generation 和输出测试保持不变。

### 6.7 `tests/test_run_model.py`

1. `_FakeBackend` 改为 `_FakeRuntime`；
2. monkeypatch `run_model.DeviceRuntime` constructor；
3. 断言 Runner 使用 `runtime=` 注入；
4. 断言 runtime 构造参数和初始化失败清理行为不变；
5. 删除 `--backend direct` 测试；
6. 测试模块描述从 backend composition 改为 device runtime composition。

## 7. 实施顺序

为避免中间状态出现循环 import 或同时维护两套类型，按以下顺序修改：

### Stage 1：迁移公共描述类型

1. 将 `KernelCase` 加入 `serving/runtime_types.py`；
2. 更新 `tests/test_runtime_types.py`；
3. 暂时允许旧 `base.py` 从新位置 import/re-export，或者在同一个补丁中立即切换所有消费者；最终状态不得
   保留 re-export。

### Stage 2：迁移底层资源模块

4. 将 device pool 移到 `serving/device_pool.py`；
5. 将 state store 迁移并重命名为 `serving/device_state_store.py`；
6. 迁移对应 pool/state store 测试；
7. 先运行 pool/state store 单元测试，确认纯资源管理行为未改变。

### Stage 3：迁移唯一执行 runtime

8. 创建 `serving/device_runtime.py`；
9. 将 `WorkerBackend`/`WorkerKernelBindings` 收敛为 `DeviceRuntime`/`KernelBindings`；
10. 更新 import、错误消息和 docstring；
11. 迁移 `tests/test_worker_backend.py` 为 `tests/test_device_runtime.py`；
12. 运行 runtime 单元测试，确认 compile、copy、state 和 lease 生命周期不变。

### Stage 4：切换 Runner 和 profiler

13. 将 profiler API 改为 runtime 命名；
14. 将 Runner 构造参数、成员和所有调用点从 backend 改为 runtime；
15. 将 `tests/test_backend.py` 的 Runner 编排部分迁移到 `tests/test_runner.py`；
16. 删除 factory 专属测试；
17. 运行 Runner 和 profiler 相关测试。

### Stage 5：切换 composition/CLI

18. 修改 `generate.py`，直接构造 `DeviceRuntime` 并删除 `--backend`；
19. 修改 `run_model.py`，执行相同收敛；
20. 更新 `tests/test_generate.py` 和 `tests/test_run_model.py`；
21. 验证无 `--backend` 命令正常，显式传入旧参数时由 argparse 明确失败。

### Stage 6：删除旧实现与清理残留

22. 删除 `base.py`、`factory.py` 和 backends package 下的旧模块；
23. 删除/迁移旧名称测试文件；
24. 删除空的 `serving/backends/` 源码目录；
25. 全仓搜索生产代码和测试中的残留：

   ```bash
   rg -n "serving\.backends|BackendName|Backend\b|create_backend|WorkerBackend|WorkerKernelBindings|WorkerStateStore|WorkerStatePair|backend_timer|--backend" serving tests
   ```

26. 搜索结果必须为空；真实 `ChipWorker`、`FakeChipWorker`、`_worker` 和 `worker_id` 不属于残留。

### Stage 7：完整验证

27. 运行格式/静态检查（以仓库当前可用检查项为准）；
28. 运行相关单元测试；
29. 运行完整本地测试；
30. 同步到 Ascend 服务器执行最小 prefill/decode 和多层 smoke；
31. 对比重构前后的输出 shape、dtype、finite、profile event 和 pool 释放结果。

## 8. 验证方案

### 8.1 本地定向测试

```bash
pytest -q \
  tests/test_runtime_types.py \
  tests/test_device_pool.py \
  tests/test_device_state_store.py \
  tests/test_device_runtime.py \
  tests/test_runner.py \
  tests/test_generate.py \
  tests/test_run_model.py
```

### 8.2 本地完整测试

```bash
pytest -q tests
```

### 8.3 CLI 检查

```bash
python serving/run_model.py --help
python serving/generate.py --help
```

验收要求：help 中不再出现 `--backend`，其他 runtime 参数仍存在。

### 8.4 Ascend 远端 smoke

使用现有 checkpoint 和最终 packed expert cache，至少执行：

```bash
python serving/run_model.py \
  --checkpoint /data/wuzhifeng/dsv4_ckpt \
  --expert-cache-dir /data/wuzhifeng/dsv4_bf16_expert_cache \
  --max-layers 1 --no-head -s 1 --decode-steps 1
```

随后增加到 5 层并开启 head：

```bash
python serving/run_model.py \
  --checkpoint /data/wuzhifeng/dsv4_ckpt \
  --expert-cache-dir /data/wuzhifeng/dsv4_bf16_expert_cache \
  --max-layers 5 -s 1 --decode-steps 1
```

本次是纯命名和模块边界重构，不要求重新跑完整 43 层性能 A/B；但 profile smoke 必须确认 event 名称和字段
没有变化。若 wall time 出现明显变化，则说明重构意外改变了初始化或资源生命周期，应停止收敛并定位。

## 9. 验收标准

重构完成必须同时满足：

- `serving/backends/` 不再包含源码；
- 生产代码和测试不再存在 backend factory、Backend Protocol 和 worker-only 公共类型；
- 唯一公开执行类为 `DeviceRuntime`；
- Runner 使用 `runtime=` 注入且不负责创建 runtime；
- `KernelCase` 位于 `runtime_types.py`，`KernelBindings` 位于 `device_runtime.py`；
- `DeviceRuntime` 内部继续使用真实 `_worker: ChipWorker`；
- CLI 不再接受 `--backend`；
- 原有 step、state、fixed weight、staging、pool reuse 和 close 行为全部通过测试；
- Ascend prefill/decode smoke 精度、资源释放和 profile 输出通过；
- 不保留旧类名、旧模块或旧 CLI 参数的兼容层。

## 10. 文档处理原则

`device_resident_backend_decoupling_plan.md`、`direct_backend_removal_plan.md` 等文件记录的是历史设计和实施
过程，不进行全局机械改写。本文作为完成单 runtime 收敛后的最新实施依据；后续活跃性能文档如需要引用
生产类型，应使用 `DeviceRuntime`、`KernelBindings` 和 `runtime` 新名称。
