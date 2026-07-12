# DirectBackend 移除方案

> 状态：已完成。WorkerBackend 已成为唯一生产 backend；worker-only `--backend` 参数继续保留。

## 1. 结论

当前可以移除 DirectBackend，并将 WorkerBackend 收敛为唯一生产 backend。

已满足的前置条件：

- Runner 只依赖 `Backend` protocol，不 import 或判断具体 backend；
- Worker 已覆盖 embedding、prefill、selected decode、head、state swap 和资源关闭；
- 4 层 Direct/Worker prefill + 多步 decode 已逐元素一致；
- Worker 完整 43 层 head/no-head 已通过，S=1024 prefill 复测通过；
- 43 层带 head warm decode 首次 A/B 中，Worker 为 4,307.312 ms，Direct 为 14,435.375 ms；
- Worker pool peak 约 26.76 GiB，满足 64 GB 设备容量；
- `generate.py` 已能显式使用 `--backend worker`。

本次只删除 Host direct execution 实现，不改变模型编排、Host layout cache、routed expert pack 或
Worker device-resident 语义。

## 2. 接口决策

### 2.1 Worker 成为唯一 backend

`Backend` protocol、`KernelCase`、`KernelBindings` 和 factory 继续保留。它们仍用于：

- 隔离 Runner 与 ChipWorker/DeviceTensor；
- 支持测试注入 fake backend；
- 保留未来新增 backend 的 composition root。

### 2.2 暂时保留 `--backend worker`

第一版移除 Direct 后仍保留 `--backend` 参数，但：

```text
choices=["worker"]
default="worker"
```

这样现有显式使用 `--backend worker` 的脚本不失效，同时不再允许 `--backend direct`。完全删除
`--backend` 参数属于后续独立的 CLI 简化，不与本次删除绑定。

### 2.3 保留 Host runtime value

不得因为删除 Direct 而删除以下内容：

- `RuntimeWeight.host_tensor`：Worker 首次上传固定权重仍需要它；
- WeightLoader Host layout cache：它是 device fixed cache 的上传源和恢复源；
- `HostStagingTensor`：routed/selected expert 仍在 Host 构建后进入 bounded device staging；
- `StepContext` 和 `KernelBindings`：Runner/Worker 生命周期仍依赖它们。

## 3. 删除文件

### 3.1 `serving/backends/direct_backend.py`

整文件删除。被删除的职责均已有 Worker 对应实现：

- Host TensorSpec materialize；
- Direct compiled-program dispatch/cache；
- Host control/output boundary；
- Direct step lifecycle；
- DirectStateStore 委托。

不得把这些逻辑迁回 Runner。

### 3.2 `serving/backends/direct_state_store.py`

整文件删除。生产 mutable state 只由 `WorkerStateStore` 的 device current/next 双缓冲管理。

## 4. 修改生产代码

### 4.1 `serving/backends/base.py`

修改：

```python
BackendName = Literal["worker"]
```

其余 `Backend` protocol、`KernelCase`、`KernelBindings` 不变。不要把 protocol 改成 WorkerBackend
concrete type，也不要在该文件 import ChipWorker。

### 4.2 `serving/backends/factory.py`

修改内容：

- 删除 `DirectBackend` import；
- 删除 `if name == "direct"` 分支；
- 保留 WorkerBackend 的 lazy import；
- 保留未知 backend 的明确 `ValueError`；
- 保留 `keep_prefill_routed_staging` 和 `runtime_cfg` 透传。

目标结构：

```python
def create_backend(name: BackendName, ... ) -> Backend:
    if name == "worker":
        from serving.backends.worker_backend import WorkerBackend

        return WorkerBackend(...)
    raise ValueError(f"unsupported backend: {name!r}")
```

factory 仍是 concrete backend 的唯一构造点。

### 4.3 `serving/run_model.py`

修改 CLI：

```python
parser.add_argument("--backend", choices=["worker"], default="worker")
```

`create_backend(args.backend, ...)`、Runner 注入、异常关闭和 `--enable-l2-swimlane` 透传均保持不变。

### 4.4 `serving/generate.py`

同样将 backend CLI 改为：

```python
parser.add_argument("--backend", choices=["worker"], default="worker")
```

prompt/prompt-file、tokenizer、generation loop、Runner 注入和关闭逻辑均不修改。

### 4.5 明确不修改的生产文件

以下文件不应因删除 Direct 而发生结构性变化：

```text
serving/runner.py
serving/state.py
serving/weight_loader.py
serving/runtime_types.py
serving/backends/worker_backend.py
serving/backends/worker_state_store.py
serving/backends/device_pool.py
models/*
```

如果实现时需要在这些文件加入 `if worker`，说明删除破坏了既有 backend 边界，应停止并重新检查。

## 5. 修改测试

### 5.1 `tests/test_backend.py`

该文件当前混合 Direct 专属测试与 backend-neutral Runner contract 测试，不能整文件删除。

删除：

- `DirectBackend` import；
- `_FakeJitFn`、`_specs()`、`_state_schemas()` 等只服务 Direct 的 helper；
- 所有 `test_direct_backend_*`；
- `test_factory_creates_direct_backend`。

保留：

- `_OpaqueTensor`、`_DelegatingBackend`；
- Runner opaque output/materialize delegation；
- public output boundary；
- begin/end step 异常安全；
- selected indices control read；
- backend state binding 注入；
- Worker factory 构造和未知 backend 拒绝测试。

将模块 docstring 从“direct execution behavior”改为“backend contract and runner orchestration”。

建议增加一个明确回归：

```python
def test_factory_rejects_removed_direct_backend():
    with pytest.raises(ValueError, match="unsupported backend: 'direct'"):
        create_backend("direct", ...)  # type: ignore[arg-type]
```

Direct 测试删除后的覆盖映射：

| 被删除的 Direct 能力 | 现有/目标覆盖 |
|---|---|
| compile cache、参数顺序 | `tests/test_worker_backend.py::test_embedding_slice_reuses_fixed_weight_compile_and_step_buffers` |
| RuntimeWeight/staging materialize | Worker fixed/staging 定向测试与 `tests/test_runtime_types.py` |
| step lifecycle/close | `test_close_cleans_active_step_state_and_worker_once` |
| control/output boundary | selected decode Worker 测试 |
| state current/next、atomic swap | `tests/test_worker_state_store.py` |
| missing/shape/dtype validation | 若 Worker 定向测试缺项，在 `tests/test_worker_backend.py` 补充，不保留 Direct helper |

### 5.2 `tests/test_run_model.py`

修改：

- 主 composition test 不再显式传 `--backend direct`；
- 断言 factory 收到默认 `"worker"`；
- 保留 platform/device/runtime cfg/staging 透传断言；
- 保留 Runner 初始化失败时 backend 关闭测试；
- 增加 `--backend direct` 被 argparse 拒绝的测试。

### 5.3 `tests/test_generate.py`

修改两处 `SimpleNamespace(backend="direct")` 为 `backend="worker"`，并同步 factory captured 断言。

增加/调整 CLI 测试：

- 未指定 `--backend` 时默认为 `worker`；
- 显式 `--backend worker` 可用；
- `--backend direct` 触发 argparse `SystemExit`；
- prompt-file 行为保持不变。

### 5.4 Worker 测试可能的补强

只在现有覆盖确有缺口时修改 `tests/test_worker_backend.py`：

- 补充 missing required value；
- Host tensor shape/dtype mismatch；
- close 后禁止 begin/materialize/run；
- factory 传入 Worker-only 配置。

不要为了保留 Direct 的具体 Host 行为而给 Worker 增加无意义测试。

## 6. 修改 reference 文档

### 6.1 `reference/device_resident_backend_decoupling_plan.md`

将其从“待删除约束”更新为“删除完成记录”：

- 当前依赖图只保留 WorkerBackend；
- 删除 Direct compatibility 章节；
- 将删除准入条件标记为已满足；
- 记录实际删除范围没有触及 Runner/WeightLoader/StatePlan；
- 保留历史 Direct/Worker A/B 结果。

### 6.2 `reference/device_resident_implementation_plan.md`

修改当前状态和 Step 10：

- Worker 标记为唯一 production backend；
- Step 10 标记完成；
- Direct 相关阶段描述保留为历史实施记录，但明确“已删除”；
- 保留数值对齐和性能 A/B 数据作为验收证据。

### 6.3 `reference/device_resident_memory_plan.md`

修改当前语义：

- mutable state 只由 WorkerStateStore 管理；
- Host layout cache 是 Worker fixed upload/recovery source，不再描述为服务 Direct；
- Direct profile 表标记为历史 baseline，保留数据。

### 6.4 历史设计文档

以下文件包含 Direct 的历史设计描述，不需要机械删除全部内容，但必须避免被误认为当前实现：

```text
reference/runtime_layout_device_resident_design.md
reference/runner_impl.md
reference/perf_optimization.md
```

处理方式：

- 在开头或相关章节标注“历史基线，DirectBackend 已移除”；
- 删除 `runner_impl.md` 中“当前默认 direct”的现态描述；
- 保留曾用于方案推导的 Direct 代码片段和 profile 数据。

### 6.5 本方案文档

实现结果：

- 删除了 `direct_backend.py`、`direct_state_store.py`、factory Direct 分支和 Direct-only 测试；
- `BackendName`、`run_model.py` 和 `generate.py` 已收敛为 worker-only，显式 `--backend worker` 继续可用；
- Runner、StatePlan、WeightLoader、runtime value、Worker 实现及模型 kernel 未因本次删除发生结构性修改；
- 本地 `compileall`、`git diff --check` 通过，完整回归为 `276 passed`；
- 远端默认 Worker 5 层带 head + 1 decode 通过，FP32 logits finite，exit=0；
- 远端默认 Worker 43 层 S=1、no-head prefill + 1 decode 通过，BF16 hidden finite，exit=0；
- 远端默认 Worker 43 层 S=1024、no-head prefill 首轮复现 507018，设备强制复位并释放锁；相同命令
  复测输出 `(1, 1024, 4, 4096)`、BF16、finite，exit=0，继续归类为已有偶发 runtime/device 问题。

## 7. 实施顺序

严格按以下顺序执行：

### Phase 1：收敛测试预期

1. 修改 `tests/test_backend.py`，移除 Direct 专属部分并保留 Runner contract；
2. 修改 `tests/test_run_model.py` 和 `tests/test_generate.py`，默认/显式 backend 改为 Worker；
3. 根据覆盖映射补齐必要的 Worker validation tests。

该阶段测试允许因生产代码尚未切换而暂时失败，但失败必须只来自 backend 预期变化。

### Phase 2：切换唯一 backend

4. 修改 `base.py` 的 `BackendName`；
5. 修改 `factory.py`，删除 Direct import/branch；
6. 修改 `run_model.py`、`generate.py`，默认和 choices 收敛到 Worker；
7. 运行 CLI/factory/Runner 定向测试。

### Phase 3：物理删除

8. 删除 `direct_backend.py`；
9. 删除 `direct_state_store.py`；
10. 全仓搜索残留 production/test import 和 `--backend direct`。

### Phase 4：本地验证

11. 运行 `python -m compileall serving models`；
12. 运行 backend/entrypoint 定向测试；
13. 运行完整 `pytest -q`；
14. 运行 `git diff --check`。

### Phase 5：远端验证

15. 不显式指定 backend，验证默认 Worker：

```bash
python serving/run_model.py -p a2a3 -d {} \
  --checkpoint /data/wuzhifeng/dsv4_ckpt \
  --expert-cache-dir /data/wuzhifeng/dsv4_bf16_expert_cache \
  --max-layers 5 --decode-steps 1
```

16. 验证完整 43 层短序列：

```bash
python serving/run_model.py -p a2a3 -d {} \
  --checkpoint /data/wuzhifeng/dsv4_ckpt \
  --expert-cache-dir /data/wuzhifeng/dsv4_bf16_expert_cache \
  --max-layers 43 --no-head -s 1 --decode-steps 1
```

17. 验证当前长序列边界：

```bash
python serving/run_model.py -p a2a3 -d {} \
  --checkpoint /data/wuzhifeng/dsv4_ckpt \
  --expert-cache-dir /data/wuzhifeng/dsv4_bf16_expert_cache \
  --max-layers 43 --no-head -s 1024 --decode-steps 0
```

远端任务均使用 `task-submit --device auto --max-time 0 --run "..."`，功能验证不开启 swimlane。

### Phase 6：更新文档

18. 更新四份 active device-resident 文档；
19. 标记历史设计文档；
20. 全仓 final audit 后提交。

## 8. 验收标准

### 8.1 静态边界

- production/test 不再 import `direct_backend` 或 `direct_state_store`；
- `BackendName` 不再包含 `direct`；
- factory 不再构造 DirectBackend；
- CLI 默认 Worker，`--backend direct` 明确失败；
- Runner、StatePlan、WeightLoader 不出现 Worker concrete type 判断；
- `RuntimeWeight` 和 Host layout cache 仍保留。

建议 audit：

```bash
rg -n "DirectBackend|DirectStateStore|direct_backend|direct_state_store" serving tests
rg -n -- "--backend direct|backend=\"direct\"" serving tests
```

第一条在 `serving tests` 应无结果；reference 中允许存在明确标记为历史的数据。

### 8.2 本地验证

- compileall 通过；
- backend/factory/Runner/CLI 定向测试通过；
- 完整 pytest 通过；
- diff check 通过。

### 8.3 远端验证

- 默认参数实际创建 WorkerBackend；
- 5 层 + head + decode 输出 finite；
- 43 层 prefill/decode 输出 finite；
- S=1024 完整 prefill 输出 finite；
- backend close 正常释放设备锁；
- `--backend worker` 显式调用仍可用。

## 9. 风险与处理

### 9.1 失去在线数值基线

Direct 删除后无法再在 serving entrypoint 中做即时 A/B。处理方式：

- 保留已完成的逐元素对齐结果和性能 A/B 文档；
- kernel correctness 继续由各模型 standalone golden tests 验证；
- Worker runner correctness 由 fake-worker 单测和远端 smoke 验证。

不为保留 A/B 而把 Direct 复制到 tools 或 tests 中形成第二套未维护实现。

### 9.2 CLI 兼容

使用 `--backend direct` 的旧脚本会明确失败，这是预期 breaking change。显式 `--backend worker` 和未指定
backend 的新默认路径都保持可用。

### 9.3 测试覆盖误删

最大的实现风险是整段删除 `tests/test_backend.py`。必须按第 5.1 节拆分，只删除 Direct concrete
behavior，保留 Runner backend-neutral contract。

### 9.4 Host cache 被误删

Direct 删除不等于 Host 权重消失。fixed weight、shared expert 和 aux 仍需从 Host cache 首次上传；routed
expert 仍按 step 在 Host 构建。任何顺手删除 Host cache 的改动都超出本方案范围。

## 10. 回退方案

本次应作为单独 commit 实现。如果远端 Worker 默认路径出现回归：

1. 回退该删除 commit；
2. 恢复 Direct CLI/factory 分支；
3. 不回退此前已验证的 WorkerBackend、device pool、state store 或 embedding 分块修复；
4. 根据失败的 Worker contract 单独修复后再重新执行本方案。
