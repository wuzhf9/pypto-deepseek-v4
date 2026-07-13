# Serving 用户文档规划

## 1. 目标

在 `docs/serving/` 下建立面向使用者的 Serving 文档，说明当前仓库的推理入口、整模编排、权重与专家缓存、Device Runtime、设备内存生命周期以及性能观测方法。

文档必须以当前代码和测试为依据，不使用 `reference/` 中的历史设计文档证明现有行为。`reference/` 仅用于保存本规划。

Serving 文档不直接套用 `docs/models/` 的模块模板：Models 文档重点解释模型语义、Kernel 接口和精度验收；Serving 文档重点解释用户工作流、Host/NPU 数据流、缓存和资源生命周期。不过，两者可以共用“代码证据优先、只记录验证方法、不记录某次验收状态”等基本写作原则。

## 2. 当前代码边界

### 2.1 在线推理与冒烟验证

```text
generate.py / smoke_model.py
        │
        ├── serving/checkpoint.py
        │
        ├── serving/device_runtime.py
        │       ├── serving/device_pool.py
        │       └── serving/device_state_store.py
        │
        └── serving/runner.py
                ├── serving/state.py
                ├── serving/weight_loader.py
                │       └── serving/expert_cache.py
                └── models/* kernels
```

### 2.2 离线专家缓存导出

```text
export_expert_cache.py
        ├── serving/checkpoint.py
        ├── serving/weight_loader.py
        └── packed BF16 expert cache
                     │
                     └── serving/expert_cache.py
```

### 2.3 各模块职责

| 模块 | 当前职责 |
|---|---|
| `generate.py` | 文本 prompt 编码、prefill/decode 生成循环、采样、EOS 处理和性能统计 |
| `smoke_model.py` | 使用随机 token 执行可裁剪层数的 prefill/decode 冒烟验证 |
| `export_expert_cache.py` | 将 routed expert 权重导出为逐层 packed BF16 磁盘缓存 |
| `serving/checkpoint.py` | 统一校验 checkpoint 目录及必需文件 |
| `serving/runner.py` | 整模编排、Block 变体选择、Kernel 参数组织和状态提交 |
| `serving/state.py` | 层类型、状态 schema、RoPE/topk Host cache 以及 prefill/decode 辅助输入 |
| `serving/weight_loader.py` | checkpoint 读取、反量化、runtime layout 转换、Host 权重缓存和专家 staging 构造 |
| `serving/expert_cache.py` | packed BF16 expert cache 的 manifest 校验、整层读取和选中专家切片读取 |
| `serving/runtime_types.py` | runner、weight loader 与 runtime 之间共享的值和生命周期描述符 |
| `serving/device_runtime.py` | ChipWorker 生命周期、Kernel 编译与 dispatch、参数物化和 step 清理 |
| `serving/device_pool.py` | DeviceTensor 分配、复用、拷贝、释放和显存统计 |
| `serving/device_state_store.py` | 以 current/next 双缓冲维护跨 step 的逐层设备状态 |
| `serving/profiler.py` | runner、weight loader 和 runtime 的分段性能事件输出 |

## 3. 建议文档清单

建议建立一个总览和 11 篇编号专题文档，共 12 个文件。

| 文档 | 主要内容 | 代码依据 |
|---|---|---|
| `docs/serving/README.md` | Serving 实现范围、模块结构、执行流、生命周期、目录导航和验证体系 | 全局 |
| `docs/serving/01_generate.md` | prompt 输入、tokenizer、prefill/decode 循环、采样、EOS 和输出 | `generate.py` |
| `docs/serving/02_smoke_model.md` | 冒烟验证、层数裁剪、head、decode steps 和硬件验证方式 | `smoke_model.py` |
| `docs/serving/03_export_expert_cache.md` | 导出命令、参数、选层、覆盖行为、原子写入和磁盘产物 | `export_expert_cache.py` |
| `docs/serving/04_expert_cache.md` | manifest、逐层文件格式、prefill 整包读取、decode 切片读取和 fallback | `serving/expert_cache.py` |
| `docs/serving/05_runner.md` | 整模编排、prefill/decode 差异、Block 选择、decode 拆分和状态提交 | `serving/runner.py` |
| `docs/serving/06_state_plan.md` | 层分类、状态 schema、RoPE/topk Host cache 和辅助输入 | `serving/state.py` |
| `docs/serving/07_weight_loader.md` | checkpoint 映射、反量化、layout、Host cache 及 routed expert 例外 | `serving/weight_loader.py`、`serving/checkpoint.py` |
| `docs/serving/08_runtime_values.md` | `RuntimeWeight`、`HostStagingTensor`、`KernelCase`、`StepContext` 契约 | `serving/runtime_types.py` |
| `docs/serving/09_device_memory.md` | allocation category、lease、buffer 复用、state 双缓冲和释放顺序 | `serving/device_pool.py`、`serving/device_state_store.py` |
| `docs/serving/10_device_runtime.md` | ChipWorker、materialize、编译缓存、Kernel dispatch 和 step 生命周期 | `serving/device_runtime.py` |
| `docs/serving/11_profiling.md` | `--profile`、事件名称、字段含义和性能分析方法 | `serving/profiler.py` 及各入口 |

以下内容不单独成文：

- `serving/__init__.py` 没有独立运行行为。
- `serving/checkpoint.py` 逻辑较小，公共约束放入 README，并在 `07_weight_loader.md` 中解释实现。
- `serving/device_state_store.py` 必须结合内存池和 state schema 才能解释其生命周期，因此与 `device_pool.py` 合并到 `09_device_memory.md`。

## 4. README 结构

`docs/serving/README.md` 保持总览和导航职责，不套用专题文档模板。建议包含：

1. Serving 的实现范围与职责边界。
2. Serving 模块结构。
3. 在线推理与离线 cache 的主调用链。
4. prefill 与 decode 的整体差异。
5. Host cache、固定 NPU 权重、staging、中间 tensor 和 state 的总览。
6. checkpoint 与 packed expert cache 的实现接口边界。
7. 性能与可观测性。
8. 当前约束和不支持的能力。
9. 专题文档导航。
10. 分层验证方法。

仓库安装、环境准备和快速运行命令属于根目录 `README.md`，不在 Serving README 中重复。

## 5. 专题文档统一结构

除 README 外，所有专题文档共享以下主干：

1. **模块定位**：解决什么问题，在 Serving 链路中的位置。
2. **使用场景与边界**：谁调用它、它负责和不负责什么。
3. **代码组成**：相关文件、类、函数及其职责。
4. **公开接口或命令行参数**：以实际签名和当前 CLI 为准。
5. **执行流程**：用代码能够证明的顺序描述主流程和分支。
6. **数据与资源生命周期**：数据位置、所有权、创建、提交、复用和释放时机。
7. **缓存、驻留和复用策略**：只在适用时保留，不适用时删除。
8. **输入约束与错误处理**：形状、dtype、路径、状态机和异常清理。
9. **性能与可观测性**：性能开关、统计字段和关键开销；不写某次性能结果。
10. **验证方法**：单元测试、Host 集成测试和远端 NPU 验证命令分开描述。
11. **相关文档**：链接上下游专题，避免重复大段内容。

文档可以根据类型调整章节侧重点，但不能改变同一术语的含义。

## 6. 两类文档 Profile

### 6.1 用户工作流文档

适用于 `01_generate.md`、`02_smoke_model.md` 和 `03_export_expert_cache.md`。

需要重点说明：

- 环境、checkpoint 和输入准备。
- 可直接执行的完整命令。
- 参数的类型、默认值、互斥或优先级规则。
- 标准输出、性能输出或磁盘产物。
- 入口内部实际执行的步骤。
- 常见错误及其触发条件。
- 与其他入口的区别和选用建议。

### 6.2 运行时组件文档

适用于 `05_runner.md` 至 `11_profiling.md`。

需要重点说明：

- 上游调用方和下游依赖。
- Host 与 NPU 之间的数据位置变化。
- 对象和 DeviceTensor 的所有权。
- persistent、staging、intermediate、scratch 等生命周期类别。
- prefill 和 decode 的行为差异。
- 创建、复用、提交、释放以及异常清理顺序。
- 对正确性或性能有实际影响的 cache key、状态机和配置项。

## 7. 与 Models 文档的区别

Serving 专题文档通常不需要包含：

- official model 数学映射。
- Golden 数学实现。
- Kernel 数值误差阈值。
- 与 Serving 职责无关的完整算子公式。

Serving 专题文档应重点包含：

- CLI 和 Python 接口。
- checkpoint/cache 目录和磁盘格式约束。
- Host 与 NPU 的数据流。
- 内存所有权和资源生命周期。
- prefill/decode 执行差异。
- 可复现的单元测试与整模硬件验证方法。
- profile 事件、计数和耗时字段的解释。

验证章节只记录验收标准和可复现方法，不记录 commit、日期、设备实例、耗时结果或某次 PASS/FAIL 状态。硬件验收只提供当前支持的平台命令，不提供 simulator 命令。

## 8. 代码证据与验证规则

编写每篇文档时按以下优先级取证：

1. 根目录入口脚本的当前 CLI 和控制流。
2. `serving/` 中目标模块的实际实现。
3. 目标模块的调用方和下游模块。
4. `models/` Kernel 的公开 spec 与入口，仅用于确认 Serving 绑定关系。
5. `tests/cli/` 和 `tests/serving/` 的当前验证覆盖。

写作和复核时必须：

- 使用 `rg` 确认重要调用或未调用结论。
- 区分 Host tensor、DeviceTensor、`RuntimeWeight` 和 `HostStagingTensor`。
- 区分 Host layout cache、Device fixed-weight cache、设备 buffer reuse 和磁盘 expert cache。
- 区分 prefill 全 routed pack 与 decode selected experts。
- 区分 step 内 intermediate、跨 step state 和 runner/runtime 全生命周期固定权重。
- 确认所有相对链接存在。
- 确认所有 CLI 参数和验证命令仍与代码一致。
- 完成文档后运行 `git diff --check`。

## 9. 建议实施顺序

1. 新建独立的 `serving-docs-writing` skill，定义 contract、两类 profile 和模板。
2. 完成 `docs/serving/README.md`，先固定术语、边界、执行流和导航。
3. 依次完成 `01_generate.md`、`02_smoke_model.md`、`03_export_expert_cache.md`。
4. 完成 `04_expert_cache.md`，连通离线导出和运行时读取流程。
5. 完成 `05_runner.md`、`06_state_plan.md`、`07_weight_loader.md`。
6. 完成 `08_runtime_values.md`、`09_device_memory.md`、`10_device_runtime.md`。
7. 最后完成 `11_profiling.md`，统一所有事件名、字段说明和性能验证入口。
8. 全量复核交叉链接、术语、CLI 命令和测试路径。

`serving-docs-writing` 可以复用 `models-docs-writing` 的证据和验收原则，但应作为独立 skill 存在，避免在同一个模板中混合模型数学文档与 Serving 生命周期文档。
