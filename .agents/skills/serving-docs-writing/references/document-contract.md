# Serving 文档写作契约

## 1. 适用范围

本契约适用于 `docs/serving/README.md` 和所有 `docs/serving/NN_*.md`。文档面向使用和理解当前 Serving 实现的用户，不是开发过程记录或设计提案。

## 2. 事实来源

按以下顺序取证：

1. 根目录入口脚本的当前参数和控制流。
2. `serving/` 中目标模块的实际实现。
3. 目标模块在其他 `serving/` 文件中的真实调用关系。
4. `models/` 的公开 spec builder 和 kernel 入口，仅用于确认 Serving 绑定。
5. `tests/cli/` 和 `tests/serving/` 中的当前行为与验证方法。

禁止用以下内容证明现状：

- `reference/` 中的设计方案或历史分析；
- 已删除 backend、兼容接口或旧 cache 格式；
- 仅存在定义但没有实际调用的代码；
- 用户口述但尚未进入代码的计划。

## 3. 术语规则

必须精确区分：

| 术语 | 文档含义 |
|---|---|
| checkpoint | 原始模型目录及其 tokenizer、索引和权重分片 |
| packed expert cache | 由导出工具生成的逐层 BF16 磁盘文件和 manifest |
| Host layout cache | `DeepSeekV4WeightLoader` 保存的 kernel-facing Host 权重布局 |
| fixed device weight | `DeviceRuntime` 按 `RuntimeWeightKey` 常驻设备的权重 |
| staging | routed expert 权重使用的有界、可复用 H2D 缓冲区 |
| state | 跨 prefill/decode step 保存的逐层可变设备数据 |
| intermediate | kernel 间传递、由 runtime 跟踪的 step 内设备 tensor |
| scratch | kernel 临时工作区，不承载跨 kernel 语义结果 |
| compile cache | runtime 按 kernel 名称、shape 和 dtype 复用编译结果 |

不要把磁盘文件、Host tensor 复用、设备 buffer 复用和编译结果复用统称为同一种 cache。

## 4. 运行路径描述

### 4.1 在线入口

描述 `generate.py` 或 `smoke_model.py` 时，应按实际顺序覆盖：

1. 参数解析和 checkpoint 校验；
2. tokenizer 或随机输入构造；
3. `DeviceRuntime` 创建；
4. `DeepSeekV4Runner` 创建及 state 准备；
5. prefill；
6. 可选的 decode 循环；
7. Host 输出或统计；
8. runner/runtime/loader 清理。

### 4.2 Prefill 与 decode

必须在相关文档中明确：

- prefill 以完整序列运行，并需要完整 routed-expert pack；
- decode 的输入 shape 为 `[1, 1]`，使用 pre-MoE 和 post-MoE 两段；
- decode 在 pre-MoE 后读取 expert indices，再构造 selected-expert staging；
- shared experts 属于固定权重路径，不应与 routed experts 混写；
- state 通过 current/next 设备 buffer 提交和交换。

## 5. 生命周期描述

涉及资源时至少回答：

1. 谁创建它；
2. 谁拥有它；
3. 它位于 Host、NPU 还是磁盘；
4. 生命周期是 kernel、step、跨 step 还是整个 runtime；
5. 何时复用或提交；
6. 正常和异常路径由谁释放。

不要仅凭变量名推断生命周期。必须结合 `materialize()`、`run()`、`begin_step()`、`end_step()`、`commit()` 和 `close()` 等实际调用。

## 6. 接口与参数

- CLI 参数表必须包含当前名称、类型、默认值、约束和作用。
- 只写当前 parser 接受的参数，不保留旧参数兼容说明。
- 参数存在组合关系时，明确互斥、优先级或共同约束。
- Python 接口只列用户或相邻组件实际需要理解的入口，不机械罗列所有私有函数。
- tensor 接口应标明 shape、dtype、Host/NPU 位置和所有权；无法从代码证明时不要猜测。

## 7. 性能与可观测性

- 解释性能开关实际传递到哪个组件。
- profile 事件名、字段和统计范围必须来自当前代码。
- 区分 Python wall time、runtime compile time、kernel run time、weight-loader 子项和 H2D/D2H 计数。
- 不在稳定用户文档中写某次 profile 数值或收益百分比。
- 性能建议必须说明适用路径，例如 prefill、warm decode 或首次编译。

## 8. 验证章节

验证章节只包含方法和预期不变量，不包含某次执行状态。按适用性分组：

1. `tests/cli/`：入口参数、流程和清理行为；
2. `tests/serving/`：Host 侧组件、状态机、缓存和生命周期；
3. 远端 NPU smoke：实际 ChipWorker 和 kernel dispatch；
4. 完整生成：tokenizer、prefill、decode 和输出链路。

硬件命令应使用仓库当前入口和参数，不提供 simulator 命令。只在任务明确要求时执行远端验证。

## 9. 写作与链接

- 正文使用中文，代码标识符和标准术语保留英文。
- 使用相对路径链接代码、测试和相邻文档。
- 不使用容易失效的代码行号。
- 仓库根目录 README 负责安装、准备和快速运行；Serving README 负责统一实现术语、架构、约束、验证体系和专题导航。
- 重复内容优先改为交叉链接。
- 不记录开发阶段、历史 stage、旧 backend 或迁移过程。

## 10. 完成检查

- 每个重要行为都能追溯到当前代码或测试。
- prefill/decode、Host/NPU、固定/staging、状态/中间量没有混淆。
- CLI 参数和命令可由当前 parser 接受。
- 所有链接存在。
- 没有验收状态、日期、设备实例或一次性性能结果。
- `git diff --check` 通过。
