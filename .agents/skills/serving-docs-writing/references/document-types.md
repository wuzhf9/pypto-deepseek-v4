# Serving 文档类型

## 1. 选择规则

先按目标选择一种类型：

| 类型 | 适用目标 | 是否使用模板 |
|---|---|---|
| Overview | `docs/serving/README.md` | 否，使用本文件检查清单 |
| Workflow | 用户直接执行的入口或完整离线流程 | 使用 workflow template |
| Runtime component | runner、state、权重、runtime、内存、profile 等内部组件 | 使用 runtime-component template |

一个文档只选择一个主类型。跨类型内容通过链接引用，不在同一篇中复制两个完整模板。

## 2. Overview Profile

`docs/serving/README.md` 负责建立 Serving 实现的全局认知，不承担仓库安装、环境准备或快速运行教程；这些内容属于仓库根目录 README。Serving README 也不按文件逐个展开实现。

建议结构：

1. 当前 Serving 实现范围与职责边界；
2. Serving 模块结构；
3. 在线推理与离线缓存调用链；
4. prefill 与 decode 总览；
5. Host/NPU 数据驻留和生命周期总览；
6. checkpoint 与 expert cache 的实现接口边界；
7. 性能与可观测性；
8. 当前约束与不支持能力；
9. 专题文档导航；
10. 分层验证方法。

README 必须做到：

- 先定义专题文档会反复使用的术语；
- 展示从入口到 runner、runtime 和 model kernel 的实现关系；
- 区分在线生成、smoke 验证和 expert cache 导出的职责边界；
- 提供架构总览，但把实现细节链接到专题文档；
- 保留用于验收 Serving 实现的测试与 NPU 验证方法；
- 不重复根目录 README 中的安装、环境准备和推荐运行步骤；
- 不成为所有专题内容的汇总副本。

## 3. Workflow Profile

适用于：

- `generate.py`；
- `smoke_model.py`；
- `export_expert_cache.py`；
- 其他用户直接执行的完整 Serving 工作流。

重点内容：

- 什么时候使用该入口；
- 前置目录和输入文件；
- 完整命令及参数表；
- 参数组合、默认值和错误条件；
- 从参数解析到资源关闭的执行顺序；
- 标准输出、生成文本或磁盘产物；
- 性能和诊断开关；
- 可复现验证方法。

不要把 CLI 文档写成函数逐项 API 参考。只解释用户行为需要的内部步骤。

## 4. Runtime Component Profile

适用于：

- `DeepSeekV4Runner` 整模编排；
- `DeepSeekV4StatePlan` 与辅助输入缓存；
- `DeepSeekV4WeightLoader` 与 checkpoint layout；
- runtime value descriptors；
- device pool、state store 和 runtime；
- profile recorder。

重点内容：

- 在调用链中的位置和职责边界；
- 上游调用者和下游依赖；
- 核心类型与相邻组件契约；
- prefill/decode 或首次/warm 路径差异；
- tensor 的表示、位置、所有权和生命周期；
- cache key、复用条件和失效/释放行为；
- 状态机与异常清理；
- 性能影响和可观测字段；
- Host 单元测试与硬件验证方法。

如果文档覆盖多个紧密相关文件，例如 device pool 与 state store，应按一个完整生命周期组织内容，而不是按源文件顺序拼接。

## 5. 当前建议映射

| 文档 | 类型 | 主要代码 |
|---|---|---|
| `README.md` | Overview | 全局 |
| `01_generate.md` | Workflow | `generate.py` |
| `02_smoke_model.md` | Workflow | `smoke_model.py` |
| `03_export_expert_cache.md` | Workflow | `export_expert_cache.py` |
| `04_expert_cache.md` | Runtime component | `serving/expert_cache.py` |
| `05_runner.md` | Runtime component | `serving/runner.py` |
| `06_state_plan.md` | Runtime component | `serving/state.py` |
| `07_weight_loader.md` | Runtime component | `serving/weight_loader.py`、`serving/checkpoint.py` |
| `08_runtime_values.md` | Runtime component | `serving/runtime_types.py` |
| `09_device_memory.md` | Runtime component | `serving/device_pool.py`、`serving/device_state_store.py` |
| `10_device_runtime.md` | Runtime component | `serving/device_runtime.py` |
| `11_profiling.md` | Runtime component | `serving/profiler.py` 及调用方 |

该映射是写作路由，不是现状证据。每次写文档仍需重新检查代码；如果目录或职责已经改变，应先更新映射。
