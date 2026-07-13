# DeepSeek V4 Flash Serving

`serving/` 提供 DeepSeek V4 Flash BF16 PyPTO 模型的单卡推理运行时。它负责从 checkpoint 准备权重和状态，通过一个长生命周期 `ChipWorker` 调度 [`models/`](../models/README.md) 中的 Kernel，并在 Host 与 NPU 之间管理固定权重、专家 staging、中间 tensor 和跨 step 状态。

当前仓库提供三个根目录入口：

- [`generate.py`](../../generate.py)：输入文本 prompt，执行完整的 43 层 prefill 与逐 token decode。
- [`smoke_model.py`](../../smoke_model.py)：使用随机 token 验证指定层数、head 和 decode step。
- [`export_expert_cache.py`](../../export_expert_cache.py)：将 routed expert 权重导出为运行时可直接读取的逐层 packed BF16 磁盘缓存。

## 实现范围

当前 Serving 路径具有以下边界：

| 项目 | 当前行为 |
|---|---|
| 模型 | DeepSeek V4 Flash BF16 PyPTO 实现 |
| Batch | 仅支持 `B=1` |
| 最大序列长度 | 固定为 `4096` |
| 设备 | 单个 `ChipWorker`、单卡逻辑 |
| 正式生成层数 | `generate.py` 固定执行配置中的 43 层 |
| Prefill 输入 | Host `int64` tensor，shape 为 `[1, S]`，`1 <= S <= 4096` |
| Decode 输入 | Host `int64` tensor，shape 为 `[1, 1]`，`0 < start_pos < 4096` |
| 模型状态 | 初始化后常驻 NPU，通过逐层 current/next 双缓冲更新 |
| 普通权重 | 在 Host 构造最终 layout，首次使用后作为 fixed weight 常驻 NPU |
| Routed experts | prefill 使用完整专家包，decode 仅 staging 被选中的专家 |
| 输出 | 每次 prefill/decode 结束时复制回 Host `torch.Tensor` |
| 并行能力 | 不实现 tensor、pipeline、expert 或 data parallel |

[`DeepSeekV4StatePlan`](../../serving/state.py) 当前要求 `batch_size=1`、`max_seq_len=4096` 和 `window_size=128`。`generate.py` 不暴露序列长度参数，内部始终使用 `DEFAULT_MAX_SEQ_LEN=4096`。

## 模块结构

```text
serving/
├── checkpoint.py          # checkpoint 目录校验
├── runner.py              # prefill/decode 整模编排
├── state.py               # state schema 与 Host 辅助输入 cache
├── weight_loader.py       # checkpoint、runtime layout 和专家 staging
├── expert_cache.py        # packed BF16 expert cache reader
├── runtime_types.py       # runner/loader/runtime 共享值类型
├── device_runtime.py      # ChipWorker、materialize、compile 和 dispatch
├── device_pool.py         # DeviceTensor 分配、复用和统计
├── device_state_store.py  # 逐层 state current/next 双缓冲
└── profiler.py            # Serving profile 事件
```

## 入口职责

| 入口 | 使用场景 | 主要输入 | 主要输出 |
|---|---|---|---|
| `generate.py` | 完整文本生成 | checkpoint、prompt、可选 expert cache | 模型回复和生成统计 |
| `smoke_model.py` | 层级裁剪、prefill/decode 和有限值检查 | checkpoint、随机输入配置、可选 expert cache | 每个阶段的 shape、dtype 和有限值状态 |
| `export_expert_cache.py` | 在线推理前准备 routed experts | checkpoint、输出目录、可选层范围 | packed BF16 文件和 manifest |

`generate.py` 和 `smoke_model.py` 创建 NPU runtime；`export_expert_cache.py` 在 Host 侧完成读取、layout 转换和磁盘写入，不创建 `ChipWorker`。具体命令行接口分别由对应的 workflow 文档说明。

## 在线推理主流程

```text
generate.py / smoke_model.py
        │
        ├── 校验 checkpoint，构造输入
        ├── 创建 DeviceRuntime 和一个 ChipWorker
        └── 创建 DeepSeekV4Runner
                │
                ├── DeepSeekV4StatePlan
                │       ├── 描述逐层设备 state
                │       └── 缓存 RoPE、topk indices 等 Host 辅助输入
                │
                ├── DeepSeekV4WeightLoader
                │       ├── 读取 checkpoint
                │       ├── 构造并缓存固定 Host runtime layout
                │       └── 从 expert cache 或 checkpoint 构造 routed experts
                │
                └── DeviceRuntime
                        ├── 将固定权重常驻 NPU
                        ├── 管理 routed-expert staging
                        ├── 管理 state、intermediate 和 scratch
                        ├── 缓存已编译 Kernel
                        └── 通过 ChipWorker 调度 models/* Kernel
```

[`DeepSeekV4Runner`](../../serving/runner.py) 只负责整模编排。模型数学仍由 [`models/`](../models/README.md) 中的 embedding、Block、split Block 和 head Kernel 实现。

离线 expert cache 路径不经过 runner 或 device runtime：

```text
export_expert_cache.py
        │
        ├── 校验 checkpoint 与目标层范围
        ├── DeepSeekV4WeightLoader 读取并转换 routed experts
        ├── 每层聚合为三个 packed BF16 tensor
        └── 原子写入 safetensors 文件并更新 manifest
```

## Prefill 与 decode

| 阶段 | Prefill | Decode |
|---|---|---|
| 输入 | `[1, S]` 的完整 prompt token | `[1, 1]` 的单个 token |
| `StepContext` | `kind=PREFILL`、`start_pos=0` | `kind=DECODE`、`start_pos>0` |
| Block 执行 | 每层执行一个完整 prefill Block | 每层拆为 pre-MoE 和 post-MoE 两个 Kernel |
| Routed experts | 每层加载全部 256 个专家的 packed 权重 | pre-MoE 先产生路由 indices，再加载 6 个 selected experts |
| 路由控制量 | 不需要在 Block 中间回 Host | indices 从 NPU 复制回 Host，用于选择专家权重 |
| State | 写入每层 next buffer 后提交 | 读取 current、写入 next，pre-MoE 后提交 |
| 层间 hidden | 由 runtime 作为 NPU intermediate 直接传递 | 由 runtime 作为 NPU intermediate 直接传递 |
| 最终结果 | logits 或 hidden 复制回 Host | logits 或 hidden 复制回 Host |

decode 拆分的目的，是在得到本 token 的路由结果后只 staging 被选中的专家，避免为每个 decode step 加载全部 256 个 routed experts。shared expert 权重不走该 staging 路径，而是作为固定权重常驻 NPU。

## 数据驻留与生命周期

Serving 中的“缓存”和“复用”分为不同层次：

| 数据或资源 | 位置 | 所有者 | 生命周期 |
|---|---|---|---|
| Checkpoint 权重分片 | 磁盘 | 用户提供 | 仓库外部管理 |
| Packed expert cache | 磁盘 | 导出工具生成、用户管理 | 跨进程保留 |
| Host runtime layout | Host | `DeepSeekV4WeightLoader` | loader 生命周期，routed experts 除外 |
| Fixed device weight | NPU | `DeviceRuntime` | 首次 materialize 至 runtime 关闭 |
| Prefill routed staging | Host + NPU staging | weight loader + runtime | Host tensor 按层构造；NPU buffer 默认在 prefill step 后释放 |
| Decode selected staging | Host + NPU staging | weight loader + runtime | 每层 decode post-MoE 使用后归还复用池 |
| Mutable state | NPU | `DeviceStateStore` | runner 初始化至 runtime 关闭，current/next 交替提交 |
| Intermediate | NPU | `DeviceRuntime` | kernel 间传递，最后消费者运行后归还 |
| Scratch | NPU | `DeviceRuntime` | 单次 Kernel dispatch |
| Active Host upload | Host 到 NPU | `DeviceRuntime` | 当前 prefill/decode step 内按 Host tensor 身份复用 |
| Compiled Kernel | Host runtime 对象 | `DeviceRuntime` | 按 kernel、shape、dtype 缓存至 runtime 关闭 |

`--keep-prefill-routed-staging` 只保留 prefill routed staging 的三块 NPU 分配以供后续同 shape staging 复用，不会把 256 个专家权重本身固定常驻 NPU。默认关闭时，这些 staging 分配在 prefill step 结束后释放。

## Checkpoint 与 expert cache 边界

三个根目录入口都通过 [`validate_checkpoint_directory()`](../../serving/checkpoint.py) 统一校验 checkpoint。校验层直接要求 `tokenizer.json` 和 `model.safetensors.index.json`；[`DeepSeekV4WeightLoader`](../../serving/weight_loader.py) 随后根据索引定位和读取权重分片。prompt encoding 直接来自仓库内的 [`official/encoding_dsv4.py`](../../official/encoding_dsv4.py)，Serving 不从 checkpoint 的额外 `encoding/` 目录加载代码。

Packed BF16 expert cache 是独立于 checkpoint 的可选磁盘布局。它由 `manifest.json` 和逐层 `layer_NNN_experts.safetensors` 组成，manifest 必须与当前 cache format version、层数、专家数、维度和 dtype 匹配。每层文件只保存 `routed_w1_t`、`routed_w2_t` 和 `routed_w3_t`：prefill 读取完整 tensor，decode 只读取路由选中的 expert slice。

没有 expert cache 时，weight loader 回退到原始 checkpoint：prefill 逐个构造全部 256 个 routed experts，decode 逐个构造当前 token 选中的 6 个 experts。shared expert 和其他普通权重始终走固定 Host layout 与 fixed device weight 路径，不属于 packed expert cache。

## 性能与诊断入口

在线入口支持以下公共诊断选项：

| 选项 | 作用 |
|---|---|
| `--profile` | 输出 runner 分段时间、runtime compile/run 时间和 weight-loader 子项 |
| `--verbose-layer-log` | 输出逐层模式、Block 变体、输入输出信息及有限值检查 |
| `--enable-l2-swimlane` | 将 `enable_l2_swimlane=True` 传给 PyPTO `RunConfig` |
| `--keep-prefill-routed-staging` | prefill 结束后保留 routed staging 的设备分配以便复用 |

首次遇到某个 kernel、shape 和 dtype 组合时会发生编译；相同 runtime 内再次命中该组合时复用编译结果。`--profile` 输出的 Python wall time、compile time、kernel run time 和权重加载子项含义不同，不应直接混为一个 Kernel 耗时。

## 当前约束

- 仅实现单卡、单 `ChipWorker` 执行，不提供并行 Serving。
- 仅支持 `B=1`。
- 当前状态布局固定 `max_seq_len=4096`，`generate.py` 不提供修改该值的 CLI 参数。
- `prompt_tokens + max_new_tokens` 不能超过 `4096`。
- `generate.py` 始终执行全部 43 层并运行 head；层数裁剪和跳过 head 仅供 `smoke_model.py` 验证使用。
- `smoke_model.py` 要求 `seq_len + decode_steps <= 4096`。
- 当前生成是单请求、逐 token decode，不包含请求调度、连续批处理或服务协议层。
- packed expert cache 必须与当前模型配置和 cache format version 完全匹配。

模型 Kernel 本身的实现范围和限制见 [`docs/models/README.md`](../models/README.md)。

## 专题文档结构

Serving 专题文档按以下边界组织：

| 文档 | 内容 |
|---|---|
| [`01_generate.md`](01_generate.md) | 文本生成入口和采样循环 |
| [`02_smoke_model.md`](02_smoke_model.md) | 整模冒烟验证入口 |
| [`03_export_expert_cache.md`](03_export_expert_cache.md) | packed expert cache 导出流程 |
| [`04_expert_cache.md`](04_expert_cache.md) | expert cache 格式和运行时读取 |
| [`05_runner.md`](05_runner.md) | prefill/decode 整模编排 |
| [`06_state_plan.md`](06_state_plan.md) | state schema 和 Host 辅助输入 cache |
| [`07_weight_loader.md`](07_weight_loader.md) | checkpoint、layout 和权重生命周期 |
| [`08_runtime_values.md`](08_runtime_values.md) | runtime value 与 step contract |
| [`09_device_memory.md`](09_device_memory.md) | DeviceTensor pool 和 state store |
| [`10_device_runtime.md`](10_device_runtime.md) | ChipWorker runtime 和 Kernel dispatch |
| [`11_profiling.md`](11_profiling.md) | profile 事件和性能分析方法 |

## 验证方法

### Host 侧 CLI 与 Serving 测试

```bash
pytest -q tests/cli tests/serving
```

这些测试覆盖入口参数、checkpoint 校验、expert cache、weight loader、state、device pool、device runtime 和 runner 编排。测试使用 Host 侧 fake runtime 验证生命周期与接口，不替代真实 NPU Kernel 验证。

### NPU 冒烟验证

在已经配置 PyPTO 和 NPU runtime 的环境中执行：

```bash
python smoke_model.py \
  --checkpoint /path/to/checkpoint \
  --expert-cache-dir /path/to/expert-cache \
  --seq-len 1 \
  --max-layers 1 \
  --decode-steps 1
```

验收时确认 prefill 和 decode 输出 shape、dtype 符合当前配置，并且 `finite=True`。

### 完整生成链路

```bash
python generate.py \
  --checkpoint /path/to/checkpoint \
  --expert-cache-dir /path/to/expert-cache \
  --prompt "你好" \
  --max-new-tokens 2
```

该方法覆盖 tokenizer、官方 prompt encoding、43 层 prefill、逐 token decode、head、采样和文本解码。模块级数值精度验收应分别按照 [`docs/models/`](../models/README.md) 的验证方法执行。
