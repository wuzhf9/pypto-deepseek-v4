# DeepSeek V4 Flash PyPTO

## 项目介绍

本项目基于 PyPTO 实现 DeepSeek V4 Flash 的单卡 BF16 推理版本，包括模型 Kernel、整模 prefill/decode 编排、权重加载、专家缓存和 Device Runtime。

当前实现采用固定的运行约束：

| 项目 | 当前配置 |
|---|---|
| 模型 | DeepSeek V4 Flash |
| 实现框架 | PyPTO |
| 运行精度 | BF16 |
| 设备模式 | 单卡 |
| Batch size | `B=1` |
| 最大序列长度 | `max_seq_len=4096` |

项目不实现多卡并行，Batch size 和最大序列长度也不作为运行时可配置项。模型 Kernel 与结构说明见 [`docs/models/`](docs/models/README.md)，整模运行时实现见 [`docs/serving/`](docs/serving/README.md)。

## 环境准备

运行本项目需要一台已安装 Ascend NPU 驱动与 Runtime 的单卡环境，并完成 PyPTO 软件栈配置。Python 环境还需要提供 PyTorch、Transformers、Safetensors 和 Hugging Face CLI；其中 Hugging Face CLI 用于下载官方 checkpoint。

本项目当前开发与验证使用的 PyPTO 软件栈版本如下：

| 组件 | 参考版本 |
|---|---|
| `pypto` | `8ebddcb8` |
| `pypto runtime` | `c94aa9f3` |
| `pto-isa` | `83d01313` |
| `ptoas` | `0.45` |

建议在复现或排查运行时问题时优先对齐上述版本，以避免编译接口、Runtime 行为或 ISA 支持差异造成影响。开始运行前，应确保当前终端能够正常导入 `torch`、`transformers`、`safetensors` 和 `pypto`，并能够执行 `hf` 命令。

## 快速开始

以下命令均在仓库根目录执行。请先完成上述运行环境配置，并将示例路径替换为实际路径。

### 1. 下载官方权重

使用 Hugging Face CLI 下载 DeepSeek V4 Flash 官方 checkpoint：

```bash
hf download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir /path/to/checkpoint
```

`/path/to/checkpoint` 目录必须包含 `tokenizer.json` 和 `model.safetensors.index.json`。

### 2. 导出 BF16 专家权重

将官方 checkpoint 中全部 routed expert 权重转换为推理时使用的 packed BF16 expert cache：

```bash
python export_expert_cache.py \
  --checkpoint /path/to/checkpoint \
  --output /path/to/expert-cache
```

导出过程按层生成 cache 文件，每层约占用 12 GiB，完整 43 层约占用 516 GiB 磁盘空间。该容量不包含官方 checkpoint，请为导出过程和其他文件额外预留空间。已有且校验通过的层文件默认会被跳过。

### 3. 运行整模冒烟验证（可选）

先运行少量层的 prefill 和 decode，验证 checkpoint、expert cache 与 Device Runtime 能够正常工作：

```bash
python smoke_model.py \
  --checkpoint /path/to/checkpoint \
  --expert-cache-dir /path/to/expert-cache \
  --max-layers 1 \
  --decode-steps 1
```

该步骤默认使用长度为 1 的随机 token 输入并执行 LM head，适合在完整推理前快速检查运行环境。

### 4. 执行完整推理

使用完整 43 层模型生成文本：

```bash
python generate.py \
  --checkpoint /path/to/checkpoint \
  --expert-cache-dir /path/to/expert-cache \
  --prompt "介绍一下DeepSeek" \
  --max-new-tokens 400
```

也可以通过 `--prompt-file /path/to/prompt.txt` 从 UTF-8 文件读取 prompt。若同时指定 `--prompt` 和 `--prompt-file`，程序会提示并优先使用 `--prompt`。

## 目录结构

```text
.
├── models/                          # DeepSeek V4 Flash BF16 模型 Kernel
│   ├── __init__.py                  # 模型包入口与公共配置导出
│   ├── config.py                    # 模型常量、层配置与静态 shape
│   ├── common.py                    # Kernel 共用的小型 Python 辅助函数
│   ├── golden.py                    # 独立 Kernel 的 golden 验证框架
│   ├── rmsnorm.py                   # RMSNorm Kernel
│   ├── linear.py                    # BF16 Linear Kernel
│   ├── rope.py                      # RoPE 表生成与旋转位置编码 Kernel
│   ├── embedding.py                 # Token embedding Kernel
│   ├── compressor_ratio128.py       # Ratio-128 HCA compressor Kernel
│   ├── compressor_ratio4.py         # Ratio-4 CSA compressor Kernel
│   ├── indexer.py                   # CSA Indexer Kernel
│   ├── attention_qkv.py             # Attention Q/KV 投影与 RoPE Kernel
│   ├── sparse_attn.py               # 稀疏 attention 基础 Kernel
│   ├── attention_out.py             # Attention 输出投影 Kernel
│   ├── attention_swa.py             # Sliding Window Attention Kernel
│   ├── attention_hca.py             # Heavily Compressed Attention Kernel
│   ├── attention_csa.py             # Compressed Sparse Attention Kernel
│   ├── gate.py                      # MoE routing gate Kernel
│   ├── expert.py                    # 单专家与专家计算 Kernel
│   ├── moe.py                       # Route-major MoE 接口与 golden 逻辑
│   ├── hc.py                        # Hyper-Connections Kernel
│   ├── split_block.py               # Selected-expert decode 拆分 Block Kernel
│   ├── block.py                     # 完整 Block 编排、Kernel 与 golden 逻辑
│   └── head.py                      # Final norm、HC head 与 LM head Kernel
├── serving/                         # 整模推理编排与 Device Runtime
│   ├── __init__.py                  # Serving 包入口
│   ├── checkpoint.py                # Checkpoint 目录与必需文件校验
│   ├── runtime_types.py             # Runner、loader 与 runtime 的公共值类型
│   ├── state.py                     # State schema、RoPE/topk Host cache
│   ├── weight_loader.py             # 权重读取、反量化与 runtime layout cache
│   ├── expert_cache.py              # Packed BF16 expert cache reader
│   ├── device_pool.py               # DeviceTensor 分配、复用、拷贝与统计
│   ├── device_state_store.py        # 跨 step 的 NPU state 双缓冲
│   ├── device_runtime.py            # ChipWorker、materialize、compile 与 dispatch
│   ├── profiler.py                  # Serving 分段性能统计
│   └── runner.py                    # Prefill/decode 整模执行编排
├── docs/                            # 面向用户的 Models 与 Serving 文档
├── official/                        # 官方模型与 prompt encoding 参考实现
├── tests/                           # Models、Serving 与 CLI 的 Host 测试
├── .agents/                         # 仓库内 Codex skills
├── generate.py                      # 完整文本生成入口
├── smoke_model.py                   # 整模 prefill/decode 冒烟验证入口
├── export_expert_cache.py           # Packed BF16 expert cache 导出入口
├── .gitignore                       # Git 忽略规则
└── README.md                        # 项目总览与使用说明
```

`models/` 只负责模型数学和 PyPTO Kernel；`serving/` 负责权重、状态、设备内存及整模执行，两者通过 Kernel specs 和 runtime values 连接。

## 推理样例

```text
User:
介绍一下DeepSeek
AI:
嗨！很高兴为你介绍我自己——DeepSeek！✨

## 我是谁？
我是由**深度求索公司**（DeepSeek）创造的AI助手，是一个纯文本大语言模型。我的使命就是帮你解答问题、提供建议、处理信息，成为你工作和生活中的得力伙伴！

## 我的核心能力
- **文本处理专家**：阅读、理解、生成各类文本内容
- **文件处理**：支持上传图像、txt、pdf、ppt、word、excel等文件，从中提取文字信息进行分析
- **超长上下文**：拥有1M的上下文窗口，可以一次性处理像《三体》三部曲那么大体量的内容！
- **联网搜索**：支持联网查询最新信息（需要你在Web/App手动开启）
- **多平台支持**：Web端、App端（支持语音输入）都能用

## 我的特点
✅ **完全免费**：没错，目前没有任何收费计划！
✅ **知识更新**：知识截止于2025年5月
✅ **热情细腻**：我会用心理解你的需求，给出贴心的回复
✅ **阅读链接**：可以访问你提供的网页链接内容

## 小提示
虽然我支持文件上传，但我是**纯文本模型**，不能直接“看”图像内容，不过可以读取图片中的文字信息哦！

有什么我可以帮你的吗？无论是学习、工作还是日常问题，尽管问我！😊
[stats]
prompt_tokens: 8
generated_tokens: 302
elapsed_s: 1018.050
output_tps: 0.297
```
