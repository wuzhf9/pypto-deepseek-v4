# DeepSeek V4 Flash 模型实现总览

## 实现定位

本仓库实现 DeepSeek V4 Flash 的 BF16 推理路径，面向 PyPTO 和 Ascend NPU。
模型结构、参数配置和计算语义以仓库中的
[`official/model.py`](../../official/model.py) 为参考，并使用 DeepSeek V4 Flash
官方 checkpoint 中的模型权重。

当前实现不是对官方 PyTorch 代码的逐行移植。`models/` 根据 PyPTO kernel
的编译和执行特点，对模型计算进行了拆分与融合；完整模型的权重加载、状态管理
和 kernel 调度由 `serving/` 负责。

## 当前运行约束

当前 `models/` kernel 和完整模型 runtime 仅支持以下运行规模：

| 约束 | 当前值 |
|---|---:|
| Execution topology | 单卡、单 NPU device |
| Batch size | `B = 1` |
| Maximum sequence length | `4096` |

当前 `models/` 只实现单卡计算逻辑。所有 kernel 都按模型完整维度处理 tensor，
不会按 rank 切分 vocabulary、attention head、hidden channel 或 routed expert；完整
模型 runtime 也不会执行 `all_reduce`、`all_gather` 等分布式集合通信。官方代码中
的 Tensor Parallel、Expert Parallel多卡执行功能均未实现，
不能通过增加 device 数量直接扩展当前模型。

所有模型 kernel 的 batch 维都固定为 1；完整模型的 prefill 长度以及
`prefill + decode` 的总位置范围不能超过 4096。虽然
[`models/config.py`](../../models/config.py) 保留官方
`max_position_embeddings=1048576` 元数据，但当前 serving state 固定使用
`max_seq_len=4096`，不支持通过运行时参数扩展该上限。

## BF16 计算路径

当前 PyPTO runtime 计算路径统一使用 BF16 权重、激活和主要中间张量，不实现
`official/model.py` 中的 FP4/FP8 runtime kernel、激活量化和量化 GEMM 路径。
同时不执行官方 `rotate_activation`，即不对 Indexer query 和 Indexer compressor
KV 应用量化前的 Hadamard rotation。
部分 kernel 会使用 FP32 完成累加或数值敏感的局部计算，再转换回接口规定的
数据类型。

为了兼容官方 checkpoint，权重加载阶段仍支持识别 FP4/FP8 权重，并在 host
侧将其反量化为 BF16。该过程属于 checkpoint 加载和权重准备，不代表模型执行
期间使用 FP4/FP8 kernel。routed expert 也会以 packed BF16 格式导出并供运行时
加载。

因此，本仓库中的“BF16 版本”具体表示：

- 模型 kernel 接收 BF16 权重，而不是量化权重及其 scale；
- runtime 不执行 FP4/FP8 激活量化或量化矩阵乘；
- runtime 不执行官方 `rotate_activation` 及对应的 Hadamard rotation；
- 量化 checkpoint 可以在加载阶段转换为 BF16 runtime 权重；
- expert cache 的运行时存储格式为 packed BF16。

## 与官方模型的对齐范围

模型主干的配置来自 [`models/config.py`](../../models/config.py)。当前配置包括：

| 配置 | 当前值 |
|---|---:|
| Hidden size | 4096 |
| 主模型层数 | 43 |
| Hash routing 层数 | 3 |
| Attention heads | 64 |
| Routed experts | 256 |
| 每个 token 激活的 routed experts | 6 |
| Shared experts | 1 |
| MoE intermediate size | 2048 |
| Vocabulary size | 129280 |

当前实现对齐官方模型的主干推理流程，包括 embedding、Hyper-Connection、不同
压缩比例下的 attention、稀疏索引、MoE 和 language-model head。同时，模型实现
包含面向当前 runtime 的 prefill/decode 分离、selected-expert decode 和 kernel
融合。

以下内容不属于当前主干推理路径：

- FP4/FP8 runtime 推理；
- 官方实现中的 Tensor Parallel、Expert Parallel；
- MTP layer。配置中保留一个 MTP layer 的官方元数据，但当前 Runner 只执行
  43 个主模型层并输出主模型 logits。

这里的“对齐”指模型参数、主干数学语义和 checkpoint 权重映射对齐，不表示
内部模块接口、kernel 边界或所有官方功能完全相同。

## 精度验证说明

模型 kernel 使用对应的 PyTorch golden 实现进行精度比较，并由各模块文档给出
具体的验收阈值和验证命令。

当前已知部分模块在相同配置的重复实机验证中，可能偶现少量输出误差超过设定
阈值。该现象作为模型 kernel 验证的全局注意事项统一记录在本页，不在每个模块
文档中重复记录单次 PASS/FAIL 状态。单次超阈值不能用于放宽模块定义的验收标准；
验收时应使用相同代码、平台和输入配置复测，持续稳定复现的超阈值应作为精度
问题进一步分析。

## `models/` 目录结构

`models/` 保存模型计算 kernel、对应的 golden 实现和单 kernel 验证入口。各文件
按职责分为以下几组。

### 配置和基础组件

| 文件 | 职责 |
|---|---|
| `config.py` | DeepSeek V4 Flash BF16 静态模型配置和派生尺寸 |
| `common.py` | shape、整除关系和 token 数量等通用工具 |
| `linear.py` | 固定模型尺寸的 BF16 linear kernel |
| `rmsnorm.py` | 不同 hidden size 的 RMSNorm kernel |
| `rope.py` | 普通和压缩位置空间使用的 RoPE table 与 kernel |
| `golden.py` | PyTorch golden、tensor spec、编译运行和精度比较基础设施 |

### 模型输入、输出和 Hyper-Connection

| 文件 | 职责 |
|---|---|
| `embedding.py` | Token embedding，并在 kernel 内扩展 Hyper-Connection 维度 |
| `hc.py` | Layer 前后的 Hyper-Connection 变换 |
| `head.py` | 最终 Hyper-Connection 聚合、RMSNorm 和 language-model head |

### Attention、压缩和索引

| 文件 | 职责 |
|---|---|
| `attention_qkv.py` | Attention Q/K/V 投影基础计算 |
| `attention_out.py` | Attention 输出投影 |
| `attention_swa.py` | Sliding-window attention 的 prefill/decode 路径 |
| `attention_csa.py` | Compressed sparse attention 的 prefill/decode 路径 |
| `attention_hca.py` | Heavily compressed attention 的 prefill/decode 路径 |
| `sparse_attn.py` | SWA、CSA 和 HCA 使用的稀疏 attention kernel |
| `compressor_ratio4.py` | Ratio-4 attention/indexer 压缩路径 |
| `compressor_ratio128.py` | Ratio-128 压缩路径 |
| `indexer.py` | 稀疏 attention 候选位置的打分和 Top-K 选择 |

### MoE

| 文件 | 职责 |
|---|---|
| `gate.py` | Hash routing 和 Top-K routing |
| `expert.py` | Shared expert 和单 routed expert 的前向计算 |
| `moe.py` | Routed expert 调度、计算与输出聚合，包括 selected-expert decode |

### 完整层组装

| 文件 | 职责 |
|---|---|
| `block.py` | 不同 attention/routing 组合的完整 prefill 和 decode layer kernel |
| `split_block.py` | Selected-expert decode 使用的 pre-MoE 和 post-MoE 分段 kernel |

## 模块文档

- [RMSNorm](01_rmsnorm.md)
- [Linear](02_linear.md)
- [RoPE](03_rope.md)
- [Embedding](04_embedding.md)
- [Ratio-128 Compressor](05_compressor_ratio128.md)
- [Ratio-4 Compressor](06_compressor_ratio4.md)
- [Indexer](07_indexer.md)
- [Attention QKV](08_attention_qkv.md)
- [Sparse Attention](09_sparse_attn.md)
- [Attention Out](10_attention_out.md)
- [Sliding Window Attention](11_attention_swa.md)
- [Heavily Compressed Attention](12_attention_hca.md)
- [Compressed Sparse Attention](13_attention_csa.md)
- [MoE Gate](14_gate.md)
- [MoE Expert](15_expert.md)
- [Mixture-of-Experts](16_moe.md)
- [Hyper-Connections](17_hc.md)
- [Split Decode Block](18_split_block.md)
- [Transformer Block](19_block.md)
- [Model Head](20_head.md)

## 主干执行流程

完整主模型由 host-side Runner 组织，主要执行流程为：

```text
input token ids
  -> embedding + Hyper-Connection expansion
  -> 43 x transformer block
       -> attention sublayer
            -> save attention residual
            -> attention Hyper-Connection pre
            -> attention RMSNorm
            -> Attention (SWA / CSA / HCA)
                 -> compressor / indexer when required
            -> attention Hyper-Connection post + residual
       -> FFN sublayer
            -> save FFN residual
            -> FFN Hyper-Connection pre
            -> FFN RMSNorm
            -> MoE gate
            -> routed experts + shared expert
            -> FFN Hyper-Connection post + residual
  -> Hyper-Connection head + RMSNorm
  -> language-model head
  -> logits
```

在每个 Block 内，attention 子层完成一次独立的
`hc_pre → RMSNorm → Attention → hc_post`，其输出再作为 FFN 子层的输入；FFN 子层
随后完成第二次独立的 `hc_pre → RMSNorm → MoE → hc_post`。两次 `hc_post` 分别与
各自子层入口保存的 residual 组合。

Attention 类型由 [`models/config.py`](../../models/config.py) 中当前层的压缩比例
决定：ratio 0 使用 SWA，ratio 4 使用 CSA，ratio 128 使用 HCA。MoE 路由方式则由
层号是否位于 hash layers 范围内决定，分别使用 hash routing 或 Top-K routing。
Prefill 和 decode 使用独立的 kernel 路径；decode 进一步拆分 selected-expert
计算，以避免为单 token 加载和执行全部 routed experts。

## 代码职责边界

`models/` 只负责模型数学计算及 kernel 验证，不负责完整推理生命周期。以下功能
位于 `serving/`：

- checkpoint 校验、tensor 读取和量化权重反量化；
- packed BF16 routed-expert cache 读取；
- NPU device-resident 权重、状态和中间 tensor 管理；
- prefill/decode 的逐层调度；
- profiling 和模型输出导出。

模型 kernel 的独立精度验证位于 `tests/models/`；完整模型调度、权重加载和运行时
行为的测试位于 `tests/serving/`。
