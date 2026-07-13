# Embedding

## 模块定位

Embedding 是完整模型的输入组件，将 tokenizer 产生的 token id 映射为 BF16 hidden
vector，并为 Hyper-Connection（HC）增加 lane 维度。它位于 prefill 和 decode 的共同
入口：

```text
input_ids [1,S]
  -> token embedding lookup [1,S,4096]
  -> HC lane expansion [1,S,4,4096]
  -> 第 0 层 Transformer Block
```

当前静态模型配置来自 [`models/config.py`](../../models/config.py)：词表大小为
129280、hidden size 为 4096、`hc_mult=4`。完整模型仅支持 batch size 1，sequence
length 受 runtime 的 4096 上限约束。

## 官方模型中的 Embedding

[`official/model.py`](../../official/model.py) 使用 `ParallelEmbedding` 实现 token
lookup。该模块沿 vocabulary 维切分权重：每个 rank 持有
`vocab_size / world_size` 行；多 rank 时先将非本 rank token mask 为 0，完成本地
`F.embedding` 后再通过 `all_reduce` 合并结果。

在单 rank 路径中，`ParallelEmbedding.forward()` 等价于：

```python
h = F.embedding(input_ids, embed.weight)
```

`Transformer.forward()` 随后在 embedding 模块之外增加 HC lane 维：

```python
h = self.embed(input_ids)
h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
```

因此官方主模型在进入第一个 Block 前，tensor shape 从 `[B,S,4096]` 变为
`[B,S,4,4096]`，四个 HC lane 的初始值完全相同。

官方还将同一个 `ParallelEmbedding` 实例绑定到 MTP layer。当前 Runner 只执行 43
个主模型层，不执行 MTP layer，因此该复用路径不属于当前完整推理范围。

## PyPTO kernel 实现

[`models/embedding.py`](../../models/embedding.py) 提供一个 inline kernel、一个独立
验收 wrapper，以及对应的 golden 和 tensor spec：

| 符号 | 职责 |
|---|---|
| `embedding_fwd` | 完成 token lookup，并将每个 hidden row 写入四个 HC lane |
| `embedding_test` | 独立编译和验收使用的 `@pl.jit` wrapper |
| `golden_embedding` | PyTorch lookup 与 HC expansion 参考实现 |
| `build_embedding_specs` | 构造动态 sequence length 对应的输入、权重和输出 spec |

`embedding_fwd` 将官方的 `ParallelEmbedding` 单 rank lookup 与紧随其后的
`unsqueeze(2).repeat(...)` 融合到同一个 kernel。kernel 不先物化
`[1,S,4096]` 的临时 tensor，而是把选中的权重分块直接写入四个 HC lane。

## 官方模块到当前实现的映射

| 官方计算 | PyPTO 实现 | 关系 | 集成位置 |
|---|---|---|---|
| 单 rank `ParallelEmbedding.forward` | `embedding_fwd` 中的 weight row 读取 | 语义等价并融合 | `models/embedding.py` |
| `h.unsqueeze(2).repeat(1,1,hc_mult,1)` | `embedding_fwd` 中按 HC lane 重复写入 | 融合内联 | `models/embedding.py` |
| Vocabulary tensor parallel mask | 无 | 当前不支持且主干不执行 | 官方多 rank 路径 |
| Vocabulary tensor parallel `all_reduce` | 无 | 当前不支持且主干不执行 | 官方多 rank 路径 |
| MTP layer 复用 embedding | 无 | 当前不支持且主干不执行 | 官方 `MTPBlock` |
| `Transformer.forward` 的模型入口 | `DeepSeekV4Runner._run_embedding` | Host 调度等价 | `serving/runner.py` |

当前 checkpoint 名称 `model.embed_tokens.weight` 在加载时规范化为
`embed.weight`。PyPTO kernel 使用完整词表权重，不做 vocabulary shard。

## 数据接口

`embedding_fwd` 的接口为：

```text
input_ids: [1,S],             INT64
weight:    [129280,4096],     BF16
out:       [1,S,4,4096],      BF16
```

其中：

- Batch 固定为 1；
- `S` 是动态 sequence/token 维度；
- 每个 `input_ids` 元素必须位于 `[0,129280)`；
- weight 的每一行对应一个 token 的 4096 维 embedding；
- 输出第三维固定为四个 HC lane，同一 token 的四个 lane 初始值相同；
- kernel 没有 bias、position 输入、持久 state 或跨调用 scratch；
- kernel 不负责 tokenizer、padding mask、position encoding 或 embedding scale。

Prefill 使用 `[1,S]` 的 prompt token ids；decode 使用 `[1,1]` 的新 token id。两条
路径调用同一个动态 shape kernel，并产生相同的 `[1,S,4,4096]` 接口，仅 sequence
length 不同。

## Kernel 实现方式

当前 tiling 参数为：

```text
T_TILE = 16
D_TILE = 128
H_BLOCKS = 4096 / 128 = 32
```

kernel 的主要步骤为：

1. 将 `input_ids` reshape 为 `[S]`，将输出 reshape 为 `[S * 4,4096]`；
2. 按最多 16 个 token 和 128 个 hidden channel 组成工作块；
3. 通过 `token_blocks * H_BLOCKS` 个 SPMD work item 分配 token/hidden block；
4. 从 `input_ids` 读取 token id，并转换为 `pl.INDEX`；
5. 从 `weight[token_id, h0:h0+128]` 读取对应 embedding tile；
6. 将同一个 tile 分别写入该 token 的四个 HC lane；
7. 将输出恢复为 `[1,S,4,4096]`。

最后一个 token block 使用 `valid_tok=min(16,S-t0)` 处理 tail，因此 `S` 不要求是
16 的整数倍。hidden size 由模块导入时的整除断言保证可被 128 整除。

直接写入最终 HC layout 避免了中间 `[1,S,4096]` tensor 以及一次独立 repeat
kernel，但仍会从 embedding weight 为每个 HC lane 读取和复制对应 tile。kernel
本身不检查 token id 越界，合法词表范围由上层 tokenizer 和输入校验保证。

## 实现差异与限制

当前实现与官方路径的主要差异如下：

- 官方 `ParallelEmbedding` 可沿 vocabulary 维执行 tensor parallel；当前仅使用完整
  `[129280,4096]` 权重，不实现 mask 和 `all_reduce`；
- 官方将 lookup 和 HC expansion 表达为两个 PyTorch 操作；当前将二者融合为一个
  PyPTO kernel；
- 当前固定 `B=1`、hidden size 4096 和 `hc_mult=4`，不是通用 embedding kernel；
- 当前输出直接采用 Block 所需的四维 HC layout，不提供独立的 `[1,S,4096]` 输出；
- 当前只覆盖主模型 embedding，不执行官方 MTP layer 的 embedding 复用；
- 当前 kernel 接收已经完成 tokenization 的 INT64 ids，不包含 tokenizer 或 prompt
  处理逻辑；
- 当前 runtime 的 sequence position 上限为 4096，但 embedding kernel 本身不读取
  position，也不执行 RoPE。

## Golden 参考实现

`models/embedding.py::golden_embedding` 使用 PyTorch 完成参考计算：

```python
h = F.embedding(input_ids.long(), weight)
out = h.unsqueeze(2).repeat(1, 1, 4, 1)
```

Golden 使用与 kernel 相同的 BF16 weight，`F.embedding` 只做 row lookup，不包含
浮点归约。HC expansion 也只复制数据，因此理论上 kernel 输出应与 golden 逐元素
一致。

[`test_embedding.py`](../../tests/models/test_embedding.py) 还会在单 rank 配置下，将
该 golden 与官方 `ParallelEmbedding` 输出加 HC expansion 的结果比较，从而验证
weight row 语义和四个 lane 的布局。

## 精度验收标准

独立 embedding kernel 使用逐元素精确标准：

| 项目 | 标准 |
|---|---:|
| Absolute tolerance | `0` |
| Relative tolerance | `0` |
| 允许超出容差的元素比例 | `0` |
| NaN/Inf | 不允许 |

即每个 BF16 输出元素必须与 PyTorch golden 完全一致。该标准符合当前实现只执行
BF16 row copy、不进行浮点运算的特征。

## 验收方法

在 Ascend A2/A3 实机上验证默认 sequence length：

```bash
python models/embedding.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 8
```

使用非 tile 对齐长度验证动态 tail：

```bash
python models/embedding.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 13
```

使用完整 runtime 支持的最大 prefill 长度验证长序列工作划分：

```bash
python models/embedding.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 4096
```

如需仅检查编译，可增加 `--compile-only`。如需启用 L2 swimlane，可增加
`--enable-l2-swimlane`；该参数会直接传入 PyPTO `RunConfig`。

## 集成验证范围

### 独立 kernel 验收

`models/embedding.py::main()` 负责构造完整配置下的 input ids、BF16 weight 和输出，
编译并执行 `embedding_test`，再按逐元素精确标准与 golden 比较。

### Host 语义与接口覆盖

[`test_embedding.py`](../../tests/models/test_embedding.py) 覆盖：

- sequence length 1、5、13 和 1024；
- golden lookup 与官方单 rank `ParallelEmbedding` 的一致性；
- HC lane expansion 的 shape 和逐元素值；
- input、weight 和 output tensor spec 的 shape 与 dtype。

这些 host 测试不编译或执行 NPU kernel，不能替代独立 kernel 的实机验收。

### Serving 集成

- [`test_runner.py`](../../tests/serving/test_runner.py) 验证 Runner 将 input ids 和
  `RuntimeWeight` 交给 runtime，并将 embedding 输出作为不透明 device tensor 传给
  后续路径；
- [`test_weight_loader.py`](../../tests/serving/test_weight_loader.py) 验证
  `model.embed_tokens.weight` 到 `embed.weight` 的名称映射、BF16 内容和 host layout
  cache；
- [`test_device_runtime.py`](../../tests/serving/test_device_runtime.py) 验证 embedding
  固定权重、编译结果和 step buffer 的跨 step 复用。

完整模型的 prefill 和 decode 都从 `DeepSeekV4Runner._run_embedding()` 开始。weight
作为 fixed weight 在 device runtime 中复用，输出保留在 NPU 并直接进入第一个
Transformer Block；只有公共模型输出边界会导回 host。
