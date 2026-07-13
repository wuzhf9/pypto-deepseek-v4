# RMSNorm

## 模块定位

RMSNorm 在 DeepSeek V4 Flash 中用于稳定 hidden state、attention 低秩投影和
压缩 KV。与 LayerNorm 不同，RMSNorm 不减去均值，只根据输入的均方根完成缩放，
再乘以可学习权重。

对于最后一维长度为 $D$ 的输入 $x$ 和可学习权重 $w$，本仓库使用的定义为：

$$
\operatorname{RMSNorm}(x, w)
= x \cdot \frac{1}{\sqrt{\frac{1}{D}\sum_{i=1}^{D}x_i^2 + \epsilon}} \cdot w
$$

其中 `epsilon = 1e-6`，来自
[`models/config.py`](../../models/config.py) 中的 `rms_norm_eps`。

## 官方模型中的 RMSNorm

[`official/model.py`](../../official/model.py) 定义了一个通用 `RMSNorm` module。
主模型根据计算位置，以 4096、1024、512 和 128 四种最后维度实例化或执行相同
的数学操作。

### 带可学习权重的 RMSNorm

| 官方位置 | 维度 | 用途 | 当前主干状态 |
|---|---:|---|---|
| `Block.attn_norm` | 4096 | Hyper-Connection attention 分支进入 Attention 前的归一化 | 已执行 |
| `Block.ffn_norm` | 4096 | Hyper-Connection FFN 分支进入 MoE 前的归一化 | 已执行 |
| `Transformer.norm` | 4096 | 最终 language-model head 前的归一化 | 已执行 |
| `Attention.q_norm` | 1024 | `wq_a` 低秩投影后的归一化 | 已执行 |
| `Attention.kv_norm` | 512 | `wkv` 投影后的归一化 | 已执行 |
| Attention `Compressor.norm` | 512 | Attention compressed KV pooling 后的归一化 | 已执行，融合实现 |
| Indexer `Compressor.norm` | 128 | Indexer compressed KV pooling 后的归一化 | 已执行，融合实现 |
| `MTPBlock.enorm` | 4096 | MTP embedding 分支投影前的归一化 | 当前 Runner 不执行 MTP |
| `MTPBlock.hnorm` | 4096 | MTP hidden 分支投影前的归一化 | 当前 Runner 不执行 MTP |
| `MTPBlock.norm` | 4096 | MTP head 前的归一化 | 当前 Runner 不执行 MTP |

“融合实现”表示 PyPTO kernel 保留相同的 RMSNorm 数学语义，但为了避免额外中间
tensor 和 kernel 边界，将计算直接写在 compressor kernel 内，而不是调用
`models/rmsnorm.py` 中的独立函数。

### 不带可学习权重的 RMS normalization

`official/model.py` 还包含若干
`x * rsqrt(mean(x²) + epsilon)` 形式的归一化。这些操作没有可学习权重，因此不属于
`RMSNorm` module，也不应映射到本页介绍的四个带权重 kernel：

- Attention 中 `wq_b` 后，每个 512 维 query head 的归一化；
- `Block.hc_pre` 中，计算 Hyper-Connection mixing 参数前对展平 HC hidden state
  的归一化；
- `ParallelHead.hc_head` 中，计算最终 HC mixing 参数前的归一化。

这些操作分别内联在 attention、HC 和 head kernel 中。

## PyPTO kernel 实现

[`models/rmsnorm.py`](../../models/rmsnorm.py) 提供四个 `@pl.jit.inline` kernel：

| Kernel | 输入最后维度 | 主要对应位置 |
|---|---:|---|
| `rmsnorm_4096` | 4096 | Block attention/FFN norm、最终模型 norm |
| `rmsnorm_1024` | 1024 | `Attention.q_norm` |
| `rmsnorm_512` | 512 | `Attention.kv_norm`；也可表达 Attention compressor norm |
| `rmsnorm_128` | 128 | 可表达 Indexer compressor norm |

每个 inline kernel 都有一个独立的顶层验证 wrapper：

| Inline kernel | 验证 wrapper | Tensor spec builder |
|---|---|---|
| `rmsnorm_4096` | `rmsnorm_4096_test` | `build_4096_specs` |
| `rmsnorm_1024` | `rmsnorm_1024_test` | `build_1024_specs` |
| `rmsnorm_512` | `rmsnorm_512_test` | `build_512_specs` |
| `rmsnorm_128` | `rmsnorm_128_test` | `build_128_specs` |

顶层 wrapper 只用于独立编译和精度验证；完整模型调用的是 inline kernel 或对应
的融合实现。

## 官方模块到当前实现的映射

| 官方计算 | PyPTO 实现 | 关系 | 集成位置 |
|---|---|---|---|
| `Block.attn_norm` | `rmsnorm_4096` | 直接调用 | `models/block.py`、`models/split_block.py` |
| `Block.ffn_norm` | `rmsnorm_4096` | 直接调用 | `models/block.py`、`models/split_block.py` |
| `Transformer.norm` | `rmsnorm_4096` | 直接调用 | `models/head.py::head_fwd` |
| `Attention.q_norm` | `rmsnorm_1024` | 直接调用 | `models/attention_qkv.py::attention_qkv_fwd` |
| `Attention.kv_norm` | `rmsnorm_512` | 直接调用 | `models/attention_qkv.py::attention_qkv_fwd` |
| Ratio-4 Attention `Compressor.norm` | 512 维等价计算 | 融合内联 | `models/compressor_ratio4.py` |
| Ratio-128 Attention `Compressor.norm` | 512 维等价计算 | 融合内联 | `models/compressor_ratio128.py` |
| Ratio-4 Indexer `Compressor.norm` | 128 维等价计算 | 融合内联 | `models/compressor_ratio4.py` |
| `MTPBlock.enorm/hnorm/norm` | 可由 `rmsnorm_4096` 表达 | 不支持/未执行 | 当前 Runner 不执行 MTP |
| Attention `wq_b` 后的无权重归一化 | 512 维按 head 计算 | 融合内联 | `models/attention_qkv.py` |
| `Block.hc_pre` 的无权重归一化 | 16384 维 HC hidden 计算 | 融合内联 | `models/hc.py::hc_pre_fwd` |
| `ParallelHead.hc_head` 的无权重归一化 | 16384 维 HC hidden 计算 | 融合内联 | `models/head.py::hc_head_fwd` |

`rmsnorm_128` 当前没有被完整模型 kernel 直接调用。它保留独立实现和验证入口，
作为 128 维 RMSNorm 的基础 kernel；当前 Indexer 路径采用融合版本。

## 数据接口

四个独立 kernel 使用相同的接口形式：

```text
x:      [1, S, D], BF16
norm_w: [D],       BF16
out:    [1, S, D], BF16
```

其中：

- Batch 固定为 1；
- `S` 是动态 sequence/token 维度；
- `D` 分别为 4096、1024、512 或 128；
- 输入、权重和输出均为 BF16；
- 均方、`epsilon`、`rsqrt` 和权重乘法使用 FP32 中间计算。

该接口不接受 bias，也不接受独立 scale tensor。checkpoint 中对应 RMSNorm weight
由 [`serving/weight_loader.py`](../../serving/weight_loader.py) 加载为 BF16，再由
[`serving/runner.py`](../../serving/runner.py) 绑定到对应的完整模型 kernel。

独立 RMSNorm kernel 不拥有持久 state 或 cache，也没有跨调用复用的中间 tensor。
`out` 由调用方提供；展平 view、FP32 平方和、均方根倒数和归一化 tile 都是单次
调用内部的 scratch 数据。

## Kernel 实现方式

四种尺寸采用相同的两遍计算结构：

1. 将 `[1, S, D]` reshape 为 `[S, D]`；
2. 每个任务处理最多 8 个 token；
3. 按 128 个 hidden channel 分块读取 BF16 输入；
4. 转为 FP32，累加每个 token 的平方和；
5. 计算 `rsqrt(mean(x²) + epsilon)`；
6. 再次分块读取输入和权重，在 FP32 中完成归一化及权重乘法；
7. 使用 round-to-nearest 模式转换为 BF16 并写回有效 token。

当前公共 tiling 参数为：

```text
T_TILE = 8
D_TILE = 128
```

实现对动态 `S` 的尾块使用 `valid_shape`，因此 sequence length 不要求是 8 的整数倍。
`rsqrt` 使用 PyPTO high-precision 模式。

## 实现差异与限制

当前实现与 `official/model.py` 的主要差异如下：

- 官方 `RMSNorm` 是任意维度的 PyTorch module；当前 PyPTO 实现针对模型实际使用
  的四种固定维度分别编译；
- 官方 module 将 parameter 保存在 FP32 以便 PyTorch 计算，checkpoint 中的
  RMSNorm weight 为 BF16；当前 kernel 接收 BF16 weight，并在 kernel 内转为
  FP32 参与计算；
- Compressor 中的 512/128 维 RMSNorm 已与 pooling 和 RoPE 路径融合；
- 当前独立 kernel 固定 batch size 为 1；
- MTP 使用的 4096 维 RMSNorm 可以由 `rmsnorm_4096` 表达，但 MTP 不属于当前
  Runner 的执行路径。

## Golden 参考实现

`models/rmsnorm.py::golden_rmsnorm` 是四种尺寸共用的 PyTorch 参考实现。Golden
按以下顺序计算：

```python
x = x.float()
norm_w = norm_w.float()
inv_rms = torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon)
out = (x * inv_rms * norm_w).to(torch.bfloat16)
```

Golden 输入来自与 kernel 相同的 BF16 tensor snapshot，最终输出转换为 BF16，
避免把输入生成误差或不同输出 dtype 混入比较结果。

## 精度验收标准

四种独立 kernel 使用同一验收标准：

| 项目 | 标准 |
|---|---:|
| Absolute tolerance | `1e-4` |
| Relative tolerance | `1 / 128`，约为 `0.0078125` |
| 允许超出容差的元素比例 | `0` |
| NaN/Inf | 不允许 |

对每个输出元素，验收条件为：

```text
abs(actual - expected) <= 1e-4 + (1 / 128) * abs(expected)
```

`max_error_ratio=0.0` 表示所有元素都必须满足上述条件，不允许通过忽略少量异常点
完成验收。

## 验收方法

`models/rmsnorm.py` 的命令行入口会依次编译并验证 4096、1024、512 和 128 四种
尺寸。Ascend A2/A3 实机验证命令为：

```bash
python models/rmsnorm.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 8
```

非 tile 对齐的动态 sequence length 可使用例如：

```bash
python models/rmsnorm.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 7
```

如需仅验证编译，可增加 `--compile-only`。如需启用 L2 swimlane，可增加
`--enable-l2-swimlane`；该参数会直接传入 PyPTO `RunConfig`。

## 集成验证范围

### 独立 kernel 验收

独立 RMSNorm kernel 目前没有对应的 `tests/models/test_rmsnorm.py`。其直接编译、
实机执行和 golden 比较入口是 `models/rmsnorm.py::main()`，会依次覆盖四种固定
维度。

### Host 侧官方语义覆盖

RMSNorm 同时受到以下上层路径的集成覆盖：

- [`test_block.py`](../../tests/models/test_block.py) 和
  [`test_split_block.py`](../../tests/models/test_split_block.py)：4096 维
  attention/FFN norm；
- [`test_attention_swa.py`](../../tests/models/test_attention_swa.py)、
  [`test_attention_csa.py`](../../tests/models/test_attention_csa.py) 和
  [`test_attention_hca.py`](../../tests/models/test_attention_hca.py)：1024 维 q norm、
  512 维 KV norm 和 compressor norm；
- [`test_compressor_ratio4.py`](../../tests/models/test_compressor_ratio4.py)、
  [`test_compressor_ratio128.py`](../../tests/models/test_compressor_ratio128.py) 和
  [`test_indexer.py`](../../tests/models/test_indexer.py)：融合的 512/128 维
  compressor norm；
- [`test_head.py`](../../tests/models/test_head.py)：最终 4096 维模型 norm。

这些 host 测试比较组合模块的 PyTorch golden 与 `official/model.py` 语义，验证
RMSNorm 所在的上层数据流和权重映射。它们不编译或执行 NPU RMSNorm kernel，
因此不能替代四个独立 kernel 的实机 golden 验收。

### 完整模型集成

Serving 层加载并绑定所有 norm weight，完整模型的 block、split-block 和 head
路径最终覆盖直接调用及 compressor 融合实现。完整模型验证可以检查跨层权重、
状态和调度是否正确，但不能替代独立 kernel 对单一数值误差来源的定位。
