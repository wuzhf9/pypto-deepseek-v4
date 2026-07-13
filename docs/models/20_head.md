# Model Head

## 模块定位

Model Head 把最后一个 Transformer Block 输出的 4 份 Hyper-Connection（HC）streams
转换为下一 token 的 vocabulary logits。完整流程由三部分组成：

```text
x [1,S,4,4096], BF16
  -> HC head reduction
       -> hc_out [1,S,4096], BF16
  -> final RMSNorm
       -> normed [1,S,4096], BF16
  -> select last token normed[:,S-1,:]
  -> language-model projection
       -> logits [1,129280], FP32
```

HC head 只生成 4 个 pre mixing weights 并把 4 streams 归约为 1 个 hidden state，不生成
Block HC 使用的 post scaling 或 combination matrix。因此
[`models/head.py`](../../models/head.py) 的 `hc_head_fwd` 是独立的 pre-only 实现，不
调用 [`models/hc.py`](../../models/hc.py) 的 `hc_pre_fwd/hc_post_fwd`。

Prefill 时 `S` 是 prompt length，但只输出最后一个 prompt token 的 logits；decode 时
`S=1`。当前 Runner 可通过配置跳过 head 以返回 hidden states，正常文本生成路径会执行
head 并用 logits 选择下一个 token。

## 官方模型中的 Head

[`official/model.py`](../../official/model.py) 的 `Transformer` 持有全局 HC head 参数、
final RMSNorm 和 `ParallelHead`：

| 官方参数/模块 | Shape | Dtype | 职责 |
|---|---:|---|---|
| `hc_head_fn` | `[4,16384]` | FP32 | 从 flattened 4-stream hidden 生成 4 个 pre logits |
| `hc_head_scale` | `[1]` | FP32 | 缩放 HC mixing projection |
| `hc_head_base` | `[4]` | FP32 | HC pre logits additive base |
| `norm.weight` | `[4096]` | BF16 | Final RMSNorm weight |
| `ParallelHead.weight` | `[vocab/world_size,4096]` | FP32 parameter | Vocabulary projection |

官方 `ParallelHead.hc_head` 先把输入最后两维展平为 16384，并在 FP32 中计算：

$$
x_f=\operatorname{FP32}(\operatorname{flatten}(x)),\qquad
r=\frac{1}{\sqrt{\operatorname{mean}(x_f^2)+10^{-6}}}
$$

$$
m=\operatorname{Linear}(x_f,W_{hc})\cdot r
$$

$$
pre=\sigma(m\cdot scale+base)+10^{-6}
$$

$$
h=\operatorname{BF16}\left(\sum_{i=0}^{3}pre_i x_i\right)
$$

`ParallelHead.forward` 随后执行 final RMSNorm，并由 `get_logits` 只读取最后一个 token：

$$
logits=\operatorname{Linear}
\left(\operatorname{FP32}(\operatorname{RMSNorm}(h)_{:,S-1,:}),W_{vocab}\right)
$$

官方 Tensor Parallel 模式下，每个 rank 只持有一段 vocabulary weight，最后通过
`all_gather` 拼接完整 logits。官方注释说明 checkpoint 中 LM head weight 存储为
BF16，但 `ParallelHead.weight` 使用 FP32 parameter 便于 logits 计算。

## PyPTO kernel 实现

[`models/head.py`](../../models/head.py) 提供以下符号：

| 符号 | 类型 | 职责 |
|---|---|---|
| `hc_head_fwd` | `@pl.jit.inline` | Padded sequence 上计算 HC pre weights 并归约 4 streams |
| `lm_head_fwd` | `@pl.jit.inline` | 只对最后一个 token 执行 full-vocabulary FP32 projection |
| `head_fwd` | `@pl.jit.inline` | 组合 HC head、`rmsnorm_4096` 和 LM head |
| `head_test` | `@pl.jit` | Standalone top-level 验收 wrapper |
| `golden_head` | PyTorch golden | 完整 Head 参考实现 |
| `build_head_specs` | Host spec builder | 构造实际 `S`、16-aligned scratch 和 padded HC weights |

`head_fwd` 在 kernel 内创建 BF16 `hc_out` 和 `normed`，直接调用
[`models/rmsnorm.py`](../../models/rmsnorm.py) 的 `rmsnorm_4096`。完整模型由
[`serving/runner.py`](../../serving/runner.py) 的 `_run_head()` dispatch `head_test`；
Block、Split Block 和 Head 是三个独立 top-level dispatch 边界。

## 官方模块到当前实现的映射

| 官方计算 | 当前实现 | 关系/状态 |
|---|---|---|
| `x.flatten(2).float()` | `x_pad` flattened view + FP32 tile cast | 语义等价；额外 sequence padding |
| Flattened RMS scaling | `hc_head_fwd::head_hc_pre` square-sum + high-precision `rsqrt` | 融合内联 |
| `F.linear(x,hc_fn)` | `x_flat @ hc_fn_t` | 语义等价；weight 转置并 pad 到 16 列 |
| `sigmoid(mix*scale+base)+eps` | `head_hc_pre` vector math | 融合内联、FP32 |
| `sum(pre*x,dim=2)` | `head_hc_reduce` | 融合内联；FP32 sum 后 BF16 `rint` |
| Final `RMSNorm(4096)` | `rmsnorm_4096` | 直接调用 |
| `x[:, -1]` | `lm_head_fwd::head_lm_last` | 直接对应；只保留最后一个有效 token |
| `F.linear(last_hidden,head.weight)` | Vocab-parallel blocks 内的 FP32 matmul | 语义等价；当前“parallel”仅指 kernel blocks |
| Official per-rank vocabulary shard | Full `[129280,4096]` weight | 接口差异；当前单卡持有完整 vocabulary |
| Official logits `all_gather` | 无 | 不支持或未执行；单卡已产生完整 logits |
| MTP 使用 `ParallelHead` | 无 | 不支持或未执行；当前 Runner 不执行 MTP layer |

## 数据接口

### 公共接口

```text
x:        [1,S,4,4096],   BF16
hc_fn_t:  [16384,16],     FP32
hc_scale: [1],            FP32
hc_base:  [16],           FP32
norm_w:   [4096],         BF16
head_w:   [129280,4096],  FP32
logits:   [1,129280],     FP32
```

Batch 固定为 1，`S` 是动态 sequence 维。`logits` 没有 sequence 维，始终只对应
`x[:,S-1,:,:]`。调用方必须保证 `S>0`；kernel 使用 `tokens-1` 选择最后一行。

Checkpoint `hc_head_fn [4,16384]` 在 host 上转置并 pad 为 `[16384,16]`；只有前 4
列有模型语义。`hc_head_base` 同样从 `[4]` pad 到 `[16]`。Padding 列由
[`serving/weight_loader.py`](../../serving/weight_loader.py) 初始化为零。`hc_scale` 不做
padding。

`head_w` 保持 vocabulary-major `[129280,4096]`，不转置。Weight loader 将 checkpoint
值转换为 FP32 runtime weight；kernel matmul 使用 `b_trans=True` 解释该布局。

### Scratch 与内部 intermediates

Host spec builder 计算：

```text
S_PAD = ceil_div(S,16) * 16
```

调用方提供：

```text
x_pad:     [1,S_PAD,4,4096], BF16
pre:       [1,S_PAD,16],     FP32
hc_out_pad:[1,S_PAD,4096],   BF16
```

Kernel 内部创建：

```text
hc_out: [1,S,4096], BF16
normed: [1,S,4096], BF16
last_hidden: [16,4096], BF16
```

`pre` 的前 4 列参与 stream reduction；后 12 列是 padded computation lanes，不被
`head_hc_reduce` 消费。`last_hidden` 只在第 0 行放置最后一个 token，其他行补零；LM
head output 只取该有效行。

### State 与 runtime ownership

Head 没有跨 layer、跨 step state 或 cache。所有 5 组 weights 都是 `RuntimeWeight`，
首次 materialize 后作为 fixed weights 常驻 device。`x` 是最后一个 Block 留在 device
上的 output；`hc_out`、`normed` 和 scratch 只在当前 head dispatch 内存在。

Runner 在 head 完成后才把 FP32 logits 导出到 host 作为公开模型 output。若
`run_head=False`，Runner 不 dispatch Head，而是直接导出最后一层 4-stream hidden。

## 实现方式

### Sequence padding 与 HC projection

`hc_head_fwd` 先把实际 `[S,16384]` flattened input 拷贝到 `x_pad`，将
`[S,S_PAD)` tail rows 补零。核心 HC computation 始终按 16-token tile 运行，因此
aligned 与 non-aligned `S` 使用同一条路径，最后只拷回前 `S` 行。

对每个 token tile：

1. 沿 16384 channels 以 `RMS_K_CHUNK=256` 累加 FP32 square sum；
2. 乘 `1/16384`、加 `1e-6`，通过 high-precision `rsqrt` 得到 RMS scale；
3. 以 `LINEAR_K_CHUNK=256` 执行 `x @ hc_fn_t` FP32 accumulation；
4. 乘 RMS scale，再应用 FP32 `hc_scale/hc_base`、sigmoid 和 `hc_eps`；
5. 把 16-wide pre tile 写入 scratch。

该 RMS scaling 没有 learned weight，与后续 final RMSNorm 是两个不同操作。

### Four-stream reduction

`head_hc_reduce` 按 16 tokens × 128 hidden channels，把 4 份 BF16 streams 转为 FP32，
分别乘 `pre[...,0:4]` 后求和，再以 `rint` 转为 BF16 `hc_out_pad`。4096 hidden
channels 分为 32 个 `D_CHUNK=128` blocks。

Head HC 不执行 Sinkhorn、post scaling 或 residual combination；这些只属于 Block HC
pre/post。Head reduction 后 4-stream 维被永久消除。

### Final RMSNorm

`head_fwd` 对全部 `S` 个 HC-reduced tokens 调用 `rmsnorm_4096`，使用 BF16
`norm_w`、FP32 square-sum/scale 和 BF16 output。虽然 LM projection 只消费最后一个
token，当前实现仍 materialize 全部 `[1,S,4096]` normalized hidden。

### Last-token vocabulary projection

`lm_head_fwd` 把 `normed[S-1,:]` 复制到 `[16,4096]` BF16 tile 的第一行。Vocabulary
129280 按 `VOCAB_TILE=128` 分为 1010 个 parallel blocks；每个 block 沿 hidden 4096
以 `K_TILE=128` 做 32 次 FP32 matmul accumulation，输出 128 个 FP32 logits。

最终输出保持 FP32，不转换回 BF16。这里的 `pl.parallel` 是单卡 kernel 内的 vocabulary
block parallelism，不是官方 Tensor Parallel rank。

## 实现差异与限制

- 当前只支持 `B=1`、HC streams 4、hidden 4096、vocabulary 129280 和 FP32 logits；
- `S` 必须大于 0，完整 runtime 的 sequence 上限为 4096；
- HC function/base runtime layout pad 到 16，仅前 4 列/元素有模型语义；
- HC head 是 pre-only reduction，不执行 Block HC 的 Sinkhorn、post 或 comb；
- 当前 final RMSNorm materialize 全部 `S` 个 tokens，但 LM projection 只读取最后一个；
- LM head weight 使用完整 `[129280,4096]` FP32 layout 并常驻单卡 device；
- 当前不实现 vocabulary Tensor Parallel、per-rank weight shard 或 logits `all_gather`；
- 当前不执行 FP4/FP8 head kernel；checkpoint head weight 由 host 加载路径转换为 FP32；
- Head 没有持久 state，也不修改 Attention cache；
- 当前 Runner 不执行 MTP layer，因此不覆盖 MTPBlock 对 `ParallelHead` 的复用。

## Golden 参考实现

`models/head.py::golden_head` 从 BF16 `x` 和 kernel-facing weights 开始。它在 FP32 中
计算 flattened RMS scale，从 padded `hc_fn_t/hc_base` 只提取前 4 个有效 lanes，执行
HC projection、sigmoid pre mixing 和 4-stream reduction，再把 `hc_out` 转回 BF16。

Golden 随后在 FP32 中计算 final RMSNorm scale，乘 BF16 `norm_w` 后把 normalized
hidden 转为 BF16。最后只选择 `normed[:,-1]`，与 FP32 `head_w` 执行
`F.linear`，写出 `[1,129280]` FP32 logits。

Golden 不模拟 sequence padding、16-row `last_hidden` tile、vocabulary tiling 或
device-resident weight lifecycle；它直接复现相同 dtype/rounding boundaries。

## 精度验收标准

Standalone Head 只比较 FP32 logits：

| 输出 | 验收方式 | Absolute tolerance | Relative tolerance | 允许超出容差的元素比例 |
|---|---|---:|---:|---:|
| `logits` | `ratio_allclose` | `1e-4` | `1/128` | `0.001` |

逐元素条件为：

```text
abs(actual - expected) <= 1e-4 + (1/128) * abs(expected)
```

允许最多 0.1% 的 logits 元素超出该条件。Actual logits 中出现任何 NaN 或 Inf 都会
直接判为不合法。

## 验收方法

在 Ascend A2/A3 实机上验证常规 prompt length：

```bash
python models/head.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 8
```

使用非 16 对齐 sequence length 验证 padding/copy-out，并验证 decode 的单 token
形态：

```bash
python models/head.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 13

python models/head.py \
  --platform a2a3 \
  --device 0 \
  --seq-len 1
```

增加 `--compile-only` 可仅检查编译；增加 `--enable-l2-swimlane` 会把相应选项传入
PyPTO `RunConfig`。

Host-side Head golden 与官方 `ParallelHead` 的比较可运行：

```bash
pytest -q tests/models/test_head.py
```

## 集成验证范围

### Standalone Head 验收

`models/head.py::main()` 编译和执行 `head_test`，比较最终 FP32 logits。

[`test_head.py`](../../tests/models/test_head.py) 使用缩小 hidden/vocabulary shape，在
sequence length 1、5、13 上逐元素比较 `golden_head` 与官方 `ParallelHead`；另一个
case 检查 padded HC weight、scratch、norm/head weights 和 logits 的 shape/dtype。该
host test 不执行 PyPTO NPU kernel，不能替代 standalone 实机验收。

### Serving 集成

- [`test_weight_loader.py`](../../tests/serving/test_weight_loader.py) 覆盖 HC function
  transpose+padding、HC base padding、FP32 head weight 和重复读取的 runtime weight
  identity；
- [`serving/runner.py`](../../serving/runner.py) 在 `run_head=True` 时把最后一个 Block
  device output 直接绑定到 `head_test`，并在 kernel 完成后导出 logits；
- [`generate.py`](../../generate.py) 使用 Runner 输出的 logits 完成 greedy 或
  temperature sampling。

当前 serving unit tests 主要覆盖 weight layout 和 Runner 的公共 output boundary；
完整 Head 数学精度由 standalone NPU 验收与 `test_head.py` 分别覆盖。
