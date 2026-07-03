# Hyper-Connections PyPTO 实现说明

本文记录当前 `models/hc.py` 中 `hc_pre_fwd` 的实现方式，后续实现
`hc_post` 和 `block` 时应沿用这里的接口约定。

## 官方语义

`Block.hc_pre` 对应 `official/model.py` 中的逻辑：

```python
shape, dtype = x.size(), x.dtype
x = x.flatten(2).float()
rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
mixes = F.linear(x, hc_fn) * rsqrt
pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, ...)
y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
return y.to(dtype), post, comb
```

输入输出语义：

```text
x:       [B, S, HC_MULT, HIDDEN] bf16
x_mixed: [B, S, HIDDEN] bf16
post:    [B, S, HC_MULT] fp32
comb:    [B, S, HC_MULT, HC_MULT] fp32
```

当前模型固定：

```text
B = 1
HC_MULT = 4
HIDDEN = 4096
HC_DIM = HC_MULT * HIDDEN = 16384
MIX_HC = (2 + HC_MULT) * HC_MULT = 24
```

## 当前 PyPTO 接口

`hc_pre_fwd` 对外保留实际序列长度 `S_DYN`，同时要求调用方传入一组
padded scratch tensor：

```text
x:              [B, S_DYN,     HC_MULT, HIDDEN] bf16
x_pad:          [B, S_PAD_DYN, HC_MULT, HIDDEN] bf16
mixes:          [B, S_PAD_DYN, MIX_PAD] fp32
pre:            [B, S_PAD_DYN, HC_PAD] fp32
comb_logits:    [B, S_PAD_DYN, HC_MULT * HC_MULT] fp32
x_mixed_pad:    [B, S_PAD_DYN, HIDDEN] bf16
post_pad:       [B, S_PAD_DYN, HC_PAD] fp32
comb_pad:       [B, S_PAD_DYN, HC_MULT * HC_MULT] fp32
x_mixed:        [B, S_DYN,     HIDDEN] bf16
post:           [B, S_DYN,     HC_PAD] fp32
comb:           [B, S_DYN,     HC_MULT * HC_MULT] fp32
```

`S_PAD_DYN` 由 host 侧按下面方式计算：

```python
S_PAD = ceil_div(S, 16) * 16
```

## Padding 设计

### 序列维 padding

PyPTO 的 HC 核心计算使用 `T_TILE = 16`。如果直接在最后一个 tile 上使用
动态 `valid_tok = min(16, S - t0)`，`hc_pre_split` 中的小宽度 tile subview 会触发
valid shape 编译问题；如果把 tile 行数改成 1 或 8，又会触发 32-byte 对齐或
boxed tile 行数约束。

因此当前实现把实际 `S` pad 到 `S_PAD`：

1. 将实际 `x` 拷贝到 `x_pad` 的前 `S` 行。
2. `x_pad[S:S_PAD]` 补 0。
3. HC 核心计算只处理完整 16-token tile。
4. 计算结束后只把前 `S` 行拷回实际输出。

这样 `S=1`、`S=13`、`S=16`、`S=32`、`S=128` 都走同一条核心路径。

### MIX_HC padding

官方 `mixes` 最后一维是 `MIX_HC = 24`，但 PyPTO vector tile 更适合 32-byte
对齐宽度。当前将其 pad 到：

```text
MIX_PAD = 32
mixes: [B, S_PAD, 32]
```

前 24 列是有效的 `pre/post/comb` logits，后 8 列只用于满足 tile 宽度。

### HC_MULT padding

官方 `pre/post` 宽度是 `HC_MULT = 4`。直接写 `[*, 4]` 的 fp32 tile 会出现
行宽 16 bytes，不满足 32-byte 对齐。因此当前使用：

```text
HC_PAD = 8
pre:      [B, S_PAD, 8]
post_pad: [B, S_PAD, 8]
post:     [B, S,     8]
```

其中前 4 列是官方语义，后 4 列是 padding。后续 `hc_post` 应只消费
`post[..., :HC_MULT]`。

`comb` 不需要额外 pad 到 32，因为 flat 后宽度是 `HC_MULT * HC_MULT = 16`，
fp32 行宽为 64 bytes，已经满足对齐。

### comb_logits scratch

`comb_logits` 是必要的 GM scratch，不是冗余输出。

Sinkhorn 需要把 `comb_logits` 的四行分别读成 padded tile：

```text
row0 = comb_logits[..., 0:4]   -> tile [16, 8] valid [16, 4]
row1 = comb_logits[..., 4:8]   -> tile [16, 8] valid [16, 4]
row2 = comb_logits[..., 8:12]  -> tile [16, 8] valid [16, 4]
row3 = comb_logits[..., 12:16] -> tile [16, 8] valid [16, 4]
```

如果直接从本地 tile `comb_logits_tile` 切 row，会遇到 PyPTO 的 Tile/Tensor 类型和
valid shape 编译问题。把 logits 先写到 GM tensor，再使用 `pl.load(...,
valid_shapes=[T_TILE, HC_MULT])` 是当前验证通过的稳定写法，也和
`pypto-serving` 的 HC 实现思路一致。

## 计算流程

当前 `hc_pre_fwd` 的计算顺序是：

1. `hc_pre_pad_x`
   - `x -> x_pad`
   - tail 位置补 0
2. `hc_pre_linear`
   - `x_pad.flatten(2)` 上计算 RMS square sum
   - 计算 `x_flat @ hc_fn.T`
   - 乘 `rsqrt(mean_square + eps)`
   - 写入 padded `mixes`
3. `hc_pre_split`
   - 从 `mixes` 中切出 `pre/post/comb` 三段
   - 计算 `pre = sigmoid(...) + hc_eps`
   - 计算 `post = 2 * sigmoid(...)`
   - 计算 `comb_logits`
   - 对 `comb_logits` 执行官方 `hc_split_sinkhorn` 逻辑
   - 写入 `post_pad` 和 `comb_pad`
4. `hc_pre_mix_x`
   - 用 `pre[..., :4]` 对 4 份 hidden 加权求和
   - 写入 `x_mixed_pad`
5. `hc_pre_copy_out`
   - 拷贝 `x_mixed_pad[:S] -> x_mixed`
   - 拷贝 `post_pad[:S] -> post`
   - 拷贝 `comb_pad[:S] -> comb`

## Golden 和验证

`golden_hc_pre` 直接用 PyTorch 实现官方语义，不模拟内部 padding。为了匹配当前
PyPTO 输出接口：

- `post` 输出如果是 `[B, S, HC_PAD]`，golden 只填前 4 列，后 4 列置 0。
- `mixes` 和 `pre` scratch 如果是 padded shape，只填前 `S` 行。

## hc_post 实现

`hc_post_fwd` 已在 `models/hc.py` 中实现。官方语义为：

```python
y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
    comb.unsqueeze(-1) * residual.unsqueeze(-2),
    dim=2,
)
return y.type_as(x)
```

当前 PyPTO 接口：

```text
x:        [B, S_DYN, HIDDEN] bf16
residual: [B, S_DYN, HC_MULT, HIDDEN] bf16
post:     [B, S_DYN, HC_PAD] fp32
comb:     [B, S_DYN, HC_MULT * HC_MULT] fp32
out:      [B, S_DYN, HC_MULT, HIDDEN] bf16
```

`hc_post_fwd` 不使用 `S_PAD_DYN`，直接按 token 循环处理动态 `S_DYN`。原因是
它只读取 `post` 和 `comb` 的标量，并按 `D_TILE = 512` 读写 hidden tile，不会像
`hc_pre` 一样在小宽度 tile 上做复杂 subview 和 Sinkhorn。

`comb` 的 flat layout 仍按 row-major `[HC_MULT, HC_MULT]`。需要注意官方求和方向：

```text
out[..., j, :] = post[..., j] * x
              + sum_k comb[..., k, j] * residual[..., k, :]
```

也就是输出第 `j` 份 HC residual 使用 `comb` 的第 `j` 列，而不是第 `j` 行。

`golden_hc_post` 直接对齐官方 `Block.hc_post`，并支持 padded `post` 输入；只消费
`post[..., :HC_MULT]`。

当前已验证：

```text
pytest -q tests
python models/hc.py -p a2a3 -d {} -s 1
python models/hc.py -p a2a3 -d {} -s 13
python models/hc.py -p a2a3 -d {} -s 16
python models/hc.py -p a2a3 -d {} -s 32
python models/hc.py -p a2a3 -d {} -s 128
python models/hc.py -p a2a3 -d {} --case post -s 1
python models/hc.py -p a2a3 -d {} --case post -s 13
python models/hc.py -p a2a3 -d {} --case post -s 128
```

上述用例均通过。

## 后续注意事项

- `hc_post` 的 `post` 输入已按 `[B, S, HC_PAD]` 接口设计，只消费前 4 列。
- `comb` 仍按 `[B, S, 16]` flat layout 传递；如需要官方形态，可 reshape 为
  `[B, S, 4, 4]`。
- `hc_pre_fwd` 的 scratch tensor 都是函数式入参。上层 `block` 需要为 attention
  分支和 ffn 分支分别准备 scratch，避免同一 kernel 调用内读写别名不清晰。
