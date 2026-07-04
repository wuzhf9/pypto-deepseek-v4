# DeepSeek V4 Flash Block PyPTO 实现方案

本文定义普通 Transformer Block 的 PyPTO kernel 入口形态和接口边界。目标是对齐
`official/model.py` 中 `Block.forward` 的单卡 bf16 计算逻辑，并复用当前仓库已经实现的
HC、Attention 和 MoE 子 kernel。

MTP 不在本文范围内。官方配置中 `num_hidden_layers = 43`，
`num_nextn_predict_layers = 1`，`compress_ratios` 长度为 44。正常 Block 只使用
`compress_ratios[0:43]`，最后一个 `compress_ratios[43] = 0` 由 `MTPBlock` 使用。

## 官方 Block 逻辑

`official/model.py` 中普通 Block 的计算顺序是：

```python
residual = x
x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
x = self.attn_norm(x)
x = self.attn(x, start_pos)
x = self.hc_post(x, residual, post, comb)

residual = x
x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
x = self.ffn_norm(x)
x = self.ffn(x, input_ids)
x = self.hc_post(x, residual, post, comb)
return x
```

因此每个 Block 入口都由两段组成：

- attention 段：`hc_pre -> rmsnorm_4096 -> attention_* -> hc_post`
- ffn 段：`hc_pre -> rmsnorm_4096 -> moe_* -> hc_post`

Block 的输入输出 hidden state 维度保持为：

```text
x:   [B, S_DYN, HC_MULT, HIDDEN] BF16
out: [B, S_DYN, HC_MULT, HIDDEN] BF16
```

其中：

```text
B = 1
HC_MULT = 4
HIDDEN = 4096
S_DYN 同时覆盖 prefill 和 decode；decode 时实际 S=1
```

## Block 形态

普通 43 层按 attention `compress_ratio` 和 MoE gate 路径组合后，共有 4 种高层形态：

| Layer IDs | compress_ratio | Attention | MoE gate | 数量 |
|---|---:|---|---|---:|
| `0, 1` | `0` | SWA | hash | 2 |
| `2` | `4` | CSA | hash | 1 |
| `3,5,...,41` | `128` | HCA | topk | 20 |
| `4,6,...,42` | `4` | CSA | topk | 20 |

落到 PyPTO 入口时，prefill 和 decode 的 cache/state 输入输出不同，应拆成 8 个入口：

```text
block_swa_hash_prefill_fwd
block_swa_hash_decode_fwd
block_csa_hash_prefill_fwd
block_csa_hash_decode_fwd
block_hca_topk_prefill_fwd
block_hca_topk_decode_fwd
block_csa_topk_prefill_fwd
block_csa_topk_decode_fwd
```

不需要实现普通 Block 的 `swa_topk` 或 `hca_hash`。这两种组合在官方普通层配置中不存在。

## 公共参数组

为了避免每个 Block 入口重复描述上百个 scratch buffer，本文把接口按参数组描述。实际
`models/block.py` 实现时可以继续沿用子 kernel 的 `build_*_specs` 创建 scratch tensor。

### Hidden 输入输出

所有 Block 入口都有：

```text
x:       [B, S_DYN, HC_MULT, HIDDEN] BF16
out:     [B, S_DYN, HC_MULT, HIDDEN] BF16
```

### Attention HC 参数

attention 段的 `hc_pre/hc_post` 使用：

```text
hc_attn_fn:    [MIX_HC, HC_MULT * HIDDEN] FP32
hc_attn_scale: [3] FP32
hc_attn_base:  [MIX_HC] FP32
```

其中：

```text
MIX_HC = (2 + HC_MULT) * HC_MULT = 24
```

Block kernel 内还需要 attention 段 HC scratch：

```text
attn_hc_x_pad
attn_hc_mixes
attn_hc_pre
attn_hc_comb_logits
attn_hc_x_mixed_pad
attn_hc_post_pad
attn_hc_comb_pad
attn_hc_x_mixed
attn_hc_post
attn_hc_comb
```

这些 scratch 的具体 padded shape 以 `models/hc.py` 的 `hc_pre_fwd` 约定为准。

### FFN HC 参数

ffn 段的 `hc_pre/hc_post` 使用：

```text
hc_ffn_fn:    [MIX_HC, HC_MULT * HIDDEN] FP32
hc_ffn_scale: [3] FP32
hc_ffn_base:  [MIX_HC] FP32
```

Block kernel 内还需要与 attention 段相同结构的一组 ffn HC scratch。

### RMSNorm 参数

两个 RMSNorm 权重分别对应官方 `attn_norm` 和 `ffn_norm`：

```text
attn_norm_w: [HIDDEN] BF16
ffn_norm_w:  [HIDDEN] BF16
```

Block kernel 内部应调用当前 `models/rmsnorm.py` 的 `rmsnorm_4096` 路径，计算：

```text
attn_normed: [B, S_DYN, HIDDEN] BF16
ffn_normed:  [B, S_DYN, HIDDEN] BF16
```

### MoE hash 参数

hash gate 形态使用 `moe_hash_fwd`：

```text
gate_w_t:      [HIDDEN, N_EXPERTS] BF16
tid2eid:       [VOCAB, TOPK] INT32
input_ids:     [B, S_DYN] INT64
routed_w1_t:   [N_EXPERTS, HIDDEN, MOE_INTER_DIM] BF16
routed_w2_t:   [N_EXPERTS, MOE_INTER_DIM, HIDDEN] BF16
routed_w3_t:   [N_EXPERTS, HIDDEN, MOE_INTER_DIM] BF16
shared_w1_t:   [HIDDEN, MOE_INTER_DIM] BF16
shared_w2_t:   [MOE_INTER_DIM, HIDDEN] BF16
shared_w3_t:   [HIDDEN, MOE_INTER_DIM] BF16
```

以及 MoE scratch：

```text
logits:        [B, S_DYN, N_EXPERTS] FP32
scores:        [B, S_DYN, N_EXPERTS] FP32
indices:       [B, S_DYN, TOPK] INT32
weights:       [B, S_DYN, TOPK] FP32
route_y:       [B, S_DYN, TOPK, HIDDEN] BF16
shared_gate:   [B, S_DYN, MOE_INTER_DIM] BF16
shared_up:     [B, S_DYN, MOE_INTER_DIM] BF16
shared_hidden: [B, S_DYN, MOE_INTER_DIM] BF16
shared_y:      [B, S_DYN, HIDDEN] BF16
moe_out:       [B, S_DYN, HIDDEN] BF16
```

### MoE topk 参数

topk gate 形态使用 `moe_topk_fwd`，比 hash 多一个 gate bias，不需要 `tid2eid/input_ids`
参与 gate：

```text
gate_w_t:      [HIDDEN, N_EXPERTS] BF16
gate_bias:     [N_EXPERTS] FP32
routed_w1_t:   [N_EXPERTS, HIDDEN, MOE_INTER_DIM] BF16
routed_w2_t:   [N_EXPERTS, MOE_INTER_DIM, HIDDEN] BF16
routed_w3_t:   [N_EXPERTS, HIDDEN, MOE_INTER_DIM] BF16
shared_w1_t:   [HIDDEN, MOE_INTER_DIM] BF16
shared_w2_t:   [MOE_INTER_DIM, HIDDEN] BF16
shared_w3_t:   [HIDDEN, MOE_INTER_DIM] BF16
```

scratch 与 hash MoE 相同。

## Attention 参数组

### SWA attention

SWA 使用 `attention_swa_prefill_fwd/decode_fwd`，对应 `compress_ratio == 0`。

公共 attention 权重：

```text
wq_a_t:    [HIDDEN, Q_LORA_RANK] BF16
q_norm_w:  [Q_LORA_RANK] BF16
wq_b_t:    [Q_LORA_RANK, N_HEADS * HEAD_DIM] BF16
wkv_t:     [HIDDEN, HEAD_DIM] BF16
kv_norm_w: [HEAD_DIM] BF16
attn_sink: [N_HEADS] FP32
wo_a_t:    [HEADS_PER_GROUP * HEAD_DIM, O_GROUPS * O_LORA_RANK] BF16
wo_b_t:    [O_GROUPS * O_LORA_RANK, HIDDEN] BF16
cos:       [S_DYN, ROPE_HEAD_DIM / 2] FP32
sin:       [S_DYN, ROPE_HEAD_DIM / 2] FP32
```

prefill 额外输入：

```text
window_topk_idxs: [B, S_DYN, WINDOW_SIZE] INT32
```

prefill 输出状态：

```text
kv_cache_out: [B, WINDOW_SIZE, HEAD_DIM] BF16
```

decode 额外输入状态：

```text
kv_cache:        [B, WINDOW_SIZE, HEAD_DIM] BF16
cache_pos:       [1] INT32
window_topk_idxs:[B, S_DYN, WINDOW_SIZE] INT32
```

decode 输出状态：

```text
kv_cache_out: [B, WINDOW_SIZE, HEAD_DIM] BF16
```

### HCA attention

HCA 使用 `attention_hca_prefill_fwd/decode_fwd`，对应 `compress_ratio == 128`。

HCA 包含 SWA 的公共 attention 权重，另外包含 ratio-128 compressor 权重：

```text
comp_wkv_t:   [HIDDEN, HEAD_DIM] BF16
comp_wgate_t: [HIDDEN, HEAD_DIM] BF16
comp_ape:     [128, HEAD_DIM] FP32
comp_norm_w:  [HEAD_DIM] BF16
comp_cos:     [C_DYN, ROPE_HEAD_DIM / 2] FP32     # prefill
comp_sin:     [C_DYN, ROPE_HEAD_DIM / 2] FP32     # prefill
comp_cos:     [1, ROPE_HEAD_DIM / 2] FP32         # decode
comp_sin:     [1, ROPE_HEAD_DIM / 2] FP32         # decode
```

prefill 额外输入：

```text
topk_idxs:         [B, S_DYN, WINDOW_SIZE + TOPK_HCA] INT32
comp_block_count:  [1] INT32
```

prefill 输出状态：

```text
kv_cache_out:         [B, WINDOW_SIZE, HEAD_DIM] BF16
comp_kv_state_out:    [B, 128, HEAD_DIM] FP32
comp_score_state_out: [B, 128, HEAD_DIM] FP32
comp_cache_out:       [B, TOPK_HCA, HEAD_DIM] BF16
```

decode 额外输入状态：

```text
kv_cache:             [B, WINDOW_SIZE, HEAD_DIM] BF16
comp_kv_state:        [B, 128, HEAD_DIM] FP32
comp_score_state:     [B, 128, HEAD_DIM] FP32
comp_cache:           [B, TOPK_HCA, HEAD_DIM] BF16
cache_pos:            [1] INT32
comp_slot:            [1] INT32
comp_cache_slot:      [1] INT32
comp_should_compress: [1] INT32
topk_idxs:            [B, S_DYN, WINDOW_SIZE + TOPK_HCA] INT32
```

decode 输出状态与 prefill 相同。

### CSA attention

CSA 使用 `attention_csa_prefill_fwd/decode_fwd`，对应 `compress_ratio == 4`。它同时包含：

- 普通 attention Q/KV/O 投影。
- attention 侧 ratio-4 compressor，用于生成 attention compressed KV。
- indexer，用于生成 compressed sparse attention 的 topk indices。
- indexer 内部的 ratio-4 compressor。

CSA 包含 SWA 的公共 attention 权重，另外包含 attention compressor 权重：

```text
attn_comp_wkv_t:   [HIDDEN, 2 * HEAD_DIM] BF16
attn_comp_wgate_t: [HIDDEN, 2 * HEAD_DIM] BF16
attn_comp_ape:     [4, 2 * HEAD_DIM] FP32
attn_comp_norm_w:  [HEAD_DIM] BF16
attn_comp_cos/sin: [C_DYN, ROPE_HEAD_DIM / 2] FP32  # prefill
attn_comp_cos/sin: [1, ROPE_HEAD_DIM / 2] FP32      # decode
```

以及 indexer 权重：

```text
idx_wq_b_t:       [Q_LORA_RANK, INDEX_N_HEADS * INDEX_HEAD_DIM] BF16
idx_weights_proj_t:[HIDDEN, INDEX_N_HEADS] BF16
idx_offset:       [1] INT32
idx_comp_wkv_t:   [HIDDEN, 2 * INDEX_HEAD_DIM] BF16
idx_comp_wgate_t: [HIDDEN, 2 * INDEX_HEAD_DIM] BF16
idx_comp_ape:     [4, 2 * INDEX_HEAD_DIM] FP32
idx_comp_norm_w:  [INDEX_HEAD_DIM] BF16
idx_comp_cos/sin: [C_DYN, ROPE_HEAD_DIM / 2] FP32   # prefill
idx_comp_cos/sin: [1, ROPE_HEAD_DIM / 2] FP32       # decode
```

prefill 额外输入：

```text
window_topk_idxs:       [B, S_DYN, WINDOW_SIZE] INT32
attn_comp_block_count:  [1] INT32
idx_comp_block_count:   [1] INT32
```

prefill 输出状态：

```text
kv_cache_out:                  [B, WINDOW_SIZE, HEAD_DIM] BF16
attn_comp_kv_state_out:        [B, 8, 2 * HEAD_DIM] FP32
attn_comp_score_state_out:     [B, 8, 2 * HEAD_DIM] FP32
attn_comp_cache_out:           [B, TOPK_CSA_COMPRESSED, HEAD_DIM] BF16
idx_kv_cache_out:              [B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM] BF16
idx_comp_kv_state_out:         [B, 8, 2 * INDEX_HEAD_DIM] FP32
idx_comp_score_state_out:      [B, 8, 2 * INDEX_HEAD_DIM] FP32
```

decode 额外输入状态：

```text
kv_cache:                 [B, WINDOW_SIZE, HEAD_DIM] BF16
attn_comp_kv_state:       [B, 8, 2 * HEAD_DIM] FP32
attn_comp_score_state:    [B, 8, 2 * HEAD_DIM] FP32
attn_comp_cache:          [B, TOPK_CSA_COMPRESSED, HEAD_DIM] BF16
idx_kv_cache_in:          [B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM] BF16
idx_comp_kv_state:        [B, 8, 2 * INDEX_HEAD_DIM] FP32
idx_comp_score_state:     [B, 8, 2 * INDEX_HEAD_DIM] FP32
cache_pos:                [1] INT32
comp_slot:                [1] INT32
comp_cache_slot:          [1] INT32
comp_should_compress:     [1] INT32
window_topk_idxs:         [B, S_DYN, WINDOW_SIZE] INT32
```

decode 输出状态与 prefill 相同。

## 8 个 Block 入口

### `block_swa_hash_prefill_fwd`

对应 layer `0, 1` 的 prefill。

输入参数组：

- Hidden 输入输出。
- Attention HC 参数。
- `attn_norm_w`。
- SWA attention prefill 参数。
- FFN HC 参数。
- `ffn_norm_w`。
- MoE hash 参数。

输出：

```text
kv_cache_out
out
```

其中 `out` 是完整 Block 输出 `[B, S_DYN, HC_MULT, HIDDEN] BF16`。

### `block_swa_hash_decode_fwd`

对应 layer `0, 1` 的 decode。

输入参数组：

- Hidden 输入输出。
- Attention HC 参数。
- `attn_norm_w`。
- SWA attention decode 参数。
- FFN HC 参数。
- `ffn_norm_w`。
- MoE hash 参数。

输出：

```text
kv_cache_out
out
```

### `block_csa_hash_prefill_fwd`

对应 layer `2` 的 prefill。

输入参数组：

- Hidden 输入输出。
- Attention HC 参数。
- `attn_norm_w`。
- CSA attention prefill 参数。
- FFN HC 参数。
- `ffn_norm_w`。
- MoE hash 参数。

输出：

```text
kv_cache_out
attn_comp_kv_state_out
attn_comp_score_state_out
attn_comp_cache_out
idx_kv_cache_out
idx_comp_kv_state_out
idx_comp_score_state_out
out
```

### `block_csa_hash_decode_fwd`

对应 layer `2` 的 decode。

输入参数组：

- Hidden 输入输出。
- Attention HC 参数。
- `attn_norm_w`。
- CSA attention decode 参数。
- FFN HC 参数。
- `ffn_norm_w`。
- MoE hash 参数。

输出与 `block_csa_hash_prefill_fwd` 相同。

### `block_hca_topk_prefill_fwd`

对应 layer `3,5,...,41` 的 prefill。

输入参数组：

- Hidden 输入输出。
- Attention HC 参数。
- `attn_norm_w`。
- HCA attention prefill 参数。
- FFN HC 参数。
- `ffn_norm_w`。
- MoE topk 参数。

输出：

```text
kv_cache_out
comp_kv_state_out
comp_score_state_out
comp_cache_out
out
```

### `block_hca_topk_decode_fwd`

对应 layer `3,5,...,41` 的 decode。

输入参数组：

- Hidden 输入输出。
- Attention HC 参数。
- `attn_norm_w`。
- HCA attention decode 参数。
- FFN HC 参数。
- `ffn_norm_w`。
- MoE topk 参数。

输出与 `block_hca_topk_prefill_fwd` 相同。

### `block_csa_topk_prefill_fwd`

对应 layer `4,6,...,42` 的 prefill。

输入参数组：

- Hidden 输入输出。
- Attention HC 参数。
- `attn_norm_w`。
- CSA attention prefill 参数。
- FFN HC 参数。
- `ffn_norm_w`。
- MoE topk 参数。

输出与 `block_csa_hash_prefill_fwd` 相同。

### `block_csa_topk_decode_fwd`

对应 layer `4,6,...,42` 的 decode。

输入参数组：

- Hidden 输入输出。
- Attention HC 参数。
- `attn_norm_w`。
- CSA attention decode 参数。
- FFN HC 参数。
- `ffn_norm_w`。
- MoE topk 参数。

输出与 `block_csa_hash_prefill_fwd` 相同。

## 实现注意事项

- Block 入口不要重新实现 attention、compressor、indexer、MoE 的内部逻辑，应调用已经验证过的
  inline kernel。
- prefill 和 decode 应保持独立入口。二者的高层计算顺序一致，但 cache/state 输入输出不同，
  强行合并会让接口和验证逻辑变复杂。
- `input_ids` 只对 hash MoE 必需；topk MoE 不需要 `input_ids` 参与 gate。
- 所有 linear 权重继续沿用当前仓库约定，传入转置后的 `*_t` 权重。
- Block golden 应直接按 `official.model.Block.forward` 的顺序写完整函数，attention 和 MoE
  可复用当前各模块的 golden，但最终应验证完整 Block 输出和状态更新。
- 独立验证时先覆盖每种入口的代表层：layer 0、layer 2、layer 3、layer 4。整网串联时再按
  layer id 选择对应入口。

## 遗留问题

当前 Block kernel 接口保留了大量 scratch buffer，主要原因是部分 scratch shape 带有
`S_DYN`、动态 padding 后的 token 数，或者子 kernel 内部定义的动态维度。此前在调试中遇到过
子 kernel 返回 tensor 后，外层变量重新绑定时动态维符号不一致的问题，因此当前实现优先选择把
scratch 作为外部参数传入，并通过子 kernel 写入已有 buffer，避免返回内部动态 shape tensor。

`../pypto-serving` 中 Qwen3 路径存在 kernel 内部使用运行时维度创建 tensor 的写法，例如通过
`pl.tensor.dim(...)` 得到 token 数后再调用 `pl.create_tensor([tokens, HIDDEN], ...)`。这说明
PyPTO 并非完全不支持在 kernel 内创建带动态维度的 tensor。但 DeepSeek v4 路径中的 scratch
大多仍基于固定编译 shape 或固定 tile shape，不能直接证明当前 Block 中所有动态 padding scratch
都可以安全迁移到 kernel 内部。

已在 `models/head.py` 中验证过 `[B, S_DYN, HIDDEN]` 这类直接动态维 scratch：`head_fwd`
内部用 `tokens = pl.tensor.dim(x, 1)` 创建 `hc_out/normed = pl.create_tensor([B, tokens,
HIDDEN], dtype=pl.BF16)`，并把它们继续传给 `hc_head_fwd`、`rmsnorm_4096` 和 `lm_head_fwd`。
固定 shape 的 `logits_pad = pl.create_tensor([T_TILE, VOCAB], dtype=pl.FP32)` 也已迁入内部。
远端 Ascend 已通过默认 `S=8`、`S=13` 和 `S=1` 验证。因此不带 padding 表达式、动态维直接
来自已有输入的中间 tensor，可以优先迁移到 kernel 内部。

已在 `models/head.py` 中做过一次针对 padded dynamic scratch 的实验：在 `head_fwd` 内部用
`tokens = pl.tensor.dim(x, 1)` 计算
`padded_tokens = ((tokens + T_TILE - 1) // T_TILE) * T_TILE`，再创建
`x_pad/pre/hc_out_pad` 并传入 `hc_head_fwd`。本地 golden 测试可以通过，但远端 Ascend
编译失败，报错为：

```text
@pl.jit: missing inferred tensor metadata for parameter 'x_pad'
```

这个结果说明当前 PyPTO 可以支持由已有动态维直接创建 tensor，但对于经过表达式计算得到的新
padded dynamic 维度，作为子 kernel 参数传递时还无法稳定推断 tensor metadata。因此 Block
中这类 padded scratch 仍应保留为外部参数，或者把使用 scratch 的逻辑直接展开在同一个 kernel
内，避免把内部创建的 padded dynamic tensor 再传给子 kernel。

后续如果要继续精简 Block 接口，应先做独立验证：

1. 在最小 kernel 中创建 `[B, tokens, HIDDEN]` 这种直接来自 `pl.tensor.dim` 的 scratch。
2. 再验证 `tokens` 经过向上取整后的 padded shape，例如 `[B, padded_tokens, ...]`。
3. 验证通过后，再逐步把 HC、attention、MoE 的中间 scratch 移入 Block kernel 内部。
4. 即使迁入内部，也应避免把内部动态 shape tensor 作为子 kernel 返回值重新绑定到外层变量；
   优先保持写入外部已知 shape buffer 或只返回最终输出/state。

## Golden 设计

Block 的 golden 不需要为 8 个入口各写一份完整计算逻辑。官方 `Block.forward` 的骨架对所有
普通层完全一致：

```text
hc_pre -> attn_norm -> attention -> hc_post
hc_pre -> ffn_norm  -> moe       -> hc_post
```

8 个入口的差异只来自：

- attention 路径：`swa`、`csa`、`hca`
- MoE gate 路径：`hash`、`topk`
- prefill/decode：通过 `start_pos == 0` 或 `start_pos > 0` 决定 attention cache/state 行为

因此应实现一份核心 golden：

```python
def golden_block_forward(
    tensors,
    *,
    start_pos: int,
    attention_kind: str,
    hash_route: bool,
):
    ...
```

核心 golden 内部按官方 `Block.forward` 顺序展开：

1. 调用 `golden_hc_pre` 得到 attention 输入、`post` 和 `comb`。
2. 调用 hidden RMSNorm golden，得到 `attn_normed`。
3. 根据 `attention_kind` 调用对应 attention golden：
   - `swa` -> `golden_attention_swa_forward`
   - `hca` -> `golden_attention_hca_forward`
   - `csa` -> `golden_attention_csa_forward`
4. 调用 `golden_hc_post` 得到 attention 段输出。
5. 再次调用 `golden_hc_pre` 得到 FFN 输入、`post` 和 `comb`。
6. 调用 hidden RMSNorm golden，得到 `ffn_normed`。
7. 调用 `golden_moe_forward(..., hash_route=hash_route)`。
8. 调用 `golden_hc_post` 得到 Block 输出。

为了对齐每个 PyPTO kernel 的输出 tuple，仍然保留 8 个很薄的 wrapper：

```python
golden_block_swa_hash_prefill(tensors)
golden_block_swa_hash_decode(tensors)
golden_block_csa_hash_prefill(tensors)
golden_block_csa_hash_decode(tensors)
golden_block_hca_topk_prefill(tensors)
golden_block_hca_topk_decode(tensors)
golden_block_csa_topk_prefill(tensors)
golden_block_csa_topk_decode(tensors)
```

这些 wrapper 不重复实现计算逻辑，只负责传入固定参数：

```python
return golden_block_forward(
    tensors,
    start_pos=...,
    attention_kind="csa",
    hash_route=True,
)
```

并按对应 kernel 的输出顺序返回结果。

这种组织方式和当前仓库中 `attention_*`、`compressor_ratio*`、`moe.py` 的 golden 风格一致：

- 核心语义只维护一份，降低后续改动时的偏差风险。
- prefill/decode 和 hash/topk wrapper 保持独立，方便 runner 和 `run_jit` 精度验证。
- 不把所有 case 合并成一个测试入口，避免不同 attention state/cache 输出混在一起。
