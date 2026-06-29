# DeepSeek V4 Flash 模型结构与计算逻辑

本文总结 `../deepseek_v4_flash` 中 DeepSeek V4 Flash 的模型结构和主要计算逻辑。

参考代码：

- 官方开源模型定义：`../deepseek_v4_flash/inference/model.py`
- 官方开源 kernel：`../deepseek_v4_flash/inference/kernel.py`
- bf16 权重转换：`../deepseek_v4_flash/inference/convert_bf16_index.py`
- bf16 推理实现：
  - `../deepseek_v4_flash/inference/low_vram_executor.py`
  - `../deepseek_v4_flash/inference/low_vram_attention.py`
  - `../deepseek_v4_flash/inference/low_vram_moe.py`
  - `../deepseek_v4_flash/inference/low_vram_base.py`
  - `../deepseek_v4_flash/inference/low_vram_kernels.py`

本文重点关注模型本身的结构和数学计算逻辑。内存 offload/cache 策略不是重点，仅在解释 bf16 权重加载边界时简要提及。

## 总体结构

DeepSeek V4 Flash 的主干结构是：

```text
input_ids
 -> token embedding
 -> expand to hc_mult hidden copies
 -> N x Transformer Block
 -> HC head reduce
 -> final RMSNorm
 -> LM head
 -> logits
```

官方 `Transformer.forward()` 的核心流程：

1. `ParallelEmbedding` 将 token id 转成 hidden。
2. hidden 扩展为 `hc_mult` 份：

   ```text
   h: [batch, seq, dim]
   h -> [batch, seq, hc_mult, dim]
   ```

3. 依次经过 `n_layers` 个 `Block`。
4. `ParallelHead` 通过 HC head 将 `[batch, seq, hc_mult, dim]` reduce 成 `[batch, seq, dim]`。
5. final `RMSNorm`。
6. LM head 只对最后一个 token 计算 logits。

## Hyper-Connections

DeepSeek V4 Flash 的 block 不是普通 transformer residual block，而是基于 **Hyper-Connections, HC** 的多副本 hidden 结构。

每个 token 的 hidden 不只保留一份，而是保留 `hc_mult` 份：

```text
x: [batch, seq, hc_mult, dim]
```

每个 block 内有两个 HC 子路径：

```text
HC-Attention path
HC-FFN path
```

### hc_pre

进入 Attention 或 FFN 前，`hc_pre()` 将多副本 hidden reduce 成单路 hidden：

```text
x: [B, S, hc_mult, D]
flatten -> [B, S, hc_mult * D]
mixes = linear(flatten(x), hc_fn) * rms_scale
pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base)
y = sum(pre[..., i] * x[..., i, :])
```

输出：

- `y`: `[B, S, D]`，送入 Attention/FFN。
- `post`: `[B, S, hc_mult]`，用于子层输出写回。
- `comb`: `[B, S, hc_mult, hc_mult]`，用于 residual 多副本混合。

`hc_split_sinkhorn()` 会把 `mixes` 拆成三部分：

- `pre`: 进入子层前的多副本加权 reduce。
- `post`: 子层输出扩展回多副本时的权重。
- `comb`: residual 多副本之间的混合矩阵。

`comb` 经过 softmax 和 Sinkhorn 迭代，近似形成行列归一化的混合矩阵。

### hc_post

子层计算完成后，`hc_post()` 将单路输出重新写回 `hc_mult` 份：

```text
y = post * sublayer_output + comb * residual
```

形状：

```text
sublayer_output: [B, S, D]
residual:        [B, S, hc_mult, D]
y:               [B, S, hc_mult, D]
```

因此一个普通 block 可以理解为：

```text
residual = x
x_single, post, comb = hc_pre_attn(x)
x_single = attn_norm(x_single)
x_single = Attention(x_single)
x = hc_post(x_single, residual, post, comb)

residual = x
x_single, post, comb = hc_pre_ffn(x)
x_single = ffn_norm(x_single)
x_single = MoE(x_single)
x = hc_post(x_single, residual, post, comb)
```

## RMSNorm

RMSNorm 逻辑是标准形式：

```text
y = x * rsqrt(mean(x^2) + eps)
out = weight * y
```

实现上通常将输入转成 fp32 做方差和归一化，再 cast 回原 dtype。

## Attention 总览

Attention 是 MLA 风格的 latent KV attention，并结合：

- low-rank query projection
- RoPE
- sliding-window sparse attention
- optional compressed KV
- learned compressed-KV indexer
- grouped low-rank output projection

输入 `x` 的形状：

```text
x: [batch, seq, dim]
```

核心流程：

```text
q path:
  x -> wq_a -> q_norm -> wq_b
  -> [B, S, n_heads, head_dim]
  -> per-head RMS normalize
  -> RoPE on tail rope_head_dim

kv path:
  x -> wkv -> kv_norm
  -> [B, S, head_dim]
  -> RoPE on tail rope_head_dim
  -> write sliding-window KV cache
  -> optional compressed KV cache

attention:
  topk_idxs = sliding window idxs + optional compressed idxs
  o = sparse_attn(q, selected kv, attn_sink, topk_idxs)
  inverse RoPE on o tail rope_head_dim

output:
  group heads
  o -> wo_a -> wo_b
```

注意：KV 是 latent KV，形状是 `[B, S, head_dim]`，不是 `[B, S, n_heads, head_dim]`。每个 query head 都和选出的 latent KV 做点积。

## RoPE 逻辑

RoPE 只作用于最后 `rope_head_dim` 个维度：

```text
q[..., -rope_head_dim:]
kv[..., -rope_head_dim:]
```

Attention 输出后还会对输出尾部做 inverse RoPE：

```text
apply_rotary_emb(o[..., -rope_head_dim:], inverse=True)
```

当该层启用 compressed KV 时，RoPE 使用 `compress_rope_theta` 和 YaRN 参数；纯 sliding-window attention 则使用基础 `rope_theta`。

## Sliding Window Sparse Attention

Attention 不对完整历史做 dense attention，而是通过 `topk_idxs` 指定每个 query 能看的 KV 位置。

基础部分是 sliding window：

```text
get_window_topk_idxs(window_size, batch, seq, start_pos)
```

它选择最近 `window_size` 个 token。decode 阶段使用环形窗口缓存：

```text
kv_cache[:, start_pos % window_size] = current_kv
```

prefill 阶段如果 prompt 长度超过 window，则只保留最后一个 window 的 KV。

## Sparse Attention 计算

官方 `kernel.py` 中 `sparse_attn` 是 TileLang kernel；bf16 版本的 `low_vram_kernels.py` 用 PyTorch 实现相同语义。

对每个 `(batch, seq)`：

1. 取该位置的 `topk_idxs`。
2. 丢弃 `-1` 无效位置。
3. gather 对应 latent KV：

   ```text
   selected = kv[batch, topk_idxs]
   ```

4. 每个 head 计算：

   ```text
   scores = q_head @ selected.T * softmax_scale
   ```

5. 追加一个 learnable `attn_sink` logit。
6. 对 `[selected scores, attn_sink]` 做 softmax。
7. 去掉 sink 概率，只用真实 KV 概率加权 selected KV：

   ```text
   out_head = probs_without_sink @ selected
   ```

`attn_sink` 的效果是允许部分注意力质量流向一个不产生 value 的 sink，从而调节实际 KV 的注意力强度。

## Compressed KV

部分层配置了 `compress_ratio`。若 `compress_ratio > 0`，Attention 除了 sliding-window KV，还会维护 compressed KV cache。

压缩逻辑由 `Compressor` 实现：

```text
x -> wkv -> kv_candidate
x -> wgate -> score
score += ape
weights = softmax(score over ratio tokens)
compressed_kv = sum(weights * kv_candidate)
compressed_kv -> RMSNorm
RoPE on tail rope_head_dim
write compressed kv_cache
```

prefill 阶段：

- 将 prompt 切成长度为 `compress_ratio` 的块。
- 每块压缩为一个 compressed KV。
- 如果最后有 remainder，则保留到 state，等待后续 decode 补齐。

decode 阶段：

- 每来一个 token，更新 compressor state。
- 当 `(start_pos + 1) % compress_ratio == 0` 时生成一个新的 compressed KV。

当 `compress_ratio == 4` 时，启用 overlap compression：

- state 中维护 overlap window 和 current window。
- 压缩时把前一窗口尾部和当前窗口拼接，减少块边界割裂。

## Compressed KV 索引

当 `compress_ratio == 4` 时，模型使用 learned `Indexer` 选择 compressed KV 的 top-k。

Indexer 逻辑：

```text
qr -> indexer.wq_b -> q_index
q_index -> RoPE
indexer.compressor(x) -> indexer compressed kv cache
weights = weights_proj(x)
index_score = q_index @ compressed_kv
index_score = relu(index_score) * weights
sum over index heads
topk -> compressed topk idxs
```

得到的 compressed topk 会追加到 sliding-window topk 后面：

```text
topk_idxs = concat(window_topk_idxs, compress_topk_idxs)
```

当 `compress_ratio` 非 0 但不是 4 时，compressed topk 使用规则索引：

```text
get_compress_topk_idxs(ratio, ...)
```

## Attention 输出投影

Sparse attention 输出形状：

```text
o: [B, S, n_heads, head_dim]
```

随后先做 inverse RoPE，再按 `o_groups` 分组：

```text
o -> [B, S, n_groups, n_heads * head_dim / n_groups]
```

输出投影是两段式低秩结构：

```text
o = einsum(o, wo_a)
out = wo_b(o)
```

其中：

- `wo_a` 是按 group 组织的 projection。
- `wo_b` 将 `n_groups * o_lora_rank` 投回 `dim`。

## MoE FFN

FFN 是 Mixture-of-Experts：

```text
x -> Gate -> top-k routed experts
selected experts -> SwiGLU expert
sum routed expert outputs
+ shared expert output
```

### Gate

Gate 先计算 expert 分数：

```text
scores = linear(x, gate.weight)
```

支持三种 score function：

- `softmax`
- `sigmoid`
- `sqrtsoftplus`

若该层是 hash routing 层：

```text
indices = tid2eid[input_ids]
```

否则：

```text
scores_for_topk = scores + bias
indices = topk(scores_for_topk)
```

注意：bias 只影响 top-k expert 选择，不影响最终 route weight。最终权重从未加 bias 的 `original_scores` 中 gather：

```text
weights = original_scores.gather(indices)
```

若 score function 不是 softmax，还会对 top-k 权重重新归一化，再乘 `route_scale`。

### Expert

每个 expert 是 SwiGLU FFN：

```text
gate = w1(x)
up   = w3(x)
hidden = silu(gate) * up
out = w2(hidden)
```

如果配置了 `swiglu_limit`，会对 `gate/up` 做 clamp。

MoE 输出：

```text
y = sum(route_weight_i * expert_i(x))
y += shared_experts(x)
```

## HC Head 与 LM Head

所有 transformer block 结束后，hidden 仍是多副本形式：

```text
h: [B, S, hc_mult, dim]
```

`ParallelHead.hc_head()` 用一组 HC head 权重将多副本 hidden reduce：

```text
x = flatten(hc copies)
mixes = linear(x, hc_head_fn) * rms_scale
pre = sigmoid(mixes * hc_head_scale + hc_head_base) + hc_eps
y = sum(pre_i * h_i)
```

然后：

```text
y = final RMSNorm(y)
logits = linear(y[:, -1], lm_head)
```

LM head 只对最后一个 token 计算 logits。

## MTP Block

`model.py` 还定义了 `MTPBlock`，用于 multi-token prediction。

MTPBlock 会融合当前 hidden 和目标 token embedding：

```text
e = embed(input_ids)
e = enorm(e)
x = hnorm(hidden)
x = e_proj(e).unsqueeze(2) + h_proj(x)
x = normal Block forward
logits = shared head(x)
```

主 `Transformer.forward()` 默认只跑主干 `layers` 并输出 logits；`mtp` 模块被构造出来，但不在主 forward 中自动执行。

## 官方低精度路径

官方 `model.py/kernel.py` 支持低精度权重和低精度 activation 计算。

### FP8 Linear

若权重 dtype 是 `torch.float8_e4m3fn`：

1. activation 先做 block-wise FP8 quant：

   ```text
   x, scale = act_quant(x, block_size=128)
   ```

2. 使用 `fp8_gemm(x, x_scale, weight, weight.scale)`。

FP8 weight scale 是按 128x128 block 组织。

### FP4 Expert Linear

若 expert 权重 dtype 是 `torch.float4_e2m1fn_x2`：

1. 权重按 K 维打包，逻辑形状 `[out, in]`，实际存储 `[out, in / 2]`。
2. 每 32 个 FP4 weight 一组 scale。
3. activation 仍先量化成 FP8。
4. 使用 `fp4_gemm()` 执行 FP8 activation x FP4 weight GEMM。

### Attention 中的 Activation Quant

官方路径里还有几处为了匹配 QAT/低精度行为的 activation quant：

普通 attention KV：

```text
act_quant(kv[..., :-rope_head_dim], 64, ...)
```

普通 compressor：

```text
act_quant(kv[..., :-rope_head_dim], 64, ...)
```

indexer query 和 indexer compressor：

```text
rotate_activation(...)
fp4_act_quant(..., fp4_block_size=32, inplace=True)
```

`rotate_activation()` 使用 Hadamard rotation，在进入 FP4 quant 前扩散激活信息。

## 当前 bf16 路径的权重转换

当前自实现 bf16 路径不直接使用官方 FP8/FP4 GEMM，而是将原始 checkpoint 中的低精度权重加载后反量化成 bf16。

`convert_bf16_index.py` 做两件事：

1. 扫描 safetensors，建立 `weight_index.json`。
2. 对低精度权重记录对应 scale tensor。

FP8 权重反量化：

```text
weight: float8_e4m3fn [out, in]
scale:  [out / 128, in / 128]

bf16_weight = weight.float() * per_block_scale
```

FP4 expert 权重反量化：

```text
packed weight: int8 [out, in / 2]
scale:         [out, in / 32]

unpack two FP4 values per byte
map E2M1 code -> float value
bf16_weight = unpacked.float() * repeat_interleave(scale, 32)
```

反量化后的权重全部作为普通 floating-point linear weight 使用。

## 当前 bf16 路径移除的低精度操作

bf16 推理路径保留模型结构和核心数学逻辑，但移除了官方低精度计算中的量化模拟和量化 GEMM。

主要差异：

| 官方路径 | bf16 路径 |
| --- | --- |
| FP8/FP4 weight + scale | 加载时反量化成 bf16 weight |
| `act_quant()` | 移除 |
| `fp4_act_quant()` | 移除 |
| `rotate_activation()` before FP4 quant | 移除 |
| `fp8_gemm()` / `fp4_gemm()` | 普通 `F.linear()` |
| TileLang sparse attention | PyTorch sparse attention 等价实现 |

具体到自实现文件：

- `low_vram_base.py` 的 `linear()` 对加载出的 bf16 weight 直接调用 `F.linear()`。
- `low_vram_attention.py` 保留 Q/KV projection、RoPE、KV cache、Compressor、Indexer、sparse attention 和 output projection，但显式删掉：
  - `act_quant(kv[..., :-rd], ...)`
  - `rotate_activation(...)`
  - `fp4_act_quant(...)`
- `low_vram_moe.py` 保留 Gate、routed experts、shared expert 和 SwiGLU 计算，expert 权重由加载阶段反量化成 bf16 后用普通 linear 计算。

因此当前 bf16 版本可以理解为：

```text
DeepSeek V4 Flash 原始模型结构
+ 官方低精度权重的 bf16 反量化
- activation quant
- FP4 前 Hadamard rotate
- FP8/FP4 GEMM kernel
```

也就是说，模型的 HC、MLA sparse attention、compressed KV、Indexer、MoE 和 head 逻辑保持不变；变化主要集中在数值表示和低精度 kernel 路径上。
