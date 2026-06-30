# DeepSeek V4 Flash Sparse Attention Plan

本文档记录 `../deepseek_v4_flash/inference/model.py` 中 `sparse_attn`
对应的 PyPTO 实现方案。目标是对齐官方 bf16 推理路径中的 attention 计算逻辑，并为
后续 `attention_swa.py`、`attention_csa.py`、`attention_hca.py` 提供稳定的
shape-specialized kernel 入口。

## 官方计算语义

官方 bf16 路径中 `sparse_attn` 的参考实现位于
`../deepseek_v4_flash/inference/low_vram_kernels.py`：

```python
selected = kv[batch_id, idxs].float()
scores = torch.einsum("hd,td->ht", q[batch_id, seq_id].float(), selected) * softmax_scale
scores = torch.cat([scores, attn_sink.float().view(n_heads, 1)], dim=1)
probs = torch.softmax(scores, dim=1)[:, :-1]
out[batch_id, seq_id] = torch.einsum("ht,td->hd", probs, selected).to(q.dtype)
```

其中：

- `q` 是每个 token 的多头 query，shape 为 `[B, S, n_heads, head_dim]`。
- `kv` 是 latent KV pool，shape 为 `[B, K, head_dim]`。
- `topk_idxs` 指定每个 query token 能访问的 KV 位置，shape 为 `[B, S, TOPK]`。
- `topk_idxs < 0` 表示 padding 或无效位置，不参与 softmax。
- `attn_sink` 是每个 head 一个额外 logit，参与 softmax，但不产生 value。
- `softmax_scale = head_dim ** -0.5`。

DeepSeek V4 Flash 固定：

```text
B = 1
n_heads = 64
head_dim = 512
window_size = 128
```

PyPTO kernel 只实现 bf16/fp32 计算路径：

```text
q input:        BF16
kv input:       BF16
attn_sink:      FP32
score compute:  FP32
softmax:        FP32
value reduce:   FP32
out:            BF16
```

不实现：

- fp8/fp4 activation quant
- paged attention
- packed prefill
- chunked prefill
- 多卡 TP/EP 并行

## Shape 变量

`sparse_attn` 概念上涉及两个可变长度：

```text
S:     query token 数
TOPK:  每个 query token 可看的 KV index 数
```

但 PyPTO kernel 中不建议把 `TOPK` 做成 dynamic shape。更稳妥的 ABI 是：

```text
S 使用 dynamic
TOPK 使用静态上限 TOPK_MAX
无效 topk 位置填 -1
```

这样可以保持 softmax 长度、score buffer 和 value reduce 循环都是编译期固定长度，同时
通过 `topk_idxs < 0` 保持和官方逻辑一致。

`TOPK_MAX` 不应作为全模型统一常量。它由 attention 路径、`window_size`、
`index_topk` 和 runner 选择的 `max_seq_len` 共同决定。整网串联时，应根据每层固定的
`compress_ratio` 选择对应的 attention kernel，而不是强行让所有层共用最大的
`TOPK_MAX`。

## `topk_idxs` 生成逻辑

官方 window topk：

```python
get_window_topk_idxs(window_size, bsz, seqlen, start_pos)
```

形状：

```text
start_pos == 0:
  [B, S, min(S, window_size)]

start_pos > 0:
  [B, 1, window_size]
```

在 PyPTO 路径中，建议 host 或上层 attention wrapper 生成固定宽度的
`[B, S, TOPK_MAX]`，不足部分填 `-1`。

对于 compressed KV：

```text
ratio = 4:
  compressed topk 来自 Indexer
  compressed_topk = min(index_topk, end_pos // 4)
  index_topk = 512

ratio = 128:
  compressed topk 来自 get_compress_topk_idxs
  compressed_topk = end_pos // 128
```

最终 topk 是 window topk 和 compressed topk 的拼接：

```text
topk_idxs = cat([window_topk, compressed_topk], dim=-1)
```

## TOPK_MAX 策略

各路径推荐的静态上限：

```text
ratio = 0:
  TOPK_MAX = window_size = 128

ratio = 4:
  TOPK_MAX = window_size + index_topk
           = 128 + 512
           = 640

ratio = 128:
  TOPK_MAX = window_size + max_seq_len // 128
```

如果强行使用全网统一上限：

```text
UNIFIED_TOPK_MAX =
  window_size + max(
    0,
    min(index_topk, max_seq_len // 4),
    max_seq_len // 128,
  )
```

这个值会被 HCA 或 CSA 的最大 compressed topk 支配。以官方
`max_position_embeddings = 1048576` 计算：

```text
ratio = 4:
  TOPK_MAX = 640

ratio = 128:
  TOPK_MAX = 128 + 8192 = 8320

unified:
  TOPK_MAX = 8320
```

这会让 `ratio=0` 的 SWA 层也承受 `[S, 8320]` 的 topk、score 和 softmax 静态形状，
不利于编译稳定性，也没有必要。当前目标优先保证正确性，但过大的静态 shape 仍可能导致
PyPTO 编译、scratch buffer 或 ptoas 问题。

因此推荐：

```text
语义统一，shape 专门化。
```

即在 `models/sparse_attn.py` 中复用同一套计算逻辑，但提供多个
shape-specialized PyPTO 入口。

## Prefill 和 Decode Shape

### SWA: `compress_ratio = 0`

prefill:

```text
q:         [1, S_DYN, 64, 512] BF16
kv:        [1, S_DYN, 512] BF16
topk_idxs: [1, S_DYN, 128] INT32
attn_sink: [64] FP32
out:       [1, S_DYN, 64, 512] BF16
```

当 `S < 128` 时，`topk_idxs` 后半部分填 `-1`。

decode:

```text
q:         [1, 1, 64, 512] BF16
kv_cache:  [1, 128, 512] BF16
topk_idxs: [1, 1, 128] INT32
attn_sink: [64] FP32
out:       [1, 1, 64, 512] BF16
```

`kv_cache` 是 sliding window ring buffer。`topk_idxs` 已经按官方
`get_window_topk_idxs` 语义转换到 cache 内位置。

### CSA: `compress_ratio = 4`

prefill:

```text
q:         [1, S_DYN, 64, 512] BF16
kv:        [1, K_CSA_PREFILL, 512] BF16
topk_idxs: [1, S_DYN, 640] INT32
attn_sink: [64] FP32
out:       [1, S_DYN, 64, 512] BF16
```

其中：

```text
K_CSA_PREFILL = S + floor(S / 4)
```

由于 PyPTO tensor shape 不适合直接表达 `S + floor(S / 4)` 这类 dynamic 派生维度，
CSA prefill 更适合由上层 wrapper 传入固定或按 runner 配置上限预分配的 KV pool，并用
`topk_idxs = -1` 屏蔽无效位置。

decode:

```text
q:         [1, 1, 64, 512] BF16
kv_pool:   [1, 128 + max_seq_len // 4, 512] BF16
topk_idxs: [1, 1, 640] INT32
attn_sink: [64] FP32
out:       [1, 1, 64, 512] BF16
```

`kv_pool` 前 `128` 个位置是 window cache，后面是 ratio=4 compressed KV cache。

### HCA: `compress_ratio = 128`

prefill:

```text
q:         [1, S_DYN, 64, 512] BF16
kv:        [1, K_HCA_PREFILL, 512] BF16
topk_idxs: [1, S_DYN, 128 + max_seq_len // 128] INT32
attn_sink: [64] FP32
out:       [1, S_DYN, 64, 512] BF16
```

其中：

```text
K_HCA_PREFILL = S + floor(S / 128)
```

decode:

```text
q:         [1, 1, 64, 512] BF16
kv_pool:   [1, 128 + max_seq_len // 128, 512] BF16
topk_idxs: [1, 1, 128 + max_seq_len // 128] INT32
attn_sink: [64] FP32
out:       [1, 1, 64, 512] BF16
```

`max_seq_len` 是 runner/compile 配置，不是模型结构常量。它同时影响 HCA compressed
KV cache size 和 HCA sparse attention 的 `TOPK_MAX`。

## 推荐文件结构

代码放在：

```text
models/sparse_attn.py
```

推荐提供的入口：

```text
sparse_attn_swa_fwd

sparse_attn_csa_fwd
sparse_attn_hca_fwd
```

其中 SWA prefill/decode 使用同一个 `sparse_attn_swa_fwd`，因为二者的 kernel ABI
可以统一为：

```text
q:         [1, S_DYN, 64, 512]
kv:        [1, K_DYN, 512]
topk_idxs: [1, S_DYN, 128]
out:       [1, S_DYN, 64, 512]
```

prefill 和 decode 只是在 host/wrapper 侧传入不同的 `kv` 长度与 topk 内容：

```text
prefill:
  kv = current prompt KV, K_DYN = S_DYN

decode:
  kv = sliding-window KV cache, K_DYN = 128
```

CSA/HCA 的入口在 compressor 和 indexer shape 稳定后再补齐。

单独验证时也不需要区分 prefill/decode 两个 JIT wrapper。可以提供一个统一测试入口：

```text
sparse_attn_swa_test
```

prefill 和 decode 通过不同 tensor specs 覆盖：

```text
prefill specs:
  kv shape = [1, S, 512]

decode specs:
  kv shape = [1, 128, 512]
```

golden 也统一读取 `kv`，不需要为 decode 单独引入 `kv_cache` 参数名。

host 侧 window topk 也使用统一 helper：

```text
build_window_topk_idxs(seq_len, start_pos)
```

其分支和官方 `get_window_topk_idxs(window_size, bsz, seqlen, start_pos)` 一致：

```text
start_pos == 0:
  生成 prefill topk，并 padding 到 [1, S, 128]

start_pos > 0:
  生成 decode topk，当前只支持 seq_len = 1
```

不要维护三份不同的业务语义。由于 PyPTO 对闭包常量和动态 shape 的支持有限，当前不使用
factory 生成通用 core helper；每个 ratio 直接提供 shape-specialized fwd，计算结构保持一致：

```text
for token:
  for head:
    gather selected kv by topk_idxs
    score = dot(q_head, kv_selected) * softmax_scale
    append attn_sink
    softmax over TOPK_MAX + 1 with invalid mask
    out_head = weighted sum selected kv
```

由于 PyPTO shape 通常需要静态常量，后续 CSA/HCA 应分别新增自己的 fwd 入口，例如
`sparse_attn_csa_fwd` 和 `sparse_attn_hca_fwd`，而不是把 `TOPK_MAX` 作为运行时动态参数。

## Invalid Topk 和 Softmax

官方实现先过滤无效 index：

```python
idxs = idxs[idxs >= 0]
```

PyPTO 中更适合保留固定长度 `TOPK_MAX`，对无效位置做 mask：

```text
valid = topk_idx >= 0
score_i = dot(q, kv[topk_idx]) * scale if valid else -inf
score_sink = attn_sink[head]
probs = softmax([score_0 ... score_TOPK_MAX-1, score_sink])
out = sum(probs_i * kv[topk_idx_i] for valid i)
```

注意：

- sink logit 始终有效。
- sink 概率不乘任何 value。
- 如果一个 token 没有任何有效 KV，softmax 仍会在 sink 上得到概率 1，输出应保持 0。
- 为避免无效 index gather 越界，无效 topk 在 gather 前应替换为安全 index，例如 0；
  对应 score 再用 mask 置为负无穷。

## 数值对齐

golden 应直接对齐官方 low-vram 逻辑，而不是复用 PyPTO kernel 的分块结构：

```text
selected = kv[batch, idxs].float()
scores = einsum(q.float(), selected) * scale
scores = cat(scores, attn_sink)
probs = softmax(scores)
out = einsum(probs_without_sink, selected).to(bf16)
```

PyPTO kernel 内建议：

- dot accumulation 使用 FP32。
- softmax 使用 FP32。
- weighted value accumulation 使用 FP32。
- 最终输出 cast BF16，使用和现有 kernel 一致的 round mode。

由于 sparse attention 包含 softmax，误差可能比 linear/RMSNorm 更容易被放大。验证阈值应以
最终业务输出 `out` 为准，必要时使用 `ratio_allclose` 允许极少量 bf16 尾差，但不应放宽到
掩盖系统性错误。

## 实现顺序

建议从 SWA 开始：

```text
1. sparse_attn_swa_fwd
2. SWA prefill/decode specs and golden validation
3. attention_swa.py 串联 attention_qkv + sparse_attn + inverse RoPE + output projection
4. compressor_ratio128.py 与 sparse_attn_hca_fwd
5. compressor_ratio4.py、indexer.py 与 sparse_attn_csa_fwd
```

其中 `sparse_attn_swa_fwd` 的 `TOPK_MAX` 固定为 `128`，可以在不引入 `max_seq_len`
的情况下同时验证 SWA prefill 和 SWA decode。

## 验证用例

SWA prefill 至少覆盖：

```text
S = 1
S = 13
S = 128
```

需要覆盖：

- `S < window_size`，大量 `topk_idxs = -1`。
- `S == window_size`，完整 window。
- 每个 token 只能看自己及之前 token，不能看未来 token。
- `attn_sink` 参与 softmax 但不产生 value。

SWA decode 至少覆盖：

```text
start_pos = 1
start_pos = 127
start_pos = 128
```

需要覆盖：

- window 未满时的 `-1` padding。
- window 满后的 ring buffer index 顺序。
- `start_pos % window_size` 写入位置和 topk index 对齐。

`start_pos = 0` 属于官方 `Attention.forward()` 的 prefill 分支，不作为 decode case。

CSA/HCA 后续验证时还应覆盖：

- compressed topk 与 window topk 拼接后的 offset。
- `ratio=4` 的 `index_topk=512` 上限。
- `ratio=128` 的 `max_seq_len // 128` 上限。
- decode boundary 上 compressor 写入新 compressed KV 的位置。
