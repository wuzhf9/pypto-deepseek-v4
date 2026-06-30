# DeepSeek V4 Flash RoPE Plan

本文档记录 `../deepseek_v4_flash/inference/model.py` 中 `apply_rotary_emb`
对应的 PyPTO 实现方案。目标是对齐官方 bf16 推理计算逻辑：RoPE 频率表在 host
侧生成，kernel 接收完整 head 维 tensor，在 kernel 内只旋转最后 `rope_head_dim`
个维度。

## 官方计算语义

官方实现的核心逻辑如下：

```python
def apply_rotary_emb(x, freqs_cis, inverse=False):
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y
```

计算类型按官方逻辑处理：

```text
x input:        BF16
x.float():      FP32
freqs_cis:      complex64，即 cos/sin 为 FP32
rotate compute: FP32
output:         cast 回 BF16
```

PyPTO 不能使用 `view_as_complex` / `view_as_real`，因此 kernel 使用 FP32
`cos/sin` 表手动实现同样的 interleaved even/odd pair 旋转。输入和输出 tensor
保持 BF16。

## Host 侧 RoPE 表

PyPTO kernel 不计算频率、YaRN 参数或三角函数。host 侧按官方
`precompute_freqs_cis` 的语义生成 complex 频率，再拆成实数表传入 kernel：

```text
cos: [S, rope_head_dim / 2] FP32
sin: [S, rope_head_dim / 2] FP32
```

DeepSeek V4 Flash 固定：

```text
rope_head_dim = 64
rope_half_dim = 32
```

因此实际 ABI 为：

```text
cos: [S_DYN, 32] FP32
sin: [S_DYN, 32] FP32
```

为了严格对齐官方路径，host 侧应先构造官方等价的 complex table，例如使用
`torch.polar(torch.ones_like(angles), angles)`，再取 `real/imag` 得到 `cos/sin`。
直接对 angle 调 `torch.cos/sin` 在数值上通常等价，但可能和官方 complex 路径存在
极小尾差；验证时应以官方 complex 路径为准。

RoPE profile 由 attention 路径决定：

```text
compress_ratio == 0:
  base = rope_theta
  original_seq_len = 0

compress_ratio > 0:
  base = compress_rope_theta
  original_seq_len = config.original_seq_len
```

运行时传入当前 token 范围对应的表：

```text
cos = cos_full[start_pos : start_pos + seqlen]
sin = sin_full[start_pos : start_pos + seqlen]
```

不需要把 half-width 表复制成 `[S, 64]`。官方 complex RoPE 每两个实数维度共享一个
频率，`[S, 32]` 已经包含完整语义。

## 官方调用点和 Shape

官方 `model.py` 中 RoPE 始终只作用于最后 `rope_head_dim=64` 个维度。PyPTO kernel
接口接收完整 head 维 tensor，内部保留前缀维度不变，只旋转最后 64 维。

按完整输入 shape 梳理，调用点如下：

| 位置 | 模块 | 输入 shape | 旋转区间 | inverse |
| --- | --- | --- | --- | --- |
| `Compressor.forward` | attention compressor | `[B, S_comp, 512]` | `[..., 448:512]` | False |
| `Compressor.forward` | indexer compressor | `[B, S_comp, 128]` | `[..., 64:128]` | False |
| `Indexer.forward` | indexer q | `[B, S, 64, 128]` | `[..., 64:128]` | False |
| `Attention.forward` | attention q | `[B, S, 64, 512]` | `[..., 448:512]` | False |
| `Attention.forward` | attention kv | `[B, S, 512]` | `[..., 448:512]` | False |
| `Attention.forward` | attention o | `[B, S, 64, 512]` | `[..., 448:512]` | True |

当前实现固定单卡、`B=1`，`S` 使用 dynamic shape。按 shape 去重后需要以下 kernel
入口：

```text
rope_3d_512_fwd:
  x:   [1, S_DYN, 512] BF16
  cos: [S_DYN, 32] FP32
  sin: [S_DYN, 32] FP32
  out: [1, S_DYN, 512] BF16

rope_3d_128_fwd:
  x:   [1, S_DYN, 128] BF16
  cos: [S_DYN, 32] FP32
  sin: [S_DYN, 32] FP32
  out: [1, S_DYN, 128] BF16

rope_4d_512_fwd:
  x:   [1, S_DYN, 64, 512] BF16
  cos: [S_DYN, 32] FP32
  sin: [S_DYN, 32] FP32
  out: [1, S_DYN, 64, 512] BF16

rope_4d_512_inv:
  x:   [1, S_DYN, 64, 512] BF16
  cos: [S_DYN, 32] FP32
  sin: [S_DYN, 32] FP32
  out: [1, S_DYN, 64, 512] BF16

rope_4d_128_fwd:
  x:   [1, S_DYN, 64, 128] BF16
  cos: [S_DYN, 32] FP32
  sin: [S_DYN, 32] FP32
  out: [1, S_DYN, 64, 128] BF16
```

官方没有 3D inverse 调用，也没有 4D 128 inverse 调用。`Attention.forward`
中 `model.py:496-504` 对应的 q/kv projection 后 RoPE 只需要：

```text
rope_4d_512_fwd  # q:  [1, S_DYN, 64, 512]
rope_3d_512_fwd  # kv: [1, S_DYN, 512]
```

## 手动旋转公式

设完整 head 维为 `head_dim`，RoPE 维度为 `rope_dim=64`：

```text
tail_offset = head_dim - rope_dim
tail = x[..., tail_offset : head_dim]
```

前缀维度直接复制：

```text
out[..., :tail_offset] = x[..., :tail_offset]
```

对 tail 内的最后一维：

```text
j:        0..63
freq_idx: j // 2
swap_idx: j ^ 1
sign[j]:  -1 for even j, +1 for odd j
```

forward RoPE：

```text
out_tail[j] = x_tail[j] * cos[freq_idx] + x_tail[swap_idx] * sign[j] * sin[freq_idx]
```

inverse RoPE 使用 complex conjugate，等价于 sin 项取反：

```text
out_tail[j] = x_tail[j] * cos[freq_idx] - x_tail[swap_idx] * sign[j] * sin[freq_idx]
```

展开成 even/odd pair，即：

```text
forward:
  y[2k]     = x[2k]     * cos_k - x[2k + 1] * sin_k
  y[2k + 1] = x[2k]     * sin_k + x[2k + 1] * cos_k

inverse:
  y[2k]     = x[2k]     * cos_k + x[2k + 1] * sin_k
  y[2k + 1] = -x[2k]    * sin_k + x[2k + 1] * cos_k
```

## Kernel 结构

RoPE 相关代码放在：

```text
models/rope.py
```

建议常量：

```text
ROPE_DIM = M.rope_head_dim       # 64
ROPE_HALF = ROPE_DIM // 2        # 32
N_HEADS = M.n_heads              # 64
ROPE_T_TILE = 1 or 16            # 以正确支持 dynamic tail write 为准
```

3D kernel 可以把输入视为 `[S_DYN, head_dim]`：

```text
for token block:
  copy prefix [0, tail_offset)
  rotate tail [tail_offset, head_dim)
  write only valid token rows
```

4D kernel 可以把输入视为 `[S_DYN, n_heads, head_dim]`，或展平成
`[S_DYN * n_heads, head_dim]`：

```text
token_idx = row // n_heads
head_idx  = row % n_heads
cos/sin   = table[token_idx]
```

cos/sin 只依赖 token，不依赖 head。实现时可以按 token block 和 head block 循环，
也可以按 flattened rows 循环；当前目标只要求计算正确，不需要做融合或性能优化。

dynamic shape 写回需要遵循已有 linear/rmsnorm 的经验：外部输出只写有效 token
范围，不能把固定 tile 的无效行整块写回到 dynamic output。

## Golden 验证

golden 应直接表达官方语义，同时保持 kernel ABI 为 FP32 half-width cos/sin：

```python
def apply_full_head_rope_golden(x, cos, sin, inverse=False):
    out = x.clone()
    tail = out[..., -64:]
    official = apply_rotary_emb_with_complex_equivalent(tail, cos, sin, inverse)
    out[..., -64:] = official
    return out
```

host 侧单测需要覆盖：

```text
1. RoPE table 生成结果与官方 complex freqs_cis 的 real/imag 一致。
2. 手写 even/odd 公式与官方 view_as_complex/view_as_real 路径一致。
3. forward 和 inverse 两种路径都验证。
4. 3D/4D、head_dim=512/128 的完整 head 输入都验证前缀不变、尾部旋转正确。
```

PyPTO kernel 验证建议覆盖非 tile 对齐和常见 prefill/decode 场景：

```text
S = 1
S = 13
S = 128
```

## 上层集成

`attention_qkv` 中可以直接复用完整 head RoPE kernel：

```text
q projection -> per-head RMSNorm -> rope_4d_512_fwd
kv projection -> RMSNorm          -> rope_3d_512_fwd
```

后续 attention 输出路径使用：

```text
o -> rope_4d_512_inv -> output projection
```

indexer 和 compressor 使用：

```text
indexer q                  -> rope_4d_128_fwd
attention compressor kv    -> rope_3d_512_fwd
indexer compressor kv      -> rope_3d_128_fwd
```

这样 RoPE kernel 的接口和上层 tensor shape 保持一致，上层模块不需要把最后
64 维切成独立 tensor，也不需要依赖可写 view 回写到原 tensor。
