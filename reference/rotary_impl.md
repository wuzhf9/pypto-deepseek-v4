# DeepSeek V4 Flash Rotary Plan

本文档记录 `../deepseek_v4_flash/inference/model.py` 中 `apply_rotary_emb`
对应的 PyPTO 实现方案。当前目标是对齐官方 bf16 推理计算逻辑，不在 kernel 内生成
RoPE 频率表，也不使用 complex tensor 视图。

## 官方计算语义

官方实现：

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
x input:       BF16
x.float():    FP32
freqs_cis:    complex64，即 cos/sin 为 FP32
rotate compute: FP32
output:       cast 回 BF16
```

因此 PyPTO kernel 的 cos/sin 入参应使用 FP32，输入输出 tensor 使用 BF16。

## Host 侧 RoPE 表

PyPTO kernel 不计算频率和三角函数。host 侧按官方 `precompute_freqs_cis` 的 YaRN
逻辑生成角度，再拆成实数表传入 kernel：

```text
cos: [S, rope_head_dim / 2] FP32
sin: [S, rope_head_dim / 2] FP32
```

DeepSeek V4 Flash 中：

```text
rope_head_dim = 64
rope_half_dim = 32
```

所以实际传入：

```text
cos: [S_DYN, 32] FP32
sin: [S_DYN, 32] FP32
```

`../pypto-serving/pypto-lib/models/deepseek/v4/rope_tables.py` 使用了
`[S, 64]` 的 full-width duplicated table：

```text
freqs_cos = cat([cos_half, cos_half], dim=-1)
freqs_sin = cat([sin_half, sin_half], dim=-1)
```

这不是官方必要语义，而是 pypto-serving kernel ABI 的工程选择。官方 complex RoPE
每两个实数维度共享一个频率，因此只需要 `[S, 32]`。本仓实现应直接使用 half-width
FP32 cos/sin 表，避免冗余复制，也更贴近官方计算逻辑。

## 官方调用点和 Shape

`model.py` 中共有 5 个 `apply_rotary_emb` 调用点，按输入 shape 去重后只需要两类
kernel shape，外加一个 4D inverse 变体。

| 位置 | 模块 | 原 tensor shape | RoPE 输入 shape | inverse |
| --- | --- | --- | --- | --- |
| line 367 | `Compressor.forward` | `kv: [B, S_comp, head_dim]` | `[B, S_comp, 64]` | False |
| line 413 | `Indexer.forward` | `q: [B, S, 64, 128]` | `[B, S, 64, 64]` | False |
| line 499 | `Attention.forward` | `q: [B, S, 64, 512]` | `[B, S, 64, 64]` | False |
| line 504 | `Attention.forward` | `kv: [B, S, 512]` | `[B, S, 64]` | False |
| line 534 | `Attention.forward` | `o: [B, S, 64, 512]` | `[B, S, 64, 64]` | True |

当前固定单卡、`B=1`，因此需要实现：

```text
rotary_3d_fwd:
  x:   [1, S_DYN, 64] BF16
  cos: [S_DYN, 32] FP32
  sin: [S_DYN, 32] FP32
  out: [1, S_DYN, 64] BF16

rotary_4d_fwd:
  x:   [1, S_DYN, 64, 64] BF16
  cos: [S_DYN, 32] FP32
  sin: [S_DYN, 32] FP32
  out: [1, S_DYN, 64, 64] BF16

rotary_4d_inv:
  x:   [1, S_DYN, 64, 64] BF16
  cos: [S_DYN, 32] FP32
  sin: [S_DYN, 32] FP32
  out: [1, S_DYN, 64, 64] BF16
```

不需要实现 `rotary_3d_inv`，因为官方当前没有 3D inverse 调用。

## 手动旋转公式

PyPTO 不能使用 `view_as_complex` / `view_as_real`，需要手动处理 interleaved
even/odd pair。

对最后一维：

```text
x_pair[k] = (x[2k], x[2k + 1])
cos_k = cos[position, k]
sin_k = sin[position, k]
```

forward RoPE：

```text
y[2k]     = x[2k]     * cos_k - x[2k + 1] * sin_k
y[2k + 1] = x[2k]     * sin_k + x[2k + 1] * cos_k
```

inverse RoPE 使用 complex conjugate，等价于 sin 项取反：

```text
y[2k]     = x[2k]     * cos_k + x[2k + 1] * sin_k
y[2k + 1] = -x[2k]    * sin_k + x[2k + 1] * cos_k
```

也可以写成 pypto-serving 使用的 swap/sign 形式。令：

```text
j:        0..63
freq_idx: j // 2
swap_idx: j ^ 1
sign[j]:  -1 for even j, +1 for odd j
```

forward：

```text
out[j] = x[j] * cos[freq_idx] + x[swap_idx] * sign[j] * sin[freq_idx]
```

inverse：

```text
out[j] = x[j] * cos[freq_idx] - x[swap_idx] * sign[j] * sin[freq_idx]
```

实现时可以直接在各个 kernel 中构造 `freq_idx / swap_idx / sign`，不需要额外抽象
`_apply_rotary_*` helper。

## Kernel 设计

建议新增文件：

```text
models/rope.py
```

公开 PyPTO 入口：

```text
rotary_3d_fwd
rotary_4d_fwd
rotary_4d_inv
```

每个入口直接实现对应 shape 和公式，不拆 `_apply_rotary_*(..., inverse_sign)` helper。
这样代码重复很少，语义清晰，也避免为当前简单逻辑引入多余抽象。

推荐常量：

```text
ROPE_DIM = M.rope_head_dim       # 64
ROPE_HALF = ROPE_DIM // 2        # 32
N_HEADS = M.n_heads              # 64
ROTARY_T_TILE = 16
```

3D kernel 可按 token 维切分：

```text
for token_block:
  valid_tok = min(ROTARY_T_TILE, S - t0)
  x_tile:   [ROTARY_T_TILE, 64]
  cos_tile: [ROTARY_T_TILE, 32]
  sin_tile: [ROTARY_T_TILE, 32]
  rotate in FP32
  write only valid_tok rows
```

4D kernel 可把 `[S, H]` 展平成 rows：

```text
x_flat:   [S_DYN * 64, 64]
out_flat: [S_DYN * 64, 64]
```

每个 row 对应：

```text
token_idx = row // 64
head_idx  = row % 64
```

cos/sin 只依赖 token，不依赖 head。实现时可以按 token block 和 head block 循环，或
按 flattened rows 循环后从 `token_idx` 读取 cos/sin。正确性优先，不需要为了性能融合
head 维。

输出写回仍需遵循 dynamic shape 尾块经验：如果 tile 行数固定为 16，外部 output 只写
`valid_tok` 或有效 row，不能把固定 tile 整块写回到动态输出。

## Golden 验证

golden 应直接复刻官方逻辑，但用 FP32 cos/sin half table，避免 complex table 作为
kernel ABI：

```python
def apply_rope_golden(x, cos, sin, inverse=False):
    x_fp32 = x.float()
    pair = x_fp32.unflatten(-1, (-1, 2))
    x0 = pair[..., 0]
    x1 = pair[..., 1]

    cos_v = cos
    sin_v = sin
    while cos_v.ndim < x0.ndim:
        cos_v = cos_v.unsqueeze(-2)
        sin_v = sin_v.unsqueeze(-2)

    if inverse:
        y0 = x0 * cos_v + x1 * sin_v
        y1 = -x0 * sin_v + x1 * cos_v
    else:
        y0 = x0 * cos_v - x1 * sin_v
        y1 = x0 * sin_v + x1 * cos_v

    return torch.stack([y0, y1], dim=-1).flatten(-2).to(x.dtype)
```

验证 case：

```text
rotary_3d_fwd
rotary_4d_fwd
rotary_4d_inv
```

每个 case 至少覆盖：

```bash
python models/rope.py -p a2a3 -d {} -s 1
python models/rope.py -p a2a3 -d {} -s 13
python models/rope.py -p a2a3 -d {} -s 128
```

其中 `S=1` 覆盖 decode，`S=13` 覆盖非 tile 对齐尾块，`S=128` 覆盖多 token block。

## 与后续 Attention 的对应关系

后续集成时按官方调用点映射：

```text
Attention q[..., -rd:]      -> rotary_4d_fwd
Attention kv[..., -rd:]     -> rotary_3d_fwd
Attention o[..., -rd:]      -> rotary_4d_inv
Indexer q[..., -rd:]        -> rotary_4d_fwd
Compressor kv[..., -rd:]    -> rotary_3d_fwd
```

host 侧需要根据当前 attention 层的 RoPE profile 生成对应 cos/sin：

```text
compress_ratio == 0:
  base = rope_theta
  original_seq_len = 0

compress_ratio > 0:
  base = compress_rope_theta
  original_seq_len = original_seq_len
```

`start_pos` 对应的 position rows 在 host 侧 materialize 成 `[S_DYN, 32]` 后传给 kernel。
