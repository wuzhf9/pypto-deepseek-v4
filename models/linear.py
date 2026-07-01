"""DeepSeek V4 Flash bf16 Linear PyPTO kernels."""

import pypto.language as pl

from models.common import assert_divisible
from models.config import FLASH_CONFIG as M


B = 1
S_DYN = pl.dynamic("S_DYN")

HIDDEN = M.dim
Q_LORA_RANK = M.q_lora_rank
HEAD_DIM = M.head_dim
INDEX_COMPRESS_OUT = 2 * M.index_head_dim
ATTN_Q_OUT = M.n_heads * M.head_dim
ATTN_OUT_IN = M.o_groups * M.o_lora_rank

T_TILE = 16
K_TILE = 128
O_TILE = 32
OUT_GROUP = 2
ATTN_Q_OUT_TILE = 64
ATTN_Q_OUT_GROUP = 2
DEFAULT_SEQ_LEN = 8

assert_divisible(HIDDEN, K_TILE, "hidden linear input size")
assert_divisible(Q_LORA_RANK, K_TILE, "q lora linear input size")
assert_divisible(ATTN_OUT_IN, K_TILE, "attention output linear input size")
assert_divisible(HEAD_DIM, O_TILE, "512 linear output size")
assert_divisible(INDEX_COMPRESS_OUT, O_TILE, "256 linear output size")
assert_divisible(Q_LORA_RANK, O_TILE, "1024 linear output size")
assert_divisible(HIDDEN, O_TILE, "4096 linear output size")
assert_divisible(ATTN_Q_OUT, ATTN_Q_OUT_TILE, "32768 linear output size")
assert_divisible(HEAD_DIM // O_TILE, OUT_GROUP, "512 output blocks")
assert_divisible(INDEX_COMPRESS_OUT // O_TILE, OUT_GROUP, "256 output blocks")
assert_divisible(Q_LORA_RANK // O_TILE, OUT_GROUP, "1024 output blocks")
assert_divisible(HIDDEN // O_TILE, OUT_GROUP, "4096 output blocks")
assert_divisible(ATTN_Q_OUT // ATTN_Q_OUT_TILE, ATTN_Q_OUT_GROUP, "32768 output tile blocks")

HIDDEN_K_BLOCKS = HIDDEN // K_TILE
Q_LORA_K_BLOCKS = Q_LORA_RANK // K_TILE
ATTN_OUT_K_BLOCKS = ATTN_OUT_IN // K_TILE
HEAD_DIM_O_BLOCKS = HEAD_DIM // O_TILE
INDEX_COMPRESS_O_BLOCKS = INDEX_COMPRESS_OUT // O_TILE
Q_LORA_O_BLOCKS = Q_LORA_RANK // O_TILE
HIDDEN_O_BLOCKS = HIDDEN // O_TILE
ATTN_Q_OUT_BLOCKS = ATTN_Q_OUT // ATTN_Q_OUT_TILE


@pl.jit.inline
def linear_4096_to_512(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    weight_t: pl.Tensor[[HIDDEN, HEAD_DIM], pl.BF16],
    out: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
):
    """Compute ``x @ weight_t`` for ``weight_t`` shape ``[4096, 512]``."""
    x.bind_dynamic(1, S_DYN)
    out.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    x_flat = pl.reshape(x, [tokens, HIDDEN])
    out_flat = pl.reshape(out, [tokens, HEAD_DIM])
    token_blocks = (tokens + T_TILE - 1) // T_TILE

    for tb in pl.range(token_blocks):
        t0 = tb * T_TILE
        valid_tok = pl.min(T_TILE, tokens - t0)
        out_tile_fp32 = pl.create_tensor([T_TILE, HEAD_DIM], dtype=pl.FP32)

        for og_idx in pl.spmd(HEAD_DIM_O_BLOCKS // OUT_GROUP, name_hint="linear_4096_to_512"):
            og = og_idx * OUT_GROUP
            for o_inner in pl.pipeline(OUT_GROUP, stage=2):
                o0 = (og + o_inner) * O_TILE
                x0 = pl.slice(x_flat, [T_TILE, K_TILE], [t0, 0], valid_shape=[valid_tok, K_TILE])
                w0 = pl.slice(weight_t, [K_TILE, O_TILE], [0, o0])
                acc = pl.matmul(x0, w0, out_dtype=pl.FP32)
                for kb in pl.pipeline(1, HIDDEN_K_BLOCKS, stage=2):
                    k0 = kb * K_TILE
                    xk = pl.slice(x_flat, [T_TILE, K_TILE], [t0, k0], valid_shape=[valid_tok, K_TILE])
                    wk = pl.slice(weight_t, [K_TILE, O_TILE], [k0, o0])
                    acc = pl.matmul_acc(acc, xk, wk)
                out_tile_fp32[:, o0 : o0 + O_TILE] = acc

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="linear_4096_to_512_write"):
            for ob in pl.range(HEAD_DIM_O_BLOCKS):
                o0 = ob * O_TILE
                out_tile = pl.cast(out_tile_fp32[:, o0 : o0 + O_TILE], target_type=pl.BF16, mode="rint")
                for row in pl.range(valid_tok):
                    out_row = pl.slice(out_tile, [1, O_TILE], [row, 0])
                    out_flat = pl.assemble(out_flat, out_row, [t0 + row, o0])

    return pl.reshape(out_flat, [B, tokens, HEAD_DIM])


@pl.jit.inline
def linear_4096_to_512_fp32(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    weight_t: pl.Tensor[[HIDDEN, HEAD_DIM], pl.BF16],
    out: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.FP32],
):
    """Compute ``x @ weight_t`` for ``weight_t`` shape ``[4096, 512]`` into FP32."""
    x.bind_dynamic(1, S_DYN)
    out.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    x_flat = pl.reshape(x, [tokens, HIDDEN])
    out_flat = pl.reshape(out, [tokens, HEAD_DIM])
    token_blocks = (tokens + T_TILE - 1) // T_TILE

    for tb in pl.range(token_blocks):
        t0 = tb * T_TILE
        valid_tok = pl.min(T_TILE, tokens - t0)
        out_tile_fp32 = pl.create_tensor([T_TILE, HEAD_DIM], dtype=pl.FP32)

        for og_idx in pl.spmd(HEAD_DIM_O_BLOCKS // OUT_GROUP, name_hint="linear_4096_to_512_fp32"):
            og = og_idx * OUT_GROUP
            for o_inner in pl.pipeline(OUT_GROUP, stage=2):
                o0 = (og + o_inner) * O_TILE
                x0 = pl.slice(x_flat, [T_TILE, K_TILE], [t0, 0], valid_shape=[valid_tok, K_TILE])
                w0 = pl.slice(weight_t, [K_TILE, O_TILE], [0, o0])
                acc = pl.matmul(x0, w0, out_dtype=pl.FP32)
                for kb in pl.pipeline(1, HIDDEN_K_BLOCKS, stage=2):
                    k0 = kb * K_TILE
                    xk = pl.slice(x_flat, [T_TILE, K_TILE], [t0, k0], valid_shape=[valid_tok, K_TILE])
                    wk = pl.slice(weight_t, [K_TILE, O_TILE], [k0, o0])
                    acc = pl.matmul_acc(acc, xk, wk)
                out_tile_fp32[:, o0 : o0 + O_TILE] = acc

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="linear_4096_to_512_fp32_write"):
            for ob in pl.range(HEAD_DIM_O_BLOCKS):
                o0 = ob * O_TILE
                out_tile = out_tile_fp32[:, o0 : o0 + O_TILE]
                for row in pl.range(valid_tok):
                    out_row = pl.slice(out_tile, [1, O_TILE], [row, 0])
                    out_flat = pl.assemble(out_flat, out_row, [t0 + row, o0])

    return pl.reshape(out_flat, [B, tokens, HEAD_DIM])


@pl.jit.inline
def linear_4096_to_256_fp32(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    weight_t: pl.Tensor[[HIDDEN, INDEX_COMPRESS_OUT], pl.BF16],
    out: pl.Tensor[[B, S_DYN, INDEX_COMPRESS_OUT], pl.FP32],
):
    """Compute ``x @ weight_t`` for ``weight_t`` shape ``[4096, 256]`` into FP32."""
    x.bind_dynamic(1, S_DYN)
    out.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    x_flat = pl.reshape(x, [tokens, HIDDEN])
    out_flat = pl.reshape(out, [tokens, INDEX_COMPRESS_OUT])
    token_blocks = (tokens + T_TILE - 1) // T_TILE

    for tb in pl.range(token_blocks):
        t0 = tb * T_TILE
        valid_tok = pl.min(T_TILE, tokens - t0)
        out_tile_fp32 = pl.create_tensor([T_TILE, INDEX_COMPRESS_OUT], dtype=pl.FP32)

        for og_idx in pl.spmd(INDEX_COMPRESS_O_BLOCKS // OUT_GROUP, name_hint="linear_4096_to_256_fp32"):
            og = og_idx * OUT_GROUP
            for o_inner in pl.pipeline(OUT_GROUP, stage=2):
                o0 = (og + o_inner) * O_TILE
                x0 = pl.slice(x_flat, [T_TILE, K_TILE], [t0, 0], valid_shape=[valid_tok, K_TILE])
                w0 = pl.slice(weight_t, [K_TILE, O_TILE], [0, o0])
                acc = pl.matmul(x0, w0, out_dtype=pl.FP32)
                for kb in pl.pipeline(1, HIDDEN_K_BLOCKS, stage=2):
                    k0 = kb * K_TILE
                    xk = pl.slice(x_flat, [T_TILE, K_TILE], [t0, k0], valid_shape=[valid_tok, K_TILE])
                    wk = pl.slice(weight_t, [K_TILE, O_TILE], [k0, o0])
                    acc = pl.matmul_acc(acc, xk, wk)
                out_tile_fp32[:, o0 : o0 + O_TILE] = acc

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="linear_4096_to_256_fp32_write"):
            for ob in pl.range(INDEX_COMPRESS_O_BLOCKS):
                o0 = ob * O_TILE
                out_tile = out_tile_fp32[:, o0 : o0 + O_TILE]
                for row in pl.range(valid_tok):
                    out_row = pl.slice(out_tile, [1, O_TILE], [row, 0])
                    out_flat = pl.assemble(out_flat, out_row, [t0 + row, o0])

    return pl.reshape(out_flat, [B, tokens, INDEX_COMPRESS_OUT])


@pl.jit.inline
def linear_4096_to_1024(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    weight_t: pl.Tensor[[HIDDEN, Q_LORA_RANK], pl.BF16],
    out: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
):
    """Compute ``x @ weight_t`` for ``weight_t`` shape ``[4096, 1024]``."""
    x.bind_dynamic(1, S_DYN)
    out.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    x_flat = pl.reshape(x, [tokens, HIDDEN])
    out_flat = pl.reshape(out, [tokens, Q_LORA_RANK])
    token_blocks = (tokens + T_TILE - 1) // T_TILE

    for tb in pl.range(token_blocks):
        t0 = tb * T_TILE
        valid_tok = pl.min(T_TILE, tokens - t0)
        out_tile_fp32 = pl.create_tensor([T_TILE, Q_LORA_RANK], dtype=pl.FP32)

        for og_idx in pl.spmd(Q_LORA_O_BLOCKS // OUT_GROUP, name_hint="linear_4096_to_1024"):
            og = og_idx * OUT_GROUP
            for o_inner in pl.pipeline(OUT_GROUP, stage=2):
                o0 = (og + o_inner) * O_TILE
                x0 = pl.slice(x_flat, [T_TILE, K_TILE], [t0, 0], valid_shape=[valid_tok, K_TILE])
                w0 = pl.slice(weight_t, [K_TILE, O_TILE], [0, o0])
                acc = pl.matmul(x0, w0, out_dtype=pl.FP32)
                for kb in pl.pipeline(1, HIDDEN_K_BLOCKS, stage=2):
                    k0 = kb * K_TILE
                    xk = pl.slice(x_flat, [T_TILE, K_TILE], [t0, k0], valid_shape=[valid_tok, K_TILE])
                    wk = pl.slice(weight_t, [K_TILE, O_TILE], [k0, o0])
                    acc = pl.matmul_acc(acc, xk, wk)
                out_tile_fp32[:, o0 : o0 + O_TILE] = acc

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="linear_4096_to_1024_write"):
            for ob in pl.range(Q_LORA_O_BLOCKS):
                o0 = ob * O_TILE
                out_tile = pl.cast(out_tile_fp32[:, o0 : o0 + O_TILE], target_type=pl.BF16, mode="rint")
                for row in pl.range(valid_tok):
                    out_row = pl.slice(out_tile, [1, O_TILE], [row, 0])
                    out_flat = pl.assemble(out_flat, out_row, [t0 + row, o0])

    return pl.reshape(out_flat, [B, tokens, Q_LORA_RANK])


@pl.jit.inline
def linear_4096_to_1024_fp32(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    weight_t: pl.Tensor[[HIDDEN, Q_LORA_RANK], pl.BF16],
    out: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.FP32],
):
    """Compute ``x @ weight_t`` for ``weight_t`` shape ``[4096, 1024]`` into FP32."""
    x.bind_dynamic(1, S_DYN)
    out.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    x_flat = pl.reshape(x, [tokens, HIDDEN])
    out_flat = pl.reshape(out, [tokens, Q_LORA_RANK])
    token_blocks = (tokens + T_TILE - 1) // T_TILE

    for tb in pl.range(token_blocks):
        t0 = tb * T_TILE
        valid_tok = pl.min(T_TILE, tokens - t0)
        out_tile_fp32 = pl.create_tensor([T_TILE, Q_LORA_RANK], dtype=pl.FP32)

        for og_idx in pl.spmd(Q_LORA_O_BLOCKS // OUT_GROUP, name_hint="linear_4096_to_1024_fp32"):
            og = og_idx * OUT_GROUP
            for o_inner in pl.pipeline(OUT_GROUP, stage=2):
                o0 = (og + o_inner) * O_TILE
                x0 = pl.slice(x_flat, [T_TILE, K_TILE], [t0, 0], valid_shape=[valid_tok, K_TILE])
                w0 = pl.slice(weight_t, [K_TILE, O_TILE], [0, o0])
                acc = pl.matmul(x0, w0, out_dtype=pl.FP32)
                for kb in pl.pipeline(1, HIDDEN_K_BLOCKS, stage=2):
                    k0 = kb * K_TILE
                    xk = pl.slice(x_flat, [T_TILE, K_TILE], [t0, k0], valid_shape=[valid_tok, K_TILE])
                    wk = pl.slice(weight_t, [K_TILE, O_TILE], [k0, o0])
                    acc = pl.matmul_acc(acc, xk, wk)
                out_tile_fp32[:, o0 : o0 + O_TILE] = acc

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="linear_4096_to_1024_fp32_write"):
            for ob in pl.range(Q_LORA_O_BLOCKS):
                o0 = ob * O_TILE
                out_tile = out_tile_fp32[:, o0 : o0 + O_TILE]
                for row in pl.range(valid_tok):
                    out_row = pl.slice(out_tile, [1, O_TILE], [row, 0])
                    out_flat = pl.assemble(out_flat, out_row, [t0 + row, o0])

    return pl.reshape(out_flat, [B, tokens, Q_LORA_RANK])


@pl.jit.inline
def linear_1024_to_32768(
    x: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    weight_t: pl.Tensor[[Q_LORA_RANK, ATTN_Q_OUT], pl.BF16],
    out: pl.Tensor[[B, S_DYN, ATTN_Q_OUT], pl.BF16],
):
    """Compute ``x @ weight_t`` for ``weight_t`` shape ``[1024, 32768]``."""
    x.bind_dynamic(1, S_DYN)
    out.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    x_flat = pl.reshape(x, [tokens, Q_LORA_RANK])
    out_flat = pl.reshape(out, [tokens, ATTN_Q_OUT])
    token_blocks = (tokens + T_TILE - 1) // T_TILE

    for tb in pl.range(token_blocks):
        t0 = tb * T_TILE
        valid_tok = pl.min(T_TILE, tokens - t0)

        for og_idx in pl.spmd(ATTN_Q_OUT_BLOCKS // ATTN_Q_OUT_GROUP, name_hint="linear_1024_to_32768"):
            og = og_idx * ATTN_Q_OUT_GROUP
            for o_inner in pl.pipeline(ATTN_Q_OUT_GROUP, stage=2):
                o0 = (og + o_inner) * ATTN_Q_OUT_TILE
                x0 = pl.slice(x_flat, [T_TILE, K_TILE], [t0, 0], valid_shape=[valid_tok, K_TILE])
                w0 = pl.slice(weight_t, [K_TILE, ATTN_Q_OUT_TILE], [0, o0])
                acc = pl.matmul(x0, w0, out_dtype=pl.FP32)
                for kb in pl.pipeline(1, Q_LORA_K_BLOCKS, stage=2):
                    k0 = kb * K_TILE
                    xk = pl.slice(x_flat, [T_TILE, K_TILE], [t0, k0], valid_shape=[valid_tok, K_TILE])
                    wk = pl.slice(weight_t, [K_TILE, ATTN_Q_OUT_TILE], [k0, o0])
                    acc = pl.matmul_acc(acc, xk, wk)
                acc_bf16 = pl.cast(acc, target_type=pl.BF16, mode="rint")
                for row in pl.range(valid_tok):
                    out_row = pl.slice(acc_bf16, [1, ATTN_Q_OUT_TILE], [row, 0])
                    out_flat = pl.assemble(out_flat, out_row, [t0 + row, o0])

    return pl.reshape(out_flat, [B, tokens, ATTN_Q_OUT])


@pl.jit.inline
def linear_8192_to_4096(
    x: pl.Tensor[[B, S_DYN, ATTN_OUT_IN], pl.BF16],
    weight_t: pl.Tensor[[ATTN_OUT_IN, HIDDEN], pl.BF16],
    out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
):
    """Compute ``x @ weight_t`` for ``weight_t`` shape ``[8192, 4096]``."""
    x.bind_dynamic(1, S_DYN)
    out.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    x_flat = pl.reshape(x, [tokens, ATTN_OUT_IN])
    out_flat = pl.reshape(out, [tokens, HIDDEN])
    token_blocks = (tokens + T_TILE - 1) // T_TILE

    for tb in pl.range(token_blocks):
        t0 = tb * T_TILE
        valid_tok = pl.min(T_TILE, tokens - t0)

        for og_idx in pl.spmd(HIDDEN_O_BLOCKS // OUT_GROUP, name_hint="linear_8192_to_4096"):
            og = og_idx * OUT_GROUP
            for o_inner in pl.pipeline(OUT_GROUP, stage=2):
                o0 = (og + o_inner) * O_TILE
                x0 = pl.slice(x_flat, [T_TILE, K_TILE], [t0, 0], valid_shape=[valid_tok, K_TILE])
                w0 = pl.slice(weight_t, [K_TILE, O_TILE], [0, o0])
                acc = pl.matmul(x0, w0, out_dtype=pl.FP32)
                for kb in pl.pipeline(1, ATTN_OUT_K_BLOCKS, stage=2):
                    k0 = kb * K_TILE
                    xk = pl.slice(x_flat, [T_TILE, K_TILE], [t0, k0], valid_shape=[valid_tok, K_TILE])
                    wk = pl.slice(weight_t, [K_TILE, O_TILE], [k0, o0])
                    acc = pl.matmul_acc(acc, xk, wk)
                acc_bf16 = pl.cast(acc, target_type=pl.BF16, mode="rint")
                for row in pl.range(valid_tok):
                    out_row = pl.slice(acc_bf16, [1, O_TILE], [row, 0])
                    out_flat = pl.assemble(out_flat, out_row, [t0 + row, o0])

    return pl.reshape(out_flat, [B, tokens, HIDDEN])


@pl.jit
def linear_4096_to_512_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    weight_t: pl.Tensor[[HIDDEN, HEAD_DIM], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16]],
):
    out = linear_4096_to_512(x, weight_t, out)
    return out


@pl.jit
def linear_4096_to_512_fp32_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    weight_t: pl.Tensor[[HIDDEN, HEAD_DIM], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, HEAD_DIM], pl.FP32]],
):
    out = linear_4096_to_512_fp32(x, weight_t, out)
    return out


@pl.jit
def linear_4096_to_256_fp32_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    weight_t: pl.Tensor[[HIDDEN, INDEX_COMPRESS_OUT], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, INDEX_COMPRESS_OUT], pl.FP32]],
):
    out = linear_4096_to_256_fp32(x, weight_t, out)
    return out


@pl.jit
def linear_4096_to_1024_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    weight_t: pl.Tensor[[HIDDEN, Q_LORA_RANK], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16]],
):
    out = linear_4096_to_1024(x, weight_t, out)
    return out


@pl.jit
def linear_4096_to_1024_fp32_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    weight_t: pl.Tensor[[HIDDEN, Q_LORA_RANK], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.FP32]],
):
    out = linear_4096_to_1024_fp32(x, weight_t, out)
    return out


@pl.jit
def linear_1024_to_32768_test(
    x: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    weight_t: pl.Tensor[[Q_LORA_RANK, ATTN_Q_OUT], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, ATTN_Q_OUT], pl.BF16]],
):
    out = linear_1024_to_32768(x, weight_t, out)
    return out


@pl.jit
def linear_8192_to_4096_test(
    x: pl.Tensor[[B, S_DYN, ATTN_OUT_IN], pl.BF16],
    weight_t: pl.Tensor[[ATTN_OUT_IN, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16]],
):
    out = linear_8192_to_4096(x, weight_t, out)
    return out


def golden_linear(tensors):
    import torch

    x = tensors["x"].float()
    weight_t = tensors["weight_t"].float()
    tensors["out"][:] = torch.matmul(x, weight_t).to(tensors["out"].dtype)


def _build_tensor_specs(in_dim: int, out_dim: int, seq_len: int, out_dtype: str = "bf16"):
    import torch

    from models.golden import TensorSpec

    def init_x():
        return torch.randn(B, seq_len, in_dim) * 0.1

    def init_weight_t():
        return torch.randn(in_dim, out_dim) * 0.02

    output_dtype = torch.float32 if out_dtype == "fp32" else torch.bfloat16

    return [
        TensorSpec("x", [B, seq_len, in_dim], torch.bfloat16, init_value=init_x),
        TensorSpec("weight_t", [in_dim, out_dim], torch.bfloat16, init_value=init_weight_t),
        TensorSpec("out", [B, seq_len, out_dim], output_dtype, is_output=True),
    ]


def build_4096_to_512_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _build_tensor_specs(HIDDEN, HEAD_DIM, seq_len)


def build_4096_to_512_fp32_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _build_tensor_specs(HIDDEN, HEAD_DIM, seq_len, out_dtype="fp32")


def build_4096_to_256_fp32_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _build_tensor_specs(HIDDEN, INDEX_COMPRESS_OUT, seq_len, out_dtype="fp32")


def build_4096_to_1024_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _build_tensor_specs(HIDDEN, Q_LORA_RANK, seq_len)


def build_4096_to_1024_fp32_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _build_tensor_specs(HIDDEN, Q_LORA_RANK, seq_len, out_dtype="fp32")


def build_1024_to_32768_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _build_tensor_specs(Q_LORA_RANK, ATTN_Q_OUT, seq_len)


def build_8192_to_4096_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _build_tensor_specs(ATTN_OUT_IN, HIDDEN, seq_len)


def main() -> int:
    import argparse

    from models.golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 Flash bf16 Linear validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3sim", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-s", "--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    cases = [
        ("4096-to-512", linear_4096_to_512_test, build_4096_to_512_specs),
        ("4096-to-512-fp32", linear_4096_to_512_fp32_test, build_4096_to_512_fp32_specs),
        ("4096-to-256-fp32", linear_4096_to_256_fp32_test, build_4096_to_256_fp32_specs),
        ("4096-to-1024", linear_4096_to_1024_test, build_4096_to_1024_specs),
        ("4096-to-1024-fp32", linear_4096_to_1024_fp32_test, build_4096_to_1024_fp32_specs),
        ("1024-to-32768", linear_1024_to_32768_test, build_1024_to_32768_specs),
        ("8192-to-4096", linear_8192_to_4096_test, build_8192_to_4096_specs),
    ]
    runtime_cfg = {
        "platform": args.platform,
        "device_id": args.device,
        "enable_l2_swimlane": args.enable_l2_swimlane,
    }
    compare_fn = {
        "out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
    }

    failed = False
    for name, fn, build_specs in cases:
        print(f"[CASE] {name}", flush=True)
        result = run_jit(
            fn=fn,
            specs=build_specs(args.seq_len),
            golden_fn=golden_linear,
            runtime_cfg=runtime_cfg,
            compile_only=args.compile_only,
            compare_fn=compare_fn,
        )
        if not result.passed:
            failed = True
            if result.error:
                print(result.error)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "B",
    "S_DYN",
    "HIDDEN",
    "Q_LORA_RANK",
    "HEAD_DIM",
    "INDEX_COMPRESS_OUT",
    "ATTN_Q_OUT",
    "ATTN_OUT_IN",
    "T_TILE",
    "K_TILE",
    "O_TILE",
    "OUT_GROUP",
    "ATTN_Q_OUT_TILE",
    "ATTN_Q_OUT_GROUP",
    "DEFAULT_SEQ_LEN",
    "linear_4096_to_512",
    "linear_4096_to_512_fp32",
    "linear_4096_to_256_fp32",
    "linear_4096_to_1024",
    "linear_4096_to_1024_fp32",
    "linear_1024_to_32768",
    "linear_8192_to_4096",
    "linear_4096_to_512_test",
    "linear_4096_to_512_fp32_test",
    "linear_4096_to_256_fp32_test",
    "linear_4096_to_1024_test",
    "linear_4096_to_1024_fp32_test",
    "linear_1024_to_32768_test",
    "linear_8192_to_4096_test",
    "golden_linear",
    "build_4096_to_512_specs",
    "build_4096_to_512_fp32_specs",
    "build_4096_to_256_fp32_specs",
    "build_4096_to_1024_specs",
    "build_4096_to_1024_fp32_specs",
    "build_1024_to_32768_specs",
    "build_8192_to_4096_specs",
]
