"""DeepSeek V4 Flash bf16 RMSNorm PyPTO kernels."""

import pypto.language as pl

from models.common import assert_divisible
from models.config import FLASH_CONFIG as M


B = 1
S_DYN = pl.dynamic("S_DYN")

D_4096 = M.dim
D_1024 = M.q_lora_rank
D_512 = M.head_dim
D_128 = M.index_head_dim
EPS = M.rms_norm_eps

D_TILE = 128
T_TILE = 8
DEFAULT_SEQ_LEN = 8

assert_divisible(D_4096, D_TILE, "4096 RMSNorm hidden size")
assert_divisible(D_1024, D_TILE, "1024 RMSNorm hidden size")
assert_divisible(D_512, D_TILE, "512 RMSNorm hidden size")
assert_divisible(D_128, D_TILE, "128 RMSNorm hidden size")

BLOCKS_4096 = D_4096 // D_TILE
BLOCKS_1024 = D_1024 // D_TILE
BLOCKS_512 = D_512 // D_TILE
BLOCKS_128 = D_128 // D_TILE

INV_4096 = 1.0 / D_4096
INV_1024 = 1.0 / D_1024
INV_512 = 1.0 / D_512
INV_128 = 1.0 / D_128


@pl.jit.inline
def rmsnorm_4096(
    x: pl.Tensor[[B, S_DYN, D_4096], pl.BF16],
    norm_w: pl.Tensor[[D_4096], pl.BF16],
    out: pl.Tensor[[B, S_DYN, D_4096], pl.BF16],
):
    """Compute bf16 RMSNorm over 4096 hidden channels."""
    x.bind_dynamic(1, S_DYN)
    out.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    x_flat = pl.reshape(x, [tokens, D_4096])
    out_flat = pl.reshape(out, [tokens, D_4096])
    token_blocks = (tokens + T_TILE - 1) // T_TILE

    for tb in pl.spmd(token_blocks, name_hint="rmsnorm_4096"):
        t0 = tb * T_TILE
        valid_tok = pl.min(T_TILE, tokens - t0)

        partial_sq = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.range(BLOCKS_4096):
            k0 = kb * D_TILE
            x_bf16 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, k0], valid_shape=[valid_tok, D_TILE])
            x_fp32 = pl.cast(x_bf16, target_type=pl.FP32)
            partial_sq = pl.add(
                partial_sq,
                pl.reshape(pl.row_sum(pl.mul(x_fp32, x_fp32)), [1, T_TILE]),
            )

        variance = pl.reshape(pl.add(pl.mul(partial_sq, INV_4096), EPS), [T_TILE, 1])
        inv_rms = pl.rsqrt(variance, high_precision=True)

        for kb in pl.range(BLOCKS_4096):
            k0 = kb * D_TILE
            x_bf16 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, k0], valid_shape=[valid_tok, D_TILE])
            x_fp32 = pl.cast(x_bf16, target_type=pl.FP32)
            weight_bf16 = pl.reshape(norm_w[k0 : k0 + D_TILE], [1, D_TILE])
            weight_fp32 = pl.cast(weight_bf16, target_type=pl.FP32)
            normed = pl.col_expand_mul(pl.row_expand_mul(x_fp32, inv_rms), weight_fp32)
            normed_bf16 = pl.cast(normed, target_type=pl.BF16, mode="rint")
            for row in pl.range(valid_tok):
                out_row = pl.slice(normed_bf16, [1, D_TILE], [row, 0], valid_shape=[1, D_TILE])
                out_flat = pl.assemble(out_flat, out_row, [t0 + row, k0])

    return pl.reshape(out_flat, [B, tokens, D_4096])


@pl.jit.inline
def rmsnorm_1024(
    x: pl.Tensor[[B, S_DYN, D_1024], pl.BF16],
    norm_w: pl.Tensor[[D_1024], pl.BF16],
    out: pl.Tensor[[B, S_DYN, D_1024], pl.BF16],
):
    """Compute bf16 RMSNorm over 1024 q-lora channels."""
    x.bind_dynamic(1, S_DYN)
    out.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    x_flat = pl.reshape(x, [tokens, D_1024])
    out_flat = pl.reshape(out, [tokens, D_1024])
    token_blocks = (tokens + T_TILE - 1) // T_TILE

    for tb in pl.spmd(token_blocks, name_hint="rmsnorm_1024"):
        t0 = tb * T_TILE
        valid_tok = pl.min(T_TILE, tokens - t0)

        partial_sq = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.range(BLOCKS_1024):
            k0 = kb * D_TILE
            x_bf16 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, k0], valid_shape=[valid_tok, D_TILE])
            x_fp32 = pl.cast(x_bf16, target_type=pl.FP32)
            partial_sq = pl.add(
                partial_sq,
                pl.reshape(pl.row_sum(pl.mul(x_fp32, x_fp32)), [1, T_TILE]),
            )

        variance = pl.reshape(pl.add(pl.mul(partial_sq, INV_1024), EPS), [T_TILE, 1])
        inv_rms = pl.rsqrt(variance, high_precision=True)

        for kb in pl.range(BLOCKS_1024):
            k0 = kb * D_TILE
            x_bf16 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, k0], valid_shape=[valid_tok, D_TILE])
            x_fp32 = pl.cast(x_bf16, target_type=pl.FP32)
            weight_bf16 = pl.reshape(norm_w[k0 : k0 + D_TILE], [1, D_TILE])
            weight_fp32 = pl.cast(weight_bf16, target_type=pl.FP32)
            normed = pl.col_expand_mul(pl.row_expand_mul(x_fp32, inv_rms), weight_fp32)
            normed_bf16 = pl.cast(normed, target_type=pl.BF16, mode="rint")
            for row in pl.range(valid_tok):
                out_row = pl.slice(normed_bf16, [1, D_TILE], [row, 0], valid_shape=[1, D_TILE])
                out_flat = pl.assemble(out_flat, out_row, [t0 + row, k0])

    return pl.reshape(out_flat, [B, tokens, D_1024])


@pl.jit.inline
def rmsnorm_512(
    x: pl.Tensor[[B, S_DYN, D_512], pl.BF16],
    norm_w: pl.Tensor[[D_512], pl.BF16],
    out: pl.Tensor[[B, S_DYN, D_512], pl.BF16],
):
    """Compute bf16 RMSNorm over 512 head channels."""
    x.bind_dynamic(1, S_DYN)
    out.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    x_flat = pl.reshape(x, [tokens, D_512])
    out_flat = pl.reshape(out, [tokens, D_512])
    token_blocks = (tokens + T_TILE - 1) // T_TILE

    for tb in pl.spmd(token_blocks, name_hint="rmsnorm_512"):
        t0 = tb * T_TILE
        valid_tok = pl.min(T_TILE, tokens - t0)

        partial_sq = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.range(BLOCKS_512):
            k0 = kb * D_TILE
            x_bf16 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, k0], valid_shape=[valid_tok, D_TILE])
            x_fp32 = pl.cast(x_bf16, target_type=pl.FP32)
            partial_sq = pl.add(
                partial_sq,
                pl.reshape(pl.row_sum(pl.mul(x_fp32, x_fp32)), [1, T_TILE]),
            )

        variance = pl.reshape(pl.add(pl.mul(partial_sq, INV_512), EPS), [T_TILE, 1])
        inv_rms = pl.rsqrt(variance, high_precision=True)

        for kb in pl.range(BLOCKS_512):
            k0 = kb * D_TILE
            x_bf16 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, k0], valid_shape=[valid_tok, D_TILE])
            x_fp32 = pl.cast(x_bf16, target_type=pl.FP32)
            weight_bf16 = pl.reshape(norm_w[k0 : k0 + D_TILE], [1, D_TILE])
            weight_fp32 = pl.cast(weight_bf16, target_type=pl.FP32)
            normed = pl.col_expand_mul(pl.row_expand_mul(x_fp32, inv_rms), weight_fp32)
            normed_bf16 = pl.cast(normed, target_type=pl.BF16, mode="rint")
            for row in pl.range(valid_tok):
                out_row = pl.slice(normed_bf16, [1, D_TILE], [row, 0], valid_shape=[1, D_TILE])
                out_flat = pl.assemble(out_flat, out_row, [t0 + row, k0])

    return pl.reshape(out_flat, [B, tokens, D_512])


@pl.jit.inline
def rmsnorm_128(
    x: pl.Tensor[[B, S_DYN, D_128], pl.BF16],
    norm_w: pl.Tensor[[D_128], pl.BF16],
    out: pl.Tensor[[B, S_DYN, D_128], pl.BF16],
):
    """Compute bf16 RMSNorm over 128 indexer-head channels."""
    x.bind_dynamic(1, S_DYN)
    out.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    x_flat = pl.reshape(x, [tokens, D_128])
    out_flat = pl.reshape(out, [tokens, D_128])
    token_blocks = (tokens + T_TILE - 1) // T_TILE

    for tb in pl.spmd(token_blocks, name_hint="rmsnorm_128"):
        t0 = tb * T_TILE
        valid_tok = pl.min(T_TILE, tokens - t0)

        partial_sq = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.range(BLOCKS_128):
            k0 = kb * D_TILE
            x_bf16 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, k0], valid_shape=[valid_tok, D_TILE])
            x_fp32 = pl.cast(x_bf16, target_type=pl.FP32)
            partial_sq = pl.add(
                partial_sq,
                pl.reshape(pl.row_sum(pl.mul(x_fp32, x_fp32)), [1, T_TILE]),
            )

        variance = pl.reshape(pl.add(pl.mul(partial_sq, INV_128), EPS), [T_TILE, 1])
        inv_rms = pl.rsqrt(variance, high_precision=True)

        for kb in pl.range(BLOCKS_128):
            k0 = kb * D_TILE
            x_bf16 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, k0], valid_shape=[valid_tok, D_TILE])
            x_fp32 = pl.cast(x_bf16, target_type=pl.FP32)
            weight_bf16 = pl.reshape(norm_w[k0 : k0 + D_TILE], [1, D_TILE])
            weight_fp32 = pl.cast(weight_bf16, target_type=pl.FP32)
            normed = pl.col_expand_mul(pl.row_expand_mul(x_fp32, inv_rms), weight_fp32)
            normed_bf16 = pl.cast(normed, target_type=pl.BF16, mode="rint")
            for row in pl.range(valid_tok):
                out_row = pl.slice(normed_bf16, [1, D_TILE], [row, 0], valid_shape=[1, D_TILE])
                out_flat = pl.assemble(out_flat, out_row, [t0 + row, k0])

    return pl.reshape(out_flat, [B, tokens, D_128])


@pl.jit
def rmsnorm_4096_test(
    x: pl.Tensor[[B, S_DYN, D_4096], pl.BF16],
    norm_w: pl.Tensor[[D_4096], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, D_4096], pl.BF16]],
):
    out = rmsnorm_4096(x, norm_w, out)
    return out


@pl.jit
def rmsnorm_1024_test(
    x: pl.Tensor[[B, S_DYN, D_1024], pl.BF16],
    norm_w: pl.Tensor[[D_1024], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, D_1024], pl.BF16]],
):
    out = rmsnorm_1024(x, norm_w, out)
    return out


@pl.jit
def rmsnorm_512_test(
    x: pl.Tensor[[B, S_DYN, D_512], pl.BF16],
    norm_w: pl.Tensor[[D_512], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, D_512], pl.BF16]],
):
    out = rmsnorm_512(x, norm_w, out)
    return out


@pl.jit
def rmsnorm_128_test(
    x: pl.Tensor[[B, S_DYN, D_128], pl.BF16],
    norm_w: pl.Tensor[[D_128], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, D_128], pl.BF16]],
):
    out = rmsnorm_128(x, norm_w, out)
    return out


hidden_rmsnorm = rmsnorm_4096
hidden_rmsnorm_test = rmsnorm_4096_test


def golden_rmsnorm(tensors):
    import torch

    x = tensors["x"].float()
    norm_w = tensors["norm_w"].float()
    inv_rms = torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    tensors["out"][:] = (x * inv_rms * norm_w).to(torch.bfloat16)


golden_hidden_rmsnorm = golden_rmsnorm


def _build_tensor_specs(dim: int, seq_len: int):
    import torch

    from models.golden import TensorSpec

    def init_x():
        return torch.randn(B, seq_len, dim) - 0.5

    def init_norm_w():
        return torch.randn(dim) * 0.1 + 1.0

    return [
        TensorSpec("x", [B, seq_len, dim], torch.bfloat16, init_value=init_x),
        TensorSpec("norm_w", [dim], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("out", [B, seq_len, dim], torch.bfloat16, is_output=True),
    ]


def build_4096_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _build_tensor_specs(D_4096, seq_len)


def build_1024_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _build_tensor_specs(D_1024, seq_len)


def build_512_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _build_tensor_specs(D_512, seq_len)


def build_128_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _build_tensor_specs(D_128, seq_len)


def main() -> int:
    import argparse

    from models.golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 Flash bf16 RMSNorm validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3sim", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-s", "--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    cases = [
        ("rmsnorm-4096", rmsnorm_4096_test, build_4096_specs),
        ("rmsnorm-1024", rmsnorm_1024_test, build_1024_specs),
        ("rmsnorm-512", rmsnorm_512_test, build_512_specs),
        ("rmsnorm-128", rmsnorm_128_test, build_128_specs),
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
            golden_fn=golden_rmsnorm,
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
    "D_4096",
    "D_1024",
    "D_512",
    "D_128",
    "EPS",
    "D_TILE",
    "T_TILE",
    "BLOCKS_4096",
    "BLOCKS_1024",
    "BLOCKS_512",
    "BLOCKS_128",
    "INV_4096",
    "INV_1024",
    "INV_512",
    "INV_128",
    "DEFAULT_SEQ_LEN",
    "rmsnorm_4096",
    "rmsnorm_1024",
    "rmsnorm_512",
    "rmsnorm_128",
    "hidden_rmsnorm",
    "rmsnorm_4096_test",
    "rmsnorm_1024_test",
    "rmsnorm_512_test",
    "rmsnorm_128_test",
    "hidden_rmsnorm_test",
    "golden_rmsnorm",
    "golden_hidden_rmsnorm",
    "build_4096_specs",
    "build_1024_specs",
    "build_512_specs",
    "build_128_specs",
]
