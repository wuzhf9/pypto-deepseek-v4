"""DeepSeek V4 Flash bf16 RMSNorm PyPTO kernel."""

import pypto.language as pl

from models.common import assert_divisible
from models.config import FLASH_CONFIG as M


B = 1
S_DYN = pl.dynamic("S_DYN")
D = M.dim
EPS = M.rms_norm_eps

D_TILE = 128
T_TILE = 8
DEFAULT_SEQ_LEN = 8

assert_divisible(D, D_TILE, "RMSNorm hidden size")
HIDDEN_BLOCKS = D // D_TILE
HIDDEN_INV = 1.0 / D


@pl.jit.inline
def hidden_rmsnorm(
    x: pl.Tensor[[B, S_DYN, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    out: pl.Tensor[[B, S_DYN, D], pl.BF16],
):
    """Two-pass hidden RMSNorm over the last dimension.

    Semantics match ``../deepseek_v4_flash/inference/model.py::RMSNorm``:
    cast activations and bf16 checkpoint weight to fp32, compute row-wise RMS,
    multiply the fp32 values, and cast the output back to bf16.
    """
    x.bind_dynamic(1, S_DYN)
    out.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    x_flat = pl.reshape(x, [tokens, D])
    out_flat = pl.reshape(out, [tokens, D])
    token_blocks = (tokens + T_TILE - 1) // T_TILE

    for tb in pl.range(token_blocks):
        t0 = tb * T_TILE
        valid_tok = pl.min(T_TILE, tokens - t0)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="hidden_rmsnorm"):
            partial_sq = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(HIDDEN_BLOCKS):
                k0 = kb * D_TILE
                x_chunk_bf16 = pl.slice(
                    x_flat,
                    [T_TILE, D_TILE],
                    [t0, k0],
                    valid_shape=[valid_tok, D_TILE],
                )
                x_chunk = pl.cast(x_chunk_bf16, target_type=pl.FP32)
                partial_sq = pl.add(
                    partial_sq,
                    pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, T_TILE]),
                )

            variance = pl.reshape(pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS), [T_TILE, 1])
            inv_rms = pl.recip(pl.sqrt(variance))

            for kb in pl.range(HIDDEN_BLOCKS):
                k0 = kb * D_TILE
                x_chunk_bf16 = pl.slice(
                    x_flat,
                    [T_TILE, D_TILE],
                    [t0, k0],
                    valid_shape=[valid_tok, D_TILE],
                )
                x_chunk = pl.cast(x_chunk_bf16, target_type=pl.FP32)
                weight_chunk_bf16 = pl.reshape(norm_w[k0 : k0 + D_TILE], [1, D_TILE])
                weight_chunk = pl.cast(weight_chunk_bf16, target_type=pl.FP32)
                normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), weight_chunk)
                normed_bf16 = pl.cast(normed, target_type=pl.BF16, mode="rint")
                out_flat = pl.assemble(out_flat, normed_bf16, [t0, k0])

    out = pl.reshape(out_flat, [B, tokens, D])
    return out


@pl.jit
def hidden_rmsnorm_test(
    x: pl.Tensor[[B, S_DYN, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, D], pl.BF16]],
):
    out = hidden_rmsnorm(x, norm_w, out)
    return out


def golden_hidden_rmsnorm(tensors):
    import torch

    x = tensors["x"].float()
    norm_w = tensors["norm_w"].float()
    inv_rms = torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    tensors["out"][:] = (x * inv_rms * norm_w).to(torch.bfloat16)


def build_tensor_specs(seq_len: int = DEFAULT_SEQ_LEN):
    import torch

    from models.golden import TensorSpec

    def init_x():
        return torch.randn(B, seq_len, D) - 0.5

    def init_norm_w():
        return torch.randn(D) * 0.1 + 1.0

    return [
        TensorSpec("x", [B, seq_len, D], torch.bfloat16, init_value=init_x),
        TensorSpec("norm_w", [D], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("out", [B, seq_len, D], torch.bfloat16, is_output=True),
    ]


def main() -> int:
    import argparse

    from models.golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 Flash hidden RMSNorm validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3sim", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-s", "--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=hidden_rmsnorm_test,
        specs=build_tensor_specs(args.seq_len),
        golden_fn=golden_hidden_rmsnorm,
        runtime_cfg={
            "platform": args.platform,
            "device_id": args.device,
            "enable_l2_swimlane": args.enable_l2_swimlane,
        },
        compile_only=args.compile_only,
        compare_fn={
            "out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "B",
    "S_DYN",
    "D",
    "EPS",
    "D_TILE",
    "T_TILE",
    "HIDDEN_BLOCKS",
    "HIDDEN_INV",
    "DEFAULT_SEQ_LEN",
    "hidden_rmsnorm",
    "hidden_rmsnorm_test",
    "golden_hidden_rmsnorm",
    "build_tensor_specs",
]
