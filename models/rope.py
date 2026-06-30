"""DeepSeek V4 Flash RoPE table generation and PyPTO kernels."""

import math
from typing import Any

import pypto.language as pl
import torch

from models.config import FLASH_CONFIG

B = 1
S_DYN = pl.dynamic("S_DYN")

ROPE_DIM = FLASH_CONFIG.rope_head_dim
ROPE_HALF = ROPE_DIM // 2
N_HEADS = FLASH_CONFIG.n_heads
ROTARY_T_TILE = 16
DEFAULT_SEQ_LEN = 8


def rope_profile_for_compress_ratio(config: Any, compress_ratio: int) -> tuple[float, int]:
    """Return ``(base_theta, original_seq_len)`` for an attention RoPE profile."""
    if compress_ratio:
        return float(config.compress_rope_theta), int(config.original_seq_len)
    return float(config.rope_theta), 0


def _linear_ramp_factor(
    low: int,
    high: int,
    dim: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if low == high:
        high = high + 0.001
    ramp = (torch.arange(dim, dtype=torch.float32, device=device) - low) / (high - low)
    return torch.clamp(ramp, 0, 1)


def _find_correction_dim(num_rotations: int, dim: int, base: float, max_seq_len: int) -> float:
    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def _find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float,
    max_seq_len: int,
) -> tuple[int, int]:
    low = math.floor(_find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(_find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def precompute_freqs_cos_sin(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 half-width RoPE tables equivalent to official ``freqs_cis``.

    Official ``apply_rotary_emb`` treats every two real channels as one complex
    value, so a full RoPE width of ``dim`` only needs ``dim // 2`` frequencies.
    The returned tensors are shaped ``[seqlen, dim // 2]`` and use FP32 to match
    the real and imaginary parts of the official complex64 table.
    """
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"RoPE dim must be a positive even integer, got {dim}")
    if seqlen <= 0:
        raise ValueError(f"RoPE sequence length must be positive, got {seqlen}")

    out_device = torch.device(device) if device is not None else None
    half_dim = dim // 2

    inv_freq = 1.0 / (
        float(base) ** (torch.arange(0, dim, 2, dtype=torch.float32, device=out_device) / dim)
    )
    if original_seq_len > 0:
        low, high = _find_correction_range(beta_fast, beta_slow, dim, float(base), int(original_seq_len))
        smooth = 1 - _linear_ramp_factor(low, high, half_dim, device=out_device)
        inv_freq = inv_freq / float(factor) * (1 - smooth) + inv_freq * smooth

    positions = torch.arange(seqlen, dtype=torch.float32, device=out_device)
    angles = torch.outer(positions, inv_freq)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis.real.contiguous(), freqs_cis.imag.contiguous()


def build_deepseek_v4_rope_tables(
    config: Any = FLASH_CONFIG,
    compress_ratio: int = 0,
    *,
    max_seq_len: int | None = None,
    rope_dim: int | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(cos, sin)`` shaped ``[max_seq_len, rope_dim // 2]`` in FP32."""
    base, original_seq_len = rope_profile_for_compress_ratio(config, compress_ratio)
    seq_len = int(max_seq_len if max_seq_len is not None else config.max_position_embeddings)
    dim = int(rope_dim if rope_dim is not None else config.rope_head_dim)

    return precompute_freqs_cos_sin(
        dim,
        seq_len,
        original_seq_len,
        base,
        float(config.rope_factor),
        int(config.beta_fast),
        int(config.beta_slow),
        device=device,
    )


def materialize_rope_range(
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    start_pos: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice a contiguous token range matching ``freqs_cis[start:start+seq]``."""
    if start_pos < 0:
        raise ValueError(f"start_pos must be non-negative, got {start_pos}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    end_pos = start_pos + seq_len
    return freqs_cos[start_pos:end_pos].contiguous(), freqs_sin[start_pos:end_pos].contiguous()


@pl.jit.inline
def rotary_3d_fwd(
    x: pl.Tensor[[B, S_DYN, ROPE_DIM], pl.BF16],
    cos: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    out: pl.Tensor[[B, S_DYN, ROPE_DIM], pl.BF16],
):
    """Apply forward RoPE to ``x`` shaped ``[1, S, 64]``."""
    x.bind_dynamic(1, S_DYN)
    cos.bind_dynamic(0, S_DYN)
    sin.bind_dynamic(0, S_DYN)
    out.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    x_flat = pl.reshape(x, [tokens, ROPE_DIM])
    out_flat = pl.reshape(out, [tokens, ROPE_DIM])
    token_blocks = (tokens + ROTARY_T_TILE - 1) // ROTARY_T_TILE

    for tb in pl.range(token_blocks):
        t0 = tb * ROTARY_T_TILE
        valid_tok = pl.min(ROTARY_T_TILE, tokens - t0)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rotary_3d_fwd"):
            ones = pl.full([ROTARY_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
            col = pl.col_expand_mul(ones, pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32))
            dup_f = pl.cast(pl.cast(pl.mul(col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
            dup_idx = pl.cast(dup_f, target_type=pl.INT32)
            lane = pl.sub(col, pl.mul(dup_f, 2.0))
            swap_idx = pl.cast(pl.sub(pl.add(col, 1.0), pl.mul(lane, 2.0)), target_type=pl.INT32)
            sign = pl.sub(pl.mul(lane, 2.0), 1.0)

            x_tile = pl.cast(
                pl.slice(x_flat, [ROTARY_T_TILE, ROPE_DIM], [t0, 0], valid_shape=[valid_tok, ROPE_DIM]),
                target_type=pl.FP32,
            )
            cos_tile = pl.slice(cos, [ROTARY_T_TILE, ROPE_HALF], [t0, 0], valid_shape=[valid_tok, ROPE_HALF])
            sin_tile = pl.slice(sin, [ROTARY_T_TILE, ROPE_HALF], [t0, 0], valid_shape=[valid_tok, ROPE_HALF])
            cos_il = pl.gather(cos_tile, dim=-1, index=dup_idx)
            sin_il = pl.gather(sin_tile, dim=-1, index=dup_idx)
            swapped = pl.gather(x_tile, dim=-1, index=swap_idx)
            rotated = pl.add(pl.mul(x_tile, cos_il), pl.mul(pl.mul(swapped, sign), sin_il))
            rotated_bf16 = pl.cast(rotated, target_type=pl.BF16, mode="rint")

            for row in pl.range(valid_tok):
                out_row = pl.slice(rotated_bf16, [1, ROPE_DIM], [row, 0], valid_shape=[1, ROPE_DIM])
                out_flat = pl.assemble(out_flat, out_row, [t0 + row, 0])

    return pl.reshape(out_flat, [B, tokens, ROPE_DIM])


@pl.jit.inline
def rotary_4d_fwd(
    x: pl.Tensor[[B, S_DYN, N_HEADS, ROPE_DIM], pl.BF16],
    cos: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    out: pl.Tensor[[B, S_DYN, N_HEADS, ROPE_DIM], pl.BF16],
):
    """Apply forward RoPE to ``x`` shaped ``[1, S, 64, 64]``."""
    x.bind_dynamic(1, S_DYN)
    cos.bind_dynamic(0, S_DYN)
    sin.bind_dynamic(0, S_DYN)
    out.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    x_flat = pl.reshape(x, [tokens, N_HEADS * ROPE_DIM])
    out_flat = pl.reshape(out, [tokens, N_HEADS * ROPE_DIM])
    token_blocks = (tokens + ROTARY_T_TILE - 1) // ROTARY_T_TILE

    for tb in pl.range(token_blocks):
        t0 = tb * ROTARY_T_TILE
        valid_tok = pl.min(ROTARY_T_TILE, tokens - t0)

        for h in pl.spmd(N_HEADS, name_hint="rotary_4d_fwd"):
            h0 = h * ROPE_DIM
            ones = pl.full([ROTARY_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
            col = pl.col_expand_mul(ones, pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32))
            dup_f = pl.cast(pl.cast(pl.mul(col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
            dup_idx = pl.cast(dup_f, target_type=pl.INT32)
            lane = pl.sub(col, pl.mul(dup_f, 2.0))
            swap_idx = pl.cast(pl.sub(pl.add(col, 1.0), pl.mul(lane, 2.0)), target_type=pl.INT32)
            sign = pl.sub(pl.mul(lane, 2.0), 1.0)

            x_tile = pl.cast(
                pl.slice(x_flat, [ROTARY_T_TILE, ROPE_DIM], [t0, h0], valid_shape=[valid_tok, ROPE_DIM]),
                target_type=pl.FP32,
            )
            cos_tile = pl.slice(cos, [ROTARY_T_TILE, ROPE_HALF], [t0, 0], valid_shape=[valid_tok, ROPE_HALF])
            sin_tile = pl.slice(sin, [ROTARY_T_TILE, ROPE_HALF], [t0, 0], valid_shape=[valid_tok, ROPE_HALF])
            cos_il = pl.gather(cos_tile, dim=-1, index=dup_idx)
            sin_il = pl.gather(sin_tile, dim=-1, index=dup_idx)
            swapped = pl.gather(x_tile, dim=-1, index=swap_idx)
            rotated = pl.add(pl.mul(x_tile, cos_il), pl.mul(pl.mul(swapped, sign), sin_il))
            rotated_bf16 = pl.cast(rotated, target_type=pl.BF16, mode="rint")

            for row in pl.range(valid_tok):
                out_row = pl.slice(rotated_bf16, [1, ROPE_DIM], [row, 0], valid_shape=[1, ROPE_DIM])
                out_flat = pl.assemble(out_flat, out_row, [t0 + row, h0])

    return pl.reshape(out_flat, [B, tokens, N_HEADS, ROPE_DIM])


@pl.jit.inline
def rotary_4d_inv(
    x: pl.Tensor[[B, S_DYN, N_HEADS, ROPE_DIM], pl.BF16],
    cos: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    out: pl.Tensor[[B, S_DYN, N_HEADS, ROPE_DIM], pl.BF16],
):
    """Apply inverse RoPE to ``x`` shaped ``[1, S, 64, 64]``."""
    x.bind_dynamic(1, S_DYN)
    cos.bind_dynamic(0, S_DYN)
    sin.bind_dynamic(0, S_DYN)
    out.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    x_flat = pl.reshape(x, [tokens, N_HEADS * ROPE_DIM])
    out_flat = pl.reshape(out, [tokens, N_HEADS * ROPE_DIM])
    token_blocks = (tokens + ROTARY_T_TILE - 1) // ROTARY_T_TILE

    for tb in pl.range(token_blocks):
        t0 = tb * ROTARY_T_TILE
        valid_tok = pl.min(ROTARY_T_TILE, tokens - t0)

        for h in pl.spmd(N_HEADS, name_hint="rotary_4d_inv"):
            h0 = h * ROPE_DIM
            ones = pl.full([ROTARY_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
            col = pl.col_expand_mul(ones, pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32))
            dup_f = pl.cast(pl.cast(pl.mul(col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
            dup_idx = pl.cast(dup_f, target_type=pl.INT32)
            lane = pl.sub(col, pl.mul(dup_f, 2.0))
            swap_idx = pl.cast(pl.sub(pl.add(col, 1.0), pl.mul(lane, 2.0)), target_type=pl.INT32)
            sign = pl.sub(pl.mul(lane, 2.0), 1.0)

            x_tile = pl.cast(
                pl.slice(x_flat, [ROTARY_T_TILE, ROPE_DIM], [t0, h0], valid_shape=[valid_tok, ROPE_DIM]),
                target_type=pl.FP32,
            )
            cos_tile = pl.slice(cos, [ROTARY_T_TILE, ROPE_HALF], [t0, 0], valid_shape=[valid_tok, ROPE_HALF])
            sin_tile = pl.slice(sin, [ROTARY_T_TILE, ROPE_HALF], [t0, 0], valid_shape=[valid_tok, ROPE_HALF])
            cos_il = pl.gather(cos_tile, dim=-1, index=dup_idx)
            sin_il = pl.gather(sin_tile, dim=-1, index=dup_idx)
            swapped = pl.gather(x_tile, dim=-1, index=swap_idx)
            rotated = pl.sub(pl.mul(x_tile, cos_il), pl.mul(pl.mul(swapped, sign), sin_il))
            rotated_bf16 = pl.cast(rotated, target_type=pl.BF16, mode="rint")

            for row in pl.range(valid_tok):
                out_row = pl.slice(rotated_bf16, [1, ROPE_DIM], [row, 0], valid_shape=[1, ROPE_DIM])
                out_flat = pl.assemble(out_flat, out_row, [t0 + row, h0])

    return pl.reshape(out_flat, [B, tokens, N_HEADS, ROPE_DIM])


@pl.jit
def rotary_3d_fwd_test(
    x: pl.Tensor[[B, S_DYN, ROPE_DIM], pl.BF16],
    cos: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    out: pl.Out[pl.Tensor[[B, S_DYN, ROPE_DIM], pl.BF16]],
):
    out = rotary_3d_fwd(x, cos, sin, out)
    return out


@pl.jit
def rotary_4d_fwd_test(
    x: pl.Tensor[[B, S_DYN, N_HEADS, ROPE_DIM], pl.BF16],
    cos: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    out: pl.Out[pl.Tensor[[B, S_DYN, N_HEADS, ROPE_DIM], pl.BF16]],
):
    out = rotary_4d_fwd(x, cos, sin, out)
    return out


@pl.jit
def rotary_4d_inv_test(
    x: pl.Tensor[[B, S_DYN, N_HEADS, ROPE_DIM], pl.BF16],
    cos: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    out: pl.Out[pl.Tensor[[B, S_DYN, N_HEADS, ROPE_DIM], pl.BF16]],
):
    out = rotary_4d_inv(x, cos, sin, out)
    return out


def _apply_rope_golden(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, inverse: bool) -> torch.Tensor:
    x_fp32 = x.float()
    pair = x_fp32.unflatten(-1, (-1, 2))
    x0 = pair[..., 0]
    x1 = pair[..., 1]

    if x.ndim == 3:
        cos_v = cos.float().view(1, x.size(1), ROPE_HALF)
        sin_v = sin.float().view(1, x.size(1), ROPE_HALF)
    else:
        cos_v = cos.float().view(1, x.size(1), 1, ROPE_HALF)
        sin_v = sin.float().view(1, x.size(1), 1, ROPE_HALF)

    if inverse:
        y0 = x0 * cos_v + x1 * sin_v
        y1 = -x0 * sin_v + x1 * cos_v
    else:
        y0 = x0 * cos_v - x1 * sin_v
        y1 = x0 * sin_v + x1 * cos_v

    return torch.stack([y0, y1], dim=-1).flatten(-2).to(x.dtype)


def golden_rotary_3d_fwd(tensors):
    tensors["out"][:] = _apply_rope_golden(tensors["x"], tensors["cos"], tensors["sin"], inverse=False)


def golden_rotary_4d_fwd(tensors):
    tensors["out"][:] = _apply_rope_golden(tensors["x"], tensors["cos"], tensors["sin"], inverse=False)


def golden_rotary_4d_inv(tensors):
    tensors["out"][:] = _apply_rope_golden(tensors["x"], tensors["cos"], tensors["sin"], inverse=True)


def _build_tensor_specs(shape: list[int], seq_len: int, start_pos: int):
    from models.golden import TensorSpec

    freqs_cos, freqs_sin = build_deepseek_v4_rope_tables(max_seq_len=start_pos + seq_len)
    local_cos, local_sin = materialize_rope_range(freqs_cos, freqs_sin, start_pos, seq_len)

    def init_x():
        return torch.randn(shape) * 0.1

    return [
        TensorSpec("x", shape, torch.bfloat16, init_value=init_x),
        TensorSpec("cos", [seq_len, ROPE_HALF], torch.float32, init_value=local_cos),
        TensorSpec("sin", [seq_len, ROPE_HALF], torch.float32, init_value=local_sin),
        TensorSpec("out", shape, torch.bfloat16, is_output=True),
    ]


def build_rotary_3d_specs(seq_len: int = DEFAULT_SEQ_LEN, start_pos: int = 0):
    return _build_tensor_specs([B, seq_len, ROPE_DIM], seq_len, start_pos)


def build_rotary_4d_specs(seq_len: int = DEFAULT_SEQ_LEN, start_pos: int = 0):
    return _build_tensor_specs([B, seq_len, N_HEADS, ROPE_DIM], seq_len, start_pos)


def main() -> int:
    import argparse

    from models.golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 Flash RoPE validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3sim", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-s", "--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--start-pos", type=int, default=7)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    cases = [
        ("rotary-3d-fwd", rotary_3d_fwd_test, build_rotary_3d_specs, golden_rotary_3d_fwd),
        ("rotary-4d-fwd", rotary_4d_fwd_test, build_rotary_4d_specs, golden_rotary_4d_fwd),
        ("rotary-4d-inv", rotary_4d_inv_test, build_rotary_4d_specs, golden_rotary_4d_inv),
    ]
    runtime_cfg = {
        "platform": args.platform,
        "device_id": args.device,
        "enable_l2_swimlane": args.enable_l2_swimlane,
    }
    compare_fn = {
        "out": ratio_allclose(atol=1e-4, rtol=5e-3, max_error_ratio=0.0),
    }

    failed = False
    for name, fn, build_specs, golden_fn in cases:
        print(f"[CASE] {name}", flush=True)
        result = run_jit(
            fn=fn,
            specs=build_specs(args.seq_len, args.start_pos),
            golden_fn=golden_fn,
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
    "ROPE_DIM",
    "ROPE_HALF",
    "N_HEADS",
    "ROTARY_T_TILE",
    "DEFAULT_SEQ_LEN",
    "rope_profile_for_compress_ratio",
    "precompute_freqs_cos_sin",
    "build_deepseek_v4_rope_tables",
    "materialize_rope_range",
    "rotary_3d_fwd",
    "rotary_4d_fwd",
    "rotary_4d_inv",
    "rotary_3d_fwd_test",
    "rotary_4d_fwd_test",
    "rotary_4d_inv_test",
    "golden_rotary_3d_fwd",
    "golden_rotary_4d_fwd",
    "golden_rotary_4d_inv",
    "build_rotary_3d_specs",
    "build_rotary_4d_specs",
]
