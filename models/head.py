"""DeepSeek V4 Flash head golden logic."""

import pypto.language as pl

from models.common import assert_divisible, ceil_div
from models.config import FLASH_CONFIG as M
from models.rmsnorm import rmsnorm_4096


B = 1
S_DYN = pl.dynamic("S_DYN")
S_PAD_DYN = pl.dynamic("S_PAD_DYN")
VOCAB = M.vocab_size
HIDDEN = M.dim
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
NORM_EPS = M.rms_norm_eps
HC_EPS = M.hc_eps
DEFAULT_SEQ_LEN = 8

K_TILE = 128
VOCAB_TILE = 128
HC_PAD = 16
T_TILE = 16
RMS_K_CHUNK = 256
LINEAR_K_CHUNK = 256
D_CHUNK = 128
X_PAD_CHUNK = 512
HC_DIM_INV = 1.0 / HC_DIM

assert HC_MULT <= HC_PAD
assert_divisible(HIDDEN, K_TILE, "head hidden size")
assert_divisible(HIDDEN, D_CHUNK, "head HC reduce hidden size")
assert_divisible(HC_DIM, K_TILE, "head HC flattened size")
assert_divisible(HC_DIM, RMS_K_CHUNK, "head HC RMS input size")
assert_divisible(HC_DIM, LINEAR_K_CHUNK, "head HC linear input size")
assert_divisible(HC_DIM, X_PAD_CHUNK, "head HC pad chunk")
assert_divisible(VOCAB, VOCAB_TILE, "head vocab size")

HIDDEN_K_BLOCKS = HIDDEN // K_TILE
RMS_K_BLOCKS = HC_DIM // RMS_K_CHUNK
LINEAR_K_BLOCKS = HC_DIM // LINEAR_K_CHUNK
D_BLOCKS = HIDDEN // D_CHUNK
X_PAD_BLOCKS = HC_DIM // X_PAD_CHUNK
VOCAB_BLOCKS = VOCAB // VOCAB_TILE


@pl.jit.inline
def hc_head_fwd(
    x: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    hc_fn: pl.Tensor[[HC_PAD, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[1], pl.FP32],
    hc_base: pl.Tensor[[HC_PAD], pl.FP32],
    pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    out_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
):
    """Run official ``head.hc_head`` on a padded token axis."""
    x.bind_dynamic(1, S_DYN)
    x_pad.bind_dynamic(1, S_PAD_DYN)
    pre.bind_dynamic(1, S_PAD_DYN)
    out_pad.bind_dynamic(1, S_PAD_DYN)
    out.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    padded_tokens = pl.tensor.dim(x_pad, 1)
    x_src_flat = pl.reshape(x, [tokens, HC_DIM])
    x_flat = pl.reshape(x_pad, [padded_tokens, HC_DIM])
    pre_flat = pl.reshape(pre, [padded_tokens, HC_PAD])
    out_pad_flat = pl.reshape(out_pad, [padded_tokens, HIDDEN])
    out_flat = pl.reshape(out, [tokens, HIDDEN])
    token_blocks = padded_tokens // T_TILE
    scale = pl.read(hc_scale, [0])

    for t in pl.range(padded_tokens):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="head_hc_pad_x"):
            for kb in pl.range(X_PAD_BLOCKS):
                k0 = kb * X_PAD_CHUNK
                if t < tokens:
                    x_row = x_src_flat[t : t + 1, k0 : k0 + X_PAD_CHUNK]
                else:
                    x_row = pl.full([1, X_PAD_CHUNK], dtype=pl.BF16, value=0.0)
                x_flat[t : t + 1, k0 : k0 + X_PAD_CHUNK] = x_row

    for tb in pl.range(token_blocks):
        t0 = tb * T_TILE

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="head_hc_pre"):
            sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.pipeline(0, RMS_K_BLOCKS, stage=2):
                k0 = kb * RMS_K_CHUNK
                x_chunk = pl.cast(
                    pl.slice(x_flat, [T_TILE, RMS_K_CHUNK], [t0, k0]),
                    target_type=pl.FP32,
                )
                sq_sum = pl.add(
                    sq_sum,
                    pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, T_TILE]),
                )
            inv = pl.reshape(pl.rsqrt(pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS), high_precision=True), [T_TILE, 1])

            x0 = pl.cast(
                pl.slice(x_flat, [T_TILE, LINEAR_K_CHUNK], [t0, 0]),
                target_type=pl.FP32,
            )
            w0 = pl.slice(hc_fn, [HC_PAD, LINEAR_K_CHUNK], [0, 0])
            acc = pl.matmul(x0, w0, b_trans=True, out_dtype=pl.FP32)
            for kb in pl.pipeline(1, LINEAR_K_BLOCKS, stage=2):
                k0 = kb * LINEAR_K_CHUNK
                x_chunk = pl.cast(
                    pl.slice(x_flat, [T_TILE, LINEAR_K_CHUNK], [t0, k0]),
                    target_type=pl.FP32,
                )
                w_chunk = pl.slice(hc_fn, [HC_PAD, LINEAR_K_CHUNK], [0, k0])
                acc = pl.matmul_acc(acc, x_chunk, w_chunk, b_trans=True)
            scaled = pl.row_expand_mul(acc, inv)
            base = pl.reshape(pl.slice(hc_base, [HC_PAD], [0]), [1, HC_PAD])
            logits = pl.add(
                pl.mul(scaled, scale),
                pl.col_expand_mul(pl.full([T_TILE, HC_PAD], dtype=pl.FP32, value=1.0), base),
            )
            pre_tile = pl.add(pl.recip(pl.add(pl.exp(pl.neg(logits)), 1.0)), HC_EPS)
            for row in pl.range(T_TILE):
                pre_row = pl.slice(pre_tile, [1, HC_PAD], [row, 0])
                pre_flat = pl.assemble(pre_flat, pre_row, [t0 + row, 0])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="head_hc_reduce"):
            pre_tile = pl.slice(pre_flat, [T_TILE, HC_PAD], [t0, 0])
            pre_tile_t = pl.transpose(pre_tile, axis1=0, axis2=1)
            pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
            pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
            pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
            pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
            for db in pl.range(D_BLOCKS):
                d0 = db * D_CHUNK
                x_h0 = pl.cast(pl.slice(x_flat, [T_TILE, D_CHUNK], [t0, 0 * HIDDEN + d0]), target_type=pl.FP32)
                x_h1 = pl.cast(pl.slice(x_flat, [T_TILE, D_CHUNK], [t0, 1 * HIDDEN + d0]), target_type=pl.FP32)
                x_h2 = pl.cast(pl.slice(x_flat, [T_TILE, D_CHUNK], [t0, 2 * HIDDEN + d0]), target_type=pl.FP32)
                x_h3 = pl.cast(pl.slice(x_flat, [T_TILE, D_CHUNK], [t0, 3 * HIDDEN + d0]), target_type=pl.FP32)
                y_tile = pl.add(
                    pl.add(pl.row_expand_mul(x_h0, pre0), pl.row_expand_mul(x_h1, pre1)),
                    pl.add(pl.row_expand_mul(x_h2, pre2), pl.row_expand_mul(x_h3, pre3)),
                )
                y_out = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
                for row in pl.range(T_TILE):
                    y_row = pl.slice(y_out, [1, D_CHUNK], [row, 0])
                    out_pad_flat = pl.assemble(out_pad_flat, y_row, [t0 + row, d0])

    for t in pl.range(tokens):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="head_hc_copy_out"):
            for db in pl.range(D_BLOCKS):
                d0 = db * D_CHUNK
                out_flat[t : t + 1, d0 : d0 + D_CHUNK] = out_pad_flat[t : t + 1, d0 : d0 + D_CHUNK]

    return pl.reshape(out_flat, [B, tokens, HIDDEN])


@pl.jit.inline
def lm_head_fwd(
    normed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    head_w: pl.Tensor[[VOCAB, HIDDEN], pl.FP32],
    logits_pad: pl.Tensor[[T_TILE, VOCAB], pl.FP32],
    logits: pl.Tensor[[B, VOCAB], pl.FP32],
):
    """Run official ``head.get_logits`` for the last token only."""
    normed.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(normed, 1)
    last_t = tokens - 1
    normed_flat = pl.reshape(normed, [tokens, HIDDEN])
    last_hidden = pl.create_tensor([T_TILE, HIDDEN], dtype=pl.BF16)

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="head_lm_last"):
        for kb in pl.range(HIDDEN_K_BLOCKS):
            k0 = kb * K_TILE
            zero_chunk = pl.full([T_TILE, K_TILE], dtype=pl.BF16, value=0.0)
            last_hidden = pl.assemble(last_hidden, zero_chunk, [0, k0])
            hidden_chunk = pl.slice(normed_flat, [1, K_TILE], [last_t, k0])
            last_hidden = pl.assemble(last_hidden, hidden_chunk, [0, k0])

    for vb in pl.parallel(VOCAB_BLOCKS):
        v0 = vb * VOCAB_TILE
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="head_lm"):
            x0 = pl.cast(pl.slice(last_hidden, [T_TILE, K_TILE], [0, 0]), target_type=pl.FP32)
            w0 = pl.slice(head_w, [VOCAB_TILE, K_TILE], [v0, 0])
            acc = pl.matmul(x0, w0, b_trans=True, out_dtype=pl.FP32)
            for kb in pl.range(1, HIDDEN_K_BLOCKS):
                k0 = kb * K_TILE
                xk = pl.cast(pl.slice(last_hidden, [T_TILE, K_TILE], [0, k0]), target_type=pl.FP32)
                wk = pl.slice(head_w, [VOCAB_TILE, K_TILE], [v0, k0])
                acc = pl.matmul_acc(acc, xk, wk, b_trans=True)
            logits_pad = pl.assemble(logits_pad, acc, [0, v0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="head_lm_store"):
        for vb in pl.range(VOCAB_BLOCKS):
            v0 = vb * VOCAB_TILE
            logits_tile = pl.slice(logits_pad, [1, VOCAB_TILE], [0, v0])
            logits = pl.assemble(logits, logits_tile, [0, v0])

    return logits


@pl.jit.inline
def head_fwd(
    x: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    hc_fn: pl.Tensor[[HC_PAD, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[1], pl.FP32],
    hc_base: pl.Tensor[[HC_PAD], pl.FP32],
    norm_w: pl.Tensor[[HIDDEN], pl.BF16],
    head_w: pl.Tensor[[VOCAB, HIDDEN], pl.FP32],
    pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    hc_out_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    logits: pl.Tensor[[B, VOCAB], pl.FP32],
):
    """Run single-card DeepSeek head: HC head, final RMSNorm, and LM head."""
    x.bind_dynamic(1, S_DYN)
    x_pad.bind_dynamic(1, S_PAD_DYN)
    pre.bind_dynamic(1, S_PAD_DYN)
    hc_out_pad.bind_dynamic(1, S_PAD_DYN)

    tokens = pl.tensor.dim(x, 1)
    hc_out = pl.create_tensor([B, tokens, HIDDEN], dtype=pl.BF16)
    normed = pl.create_tensor([B, tokens, HIDDEN], dtype=pl.BF16)
    logits_pad = pl.create_tensor([T_TILE, VOCAB], dtype=pl.FP32)
    hc_out = hc_head_fwd(x, x_pad, hc_fn, hc_scale, hc_base, pre, hc_out_pad, hc_out)
    normed = rmsnorm_4096(hc_out, norm_w, normed)
    logits = lm_head_fwd(normed, head_w, logits_pad, logits)
    return logits


@pl.jit
def head_test(
    x: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    hc_fn: pl.Tensor[[HC_PAD, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[1], pl.FP32],
    hc_base: pl.Tensor[[HC_PAD], pl.FP32],
    norm_w: pl.Tensor[[HIDDEN], pl.BF16],
    head_w: pl.Tensor[[VOCAB, HIDDEN], pl.FP32],
    pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    hc_out_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    logits: pl.Out[pl.Tensor[[B, VOCAB], pl.FP32]],
):
    logits = head_fwd(
        x,
        x_pad,
        hc_fn,
        hc_scale,
        hc_base,
        norm_w,
        head_w,
        pre,
        hc_out_pad,
        logits,
    )
    return logits


def golden_head(tensors):
    import torch
    import torch.nn.functional as F

    x = tensors["x"]
    shape, dtype = x.size(), x.dtype
    x_flat = x.flatten(2).float()

    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + NORM_EPS)
    hc_fn = tensors["hc_fn"][:HC_MULT].float()
    hc_base = tensors["hc_base"][:HC_MULT].float()
    mixes = F.linear(x_flat, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * tensors["hc_scale"].float() + hc_base) + HC_EPS
    hc_out = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2).to(dtype)

    normed_float = hc_out.float()
    inv_rms = torch.rsqrt(normed_float.square().mean(-1, keepdim=True) + NORM_EPS)
    normed = (tensors["norm_w"].float() * normed_float * inv_rms).to(dtype)

    logits = F.linear(normed[:, -1].float(), tensors["head_w"].float())

    tensors["logits"][:] = logits


def build_head_specs(seq_len: int = DEFAULT_SEQ_LEN):
    import torch

    from models.golden import TensorSpec

    padded_seq_len = ceil_div(seq_len, T_TILE) * T_TILE

    def init_x():
        return torch.randn(B, seq_len, HC_MULT, HIDDEN) * 0.2

    def init_hc_fn():
        weight = torch.zeros(HC_PAD, HC_DIM)
        weight[:HC_MULT] = torch.randn(HC_MULT, HC_DIM) * 0.02
        return weight

    def init_hc_scale():
        return torch.randn(1) * 0.1 + 1.0

    def init_hc_base():
        base = torch.zeros(HC_PAD)
        base[:HC_MULT] = torch.randn(HC_MULT) * 0.02
        return base

    def init_norm_w():
        return torch.randn(HIDDEN) * 0.1 + 1.0

    def init_head_w():
        return torch.randn(VOCAB, HIDDEN) * 0.02

    return [
        TensorSpec("x", [B, seq_len, HC_MULT, HIDDEN], torch.bfloat16, init_value=init_x),
        TensorSpec("x_pad", [B, padded_seq_len, HC_MULT, HIDDEN], torch.bfloat16),
        TensorSpec("hc_fn", [HC_PAD, HC_DIM], torch.float32, init_value=init_hc_fn),
        TensorSpec("hc_scale", [1], torch.float32, init_value=init_hc_scale),
        TensorSpec("hc_base", [HC_PAD], torch.float32, init_value=init_hc_base),
        TensorSpec("norm_w", [HIDDEN], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("head_w", [VOCAB, HIDDEN], torch.float32, init_value=init_head_w),
        TensorSpec("pre", [B, padded_seq_len, HC_PAD], torch.float32),
        TensorSpec("hc_out_pad", [B, padded_seq_len, HIDDEN], torch.bfloat16),
        TensorSpec("logits", [B, VOCAB], torch.float32, is_output=True),
    ]


def main() -> int:
    import argparse

    from models.golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 Flash head validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3sim", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-s", "--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    runtime_cfg = {
        "platform": args.platform,
        "device_id": args.device,
        "enable_l2_swimlane": args.enable_l2_swimlane,
    }
    print("[CASE] head", flush=True)
    result = run_jit(
        fn=head_test,
        specs=build_head_specs(args.seq_len),
        golden_fn=golden_head,
        runtime_cfg=runtime_cfg,
        compile_only=args.compile_only,
        compare_fn={
            "logits": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        },
    )
    if not result.passed and result.error:
        print(result.error)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "B",
    "S_DYN",
    "S_PAD_DYN",
    "VOCAB",
    "HIDDEN",
    "HC_MULT",
    "HC_DIM",
    "NORM_EPS",
    "HC_EPS",
    "DEFAULT_SEQ_LEN",
    "K_TILE",
    "VOCAB_TILE",
    "HC_PAD",
    "HC_DIM_INV",
    "HIDDEN_K_BLOCKS",
    "RMS_K_BLOCKS",
    "LINEAR_K_BLOCKS",
    "D_BLOCKS",
    "X_PAD_CHUNK",
    "X_PAD_BLOCKS",
    "VOCAB_BLOCKS",
    "hc_head_fwd",
    "lm_head_fwd",
    "head_fwd",
    "head_test",
    "golden_head",
    "build_head_specs",
]
