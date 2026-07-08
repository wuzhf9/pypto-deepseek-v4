"""DeepSeek V4 Flash Hyper-Connections PyPTO kernels."""

import torch
import torch.nn.functional as F

import pypto.language as pl

from models.common import assert_divisible, ceil_div
from models.config import FLASH_CONFIG as M


B = 1
S_DYN = pl.dynamic("S_DYN")
S_PAD_DYN = pl.dynamic("S_PAD_DYN")

HIDDEN = M.dim
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc_dim
HC_SINKHORN_ITERS = M.hc_sinkhorn_iters
RMS_NORM_EPS = M.rms_norm_eps
HC_EPS = M.hc_eps

MIX_PAD = 32
HC_PAD = 8
T_TILE = 16
K_TILE = 128
D_TILE = 512
DEFAULT_SEQ_LEN = 16

HC_DIM_INV = 1.0 / HC_DIM
HIDDEN_BLOCKS = HIDDEN // D_TILE
HC_K_BLOCKS = HC_DIM // K_TILE

assert HC_MULT == 4, "DeepSeek V4 Flash HC PyPTO kernel is specialized for hc_mult=4"
assert MIX_HC <= MIX_PAD
assert HC_MULT <= HC_PAD
assert_divisible(HIDDEN, D_TILE, "HC hidden size")
assert_divisible(HC_DIM, K_TILE, "HC flattened size")


@pl.jit.inline
def hc_pre_fwd(
    x: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    hc_fn_t: pl.Tensor[[HC_DIM, MIX_HC], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    x_mixed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    post: pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32],
    comb: pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32],
):
    """Run ``Block.hc_pre`` and produce the state required by ``hc_post``."""
    x.bind_dynamic(1, S_DYN)
    x_pad.bind_dynamic(1, S_PAD_DYN)
    mixes.bind_dynamic(1, S_PAD_DYN)
    pre.bind_dynamic(1, S_PAD_DYN)
    comb_logits.bind_dynamic(1, S_PAD_DYN)
    x_mixed_pad.bind_dynamic(1, S_PAD_DYN)
    post_pad.bind_dynamic(1, S_PAD_DYN)
    comb_pad.bind_dynamic(1, S_PAD_DYN)
    x_mixed.bind_dynamic(1, S_DYN)
    post.bind_dynamic(1, S_DYN)
    comb.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    padded_tokens = pl.tensor.dim(x_pad, 1)
    x_src_flat = pl.reshape(x, [tokens, HC_DIM])
    x_flat = pl.reshape(x_pad, [padded_tokens, HC_DIM])
    mixes_flat = pl.reshape(mixes, [padded_tokens, MIX_PAD])
    pre_flat = pl.reshape(pre, [padded_tokens, HC_PAD])
    comb_logits_flat = pl.reshape(comb_logits, [padded_tokens, HC_MULT * HC_MULT])
    x_mixed_pad_flat = pl.reshape(x_mixed_pad, [padded_tokens, HIDDEN])
    post_pad_flat = pl.reshape(post_pad, [padded_tokens, HC_PAD])
    comb_pad_flat = pl.reshape(comb_pad, [padded_tokens, HC_MULT * HC_MULT])
    x_mixed_flat = pl.reshape(x_mixed, [tokens, HIDDEN])
    post_flat = pl.reshape(post, [tokens, HC_PAD])
    comb_flat = pl.reshape(comb, [tokens, HC_MULT * HC_MULT])
    token_blocks = padded_tokens // T_TILE

    scale0 = pl.read(hc_scale, [0])
    scale1 = pl.read(hc_scale, [1])
    scale2 = pl.read(hc_scale, [2])

    for t in pl.spmd(padded_tokens, name_hint="hc_pre_pad_x"):
        for kb in pl.range(HC_DIM // D_TILE):
            k0 = kb * D_TILE
            if t < tokens:
                x_row = x_src_flat[t : t + 1, k0 : k0 + D_TILE]
            else:
                x_row = pl.full([1, D_TILE], dtype=pl.BF16, value=0.0)
            x_flat[t : t + 1, k0 : k0 + D_TILE] = x_row

    for tb in pl.range(token_blocks):
        t0 = tb * T_TILE
        valid_tok = T_TILE

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="hc_pre_linear"):
            sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
            mix_acc = pl.create_tensor([T_TILE, MIX_PAD], dtype=pl.FP32)
            for kb in pl.pipeline(0, HC_K_BLOCKS, stage=2):
                k0 = kb * K_TILE
                x_lin = pl.slice(x_flat, [T_TILE, K_TILE], [t0, k0], valid_shape=[valid_tok, K_TILE])
                x_fp32 = pl.cast(x_lin, target_type=pl.FP32)
                x_sq = pl.mul(x_fp32, x_fp32)
                sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(x_sq), [1, T_TILE]))
                w_lin = pl.slice(hc_fn_t, [K_TILE, MIX_PAD], [k0, 0], valid_shape=[K_TILE, MIX_HC])
                if kb == 0:
                    mix_acc = pl.matmul(x_fp32, w_lin, out_dtype=pl.FP32)
                else:
                    mix_acc = pl.matmul_acc(mix_acc, x_fp32, w_lin)

            mean_sq = pl.add(pl.mul(sq_sum, HC_DIM_INV), RMS_NORM_EPS)
            inv_rms = pl.reshape(pl.rsqrt(mean_sq, high_precision=True), [T_TILE, 1])
            mixes_tile = pl.row_expand_mul(mix_acc, inv_rms)
            for row in pl.range(valid_tok):
                mix_row = pl.slice(mixes_tile, [1, MIX_PAD], [row, 0], valid_shape=[1, MIX_PAD])
                mixes_flat[t0 + row : t0 + row + 1, 0:MIX_PAD] = mix_row

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="hc_pre_split"):
            pre_mix = pl.load(
                mixes_flat,
                [t0, 0],
                [T_TILE, HC_PAD],
                valid_shapes=[valid_tok, HC_PAD],
                target_memory=pl.MemorySpace.Vec,
            )
            post_mix = pl.load(
                mixes_flat,
                [t0, HC_MULT],
                [T_TILE, HC_PAD],
                valid_shapes=[valid_tok, HC_PAD],
                target_memory=pl.MemorySpace.Vec,
            )
            comb_mix = pl.load(
                mixes_flat,
                [t0, HC_MULT * 2],
                [T_TILE, HC_MULT * HC_MULT],
                valid_shapes=[valid_tok, HC_MULT * HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )

            pre_base_tile = pl.load(hc_base, [0], [HC_PAD], target_memory=pl.MemorySpace.Vec)
            pre_base = pl.reshape(pre_base_tile, [1, HC_PAD])
            pre_scaled = pl.mul(pre_mix, scale0)
            pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
            pre_sig = pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0))
            pre_tile = pl.add(pre_sig, HC_EPS)

            post_base_tile = pl.load(hc_base, [HC_MULT], [HC_PAD], target_memory=pl.MemorySpace.Vec)
            post_base = pl.reshape(post_base_tile, [1, HC_PAD])
            post_scaled = pl.mul(post_mix, scale1)
            post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
            post_sig = pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0))
            post_tile = pl.mul(post_sig, 2.0)

            comb_base_tile = pl.load(
                hc_base,
                [HC_MULT * 2],
                [HC_MULT * HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            comb_base = pl.reshape(comb_base_tile, [1, HC_MULT * HC_MULT])
            comb_scaled = pl.mul(comb_mix, scale2)
            comb_logits_tile = pl.add(comb_scaled, pl.col_expand(comb_scaled, comb_base))

            pre_out = pl.set_validshape(pre_tile, valid_tok, HC_PAD)
            post_out = pl.set_validshape(post_tile, valid_tok, HC_MULT)
            comb_logits_out = pl.set_validshape(comb_logits_tile, valid_tok, HC_MULT * HC_MULT)
            pl.store(pre_out, [t0, 0], pre_flat)
            pl.store(post_out, [t0, 0], post_pad_flat)
            pl.store(comb_logits_out, [t0, 0], comb_logits_flat)

            row0 = pl.load(
                comb_logits_flat,
                [t0, 0 * HC_MULT],
                [T_TILE, HC_PAD],
                valid_shapes=[valid_tok, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            row1 = pl.load(
                comb_logits_flat,
                [t0, 1 * HC_MULT],
                [T_TILE, HC_PAD],
                valid_shapes=[valid_tok, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            row2 = pl.load(
                comb_logits_flat,
                [t0, 2 * HC_MULT],
                [T_TILE, HC_PAD],
                valid_shapes=[valid_tok, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )
            row3 = pl.load(
                comb_logits_flat,
                [t0, 3 * HC_MULT],
                [T_TILE, HC_PAD],
                valid_shapes=[valid_tok, HC_MULT],
                target_memory=pl.MemorySpace.Vec,
            )

            row0_p = pl.fillpad(row0, pad_value=pl.PadValue.min)
            row1_p = pl.fillpad(row1, pad_value=pl.PadValue.min)
            row2_p = pl.fillpad(row2, pad_value=pl.PadValue.min)
            row3_p = pl.fillpad(row3, pad_value=pl.PadValue.min)

            row_max_tmp = pl.create_tile([T_TILE, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            row_sum_tmp = pl.create_tile([T_TILE, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            row0_max = pl.row_max(row0_p, row_max_tmp)
            row1_max = pl.row_max(row1_p, row_max_tmp)
            row2_max = pl.row_max(row2_p, row_max_tmp)
            row3_max = pl.row_max(row3_p, row_max_tmp)
            row0_exp = pl.exp(pl.row_expand_sub(row0_p, row0_max))
            row1_exp = pl.exp(pl.row_expand_sub(row1_p, row1_max))
            row2_exp = pl.exp(pl.row_expand_sub(row2_p, row2_max))
            row3_exp = pl.exp(pl.row_expand_sub(row3_p, row3_max))
            row0_sum = pl.row_sum(row0_exp, row_sum_tmp)
            row1_sum = pl.row_sum(row1_exp, row_sum_tmp)
            row2_sum = pl.row_sum(row2_exp, row_sum_tmp)
            row3_sum = pl.row_sum(row3_exp, row_sum_tmp)
            row0_soft = pl.add(pl.row_expand_div(row0_exp, row0_sum), HC_EPS)
            row1_soft = pl.add(pl.row_expand_div(row1_exp, row1_sum), HC_EPS)
            row2_soft = pl.add(pl.row_expand_div(row2_exp, row2_sum), HC_EPS)
            row3_soft = pl.add(pl.row_expand_div(row3_exp, row3_sum), HC_EPS)

            row0_valid = pl.set_validshape(row0_soft, T_TILE, HC_MULT)
            row1_valid = pl.set_validshape(row1_soft, T_TILE, HC_MULT)
            row2_valid = pl.set_validshape(row2_soft, T_TILE, HC_MULT)
            row3_valid = pl.set_validshape(row3_soft, T_TILE, HC_MULT)
            row0_cur = pl.fillpad(row0_valid, pad_value=pl.PadValue.zero)
            row1_cur = pl.fillpad(row1_valid, pad_value=pl.PadValue.zero)
            row2_cur = pl.fillpad(row2_valid, pad_value=pl.PadValue.zero)
            row3_cur = pl.fillpad(row3_valid, pad_value=pl.PadValue.zero)

            col_sum = pl.add(pl.add(row0_cur, row1_cur), pl.add(row2_cur, row3_cur))
            col_sum = pl.add(col_sum, HC_EPS)
            row0_cur = pl.div(row0_cur, col_sum)
            row1_cur = pl.div(row1_cur, col_sum)
            row2_cur = pl.div(row2_cur, col_sum)
            row3_cur = pl.div(row3_cur, col_sum)

            row_sum_tmp_iter = pl.create_tile([T_TILE, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
            for _sk_it in pl.pipeline(HC_SINKHORN_ITERS - 1, stage=2):
                row0_rowsum = pl.add(pl.row_sum(row0_cur, row_sum_tmp_iter), HC_EPS)
                row1_rowsum = pl.add(pl.row_sum(row1_cur, row_sum_tmp_iter), HC_EPS)
                row2_rowsum = pl.add(pl.row_sum(row2_cur, row_sum_tmp_iter), HC_EPS)
                row3_rowsum = pl.add(pl.row_sum(row3_cur, row_sum_tmp_iter), HC_EPS)
                row0_norm = pl.row_expand_div(row0_cur, row0_rowsum)
                row1_norm = pl.row_expand_div(row1_cur, row1_rowsum)
                row2_norm = pl.row_expand_div(row2_cur, row2_rowsum)
                row3_norm = pl.row_expand_div(row3_cur, row3_rowsum)
                col_sum = pl.add(pl.add(row0_norm, row1_norm), pl.add(row2_norm, row3_norm))
                col_sum = pl.add(col_sum, HC_EPS)
                row0_cur = pl.div(row0_norm, col_sum)
                row1_cur = pl.div(row1_norm, col_sum)
                row2_cur = pl.div(row2_norm, col_sum)
                row3_cur = pl.div(row3_norm, col_sum)

            row0_out = pl.set_validshape(row0_cur, valid_tok, HC_MULT)
            row1_out = pl.set_validshape(row1_cur, valid_tok, HC_MULT)
            row2_out = pl.set_validshape(row2_cur, valid_tok, HC_MULT)
            row3_out = pl.set_validshape(row3_cur, valid_tok, HC_MULT)
            pl.store(row0_out, [t0, 0 * HC_MULT], comb_pad_flat)
            pl.store(row1_out, [t0, 1 * HC_MULT], comb_pad_flat)
            pl.store(row2_out, [t0, 2 * HC_MULT], comb_pad_flat)
            pl.store(row3_out, [t0, 3 * HC_MULT], comb_pad_flat)

    for task in pl.spmd(token_blocks * HIDDEN_BLOCKS, name_hint="hc_pre_mix_x"):
        tb = task // HIDDEN_BLOCKS
        db = task % HIDDEN_BLOCKS
        t0 = tb * T_TILE
        d0 = db * D_TILE
        valid_tok = T_TILE
        mix_pre = pl.slice(pre_flat, [T_TILE, HC_PAD], [t0, 0], valid_shape=[valid_tok, HC_MULT])
        mix_pre_t = pl.transpose(mix_pre, axis1=0, axis2=1)
        pre0 = pl.reshape(mix_pre_t[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(mix_pre_t[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(mix_pre_t[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(mix_pre_t[3:4, 0:T_TILE], [T_TILE, 1])
        x0 = pl.cast(
            pl.slice(x_flat, [T_TILE, D_TILE], [t0, 0 * HIDDEN + d0], valid_shape=[valid_tok, D_TILE]),
            target_type=pl.FP32,
        )
        x1 = pl.cast(
            pl.slice(x_flat, [T_TILE, D_TILE], [t0, 1 * HIDDEN + d0], valid_shape=[valid_tok, D_TILE]),
            target_type=pl.FP32,
        )
        x2 = pl.cast(
            pl.slice(x_flat, [T_TILE, D_TILE], [t0, 2 * HIDDEN + d0], valid_shape=[valid_tok, D_TILE]),
            target_type=pl.FP32,
        )
        x3 = pl.cast(
            pl.slice(x_flat, [T_TILE, D_TILE], [t0, 3 * HIDDEN + d0], valid_shape=[valid_tok, D_TILE]),
            target_type=pl.FP32,
        )
        y0 = pl.row_expand_mul(x0, pre0)
        y1 = pl.row_expand_mul(x1, pre1)
        y2 = pl.row_expand_mul(x2, pre2)
        y3 = pl.row_expand_mul(x3, pre3)
        y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
        y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
        for row in pl.range(valid_tok):
            y_row = pl.slice(y_bf16, [1, D_TILE], [row, 0], valid_shape=[1, D_TILE])
            x_mixed_pad_flat[t0 + row : t0 + row + 1, d0 : d0 + D_TILE] = y_row

    for t in pl.range(tokens):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="hc_pre_copy_out"):
            for db in pl.range(HIDDEN_BLOCKS):
                d0 = db * D_TILE
                x_mixed_flat[t : t + 1, d0 : d0 + D_TILE] = x_mixed_pad_flat[t : t + 1, d0 : d0 + D_TILE]
            post_flat[t : t + 1, 0:HC_PAD] = post_pad_flat[t : t + 1, 0:HC_PAD]
            comb_flat[t : t + 1, 0 : HC_MULT * HC_MULT] = comb_pad_flat[t : t + 1, 0 : HC_MULT * HC_MULT]
    return pl.reshape(x_mixed_flat, [B, tokens, HIDDEN])


@pl.jit
def hc_pre_test(
    x: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    hc_fn_t: pl.Tensor[[HC_DIM, MIX_HC], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    x_mixed: pl.Out[pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16]],
    post: pl.Out[pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32]],
    comb: pl.Out[pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32]],
):
    x_mixed = hc_pre_fwd(
        x,
        x_pad,
        hc_fn_t,
        hc_scale,
        hc_base,
        mixes,
        pre,
        comb_logits,
        x_mixed_pad,
        post_pad,
        comb_pad,
        x_mixed,
        post,
        comb,
    )
    return x_mixed


@pl.jit.inline
def hc_post_fwd(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    residual: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    post: pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32],
    comb: pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32],
    out: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
):
    """Run ``Block.hc_post`` with padded ``post`` from ``hc_pre_fwd``."""
    x.bind_dynamic(1, S_DYN)
    residual.bind_dynamic(1, S_DYN)
    post.bind_dynamic(1, S_DYN)
    comb.bind_dynamic(1, S_DYN)
    out.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    x_flat = pl.reshape(x, [tokens, HIDDEN])
    residual_flat = pl.reshape(residual, [tokens * HC_MULT, HIDDEN])
    post_flat = pl.reshape(post, [tokens, HC_PAD])
    comb_flat = pl.reshape(comb, [tokens, HC_MULT * HC_MULT])
    out_flat = pl.reshape(out, [tokens * HC_MULT, HIDDEN])

    for t in pl.range(tokens):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="hc_post"):
            for j in pl.range(HC_MULT):
                post_j = pl.read(post_flat, [t, j])
                comb_0j = pl.read(comb_flat, [t, 0 * HC_MULT + j])
                comb_1j = pl.read(comb_flat, [t, 1 * HC_MULT + j])
                comb_2j = pl.read(comb_flat, [t, 2 * HC_MULT + j])
                comb_3j = pl.read(comb_flat, [t, 3 * HC_MULT + j])
                dst = t * HC_MULT + j
                src0 = t * HC_MULT + 0
                src1 = t * HC_MULT + 1
                src2 = t * HC_MULT + 2
                src3 = t * HC_MULT + 3
                for db in pl.range(HIDDEN_BLOCKS):
                    d0 = db * D_TILE
                    x_tile = pl.cast(x_flat[t : t + 1, d0 : d0 + D_TILE], target_type=pl.FP32)
                    residual0 = pl.cast(residual_flat[src0 : src0 + 1, d0 : d0 + D_TILE], target_type=pl.FP32)
                    residual1 = pl.cast(residual_flat[src1 : src1 + 1, d0 : d0 + D_TILE], target_type=pl.FP32)
                    residual2 = pl.cast(residual_flat[src2 : src2 + 1, d0 : d0 + D_TILE], target_type=pl.FP32)
                    residual3 = pl.cast(residual_flat[src3 : src3 + 1, d0 : d0 + D_TILE], target_type=pl.FP32)
                    y = pl.mul(x_tile, post_j)
                    y = pl.add(y, pl.mul(residual0, comb_0j))
                    y = pl.add(y, pl.mul(residual1, comb_1j))
                    y = pl.add(y, pl.mul(residual2, comb_2j))
                    y = pl.add(y, pl.mul(residual3, comb_3j))
                    out_flat[dst : dst + 1, d0 : d0 + D_TILE] = pl.cast(y, target_type=pl.BF16, mode="rint")

    return pl.reshape(out_flat, [B, tokens, HC_MULT, HIDDEN])


@pl.jit
def hc_post_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    residual: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    post: pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32],
    comb: pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32],
    out: pl.Out[pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16]],
):
    out = hc_post_fwd(x, residual, post, comb, out)
    return out


def split_sinkhorn_golden(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    hc_mult: int | None = None,
    sinkhorn_iters: int | None = None,
    eps: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch port of official ``hc_split_sinkhorn``."""
    hc = HC_MULT if hc_mult is None else hc_mult
    iters = HC_SINKHORN_ITERS if sinkhorn_iters is None else sinkhorn_iters
    sinkhorn_eps = HC_EPS if eps is None else eps

    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + sinkhorn_eps
    post = 2 * torch.sigmoid(mixes[..., hc : hc * 2] * hc_scale[1] + hc_base[hc : hc * 2])
    comb = (mixes[..., hc * 2 :] * hc_scale[2] + hc_base[hc * 2 :]).view(*mixes.shape[:-1], hc, hc)

    comb = torch.softmax(comb, dim=-1) + sinkhorn_eps
    comb = comb / (comb.sum(-2, keepdim=True) + sinkhorn_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + sinkhorn_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + sinkhorn_eps)
    return pre, post, comb


def golden_hc_pre(tensors):
    """Torch reference for ``Block.hc_pre``."""
    x = tensors["x"].float()
    hc_fn_t = tensors["hc_fn_t"].float()
    hc_scale = tensors["hc_scale"].float()
    hc_base = tensors["hc_base"].float()

    x_flat = x.flatten(2)
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + RMS_NORM_EPS)
    mixes = F.linear(x_flat, hc_fn_t.transpose(0, 1).contiguous()) * rsqrt
    pre, post, comb = split_sinkhorn_golden(mixes, hc_scale, hc_base)
    x_mixed = torch.sum(pre.unsqueeze(-1) * x, dim=2).to(tensors["x_mixed"].dtype)

    tensors["x_mixed"][:] = x_mixed
    if tensors["post"].shape[-1] == HC_PAD:
        tensors["post"].zero_()
        tensors["post"][..., :HC_MULT] = post.to(tensors["post"].dtype)
    else:
        tensors["post"][:] = post.to(tensors["post"].dtype)
    tensors["comb"][:] = comb.reshape(*comb.shape[:-2], HC_MULT * HC_MULT).to(tensors["comb"].dtype)
    if "mixes" in tensors:
        tensors["mixes"][:, : mixes.shape[1], :MIX_HC] = mixes.to(tensors["mixes"].dtype)
    if "pre" in tensors:
        tensors["pre"][:, : pre.shape[1], :HC_MULT] = pre.to(tensors["pre"].dtype)


def golden_hc_post(tensors):
    """Torch reference for ``Block.hc_post``."""
    x = tensors["x"]
    residual = tensors["residual"]
    post = tensors["post"][..., :HC_MULT].float()
    comb = tensors["comb"].float().view(*tensors["comb"].shape[:-1], HC_MULT, HC_MULT)

    y = post.unsqueeze(-1) * x.float().unsqueeze(-2)
    y = y + torch.sum(comb.unsqueeze(-1) * residual.float().unsqueeze(-2), dim=2)
    tensors["out"][:] = y.to(tensors["out"].dtype)


def _build_hc_pre_specs(seq_len: int):
    from models.golden import TensorSpec

    seq_pad = ceil_div(seq_len, T_TILE) * T_TILE

    def init_x():
        return (torch.randn(B, seq_len, HC_MULT, HIDDEN, dtype=torch.float32) * 0.5).to(torch.bfloat16)

    def init_hc_fn_t():
        return (torch.randn(HC_DIM, MIX_HC, dtype=torch.float32) * (HC_DIM**-0.5)).contiguous()

    def init_hc_scale():
        return torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

    def init_hc_base():
        return torch.zeros(MIX_HC, dtype=torch.float32)

    return [
        TensorSpec("x", [B, seq_len, HC_MULT, HIDDEN], torch.bfloat16, init_value=init_x),
        TensorSpec("x_pad", [B, seq_pad, HC_MULT, HIDDEN], torch.bfloat16),
        TensorSpec("hc_fn_t", [HC_DIM, MIX_HC], torch.float32, init_value=init_hc_fn_t),
        TensorSpec("hc_scale", [3], torch.float32, init_value=init_hc_scale),
        TensorSpec("hc_base", [MIX_HC], torch.float32, init_value=init_hc_base),
        TensorSpec("mixes", [B, seq_pad, MIX_PAD], torch.float32),
        TensorSpec("pre", [B, seq_pad, HC_PAD], torch.float32),
        TensorSpec("comb_logits", [B, seq_pad, HC_MULT * HC_MULT], torch.float32),
        TensorSpec("x_mixed_pad", [B, seq_pad, HIDDEN], torch.bfloat16),
        TensorSpec("post_pad", [B, seq_pad, HC_PAD], torch.float32),
        TensorSpec("comb_pad", [B, seq_pad, HC_MULT * HC_MULT], torch.float32),
        TensorSpec("x_mixed", [B, seq_len, HIDDEN], torch.bfloat16, is_output=True),
        TensorSpec("post", [B, seq_len, HC_PAD], torch.float32, is_output=True),
        TensorSpec("comb", [B, seq_len, HC_MULT * HC_MULT], torch.float32, is_output=True),
    ]


def build_hc_pre_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _build_hc_pre_specs(seq_len)


def build_hc_post_specs(seq_len: int = DEFAULT_SEQ_LEN):
    from models.golden import TensorSpec

    def init_x():
        return (torch.randn(B, seq_len, HIDDEN, dtype=torch.float32) * 0.5).to(torch.bfloat16)

    def init_residual():
        return (torch.randn(B, seq_len, HC_MULT, HIDDEN, dtype=torch.float32) * 0.5).to(torch.bfloat16)

    def init_post():
        value = torch.zeros(B, seq_len, HC_PAD, dtype=torch.float32)
        value[..., :HC_MULT] = torch.sigmoid(torch.randn(B, seq_len, HC_MULT, dtype=torch.float32)) * 2.0
        return value

    def init_comb():
        logits = torch.randn(B, seq_len, HC_MULT, HC_MULT, dtype=torch.float32)
        value = torch.softmax(logits, dim=-1)
        return value.reshape(B, seq_len, HC_MULT * HC_MULT)

    return [
        TensorSpec("x", [B, seq_len, HIDDEN], torch.bfloat16, init_value=init_x),
        TensorSpec("residual", [B, seq_len, HC_MULT, HIDDEN], torch.bfloat16, init_value=init_residual),
        TensorSpec("post", [B, seq_len, HC_PAD], torch.float32, init_value=init_post),
        TensorSpec("comb", [B, seq_len, HC_MULT * HC_MULT], torch.float32, init_value=init_comb),
        TensorSpec("out", [B, seq_len, HC_MULT, HIDDEN], torch.bfloat16, is_output=True),
    ]


def main() -> int:
    import argparse

    from models.golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 Flash Hyper-Connections validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3sim", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-s", "--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--case", choices=["all", "pre", "post"], default="all")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    runtime_cfg = {
        "platform": args.platform,
        "device_id": args.device,
        "enable_l2_swimlane": args.enable_l2_swimlane,
    }
    all_cases = {
        "pre": (
            "hc-pre",
            hc_pre_test,
            build_hc_pre_specs,
            golden_hc_pre,
            {
                "x_mixed": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
                "post": ratio_allclose(atol=2.5e-5, rtol=5e-3, max_error_ratio=0.001),
                "comb": ratio_allclose(atol=2.5e-5, rtol=5e-3, max_error_ratio=0.001),
            },
        ),
        "post": (
            "hc-post",
            hc_post_test,
            build_hc_post_specs,
            golden_hc_post,
            {
                "out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
            },
        ),
    }
    if args.case == "all":
        cases = [all_cases["pre"], all_cases["post"]]
    else:
        cases = [all_cases[args.case]]

    failed = False
    for name, fn, build_specs, golden_fn, compare_fn in cases:
        print(f"[CASE] {name}", flush=True)
        result = run_jit(
            fn=fn,
            specs=build_specs(args.seq_len),
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
    "S_PAD_DYN",
    "HIDDEN",
    "HC_MULT",
    "HC_DIM",
    "MIX_HC",
    "HC_SINKHORN_ITERS",
    "RMS_NORM_EPS",
    "HC_EPS",
    "MIX_PAD",
    "HC_PAD",
    "hc_pre_fwd",
    "hc_post_fwd",
    "hc_pre_test",
    "hc_post_test",
    "split_sinkhorn_golden",
    "golden_hc_pre",
    "golden_hc_post",
    "build_hc_pre_specs",
    "build_hc_post_specs",
]
