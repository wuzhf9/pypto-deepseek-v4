"""Split decode Block kernels and golden logic for selected-expert execution."""

import torch
import pypto.language as pl

from models import block as block_golden
from models.attention_csa import attention_csa_decode_fwd
from models.attention_hca import attention_hca_decode_fwd
from models.attention_swa import attention_swa_decode_fwd
from models.common import ceil_div
from models.gate import gate_hash_fwd, gate_topk_fwd, golden_gate_forward
from models.hc import HC_PAD, MIX_PAD, T_TILE as HC_T_TILE
from models.hc import hc_post_fwd, hc_pre_fwd
from models.moe import ROUTE_SCALE, golden_moe_selected_decode_experts_forward, moe_selected_decode_experts_fwd
from models.rmsnorm import rmsnorm_4096


B = block_golden.B
S_DYN = block_golden.S_DYN
S_PAD_DYN = block_golden.S_PAD_DYN

HIDDEN = block_golden.HIDDEN
HC_MULT = block_golden.HC_MULT
HC_DIM = block_golden.HC_DIM
MIX_HC = block_golden.MIX_HC
Q_LORA_RANK = block_golden.Q_LORA_RANK
N_HEADS = block_golden.N_HEADS
HEAD_DIM = block_golden.HEAD_DIM
ATTN_Q_OUT = block_golden.ATTN_Q_OUT
O_GROUPS = block_golden.O_GROUPS
O_LORA_RANK = block_golden.O_LORA_RANK
HEADS_PER_GROUP = block_golden.HEADS_PER_GROUP
O_GROUP_IN = block_golden.O_GROUP_IN
ATTN_OUT_IN = block_golden.ATTN_OUT_IN
ROPE_HALF = block_golden.ROPE_HALF
WINDOW_SIZE = block_golden.WINDOW_SIZE
TOPK_SWA = block_golden.TOPK_SWA
COMPRESS_RATIO128 = block_golden.COMPRESS_RATIO128
TOPK_HCA = block_golden.TOPK_HCA
TOPK_HCA_TOTAL = block_golden.TOPK_HCA_TOTAL
INDEX_N_HEADS = block_golden.INDEX_N_HEADS
INDEX_HEAD_DIM = block_golden.INDEX_HEAD_DIM
INDEX_Q_OUT = block_golden.INDEX_Q_OUT
INDEX_PROJ_DIM = block_golden.INDEX_PROJ_DIM
RATIO4_STATE_ROWS = block_golden.RATIO4_STATE_ROWS
ATTN_PROJ_DIM = block_golden.ATTN_PROJ_DIM
COMPRESS_RATIO4 = block_golden.COMPRESS_RATIO4
TOPK_CSA_COMPRESSED = block_golden.TOPK_CSA_COMPRESSED
MOE_INTER_DIM = block_golden.MOE_INTER_DIM
N_EXPERTS = block_golden.N_EXPERTS
TOPK = block_golden.TOPK
VOCAB = block_golden.VOCAB
DEFAULT_DECODE_START_POS = block_golden.DEFAULT_DECODE_START_POS


@pl.jit
def swa_hash_selected_decode_pre_moe_fwd(
    x: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    kv_cache: pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16],
    cache_pos: pl.Tensor[[1], pl.INT32],
    attn_hc_fn_t: pl.Tensor[[HC_DIM, MIX_HC], pl.FP32],
    attn_hc_scale: pl.Tensor[[3], pl.FP32],
    attn_hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[HIDDEN], pl.BF16],
    wq_a_t: pl.Tensor[[HIDDEN, Q_LORA_RANK], pl.BF16],
    q_norm_w: pl.Tensor[[Q_LORA_RANK], pl.BF16],
    wq_b_t: pl.Tensor[[Q_LORA_RANK, ATTN_Q_OUT], pl.BF16],
    wkv_t: pl.Tensor[[HIDDEN, HEAD_DIM], pl.BF16],
    kv_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[N_HEADS], pl.FP32],
    topk_idxs: pl.Tensor[[B, S_DYN, TOPK_SWA], pl.INT32],
    wo_a_t: pl.Tensor[[O_GROUP_IN, ATTN_OUT_IN], pl.BF16],
    wo_b_t: pl.Tensor[[ATTN_OUT_IN, HIDDEN], pl.BF16],
    cos: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    ffn_hc_fn_t: pl.Tensor[[HC_DIM, MIX_HC], pl.FP32],
    ffn_hc_scale: pl.Tensor[[3], pl.FP32],
    ffn_hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[HIDDEN], pl.BF16],
    gate_w_t: pl.Tensor[[HIDDEN, N_EXPERTS], pl.BF16],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[B, S_DYN], pl.INT64],
    attn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    attn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    attn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    attn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    kv_cache_out: pl.Out[pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16]],
    ffn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    ffn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    ffn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    ffn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_out: pl.Out[pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16]],
    ffn_normed: pl.Out[pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16]],
    indices: pl.Out[pl.Tensor[[B, S_DYN, TOPK], pl.INT32]],
    weights: pl.Out[pl.Tensor[[B, S_DYN, TOPK], pl.FP32]],
    ffn_hc_post: pl.Out[pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32]],
    ffn_hc_comb: pl.Out[pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32]],
):
    """Run SWA hash decode block up to FFN MoE gate."""
    x.bind_dynamic(1, S_DYN)
    tokens = pl.tensor.dim(x, 1)
    attn_hc_x_mixed = pl.create_tensor([B, tokens, HIDDEN], dtype=pl.BF16)
    attn_hc_post = pl.create_tensor([B, tokens, HC_PAD], dtype=pl.FP32)
    attn_hc_comb = pl.create_tensor([B, tokens, HC_MULT * HC_MULT], dtype=pl.FP32)
    attn_normed = pl.create_tensor([B, tokens, HIDDEN], dtype=pl.BF16)
    attn_out = pl.create_tensor([B, tokens, HIDDEN], dtype=pl.BF16)
    ffn_hc_x_mixed = pl.create_tensor([B, tokens, HIDDEN], dtype=pl.BF16)

    hc_pre_fwd(
        x,
        attn_hc_x_pad,
        attn_hc_fn_t,
        attn_hc_scale,
        attn_hc_base,
        attn_hc_mixes,
        attn_hc_pre,
        attn_hc_comb_logits,
        attn_hc_x_mixed_pad,
        attn_hc_post_pad,
        attn_hc_comb_pad,
        attn_hc_x_mixed,
        attn_hc_post,
        attn_hc_comb,
    )
    rmsnorm_4096(attn_hc_x_mixed, attn_norm_w, attn_normed)
    attention_swa_decode_fwd(
        attn_normed,
        kv_cache,
        cache_pos,
        wq_a_t,
        q_norm_w,
        wq_b_t,
        wkv_t,
        kv_norm_w,
        attn_sink,
        topk_idxs,
        wo_a_t,
        wo_b_t,
        cos,
        sin,
        kv_cache_out,
        attn_out,
    )
    hc_post_fwd(attn_out, x, attn_hc_post, attn_hc_comb, attn_hc_out)

    hc_pre_fwd(
        attn_hc_out,
        ffn_hc_x_pad,
        ffn_hc_fn_t,
        ffn_hc_scale,
        ffn_hc_base,
        ffn_hc_mixes,
        ffn_hc_pre,
        ffn_hc_comb_logits,
        ffn_hc_x_mixed_pad,
        ffn_hc_post_pad,
        ffn_hc_comb_pad,
        ffn_hc_x_mixed,
        ffn_hc_post,
        ffn_hc_comb,
    )
    rmsnorm_4096(ffn_hc_x_mixed, ffn_norm_w, ffn_normed)
    gate_hash_fwd(ffn_normed, gate_w_t, tid2eid, input_ids, indices, weights)
    return kv_cache_out, attn_hc_out, ffn_normed, indices, weights, ffn_hc_post, ffn_hc_comb


@pl.jit
def csa_hash_selected_decode_pre_moe_fwd(
    x: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    kv_cache: pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16],
    attn_comp_kv_state: pl.Tensor[[B, RATIO4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_score_state: pl.Tensor[[B, RATIO4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_cache: pl.Tensor[[B, TOPK_CSA_COMPRESSED, HEAD_DIM], pl.BF16],
    idx_kv_cache_in: pl.Tensor[[B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM], pl.BF16],
    idx_comp_kv_state: pl.Tensor[[B, RATIO4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    idx_comp_score_state: pl.Tensor[[B, RATIO4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    cache_pos: pl.Tensor[[1], pl.INT32],
    comp_slot: pl.Tensor[[1], pl.INT32],
    comp_cache_slot: pl.Tensor[[1], pl.INT32],
    comp_should_compress: pl.Tensor[[1], pl.INT32],
    attn_hc_fn_t: pl.Tensor[[HC_DIM, MIX_HC], pl.FP32],
    attn_hc_scale: pl.Tensor[[3], pl.FP32],
    attn_hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[HIDDEN], pl.BF16],
    wq_a_t: pl.Tensor[[HIDDEN, Q_LORA_RANK], pl.BF16],
    q_norm_w: pl.Tensor[[Q_LORA_RANK], pl.BF16],
    wq_b_t: pl.Tensor[[Q_LORA_RANK, ATTN_Q_OUT], pl.BF16],
    wkv_t: pl.Tensor[[HIDDEN, HEAD_DIM], pl.BF16],
    kv_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[N_HEADS], pl.FP32],
    window_topk_idxs: pl.Tensor[[B, S_DYN, TOPK_SWA], pl.INT32],
    wo_a_t: pl.Tensor[[O_GROUP_IN, ATTN_OUT_IN], pl.BF16],
    wo_b_t: pl.Tensor[[ATTN_OUT_IN, HIDDEN], pl.BF16],
    cos: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    attn_comp_wkv_t: pl.Tensor[[HIDDEN, ATTN_PROJ_DIM], pl.BF16],
    attn_comp_wgate_t: pl.Tensor[[HIDDEN, ATTN_PROJ_DIM], pl.BF16],
    attn_comp_ape: pl.Tensor[[COMPRESS_RATIO4, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_comp_cos: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    attn_comp_sin: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    idx_wq_b_t: pl.Tensor[[Q_LORA_RANK, INDEX_Q_OUT], pl.BF16],
    idx_weights_proj_t: pl.Tensor[[HIDDEN, INDEX_N_HEADS], pl.BF16],
    idx_offset: pl.Tensor[[1], pl.INT32],
    idx_comp_wkv_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    idx_comp_wgate_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    idx_comp_ape: pl.Tensor[[COMPRESS_RATIO4, INDEX_PROJ_DIM], pl.FP32],
    idx_comp_norm_w: pl.Tensor[[INDEX_HEAD_DIM], pl.BF16],
    idx_comp_cos: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    idx_comp_sin: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    ffn_hc_fn_t: pl.Tensor[[HC_DIM, MIX_HC], pl.FP32],
    ffn_hc_scale: pl.Tensor[[3], pl.FP32],
    ffn_hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[HIDDEN], pl.BF16],
    gate_w_t: pl.Tensor[[HIDDEN, N_EXPERTS], pl.BF16],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[B, S_DYN], pl.INT64],
    attn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    attn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    attn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    attn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    kv_pool: pl.Tensor[[B, WINDOW_SIZE + TOPK_CSA_COMPRESSED, HEAD_DIM], pl.BF16],
    kv_cache_out: pl.Out[pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16]],
    attn_comp_kv_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32]],
    attn_comp_score_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32]],
    attn_comp_cache_out: pl.Out[pl.Tensor[[B, TOPK_CSA_COMPRESSED, HEAD_DIM], pl.BF16]],
    idx_kv_cache_out: pl.Out[pl.Tensor[[B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM], pl.BF16]],
    idx_comp_kv_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
    idx_comp_score_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
    ffn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    ffn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    ffn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    ffn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_out: pl.Out[pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16]],
    ffn_normed: pl.Out[pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16]],
    indices: pl.Out[pl.Tensor[[B, S_DYN, TOPK], pl.INT32]],
    weights: pl.Out[pl.Tensor[[B, S_DYN, TOPK], pl.FP32]],
    ffn_hc_post: pl.Out[pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32]],
    ffn_hc_comb: pl.Out[pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32]],
):
    """Run CSA hash decode block up to FFN MoE gate."""
    x.bind_dynamic(1, S_DYN)
    tokens = pl.tensor.dim(x, 1)
    attn_hc_x_mixed = pl.create_tensor([B, tokens, HIDDEN], dtype=pl.BF16)
    attn_hc_post = pl.create_tensor([B, tokens, HC_PAD], dtype=pl.FP32)
    attn_hc_comb = pl.create_tensor([B, tokens, HC_MULT * HC_MULT], dtype=pl.FP32)
    attn_normed = pl.create_tensor([B, tokens, HIDDEN], dtype=pl.BF16)
    attn_out = pl.create_tensor([B, tokens, HIDDEN], dtype=pl.BF16)
    ffn_hc_x_mixed = pl.create_tensor([B, tokens, HIDDEN], dtype=pl.BF16)

    hc_pre_fwd(
        x,
        attn_hc_x_pad,
        attn_hc_fn_t,
        attn_hc_scale,
        attn_hc_base,
        attn_hc_mixes,
        attn_hc_pre,
        attn_hc_comb_logits,
        attn_hc_x_mixed_pad,
        attn_hc_post_pad,
        attn_hc_comb_pad,
        attn_hc_x_mixed,
        attn_hc_post,
        attn_hc_comb,
    )
    rmsnorm_4096(attn_hc_x_mixed, attn_norm_w, attn_normed)
    attention_csa_decode_fwd(
        attn_normed,
        kv_cache,
        attn_comp_kv_state,
        attn_comp_score_state,
        attn_comp_cache,
        idx_kv_cache_in,
        idx_comp_kv_state,
        idx_comp_score_state,
        cache_pos,
        comp_slot,
        comp_cache_slot,
        comp_should_compress,
        wq_a_t,
        q_norm_w,
        wq_b_t,
        wkv_t,
        kv_norm_w,
        attn_sink,
        window_topk_idxs,
        wo_a_t,
        wo_b_t,
        cos,
        sin,
        attn_comp_wkv_t,
        attn_comp_wgate_t,
        attn_comp_ape,
        attn_comp_norm_w,
        attn_comp_cos,
        attn_comp_sin,
        idx_wq_b_t,
        idx_weights_proj_t,
        idx_offset,
        idx_comp_wkv_t,
        idx_comp_wgate_t,
        idx_comp_ape,
        idx_comp_norm_w,
        idx_comp_cos,
        idx_comp_sin,
        kv_cache_out,
        attn_comp_kv_state_out,
        attn_comp_score_state_out,
        attn_comp_cache_out,
        idx_kv_cache_out,
        idx_comp_kv_state_out,
        idx_comp_score_state_out,
        attn_out,
    )
    hc_post_fwd(attn_out, x, attn_hc_post, attn_hc_comb, attn_hc_out)

    hc_pre_fwd(
        attn_hc_out,
        ffn_hc_x_pad,
        ffn_hc_fn_t,
        ffn_hc_scale,
        ffn_hc_base,
        ffn_hc_mixes,
        ffn_hc_pre,
        ffn_hc_comb_logits,
        ffn_hc_x_mixed_pad,
        ffn_hc_post_pad,
        ffn_hc_comb_pad,
        ffn_hc_x_mixed,
        ffn_hc_post,
        ffn_hc_comb,
    )
    rmsnorm_4096(ffn_hc_x_mixed, ffn_norm_w, ffn_normed)
    gate_hash_fwd(ffn_normed, gate_w_t, tid2eid, input_ids, indices, weights)
    return (
        kv_cache_out,
        attn_comp_kv_state_out,
        attn_comp_score_state_out,
        attn_comp_cache_out,
        idx_kv_cache_out,
        idx_comp_kv_state_out,
        idx_comp_score_state_out,
        attn_hc_out,
        ffn_normed,
        indices,
        weights,
        ffn_hc_post,
        ffn_hc_comb,
    )


@pl.jit
def csa_topk_selected_decode_pre_moe_fwd(
    x: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    kv_cache: pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16],
    attn_comp_kv_state: pl.Tensor[[B, RATIO4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_score_state: pl.Tensor[[B, RATIO4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_cache: pl.Tensor[[B, TOPK_CSA_COMPRESSED, HEAD_DIM], pl.BF16],
    idx_kv_cache_in: pl.Tensor[[B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM], pl.BF16],
    idx_comp_kv_state: pl.Tensor[[B, RATIO4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    idx_comp_score_state: pl.Tensor[[B, RATIO4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    cache_pos: pl.Tensor[[1], pl.INT32],
    comp_slot: pl.Tensor[[1], pl.INT32],
    comp_cache_slot: pl.Tensor[[1], pl.INT32],
    comp_should_compress: pl.Tensor[[1], pl.INT32],
    attn_hc_fn_t: pl.Tensor[[HC_DIM, MIX_HC], pl.FP32],
    attn_hc_scale: pl.Tensor[[3], pl.FP32],
    attn_hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[HIDDEN], pl.BF16],
    wq_a_t: pl.Tensor[[HIDDEN, Q_LORA_RANK], pl.BF16],
    q_norm_w: pl.Tensor[[Q_LORA_RANK], pl.BF16],
    wq_b_t: pl.Tensor[[Q_LORA_RANK, ATTN_Q_OUT], pl.BF16],
    wkv_t: pl.Tensor[[HIDDEN, HEAD_DIM], pl.BF16],
    kv_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[N_HEADS], pl.FP32],
    window_topk_idxs: pl.Tensor[[B, S_DYN, TOPK_SWA], pl.INT32],
    wo_a_t: pl.Tensor[[O_GROUP_IN, ATTN_OUT_IN], pl.BF16],
    wo_b_t: pl.Tensor[[ATTN_OUT_IN, HIDDEN], pl.BF16],
    cos: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    attn_comp_wkv_t: pl.Tensor[[HIDDEN, ATTN_PROJ_DIM], pl.BF16],
    attn_comp_wgate_t: pl.Tensor[[HIDDEN, ATTN_PROJ_DIM], pl.BF16],
    attn_comp_ape: pl.Tensor[[COMPRESS_RATIO4, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_comp_cos: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    attn_comp_sin: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    idx_wq_b_t: pl.Tensor[[Q_LORA_RANK, INDEX_Q_OUT], pl.BF16],
    idx_weights_proj_t: pl.Tensor[[HIDDEN, INDEX_N_HEADS], pl.BF16],
    idx_offset: pl.Tensor[[1], pl.INT32],
    idx_comp_wkv_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    idx_comp_wgate_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    idx_comp_ape: pl.Tensor[[COMPRESS_RATIO4, INDEX_PROJ_DIM], pl.FP32],
    idx_comp_norm_w: pl.Tensor[[INDEX_HEAD_DIM], pl.BF16],
    idx_comp_cos: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    idx_comp_sin: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    ffn_hc_fn_t: pl.Tensor[[HC_DIM, MIX_HC], pl.FP32],
    ffn_hc_scale: pl.Tensor[[3], pl.FP32],
    ffn_hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[HIDDEN], pl.BF16],
    gate_w_t: pl.Tensor[[HIDDEN, N_EXPERTS], pl.BF16],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    attn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    attn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    attn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    attn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    kv_pool: pl.Tensor[[B, WINDOW_SIZE + TOPK_CSA_COMPRESSED, HEAD_DIM], pl.BF16],
    kv_cache_out: pl.Out[pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16]],
    attn_comp_kv_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32]],
    attn_comp_score_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32]],
    attn_comp_cache_out: pl.Out[pl.Tensor[[B, TOPK_CSA_COMPRESSED, HEAD_DIM], pl.BF16]],
    idx_kv_cache_out: pl.Out[pl.Tensor[[B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM], pl.BF16]],
    idx_comp_kv_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
    idx_comp_score_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
    ffn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    ffn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    ffn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    ffn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_out: pl.Out[pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16]],
    ffn_normed: pl.Out[pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16]],
    indices: pl.Out[pl.Tensor[[B, S_DYN, TOPK], pl.INT32]],
    weights: pl.Out[pl.Tensor[[B, S_DYN, TOPK], pl.FP32]],
    ffn_hc_post: pl.Out[pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32]],
    ffn_hc_comb: pl.Out[pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32]],
):
    """Run CSA topk decode block up to FFN MoE gate."""
    x.bind_dynamic(1, S_DYN)
    tokens = pl.tensor.dim(x, 1)
    attn_hc_x_mixed = pl.create_tensor([B, tokens, HIDDEN], dtype=pl.BF16)
    attn_hc_post = pl.create_tensor([B, tokens, HC_PAD], dtype=pl.FP32)
    attn_hc_comb = pl.create_tensor([B, tokens, HC_MULT * HC_MULT], dtype=pl.FP32)
    attn_normed = pl.create_tensor([B, tokens, HIDDEN], dtype=pl.BF16)
    attn_out = pl.create_tensor([B, tokens, HIDDEN], dtype=pl.BF16)
    ffn_hc_x_mixed = pl.create_tensor([B, tokens, HIDDEN], dtype=pl.BF16)

    hc_pre_fwd(
        x,
        attn_hc_x_pad,
        attn_hc_fn_t,
        attn_hc_scale,
        attn_hc_base,
        attn_hc_mixes,
        attn_hc_pre,
        attn_hc_comb_logits,
        attn_hc_x_mixed_pad,
        attn_hc_post_pad,
        attn_hc_comb_pad,
        attn_hc_x_mixed,
        attn_hc_post,
        attn_hc_comb,
    )
    rmsnorm_4096(attn_hc_x_mixed, attn_norm_w, attn_normed)
    attention_csa_decode_fwd(
        attn_normed,
        kv_cache,
        attn_comp_kv_state,
        attn_comp_score_state,
        attn_comp_cache,
        idx_kv_cache_in,
        idx_comp_kv_state,
        idx_comp_score_state,
        cache_pos,
        comp_slot,
        comp_cache_slot,
        comp_should_compress,
        wq_a_t,
        q_norm_w,
        wq_b_t,
        wkv_t,
        kv_norm_w,
        attn_sink,
        window_topk_idxs,
        wo_a_t,
        wo_b_t,
        cos,
        sin,
        attn_comp_wkv_t,
        attn_comp_wgate_t,
        attn_comp_ape,
        attn_comp_norm_w,
        attn_comp_cos,
        attn_comp_sin,
        idx_wq_b_t,
        idx_weights_proj_t,
        idx_offset,
        idx_comp_wkv_t,
        idx_comp_wgate_t,
        idx_comp_ape,
        idx_comp_norm_w,
        idx_comp_cos,
        idx_comp_sin,
        kv_cache_out,
        attn_comp_kv_state_out,
        attn_comp_score_state_out,
        attn_comp_cache_out,
        idx_kv_cache_out,
        idx_comp_kv_state_out,
        idx_comp_score_state_out,
        attn_out,
    )
    hc_post_fwd(attn_out, x, attn_hc_post, attn_hc_comb, attn_hc_out)

    hc_pre_fwd(
        attn_hc_out,
        ffn_hc_x_pad,
        ffn_hc_fn_t,
        ffn_hc_scale,
        ffn_hc_base,
        ffn_hc_mixes,
        ffn_hc_pre,
        ffn_hc_comb_logits,
        ffn_hc_x_mixed_pad,
        ffn_hc_post_pad,
        ffn_hc_comb_pad,
        ffn_hc_x_mixed,
        ffn_hc_post,
        ffn_hc_comb,
    )
    rmsnorm_4096(ffn_hc_x_mixed, ffn_norm_w, ffn_normed)
    gate_topk_fwd(ffn_normed, gate_w_t, gate_bias, indices, weights)
    return (
        kv_cache_out,
        attn_comp_kv_state_out,
        attn_comp_score_state_out,
        attn_comp_cache_out,
        idx_kv_cache_out,
        idx_comp_kv_state_out,
        idx_comp_score_state_out,
        attn_hc_out,
        ffn_normed,
        indices,
        weights,
        ffn_hc_post,
        ffn_hc_comb,
    )


@pl.jit
def hca_topk_selected_decode_pre_moe_fwd(
    x: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    kv_cache: pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16],
    comp_kv_state: pl.Tensor[[B, COMPRESS_RATIO128, HEAD_DIM], pl.FP32],
    comp_score_state: pl.Tensor[[B, COMPRESS_RATIO128, HEAD_DIM], pl.FP32],
    comp_cache: pl.Tensor[[B, TOPK_HCA, HEAD_DIM], pl.BF16],
    cache_pos: pl.Tensor[[1], pl.INT32],
    comp_slot: pl.Tensor[[1], pl.INT32],
    comp_cache_slot: pl.Tensor[[1], pl.INT32],
    comp_should_compress: pl.Tensor[[1], pl.INT32],
    attn_hc_fn_t: pl.Tensor[[HC_DIM, MIX_HC], pl.FP32],
    attn_hc_scale: pl.Tensor[[3], pl.FP32],
    attn_hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[HIDDEN], pl.BF16],
    wq_a_t: pl.Tensor[[HIDDEN, Q_LORA_RANK], pl.BF16],
    q_norm_w: pl.Tensor[[Q_LORA_RANK], pl.BF16],
    wq_b_t: pl.Tensor[[Q_LORA_RANK, ATTN_Q_OUT], pl.BF16],
    wkv_t: pl.Tensor[[HIDDEN, HEAD_DIM], pl.BF16],
    kv_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[N_HEADS], pl.FP32],
    topk_idxs: pl.Tensor[[B, S_DYN, TOPK_HCA_TOTAL], pl.INT32],
    wo_a_t: pl.Tensor[[O_GROUP_IN, ATTN_OUT_IN], pl.BF16],
    wo_b_t: pl.Tensor[[ATTN_OUT_IN, HIDDEN], pl.BF16],
    cos: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    comp_wkv_t: pl.Tensor[[HIDDEN, HEAD_DIM], pl.BF16],
    comp_wgate_t: pl.Tensor[[HIDDEN, HEAD_DIM], pl.BF16],
    comp_ape: pl.Tensor[[COMPRESS_RATIO128, HEAD_DIM], pl.FP32],
    comp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    comp_cos: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    comp_sin: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    ffn_hc_fn_t: pl.Tensor[[HC_DIM, MIX_HC], pl.FP32],
    ffn_hc_scale: pl.Tensor[[3], pl.FP32],
    ffn_hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[HIDDEN], pl.BF16],
    gate_w_t: pl.Tensor[[HIDDEN, N_EXPERTS], pl.BF16],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    attn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    attn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    attn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    attn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    kv_pool: pl.Tensor[[B, WINDOW_SIZE + TOPK_HCA, HEAD_DIM], pl.BF16],
    kv_cache_out: pl.Out[pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16]],
    comp_kv_state_out: pl.Out[pl.Tensor[[B, COMPRESS_RATIO128, HEAD_DIM], pl.FP32]],
    comp_score_state_out: pl.Out[pl.Tensor[[B, COMPRESS_RATIO128, HEAD_DIM], pl.FP32]],
    comp_cache_out: pl.Out[pl.Tensor[[B, TOPK_HCA, HEAD_DIM], pl.BF16]],
    ffn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    ffn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    ffn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    ffn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_out: pl.Out[pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16]],
    ffn_normed: pl.Out[pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16]],
    indices: pl.Out[pl.Tensor[[B, S_DYN, TOPK], pl.INT32]],
    weights: pl.Out[pl.Tensor[[B, S_DYN, TOPK], pl.FP32]],
    ffn_hc_post: pl.Out[pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32]],
    ffn_hc_comb: pl.Out[pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32]],
):
    """Run HCA topk decode block up to FFN MoE gate."""
    x.bind_dynamic(1, S_DYN)
    tokens = pl.tensor.dim(x, 1)
    attn_hc_x_mixed = pl.create_tensor([B, tokens, HIDDEN], dtype=pl.BF16)
    attn_hc_post = pl.create_tensor([B, tokens, HC_PAD], dtype=pl.FP32)
    attn_hc_comb = pl.create_tensor([B, tokens, HC_MULT * HC_MULT], dtype=pl.FP32)
    attn_normed = pl.create_tensor([B, tokens, HIDDEN], dtype=pl.BF16)
    attn_out = pl.create_tensor([B, tokens, HIDDEN], dtype=pl.BF16)
    ffn_hc_x_mixed = pl.create_tensor([B, tokens, HIDDEN], dtype=pl.BF16)

    hc_pre_fwd(
        x,
        attn_hc_x_pad,
        attn_hc_fn_t,
        attn_hc_scale,
        attn_hc_base,
        attn_hc_mixes,
        attn_hc_pre,
        attn_hc_comb_logits,
        attn_hc_x_mixed_pad,
        attn_hc_post_pad,
        attn_hc_comb_pad,
        attn_hc_x_mixed,
        attn_hc_post,
        attn_hc_comb,
    )
    rmsnorm_4096(attn_hc_x_mixed, attn_norm_w, attn_normed)
    attention_hca_decode_fwd(
        attn_normed,
        kv_cache,
        comp_kv_state,
        comp_score_state,
        comp_cache,
        cache_pos,
        comp_slot,
        comp_cache_slot,
        comp_should_compress,
        wq_a_t,
        q_norm_w,
        wq_b_t,
        wkv_t,
        kv_norm_w,
        attn_sink,
        topk_idxs,
        wo_a_t,
        wo_b_t,
        cos,
        sin,
        comp_wkv_t,
        comp_wgate_t,
        comp_ape,
        comp_norm_w,
        comp_cos,
        comp_sin,
        kv_cache_out,
        comp_kv_state_out,
        comp_score_state_out,
        comp_cache_out,
        attn_out,
    )
    hc_post_fwd(attn_out, x, attn_hc_post, attn_hc_comb, attn_hc_out)

    hc_pre_fwd(
        attn_hc_out,
        ffn_hc_x_pad,
        ffn_hc_fn_t,
        ffn_hc_scale,
        ffn_hc_base,
        ffn_hc_mixes,
        ffn_hc_pre,
        ffn_hc_comb_logits,
        ffn_hc_x_mixed_pad,
        ffn_hc_post_pad,
        ffn_hc_comb_pad,
        ffn_hc_x_mixed,
        ffn_hc_post,
        ffn_hc_comb,
    )
    rmsnorm_4096(ffn_hc_x_mixed, ffn_norm_w, ffn_normed)
    gate_topk_fwd(ffn_normed, gate_w_t, gate_bias, indices, weights)
    return (
        kv_cache_out,
        comp_kv_state_out,
        comp_score_state_out,
        comp_cache_out,
        attn_hc_out,
        ffn_normed,
        indices,
        weights,
        ffn_hc_post,
        ffn_hc_comb,
    )


@pl.jit
def swa_hash_selected_decode_post_moe_fwd(
    ffn_normed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    weights: pl.Tensor[[B, S_DYN, TOPK], pl.FP32],
    selected_w1_t: pl.Tensor[[TOPK, HIDDEN, MOE_INTER_DIM], pl.BF16],
    selected_w2_t: pl.Tensor[[TOPK, MOE_INTER_DIM, HIDDEN], pl.BF16],
    selected_w3_t: pl.Tensor[[TOPK, HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w1_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w2_t: pl.Tensor[[MOE_INTER_DIM, HIDDEN], pl.BF16],
    shared_w3_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    attn_hc_out: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    ffn_hc_post: pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb: pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32],
    out: pl.Out[pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16]],
):
    """Run selected FFN MoE and HC post for SWA hash decode block."""
    ffn_normed.bind_dynamic(1, S_DYN)
    tokens = pl.tensor.dim(ffn_normed, 1)
    moe_out = pl.create_tensor([B, tokens, HIDDEN], dtype=pl.BF16)
    moe_selected_decode_experts_fwd(
        ffn_normed,
        weights,
        selected_w1_t,
        selected_w2_t,
        selected_w3_t,
        shared_w1_t,
        shared_w2_t,
        shared_w3_t,
        moe_out,
    )
    hc_post_fwd(moe_out, attn_hc_out, ffn_hc_post, ffn_hc_comb, out)
    return out


def _golden_selected_decode_pre_moe(
    tensors: dict[str, torch.Tensor],
    *,
    start_pos: int,
    attention_kind: str,
    hash_route: bool,
) -> None:
    """Run split decode block up to FFN MoE gate."""
    if start_pos <= 0:
        raise ValueError(f"decode start_pos must be positive, got {start_pos}")
    x = tensors["x"]
    if x.shape[1] != 1:
        raise ValueError(f"decode expects seq_len=1, got {x.shape[1]}")

    attn_residual = x
    attn_x, attn_post, attn_comb = block_golden._run_hc_pre(tensors, prefix="attn", x=x)
    attn_normed = block_golden._run_hidden_rmsnorm(
        tensors,
        x=attn_x,
        norm_w_key="attn_norm_w",
        out_key="attn_normed",
    )
    attn_out = block_golden._run_attention(
        tensors,
        x=attn_normed,
        start_pos=start_pos,
        attention_kind=attention_kind,
    )
    attn_hc_out = block_golden._run_hc_post(
        tensors,
        prefix="attn",
        x=attn_out,
        residual=attn_residual,
        post=attn_post,
        comb=attn_comb,
    )

    ffn_x, ffn_post, ffn_comb = block_golden._run_hc_pre(tensors, prefix="ffn", x=attn_hc_out)
    ffn_normed = block_golden._run_hidden_rmsnorm(
        tensors,
        x=ffn_x,
        norm_w_key="ffn_norm_w",
        out_key="ffn_normed",
    )

    gate_tensors = dict(tensors)
    gate_tensors["x"] = ffn_normed
    gate_tensors["indices"] = tensors["indices"]
    gate_tensors["weights"] = tensors["weights"]
    golden_gate_forward(gate_tensors, hash_route=hash_route)

    tensors["attn_hc_out"][:] = attn_hc_out
    tensors["ffn_normed"][:] = ffn_normed
    tensors["ffn_hc_post"][:] = ffn_post
    tensors["ffn_hc_comb"][:] = ffn_comb
    tensors["indices"][:] = gate_tensors["indices"]
    tensors["weights"][:] = gate_tensors["weights"]


def golden_swa_hash_selected_decode_pre_moe(tensors: dict[str, torch.Tensor], start_pos: int) -> None:
    """Run SWA hash decode block up to FFN MoE gate."""
    _golden_selected_decode_pre_moe(tensors, start_pos=start_pos, attention_kind="swa", hash_route=True)


def golden_csa_hash_selected_decode_pre_moe(tensors: dict[str, torch.Tensor], start_pos: int) -> None:
    """Run CSA hash decode block up to FFN MoE gate."""
    _golden_selected_decode_pre_moe(tensors, start_pos=start_pos, attention_kind="csa", hash_route=True)


def golden_hca_topk_selected_decode_pre_moe(tensors: dict[str, torch.Tensor], start_pos: int) -> None:
    """Run HCA topk decode block up to FFN MoE gate."""
    _golden_selected_decode_pre_moe(tensors, start_pos=start_pos, attention_kind="hca", hash_route=False)


def golden_csa_topk_selected_decode_pre_moe(tensors: dict[str, torch.Tensor], start_pos: int) -> None:
    """Run CSA topk decode block up to FFN MoE gate."""
    _golden_selected_decode_pre_moe(tensors, start_pos=start_pos, attention_kind="csa", hash_route=False)


def _golden_selected_decode_post_moe(tensors: dict[str, torch.Tensor]) -> None:
    """Run selected FFN MoE and HC post for split decode block."""
    moe_tensors = {
        "x": tensors["ffn_normed"],
        "weights": tensors["weights"],
        "selected_w1_t": tensors["selected_w1_t"],
        "selected_w2_t": tensors["selected_w2_t"],
        "selected_w3_t": tensors["selected_w3_t"],
        "shared_w1_t": tensors["shared_w1_t"],
        "shared_w2_t": tensors["shared_w2_t"],
        "shared_w3_t": tensors["shared_w3_t"],
        "out": tensors["moe_out"] if "moe_out" in tensors else torch.empty_like(tensors["ffn_normed"]),
    }
    golden_moe_selected_decode_experts_forward(moe_tensors)

    out = block_golden._run_hc_post(
        tensors,
        prefix="ffn",
        x=moe_tensors["out"],
        residual=tensors["attn_hc_out"],
        post=tensors["ffn_hc_post"],
        comb=tensors["ffn_hc_comb"],
    )
    tensors["out"][:] = out


def golden_swa_hash_selected_decode_post_moe(tensors: dict[str, torch.Tensor]) -> None:
    """Run selected FFN MoE and HC post for SWA hash decode block."""
    _golden_selected_decode_post_moe(tensors)


def golden_csa_hash_selected_decode_post_moe(tensors: dict[str, torch.Tensor]) -> None:
    """Run selected FFN MoE and HC post for CSA hash decode block."""
    _golden_selected_decode_post_moe(tensors)


def golden_hca_topk_selected_decode_post_moe(tensors: dict[str, torch.Tensor]) -> None:
    """Run selected FFN MoE and HC post for HCA topk decode block."""
    _golden_selected_decode_post_moe(tensors)


def golden_csa_topk_selected_decode_post_moe(tensors: dict[str, torch.Tensor]) -> None:
    """Run selected FFN MoE and HC post for CSA topk decode block."""
    _golden_selected_decode_post_moe(tensors)


def _spec_map(specs):
    return {spec.name: spec for spec in specs}


def build_swa_hash_selected_decode_pre_moe_specs(start_pos: int = DEFAULT_DECODE_START_POS):
    from models.golden import TensorSpec

    seq_len = 1
    base = _spec_map(block_golden.build_swa_hash_decode_specs(start_pos))
    names = [
        "x",
        "kv_cache",
        "cache_pos",
        "attn_hc_fn_t",
        "attn_hc_scale",
        "attn_hc_base",
        "attn_norm_w",
        "wq_a_t",
        "q_norm_w",
        "wq_b_t",
        "wkv_t",
        "kv_norm_w",
        "attn_sink",
        "topk_idxs",
        "wo_a_t",
        "wo_b_t",
        "cos",
        "sin",
        "ffn_hc_fn_t",
        "ffn_hc_scale",
        "ffn_hc_base",
        "ffn_norm_w",
        "gate_w_t",
        "tid2eid",
        "input_ids",
        "attn_hc_x_pad",
        "attn_hc_mixes",
        "attn_hc_pre",
        "attn_hc_comb_logits",
        "attn_hc_x_mixed_pad",
        "attn_hc_post_pad",
        "attn_hc_comb_pad",
        "kv_cache_out",
        "ffn_hc_x_pad",
        "ffn_hc_mixes",
        "ffn_hc_pre",
        "ffn_hc_comb_logits",
        "ffn_hc_x_mixed_pad",
        "ffn_hc_post_pad",
        "ffn_hc_comb_pad",
    ]
    specs = [base[name] for name in names]
    specs.extend(
        [
            TensorSpec("attn_hc_out", [B, seq_len, HC_MULT, HIDDEN], torch.bfloat16, is_output=True),
            TensorSpec("ffn_normed", [B, seq_len, HIDDEN], torch.bfloat16, is_output=True),
            TensorSpec("indices", [B, seq_len, TOPK], torch.int32, is_output=True),
            TensorSpec("weights", [B, seq_len, TOPK], torch.float32, is_output=True),
            TensorSpec("ffn_hc_post", [B, seq_len, HC_PAD], torch.float32, is_output=True),
            TensorSpec("ffn_hc_comb", [B, seq_len, HC_MULT * HC_MULT], torch.float32, is_output=True),
        ]
    )
    return specs


def build_csa_hash_selected_decode_pre_moe_specs(start_pos: int = DEFAULT_DECODE_START_POS):
    from models.golden import TensorSpec

    seq_len = 1
    base = _spec_map(block_golden.build_csa_hash_decode_specs(start_pos))
    names = [
        "x",
        "kv_cache",
        "attn_comp_kv_state",
        "attn_comp_score_state",
        "attn_comp_cache",
        "idx_kv_cache_in",
        "idx_comp_kv_state",
        "idx_comp_score_state",
        "cache_pos",
        "comp_slot",
        "comp_cache_slot",
        "comp_should_compress",
        "attn_hc_fn_t",
        "attn_hc_scale",
        "attn_hc_base",
        "attn_norm_w",
        "wq_a_t",
        "q_norm_w",
        "wq_b_t",
        "wkv_t",
        "kv_norm_w",
        "attn_sink",
        "window_topk_idxs",
        "wo_a_t",
        "wo_b_t",
        "cos",
        "sin",
        "attn_comp_wkv_t",
        "attn_comp_wgate_t",
        "attn_comp_ape",
        "attn_comp_norm_w",
        "attn_comp_cos",
        "attn_comp_sin",
        "idx_wq_b_t",
        "idx_weights_proj_t",
        "idx_offset",
        "idx_comp_wkv_t",
        "idx_comp_wgate_t",
        "idx_comp_ape",
        "idx_comp_norm_w",
        "idx_comp_cos",
        "idx_comp_sin",
        "ffn_hc_fn_t",
        "ffn_hc_scale",
        "ffn_hc_base",
        "ffn_norm_w",
        "gate_w_t",
        "tid2eid",
        "input_ids",
        "attn_hc_x_pad",
        "attn_hc_mixes",
        "attn_hc_pre",
        "attn_hc_comb_logits",
        "attn_hc_x_mixed_pad",
        "attn_hc_post_pad",
        "attn_hc_comb_pad",
        "kv_pool",
        "kv_cache_out",
        "attn_comp_kv_state_out",
        "attn_comp_score_state_out",
        "attn_comp_cache_out",
        "idx_kv_cache_out",
        "idx_comp_kv_state_out",
        "idx_comp_score_state_out",
        "ffn_hc_x_pad",
        "ffn_hc_mixes",
        "ffn_hc_pre",
        "ffn_hc_comb_logits",
        "ffn_hc_x_mixed_pad",
        "ffn_hc_post_pad",
        "ffn_hc_comb_pad",
    ]
    specs = [base[name] for name in names]
    specs.extend(
        [
            TensorSpec("attn_hc_out", [B, seq_len, HC_MULT, HIDDEN], torch.bfloat16, is_output=True),
            TensorSpec("ffn_normed", [B, seq_len, HIDDEN], torch.bfloat16, is_output=True),
            TensorSpec("indices", [B, seq_len, TOPK], torch.int32, is_output=True),
            TensorSpec("weights", [B, seq_len, TOPK], torch.float32, is_output=True),
            TensorSpec("ffn_hc_post", [B, seq_len, HC_PAD], torch.float32, is_output=True),
            TensorSpec("ffn_hc_comb", [B, seq_len, HC_MULT * HC_MULT], torch.float32, is_output=True),
        ]
    )
    return specs


def build_csa_topk_selected_decode_pre_moe_specs(start_pos: int = DEFAULT_DECODE_START_POS):
    from models.golden import TensorSpec

    seq_len = 1
    base = _spec_map(block_golden.build_csa_topk_decode_specs(start_pos))
    names = [
        "x",
        "kv_cache",
        "attn_comp_kv_state",
        "attn_comp_score_state",
        "attn_comp_cache",
        "idx_kv_cache_in",
        "idx_comp_kv_state",
        "idx_comp_score_state",
        "cache_pos",
        "comp_slot",
        "comp_cache_slot",
        "comp_should_compress",
        "attn_hc_fn_t",
        "attn_hc_scale",
        "attn_hc_base",
        "attn_norm_w",
        "wq_a_t",
        "q_norm_w",
        "wq_b_t",
        "wkv_t",
        "kv_norm_w",
        "attn_sink",
        "window_topk_idxs",
        "wo_a_t",
        "wo_b_t",
        "cos",
        "sin",
        "attn_comp_wkv_t",
        "attn_comp_wgate_t",
        "attn_comp_ape",
        "attn_comp_norm_w",
        "attn_comp_cos",
        "attn_comp_sin",
        "idx_wq_b_t",
        "idx_weights_proj_t",
        "idx_offset",
        "idx_comp_wkv_t",
        "idx_comp_wgate_t",
        "idx_comp_ape",
        "idx_comp_norm_w",
        "idx_comp_cos",
        "idx_comp_sin",
        "ffn_hc_fn_t",
        "ffn_hc_scale",
        "ffn_hc_base",
        "ffn_norm_w",
        "gate_w_t",
        "gate_bias",
        "attn_hc_x_pad",
        "attn_hc_mixes",
        "attn_hc_pre",
        "attn_hc_comb_logits",
        "attn_hc_x_mixed_pad",
        "attn_hc_post_pad",
        "attn_hc_comb_pad",
        "kv_pool",
        "kv_cache_out",
        "attn_comp_kv_state_out",
        "attn_comp_score_state_out",
        "attn_comp_cache_out",
        "idx_kv_cache_out",
        "idx_comp_kv_state_out",
        "idx_comp_score_state_out",
        "ffn_hc_x_pad",
        "ffn_hc_mixes",
        "ffn_hc_pre",
        "ffn_hc_comb_logits",
        "ffn_hc_x_mixed_pad",
        "ffn_hc_post_pad",
        "ffn_hc_comb_pad",
    ]
    specs = [base[name] for name in names]
    specs.extend(
        [
            TensorSpec("attn_hc_out", [B, seq_len, HC_MULT, HIDDEN], torch.bfloat16, is_output=True),
            TensorSpec("ffn_normed", [B, seq_len, HIDDEN], torch.bfloat16, is_output=True),
            TensorSpec("indices", [B, seq_len, TOPK], torch.int32, is_output=True),
            TensorSpec("weights", [B, seq_len, TOPK], torch.float32, is_output=True),
            TensorSpec("ffn_hc_post", [B, seq_len, HC_PAD], torch.float32, is_output=True),
            TensorSpec("ffn_hc_comb", [B, seq_len, HC_MULT * HC_MULT], torch.float32, is_output=True),
        ]
    )
    return specs


def build_hca_topk_selected_decode_pre_moe_specs(start_pos: int = DEFAULT_DECODE_START_POS):
    from models.golden import TensorSpec

    seq_len = 1
    base = _spec_map(block_golden.build_hca_topk_decode_specs(start_pos))
    names = [
        "x",
        "kv_cache",
        "comp_kv_state",
        "comp_score_state",
        "comp_cache",
        "cache_pos",
        "comp_slot",
        "comp_cache_slot",
        "comp_should_compress",
        "attn_hc_fn_t",
        "attn_hc_scale",
        "attn_hc_base",
        "attn_norm_w",
        "wq_a_t",
        "q_norm_w",
        "wq_b_t",
        "wkv_t",
        "kv_norm_w",
        "attn_sink",
        "topk_idxs",
        "wo_a_t",
        "wo_b_t",
        "cos",
        "sin",
        "comp_wkv_t",
        "comp_wgate_t",
        "comp_ape",
        "comp_norm_w",
        "comp_cos",
        "comp_sin",
        "ffn_hc_fn_t",
        "ffn_hc_scale",
        "ffn_hc_base",
        "ffn_norm_w",
        "gate_w_t",
        "gate_bias",
        "attn_hc_x_pad",
        "attn_hc_mixes",
        "attn_hc_pre",
        "attn_hc_comb_logits",
        "attn_hc_x_mixed_pad",
        "attn_hc_post_pad",
        "attn_hc_comb_pad",
        "kv_pool",
        "kv_cache_out",
        "comp_kv_state_out",
        "comp_score_state_out",
        "comp_cache_out",
        "ffn_hc_x_pad",
        "ffn_hc_mixes",
        "ffn_hc_pre",
        "ffn_hc_comb_logits",
        "ffn_hc_x_mixed_pad",
        "ffn_hc_post_pad",
        "ffn_hc_comb_pad",
    ]
    specs = [base[name] for name in names]
    specs.extend(
        [
            TensorSpec("attn_hc_out", [B, seq_len, HC_MULT, HIDDEN], torch.bfloat16, is_output=True),
            TensorSpec("ffn_normed", [B, seq_len, HIDDEN], torch.bfloat16, is_output=True),
            TensorSpec("indices", [B, seq_len, TOPK], torch.int32, is_output=True),
            TensorSpec("weights", [B, seq_len, TOPK], torch.float32, is_output=True),
            TensorSpec("ffn_hc_post", [B, seq_len, HC_PAD], torch.float32, is_output=True),
            TensorSpec("ffn_hc_comb", [B, seq_len, HC_MULT * HC_MULT], torch.float32, is_output=True),
        ]
    )
    return specs


def build_swa_hash_selected_decode_post_moe_specs(_start_pos: int = DEFAULT_DECODE_START_POS):
    from models.golden import TensorSpec

    seq_len = 1

    def init_ffn_normed():
        return (torch.randn(B, seq_len, HIDDEN, dtype=torch.float32) * 0.05).to(torch.bfloat16)

    def init_weights():
        weights = torch.rand(B, seq_len, TOPK, dtype=torch.float32)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights * ROUTE_SCALE

    def init_selected_w1_t():
        return torch.randn(TOPK, HIDDEN, MOE_INTER_DIM, dtype=torch.float32) * 0.005

    def init_selected_w2_t():
        return torch.randn(TOPK, MOE_INTER_DIM, HIDDEN, dtype=torch.float32) * 0.005

    def init_selected_w3_t():
        return torch.randn(TOPK, HIDDEN, MOE_INTER_DIM, dtype=torch.float32) * 0.005

    def init_shared_w1_t():
        return torch.randn(HIDDEN, MOE_INTER_DIM, dtype=torch.float32) * 0.005

    def init_shared_w2_t():
        return torch.randn(MOE_INTER_DIM, HIDDEN, dtype=torch.float32) * 0.005

    def init_shared_w3_t():
        return torch.randn(HIDDEN, MOE_INTER_DIM, dtype=torch.float32) * 0.005

    def init_attn_hc_out():
        return (torch.randn(B, seq_len, HC_MULT, HIDDEN, dtype=torch.float32) * 0.05).to(torch.bfloat16)

    def init_ffn_hc_post():
        value = torch.zeros(B, seq_len, HC_PAD, dtype=torch.float32)
        value[..., :HC_MULT] = torch.rand(B, seq_len, HC_MULT, dtype=torch.float32) * 2.0
        return value

    def init_ffn_hc_comb():
        return torch.randn(B, seq_len, HC_MULT * HC_MULT, dtype=torch.float32) * 0.1

    return [
        TensorSpec("ffn_normed", [B, seq_len, HIDDEN], torch.bfloat16, init_value=init_ffn_normed),
        TensorSpec("weights", [B, seq_len, TOPK], torch.float32, init_value=init_weights),
        TensorSpec("selected_w1_t", [TOPK, HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=init_selected_w1_t),
        TensorSpec("selected_w2_t", [TOPK, MOE_INTER_DIM, HIDDEN], torch.bfloat16, init_value=init_selected_w2_t),
        TensorSpec("selected_w3_t", [TOPK, HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=init_selected_w3_t),
        TensorSpec("shared_w1_t", [HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=init_shared_w1_t),
        TensorSpec("shared_w2_t", [MOE_INTER_DIM, HIDDEN], torch.bfloat16, init_value=init_shared_w2_t),
        TensorSpec("shared_w3_t", [HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=init_shared_w3_t),
        TensorSpec("attn_hc_out", [B, seq_len, HC_MULT, HIDDEN], torch.bfloat16, init_value=init_attn_hc_out),
        TensorSpec("ffn_hc_post", [B, seq_len, HC_PAD], torch.float32, init_value=init_ffn_hc_post),
        TensorSpec("ffn_hc_comb", [B, seq_len, HC_MULT * HC_MULT], torch.float32, init_value=init_ffn_hc_comb),
        TensorSpec("out", [B, seq_len, HC_MULT, HIDDEN], torch.bfloat16, is_output=True),
    ]


def main() -> int:
    import argparse
    import time

    from models.golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone split decode Block validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3sim", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--decode-start-pos", type=int, default=DEFAULT_DECODE_START_POS)
    parser.add_argument(
        "--case",
        choices=[
            "all",
            "pre",
            "post",
            "swa-hash-pre",
            "swa-hash-post",
            "csa-hash-pre",
            "csa-hash-post",
            "hca-topk-pre",
            "hca-topk-post",
            "csa-topk-pre",
            "csa-topk-post",
        ],
        default="all",
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    runtime_cfg = {
        "platform": args.platform,
        "device_id": args.device,
        "enable_l2_swimlane": args.enable_l2_swimlane,
    }
    compare_fn = {
        "kv_cache_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "attn_comp_kv_state_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "attn_comp_score_state_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "attn_comp_cache_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "idx_kv_cache_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "idx_comp_kv_state_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "idx_comp_score_state_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "comp_kv_state_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "comp_score_state_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "comp_cache_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "attn_hc_out": ratio_allclose(atol=1e-3, rtol=2.0 / 128, max_error_ratio=0.005),
        "ffn_normed": ratio_allclose(atol=1e-3, rtol=2.0 / 128, max_error_ratio=0.005),
        "weights": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "ffn_hc_post": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "ffn_hc_comb": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "out": ratio_allclose(atol=1e-3, rtol=2.0 / 128, max_error_ratio=0.005),
    }

    cases = []
    if args.case in ("all", "pre", "swa-hash-pre"):
        cases.append(
            (
                "swa-hash-pre",
                swa_hash_selected_decode_pre_moe_fwd,
                lambda: build_swa_hash_selected_decode_pre_moe_specs(args.decode_start_pos),
                lambda tensors: golden_swa_hash_selected_decode_pre_moe(tensors, args.decode_start_pos),
            )
        )
    if args.case in ("all", "post", "swa-hash-post"):
        cases.append(
            (
                "swa-hash-post",
                swa_hash_selected_decode_post_moe_fwd,
                lambda: build_swa_hash_selected_decode_post_moe_specs(args.decode_start_pos),
                golden_swa_hash_selected_decode_post_moe,
            )
        )
    if args.case in ("all", "pre", "csa-hash-pre"):
        cases.append(
            (
                "csa-hash-pre",
                csa_hash_selected_decode_pre_moe_fwd,
                lambda: build_csa_hash_selected_decode_pre_moe_specs(args.decode_start_pos),
                lambda tensors: golden_csa_hash_selected_decode_pre_moe(tensors, args.decode_start_pos),
            )
        )
    if args.case in ("all", "post", "csa-hash-post"):
        cases.append(
            (
                "csa-hash-post",
                swa_hash_selected_decode_post_moe_fwd,
                lambda: build_swa_hash_selected_decode_post_moe_specs(args.decode_start_pos),
                golden_csa_hash_selected_decode_post_moe,
            )
        )
    if args.case in ("all", "pre", "hca-topk-pre"):
        cases.append(
            (
                "hca-topk-pre",
                hca_topk_selected_decode_pre_moe_fwd,
                lambda: build_hca_topk_selected_decode_pre_moe_specs(args.decode_start_pos),
                lambda tensors: golden_hca_topk_selected_decode_pre_moe(tensors, args.decode_start_pos),
            )
        )
    if args.case in ("all", "post", "hca-topk-post"):
        cases.append(
            (
                "hca-topk-post",
                swa_hash_selected_decode_post_moe_fwd,
                lambda: build_swa_hash_selected_decode_post_moe_specs(args.decode_start_pos),
                golden_hca_topk_selected_decode_post_moe,
            )
        )
    if args.case in ("all", "pre", "csa-topk-pre"):
        cases.append(
            (
                "csa-topk-pre",
                csa_topk_selected_decode_pre_moe_fwd,
                lambda: build_csa_topk_selected_decode_pre_moe_specs(args.decode_start_pos),
                lambda tensors: golden_csa_topk_selected_decode_pre_moe(tensors, args.decode_start_pos),
            )
        )
    if args.case in ("all", "post", "csa-topk-post"):
        cases.append(
            (
                "csa-topk-post",
                swa_hash_selected_decode_post_moe_fwd,
                lambda: build_swa_hash_selected_decode_post_moe_specs(args.decode_start_pos),
                golden_csa_topk_selected_decode_post_moe,
            )
        )

    failed = False
    for name, fn, build_specs, golden_fn in cases:
        print(f"[CASE] {name}", flush=True)
        result = run_jit(
            fn=fn,
            specs=build_specs(),
            golden_fn=golden_fn,
            runtime_cfg=runtime_cfg,
            compile_only=args.compile_only,
            compare_fn=compare_fn,
        )
        if not result.passed:
            failed = True
            if result.error:
                print(result.error)
        time.sleep(0.2)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "swa_hash_selected_decode_pre_moe_fwd",
    "csa_hash_selected_decode_pre_moe_fwd",
    "csa_topk_selected_decode_pre_moe_fwd",
    "hca_topk_selected_decode_pre_moe_fwd",
    "swa_hash_selected_decode_post_moe_fwd",
    "golden_swa_hash_selected_decode_pre_moe",
    "golden_swa_hash_selected_decode_post_moe",
    "golden_csa_hash_selected_decode_pre_moe",
    "golden_csa_hash_selected_decode_post_moe",
    "golden_hca_topk_selected_decode_pre_moe",
    "golden_hca_topk_selected_decode_post_moe",
    "golden_csa_topk_selected_decode_pre_moe",
    "golden_csa_topk_selected_decode_post_moe",
    "build_swa_hash_selected_decode_pre_moe_specs",
    "build_csa_hash_selected_decode_pre_moe_specs",
    "build_csa_topk_selected_decode_pre_moe_specs",
    "build_hca_topk_selected_decode_pre_moe_specs",
    "build_swa_hash_selected_decode_post_moe_specs",
]
