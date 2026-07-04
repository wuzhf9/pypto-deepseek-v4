"""DeepSeek V4 Flash Block golden logic and PyPTO kernels."""

import torch
import pypto.language as pl

from models.attention_csa import attention_csa_decode_fwd, attention_csa_prefill_fwd
from models.attention_hca import attention_hca_decode_fwd, attention_hca_prefill_fwd
from models.attention_swa import attention_swa_decode_fwd, attention_swa_prefill_fwd
from models.attention_csa import golden_attention_csa_forward
from models.attention_hca import golden_attention_hca_forward
from models.attention_swa import golden_attention_swa_forward
from models.common import ceil_div
from models.config import FLASH_CONFIG as M
from models.hc import HC_PAD, MIX_PAD, T_TILE as HC_T_TILE
from models.hc import golden_hc_post, golden_hc_pre, hc_post_fwd, hc_pre_fwd
from models.moe import golden_moe_forward, moe_hash_fwd, moe_topk_fwd
from models.rmsnorm import golden_rmsnorm, rmsnorm_4096
from models.rope import build_deepseek_v4_rope_tables, materialize_compressor_rope, materialize_rope_range
from models.sparse_attn import build_compress_topk_idxs, build_window_topk_idxs


B = 1
S_DYN = pl.dynamic("S_DYN")
S_PAD_DYN = pl.dynamic("S_PAD_DYN")
C_DYN = pl.dynamic("C_DYN")
K_DYN = pl.dynamic("K_DYN")

HIDDEN = M.dim
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc_dim
Q_LORA_RANK = M.q_lora_rank
N_HEADS = M.n_heads
HEAD_DIM = M.head_dim
ATTN_Q_OUT = N_HEADS * HEAD_DIM
O_GROUPS = M.o_groups
O_LORA_RANK = M.o_lora_rank
HEADS_PER_GROUP = M.heads_per_o_group
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
ATTN_OUT_IN = O_GROUPS * O_LORA_RANK
ROPE_HALF = M.rope_head_dim // 2
WINDOW_SIZE = M.window_size
TOPK_SWA = WINDOW_SIZE
COMPRESS_RATIO128 = 128
HCA_MAX_POSITION_EMBEDDINGS = 4096
TOPK_HCA = HCA_MAX_POSITION_EMBEDDINGS // COMPRESS_RATIO128
TOPK_HCA_TOTAL = TOPK_SWA + TOPK_HCA
INDEX_N_HEADS = M.index_n_heads
INDEX_HEAD_DIM = M.index_head_dim
INDEX_Q_OUT = INDEX_N_HEADS * INDEX_HEAD_DIM
INDEX_PROJ_DIM = 2 * INDEX_HEAD_DIM
INDEX_TOPK = M.index_topk
INDEX_MAX_POSITION_EMBEDDINGS = 4096
INDEX_SCORE_LEN = INDEX_MAX_POSITION_EMBEDDINGS // 4
COMPRESS_RATIO4 = 4
RATIO4_STATE_ROWS = 2 * COMPRESS_RATIO4
ATTN_PROJ_DIM = 2 * HEAD_DIM
TOPK_CSA = INDEX_TOPK
TOPK_CSA_TOTAL = TOPK_SWA + TOPK_CSA
TOPK_CSA_COMPRESSED = INDEX_SCORE_LEN
MOE_INTER_DIM = M.moe_inter_dim
N_EXPERTS = M.n_routed_experts
TOPK = M.n_activated_experts
VOCAB = M.vocab_size

DEFAULT_SEQ_LEN = 8
DEFAULT_DECODE_START_POS = 1

assert HC_MULT == 4, "Block kernel currently expects DeepSeek V4 Flash hc_mult=4"


@pl.jit
def block_swa_hash_prefill_fwd(
    x: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    attn_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
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
    ffn_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    ffn_hc_scale: pl.Tensor[[3], pl.FP32],
    ffn_hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[HIDDEN], pl.BF16],
    gate_w_t: pl.Tensor[[HIDDEN, N_EXPERTS], pl.BF16],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[B, S_DYN], pl.INT64],
    routed_w1_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    routed_w2_t: pl.Tensor[[N_EXPERTS, MOE_INTER_DIM, HIDDEN], pl.BF16],
    routed_w3_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w1_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w2_t: pl.Tensor[[MOE_INTER_DIM, HIDDEN], pl.BF16],
    shared_w3_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    attn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    attn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    attn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    attn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_x_mixed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    attn_hc_post: pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32],
    attn_hc_comb: pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_normed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    q_a: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    q_proj: pl.Tensor[[B, S_DYN, ATTN_Q_OUT], pl.BF16],
    kv_proj: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    kv_normed: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    q: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    kv_cache_out: pl.Out[pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16]],
    attn_o: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    o_inv: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    proj: pl.Tensor[[B, S_DYN, ATTN_OUT_IN], pl.BF16],
    attn_out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    attn_hc_out: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    ffn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    ffn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    ffn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    ffn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_hc_x_mixed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    ffn_hc_post: pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb: pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_normed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    logits: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    scores: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    indices: pl.Tensor[[B, S_DYN, TOPK], pl.INT32],
    weights: pl.Tensor[[B, S_DYN, TOPK], pl.FP32],
    route_y: pl.Tensor[[B, S_DYN, TOPK, HIDDEN], pl.BF16],
    shared_gate: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_up: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_hidden: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_y: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    moe_out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16]],
):
    """Run ordinary Block.forward for SWA attention + hash MoE, prefill path."""
    hc_pre_fwd(
        x,
        attn_hc_x_pad,
        attn_hc_fn,
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
    attention_swa_prefill_fwd(
        attn_normed,
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
        q_a,
        q_proj,
        kv_proj,
        kv_normed,
        qr,
        q,
        kv,
        kv_cache_out,
        attn_o,
        o_inv,
        proj,
        attn_out,
    )
    hc_post_fwd(attn_out, x, attn_hc_post, attn_hc_comb, attn_hc_out)

    hc_pre_fwd(
        attn_hc_out,
        ffn_hc_x_pad,
        ffn_hc_fn,
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
    moe_hash_fwd(
        ffn_normed,
        gate_w_t,
        tid2eid,
        input_ids,
        routed_w1_t,
        routed_w2_t,
        routed_w3_t,
        shared_w1_t,
        shared_w2_t,
        shared_w3_t,
        logits,
        scores,
        indices,
        weights,
        route_y,
        shared_gate,
        shared_up,
        shared_hidden,
        shared_y,
        moe_out,
    )
    hc_post_fwd(moe_out, attn_hc_out, ffn_hc_post, ffn_hc_comb, out)
    return kv_cache_out, out


@pl.jit
def block_swa_hash_decode_fwd(
    x: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    kv_cache: pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16],
    cache_pos: pl.Tensor[[1], pl.INT32],
    attn_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
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
    ffn_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    ffn_hc_scale: pl.Tensor[[3], pl.FP32],
    ffn_hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[HIDDEN], pl.BF16],
    gate_w_t: pl.Tensor[[HIDDEN, N_EXPERTS], pl.BF16],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[B, S_DYN], pl.INT64],
    routed_w1_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    routed_w2_t: pl.Tensor[[N_EXPERTS, MOE_INTER_DIM, HIDDEN], pl.BF16],
    routed_w3_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w1_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w2_t: pl.Tensor[[MOE_INTER_DIM, HIDDEN], pl.BF16],
    shared_w3_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    attn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    attn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    attn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    attn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_x_mixed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    attn_hc_post: pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32],
    attn_hc_comb: pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_normed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    q_a: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    q_proj: pl.Tensor[[B, S_DYN, ATTN_Q_OUT], pl.BF16],
    kv_proj: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    kv_normed: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    q: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    kv_cache_out: pl.Out[pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16]],
    attn_o: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    o_inv: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    proj: pl.Tensor[[B, S_DYN, ATTN_OUT_IN], pl.BF16],
    attn_out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    attn_hc_out: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    ffn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    ffn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    ffn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    ffn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_hc_x_mixed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    ffn_hc_post: pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb: pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_normed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    logits: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    scores: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    indices: pl.Tensor[[B, S_DYN, TOPK], pl.INT32],
    weights: pl.Tensor[[B, S_DYN, TOPK], pl.FP32],
    route_y: pl.Tensor[[B, S_DYN, TOPK, HIDDEN], pl.BF16],
    shared_gate: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_up: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_hidden: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_y: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    moe_out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16]],
):
    """Run ordinary Block.forward for SWA attention + hash MoE, decode path."""
    hc_pre_fwd(
        x,
        attn_hc_x_pad,
        attn_hc_fn,
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
        q_a,
        q_proj,
        kv_proj,
        kv_normed,
        qr,
        q,
        kv,
        kv_cache_out,
        attn_o,
        o_inv,
        proj,
        attn_out,
    )
    hc_post_fwd(attn_out, x, attn_hc_post, attn_hc_comb, attn_hc_out)

    hc_pre_fwd(
        attn_hc_out,
        ffn_hc_x_pad,
        ffn_hc_fn,
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
    moe_hash_fwd(
        ffn_normed,
        gate_w_t,
        tid2eid,
        input_ids,
        routed_w1_t,
        routed_w2_t,
        routed_w3_t,
        shared_w1_t,
        shared_w2_t,
        shared_w3_t,
        logits,
        scores,
        indices,
        weights,
        route_y,
        shared_gate,
        shared_up,
        shared_hidden,
        shared_y,
        moe_out,
    )
    hc_post_fwd(moe_out, attn_hc_out, ffn_hc_post, ffn_hc_comb, out)
    return kv_cache_out, out


@pl.jit
def block_csa_hash_prefill_fwd(
    x: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    attn_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
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
    attn_comp_cos: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    attn_comp_sin: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    attn_comp_block_count: pl.Tensor[[1], pl.INT32],
    idx_wq_b_t: pl.Tensor[[Q_LORA_RANK, INDEX_Q_OUT], pl.BF16],
    idx_weights_proj_t: pl.Tensor[[HIDDEN, INDEX_N_HEADS], pl.BF16],
    idx_offset: pl.Tensor[[1], pl.INT32],
    idx_comp_wkv_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    idx_comp_wgate_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    idx_comp_ape: pl.Tensor[[COMPRESS_RATIO4, INDEX_PROJ_DIM], pl.FP32],
    idx_comp_norm_w: pl.Tensor[[INDEX_HEAD_DIM], pl.BF16],
    idx_comp_cos: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    idx_comp_sin: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    idx_comp_block_count: pl.Tensor[[1], pl.INT32],
    ffn_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    ffn_hc_scale: pl.Tensor[[3], pl.FP32],
    ffn_hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[HIDDEN], pl.BF16],
    gate_w_t: pl.Tensor[[HIDDEN, N_EXPERTS], pl.BF16],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[B, S_DYN], pl.INT64],
    routed_w1_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    routed_w2_t: pl.Tensor[[N_EXPERTS, MOE_INTER_DIM, HIDDEN], pl.BF16],
    routed_w3_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w1_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w2_t: pl.Tensor[[MOE_INTER_DIM, HIDDEN], pl.BF16],
    shared_w3_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    attn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    attn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    attn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    attn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_x_mixed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    attn_hc_post: pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32],
    attn_hc_comb: pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_normed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    q_a: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    q_proj: pl.Tensor[[B, S_DYN, ATTN_Q_OUT], pl.BF16],
    kv_proj: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    kv_normed: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    q: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    attn_comp_kv_proj: pl.Tensor[[B, S_DYN, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_score_proj: pl.Tensor[[B, S_DYN, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_pooled: pl.Tensor[[B, C_DYN, HEAD_DIM], pl.BF16],
    attn_comp_normed: pl.Tensor[[B, C_DYN, HEAD_DIM], pl.BF16],
    attn_compressed: pl.Tensor[[B, C_DYN, HEAD_DIM], pl.BF16],
    kv_pool: pl.Tensor[[B, K_DYN, HEAD_DIM], pl.BF16],
    kv_cache_out: pl.Out[pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16]],
    attn_comp_kv_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32]],
    attn_comp_score_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32]],
    attn_comp_cache_out: pl.Out[pl.Tensor[[B, TOPK_CSA_COMPRESSED, HEAD_DIM], pl.BF16]],
    idx_q_proj: pl.Tensor[[B, S_DYN, INDEX_Q_OUT], pl.BF16],
    idx_q_rope: pl.Tensor[[B, S_DYN, INDEX_N_HEADS, INDEX_HEAD_DIM], pl.BF16],
    idx_weights: pl.Tensor[[B, S_DYN, INDEX_N_HEADS], pl.BF16],
    idx_comp_kv_proj: pl.Tensor[[B, S_DYN, INDEX_PROJ_DIM], pl.FP32],
    idx_comp_score_proj: pl.Tensor[[B, S_DYN, INDEX_PROJ_DIM], pl.FP32],
    idx_comp_pooled: pl.Tensor[[B, C_DYN, INDEX_HEAD_DIM], pl.BF16],
    idx_comp_normed: pl.Tensor[[B, C_DYN, INDEX_HEAD_DIM], pl.BF16],
    idx_score: pl.Tensor[[B, S_DYN, INDEX_SCORE_LEN], pl.FP32],
    idx_topk_idxs: pl.Tensor[[B, S_DYN, INDEX_TOPK], pl.INT32],
    idx_kv_cache_out: pl.Out[pl.Tensor[[B, INDEX_SCORE_LEN, INDEX_HEAD_DIM], pl.BF16]],
    idx_comp_kv_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
    idx_comp_score_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
    csa_topk_idxs: pl.Tensor[[B, S_DYN, TOPK_CSA_TOTAL], pl.INT32],
    attn_o: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    o_inv: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    proj: pl.Tensor[[B, S_DYN, ATTN_OUT_IN], pl.BF16],
    attn_out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    attn_hc_out: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    ffn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    ffn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    ffn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    ffn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_hc_x_mixed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    ffn_hc_post: pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb: pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_normed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    logits: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    scores: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    indices: pl.Tensor[[B, S_DYN, TOPK], pl.INT32],
    weights: pl.Tensor[[B, S_DYN, TOPK], pl.FP32],
    route_y: pl.Tensor[[B, S_DYN, TOPK, HIDDEN], pl.BF16],
    shared_gate: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_up: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_hidden: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_y: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    moe_out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16]],
):
    """Run ordinary Block.forward for CSA attention + hash MoE, prefill path."""
    hc_pre_fwd(
        x,
        attn_hc_x_pad,
        attn_hc_fn,
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
    attention_csa_prefill_fwd(
        attn_normed,
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
        attn_comp_block_count,
        idx_wq_b_t,
        idx_weights_proj_t,
        idx_offset,
        idx_comp_wkv_t,
        idx_comp_wgate_t,
        idx_comp_ape,
        idx_comp_norm_w,
        idx_comp_cos,
        idx_comp_sin,
        idx_comp_block_count,
        q_a,
        q_proj,
        kv_proj,
        kv_normed,
        qr,
        q,
        kv,
        attn_comp_kv_proj,
        attn_comp_score_proj,
        attn_comp_pooled,
        attn_comp_normed,
        attn_compressed,
        kv_pool,
        kv_cache_out,
        attn_comp_kv_state_out,
        attn_comp_score_state_out,
        attn_comp_cache_out,
        idx_q_proj,
        idx_q_rope,
        idx_weights,
        idx_comp_kv_proj,
        idx_comp_score_proj,
        idx_comp_pooled,
        idx_comp_normed,
        idx_score,
        idx_topk_idxs,
        idx_kv_cache_out,
        idx_comp_kv_state_out,
        idx_comp_score_state_out,
        csa_topk_idxs,
        attn_o,
        o_inv,
        proj,
        attn_out,
    )
    hc_post_fwd(attn_out, x, attn_hc_post, attn_hc_comb, attn_hc_out)

    hc_pre_fwd(
        attn_hc_out,
        ffn_hc_x_pad,
        ffn_hc_fn,
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
    moe_hash_fwd(
        ffn_normed,
        gate_w_t,
        tid2eid,
        input_ids,
        routed_w1_t,
        routed_w2_t,
        routed_w3_t,
        shared_w1_t,
        shared_w2_t,
        shared_w3_t,
        logits,
        scores,
        indices,
        weights,
        route_y,
        shared_gate,
        shared_up,
        shared_hidden,
        shared_y,
        moe_out,
    )
    hc_post_fwd(moe_out, attn_hc_out, ffn_hc_post, ffn_hc_comb, out)
    return (
        kv_cache_out,
        attn_comp_kv_state_out,
        attn_comp_score_state_out,
        attn_comp_cache_out,
        idx_kv_cache_out,
        idx_comp_kv_state_out,
        idx_comp_score_state_out,
        out,
    )


@pl.jit
def block_csa_hash_decode_fwd(
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
    attn_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
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
    ffn_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    ffn_hc_scale: pl.Tensor[[3], pl.FP32],
    ffn_hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[HIDDEN], pl.BF16],
    gate_w_t: pl.Tensor[[HIDDEN, N_EXPERTS], pl.BF16],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[B, S_DYN], pl.INT64],
    routed_w1_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    routed_w2_t: pl.Tensor[[N_EXPERTS, MOE_INTER_DIM, HIDDEN], pl.BF16],
    routed_w3_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w1_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w2_t: pl.Tensor[[MOE_INTER_DIM, HIDDEN], pl.BF16],
    shared_w3_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    attn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    attn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    attn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    attn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_x_mixed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    attn_hc_post: pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32],
    attn_hc_comb: pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_normed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    q_a: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    q_proj: pl.Tensor[[B, S_DYN, ATTN_Q_OUT], pl.BF16],
    kv_proj: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    kv_normed: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    q: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    attn_comp_kv_proj: pl.Tensor[[B, S_DYN, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_score_proj: pl.Tensor[[B, S_DYN, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_pooled: pl.Tensor[[B, 1, HEAD_DIM], pl.BF16],
    attn_comp_normed: pl.Tensor[[B, 1, HEAD_DIM], pl.BF16],
    attn_compressed: pl.Tensor[[B, 1, HEAD_DIM], pl.BF16],
    kv_pool: pl.Tensor[[B, WINDOW_SIZE + TOPK_CSA_COMPRESSED, HEAD_DIM], pl.BF16],
    kv_cache_out: pl.Out[pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16]],
    attn_comp_kv_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32]],
    attn_comp_score_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32]],
    attn_comp_cache_out: pl.Out[pl.Tensor[[B, TOPK_CSA_COMPRESSED, HEAD_DIM], pl.BF16]],
    idx_q_proj: pl.Tensor[[B, S_DYN, INDEX_Q_OUT], pl.BF16],
    idx_q_rope: pl.Tensor[[B, S_DYN, INDEX_N_HEADS, INDEX_HEAD_DIM], pl.BF16],
    idx_weights: pl.Tensor[[B, S_DYN, INDEX_N_HEADS], pl.BF16],
    idx_comp_kv_proj: pl.Tensor[[B, S_DYN, INDEX_PROJ_DIM], pl.FP32],
    idx_comp_score_proj: pl.Tensor[[B, S_DYN, INDEX_PROJ_DIM], pl.FP32],
    idx_comp_pooled: pl.Tensor[[B, 1, INDEX_HEAD_DIM], pl.BF16],
    idx_comp_normed: pl.Tensor[[B, 1, INDEX_HEAD_DIM], pl.BF16],
    idx_score: pl.Tensor[[B, S_DYN, INDEX_SCORE_LEN], pl.FP32],
    idx_topk_idxs: pl.Tensor[[B, S_DYN, INDEX_TOPK], pl.INT32],
    idx_kv_cache_out: pl.Out[pl.Tensor[[B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM], pl.BF16]],
    idx_comp_kv_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
    idx_comp_score_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
    csa_topk_idxs: pl.Tensor[[B, S_DYN, TOPK_CSA_TOTAL], pl.INT32],
    attn_o: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    o_inv: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    proj: pl.Tensor[[B, S_DYN, ATTN_OUT_IN], pl.BF16],
    attn_out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    attn_hc_out: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    ffn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    ffn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    ffn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    ffn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_hc_x_mixed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    ffn_hc_post: pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb: pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_normed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    logits: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    scores: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    indices: pl.Tensor[[B, S_DYN, TOPK], pl.INT32],
    weights: pl.Tensor[[B, S_DYN, TOPK], pl.FP32],
    route_y: pl.Tensor[[B, S_DYN, TOPK, HIDDEN], pl.BF16],
    shared_gate: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_up: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_hidden: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_y: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    moe_out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16]],
):
    """Run ordinary Block.forward for CSA attention + hash MoE, decode path."""
    hc_pre_fwd(
        x,
        attn_hc_x_pad,
        attn_hc_fn,
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
        q_a,
        q_proj,
        kv_proj,
        kv_normed,
        qr,
        q,
        kv,
        attn_comp_kv_proj,
        attn_comp_score_proj,
        attn_comp_pooled,
        attn_comp_normed,
        attn_compressed,
        kv_pool,
        kv_cache_out,
        attn_comp_kv_state_out,
        attn_comp_score_state_out,
        attn_comp_cache_out,
        idx_q_proj,
        idx_q_rope,
        idx_weights,
        idx_comp_kv_proj,
        idx_comp_score_proj,
        idx_comp_pooled,
        idx_comp_normed,
        idx_score,
        idx_topk_idxs,
        idx_kv_cache_out,
        idx_comp_kv_state_out,
        idx_comp_score_state_out,
        csa_topk_idxs,
        attn_o,
        o_inv,
        proj,
        attn_out,
    )
    hc_post_fwd(attn_out, x, attn_hc_post, attn_hc_comb, attn_hc_out)

    hc_pre_fwd(
        attn_hc_out,
        ffn_hc_x_pad,
        ffn_hc_fn,
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
    moe_hash_fwd(
        ffn_normed,
        gate_w_t,
        tid2eid,
        input_ids,
        routed_w1_t,
        routed_w2_t,
        routed_w3_t,
        shared_w1_t,
        shared_w2_t,
        shared_w3_t,
        logits,
        scores,
        indices,
        weights,
        route_y,
        shared_gate,
        shared_up,
        shared_hidden,
        shared_y,
        moe_out,
    )
    hc_post_fwd(moe_out, attn_hc_out, ffn_hc_post, ffn_hc_comb, out)
    return (
        kv_cache_out,
        attn_comp_kv_state_out,
        attn_comp_score_state_out,
        attn_comp_cache_out,
        idx_kv_cache_out,
        idx_comp_kv_state_out,
        idx_comp_score_state_out,
        out,
    )


@pl.jit
def block_csa_topk_prefill_fwd(
    x: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    attn_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
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
    attn_comp_cos: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    attn_comp_sin: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    attn_comp_block_count: pl.Tensor[[1], pl.INT32],
    idx_wq_b_t: pl.Tensor[[Q_LORA_RANK, INDEX_Q_OUT], pl.BF16],
    idx_weights_proj_t: pl.Tensor[[HIDDEN, INDEX_N_HEADS], pl.BF16],
    idx_offset: pl.Tensor[[1], pl.INT32],
    idx_comp_wkv_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    idx_comp_wgate_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    idx_comp_ape: pl.Tensor[[COMPRESS_RATIO4, INDEX_PROJ_DIM], pl.FP32],
    idx_comp_norm_w: pl.Tensor[[INDEX_HEAD_DIM], pl.BF16],
    idx_comp_cos: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    idx_comp_sin: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    idx_comp_block_count: pl.Tensor[[1], pl.INT32],
    ffn_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    ffn_hc_scale: pl.Tensor[[3], pl.FP32],
    ffn_hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[HIDDEN], pl.BF16],
    gate_w_t: pl.Tensor[[HIDDEN, N_EXPERTS], pl.BF16],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    routed_w1_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    routed_w2_t: pl.Tensor[[N_EXPERTS, MOE_INTER_DIM, HIDDEN], pl.BF16],
    routed_w3_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w1_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w2_t: pl.Tensor[[MOE_INTER_DIM, HIDDEN], pl.BF16],
    shared_w3_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    attn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    attn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    attn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    attn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_x_mixed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    attn_hc_post: pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32],
    attn_hc_comb: pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_normed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    q_a: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    q_proj: pl.Tensor[[B, S_DYN, ATTN_Q_OUT], pl.BF16],
    kv_proj: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    kv_normed: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    q: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    attn_comp_kv_proj: pl.Tensor[[B, S_DYN, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_score_proj: pl.Tensor[[B, S_DYN, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_pooled: pl.Tensor[[B, C_DYN, HEAD_DIM], pl.BF16],
    attn_comp_normed: pl.Tensor[[B, C_DYN, HEAD_DIM], pl.BF16],
    attn_compressed: pl.Tensor[[B, C_DYN, HEAD_DIM], pl.BF16],
    kv_pool: pl.Tensor[[B, K_DYN, HEAD_DIM], pl.BF16],
    kv_cache_out: pl.Out[pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16]],
    attn_comp_kv_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32]],
    attn_comp_score_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32]],
    attn_comp_cache_out: pl.Out[pl.Tensor[[B, TOPK_CSA_COMPRESSED, HEAD_DIM], pl.BF16]],
    idx_q_proj: pl.Tensor[[B, S_DYN, INDEX_Q_OUT], pl.BF16],
    idx_q_rope: pl.Tensor[[B, S_DYN, INDEX_N_HEADS, INDEX_HEAD_DIM], pl.BF16],
    idx_weights: pl.Tensor[[B, S_DYN, INDEX_N_HEADS], pl.BF16],
    idx_comp_kv_proj: pl.Tensor[[B, S_DYN, INDEX_PROJ_DIM], pl.FP32],
    idx_comp_score_proj: pl.Tensor[[B, S_DYN, INDEX_PROJ_DIM], pl.FP32],
    idx_comp_pooled: pl.Tensor[[B, C_DYN, INDEX_HEAD_DIM], pl.BF16],
    idx_comp_normed: pl.Tensor[[B, C_DYN, INDEX_HEAD_DIM], pl.BF16],
    idx_score: pl.Tensor[[B, S_DYN, INDEX_SCORE_LEN], pl.FP32],
    idx_topk_idxs: pl.Tensor[[B, S_DYN, INDEX_TOPK], pl.INT32],
    idx_kv_cache_out: pl.Out[pl.Tensor[[B, INDEX_SCORE_LEN, INDEX_HEAD_DIM], pl.BF16]],
    idx_comp_kv_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
    idx_comp_score_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
    csa_topk_idxs: pl.Tensor[[B, S_DYN, TOPK_CSA_TOTAL], pl.INT32],
    attn_o: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    o_inv: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    proj: pl.Tensor[[B, S_DYN, ATTN_OUT_IN], pl.BF16],
    attn_out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    attn_hc_out: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    ffn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    ffn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    ffn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    ffn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_hc_x_mixed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    ffn_hc_post: pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb: pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_normed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    logits: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    scores: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    indices: pl.Tensor[[B, S_DYN, TOPK], pl.INT32],
    weights: pl.Tensor[[B, S_DYN, TOPK], pl.FP32],
    route_y: pl.Tensor[[B, S_DYN, TOPK, HIDDEN], pl.BF16],
    shared_gate: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_up: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_hidden: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_y: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    moe_out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16]],
):
    """Run ordinary Block.forward for CSA attention + topk MoE, prefill path."""
    hc_pre_fwd(
        x,
        attn_hc_x_pad,
        attn_hc_fn,
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
    attention_csa_prefill_fwd(
        attn_normed,
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
        attn_comp_block_count,
        idx_wq_b_t,
        idx_weights_proj_t,
        idx_offset,
        idx_comp_wkv_t,
        idx_comp_wgate_t,
        idx_comp_ape,
        idx_comp_norm_w,
        idx_comp_cos,
        idx_comp_sin,
        idx_comp_block_count,
        q_a,
        q_proj,
        kv_proj,
        kv_normed,
        qr,
        q,
        kv,
        attn_comp_kv_proj,
        attn_comp_score_proj,
        attn_comp_pooled,
        attn_comp_normed,
        attn_compressed,
        kv_pool,
        kv_cache_out,
        attn_comp_kv_state_out,
        attn_comp_score_state_out,
        attn_comp_cache_out,
        idx_q_proj,
        idx_q_rope,
        idx_weights,
        idx_comp_kv_proj,
        idx_comp_score_proj,
        idx_comp_pooled,
        idx_comp_normed,
        idx_score,
        idx_topk_idxs,
        idx_kv_cache_out,
        idx_comp_kv_state_out,
        idx_comp_score_state_out,
        csa_topk_idxs,
        attn_o,
        o_inv,
        proj,
        attn_out,
    )
    hc_post_fwd(attn_out, x, attn_hc_post, attn_hc_comb, attn_hc_out)

    hc_pre_fwd(
        attn_hc_out,
        ffn_hc_x_pad,
        ffn_hc_fn,
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
    moe_topk_fwd(
        ffn_normed,
        gate_w_t,
        gate_bias,
        routed_w1_t,
        routed_w2_t,
        routed_w3_t,
        shared_w1_t,
        shared_w2_t,
        shared_w3_t,
        logits,
        scores,
        indices,
        weights,
        route_y,
        shared_gate,
        shared_up,
        shared_hidden,
        shared_y,
        moe_out,
    )
    hc_post_fwd(moe_out, attn_hc_out, ffn_hc_post, ffn_hc_comb, out)
    return (
        kv_cache_out,
        attn_comp_kv_state_out,
        attn_comp_score_state_out,
        attn_comp_cache_out,
        idx_kv_cache_out,
        idx_comp_kv_state_out,
        idx_comp_score_state_out,
        out,
    )


@pl.jit
def block_csa_topk_decode_fwd(
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
    attn_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
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
    ffn_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    ffn_hc_scale: pl.Tensor[[3], pl.FP32],
    ffn_hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[HIDDEN], pl.BF16],
    gate_w_t: pl.Tensor[[HIDDEN, N_EXPERTS], pl.BF16],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    routed_w1_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    routed_w2_t: pl.Tensor[[N_EXPERTS, MOE_INTER_DIM, HIDDEN], pl.BF16],
    routed_w3_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w1_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w2_t: pl.Tensor[[MOE_INTER_DIM, HIDDEN], pl.BF16],
    shared_w3_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    attn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    attn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    attn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    attn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_x_mixed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    attn_hc_post: pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32],
    attn_hc_comb: pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_normed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    q_a: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    q_proj: pl.Tensor[[B, S_DYN, ATTN_Q_OUT], pl.BF16],
    kv_proj: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    kv_normed: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    q: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    attn_comp_kv_proj: pl.Tensor[[B, S_DYN, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_score_proj: pl.Tensor[[B, S_DYN, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_pooled: pl.Tensor[[B, 1, HEAD_DIM], pl.BF16],
    attn_comp_normed: pl.Tensor[[B, 1, HEAD_DIM], pl.BF16],
    attn_compressed: pl.Tensor[[B, 1, HEAD_DIM], pl.BF16],
    kv_pool: pl.Tensor[[B, WINDOW_SIZE + TOPK_CSA_COMPRESSED, HEAD_DIM], pl.BF16],
    kv_cache_out: pl.Out[pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16]],
    attn_comp_kv_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32]],
    attn_comp_score_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32]],
    attn_comp_cache_out: pl.Out[pl.Tensor[[B, TOPK_CSA_COMPRESSED, HEAD_DIM], pl.BF16]],
    idx_q_proj: pl.Tensor[[B, S_DYN, INDEX_Q_OUT], pl.BF16],
    idx_q_rope: pl.Tensor[[B, S_DYN, INDEX_N_HEADS, INDEX_HEAD_DIM], pl.BF16],
    idx_weights: pl.Tensor[[B, S_DYN, INDEX_N_HEADS], pl.BF16],
    idx_comp_kv_proj: pl.Tensor[[B, S_DYN, INDEX_PROJ_DIM], pl.FP32],
    idx_comp_score_proj: pl.Tensor[[B, S_DYN, INDEX_PROJ_DIM], pl.FP32],
    idx_comp_pooled: pl.Tensor[[B, 1, INDEX_HEAD_DIM], pl.BF16],
    idx_comp_normed: pl.Tensor[[B, 1, INDEX_HEAD_DIM], pl.BF16],
    idx_score: pl.Tensor[[B, S_DYN, INDEX_SCORE_LEN], pl.FP32],
    idx_topk_idxs: pl.Tensor[[B, S_DYN, INDEX_TOPK], pl.INT32],
    idx_kv_cache_out: pl.Out[pl.Tensor[[B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM], pl.BF16]],
    idx_comp_kv_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
    idx_comp_score_state_out: pl.Out[pl.Tensor[[B, RATIO4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
    csa_topk_idxs: pl.Tensor[[B, S_DYN, TOPK_CSA_TOTAL], pl.INT32],
    attn_o: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    o_inv: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    proj: pl.Tensor[[B, S_DYN, ATTN_OUT_IN], pl.BF16],
    attn_out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    attn_hc_out: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    ffn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    ffn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    ffn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    ffn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_hc_x_mixed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    ffn_hc_post: pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb: pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_normed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    logits: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    scores: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    indices: pl.Tensor[[B, S_DYN, TOPK], pl.INT32],
    weights: pl.Tensor[[B, S_DYN, TOPK], pl.FP32],
    route_y: pl.Tensor[[B, S_DYN, TOPK, HIDDEN], pl.BF16],
    shared_gate: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_up: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_hidden: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_y: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    moe_out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16]],
):
    """Run ordinary Block.forward for CSA attention + topk MoE, decode path."""
    hc_pre_fwd(
        x,
        attn_hc_x_pad,
        attn_hc_fn,
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
        q_a,
        q_proj,
        kv_proj,
        kv_normed,
        qr,
        q,
        kv,
        attn_comp_kv_proj,
        attn_comp_score_proj,
        attn_comp_pooled,
        attn_comp_normed,
        attn_compressed,
        kv_pool,
        kv_cache_out,
        attn_comp_kv_state_out,
        attn_comp_score_state_out,
        attn_comp_cache_out,
        idx_q_proj,
        idx_q_rope,
        idx_weights,
        idx_comp_kv_proj,
        idx_comp_score_proj,
        idx_comp_pooled,
        idx_comp_normed,
        idx_score,
        idx_topk_idxs,
        idx_kv_cache_out,
        idx_comp_kv_state_out,
        idx_comp_score_state_out,
        csa_topk_idxs,
        attn_o,
        o_inv,
        proj,
        attn_out,
    )
    hc_post_fwd(attn_out, x, attn_hc_post, attn_hc_comb, attn_hc_out)

    hc_pre_fwd(
        attn_hc_out,
        ffn_hc_x_pad,
        ffn_hc_fn,
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
    moe_topk_fwd(
        ffn_normed,
        gate_w_t,
        gate_bias,
        routed_w1_t,
        routed_w2_t,
        routed_w3_t,
        shared_w1_t,
        shared_w2_t,
        shared_w3_t,
        logits,
        scores,
        indices,
        weights,
        route_y,
        shared_gate,
        shared_up,
        shared_hidden,
        shared_y,
        moe_out,
    )
    hc_post_fwd(moe_out, attn_hc_out, ffn_hc_post, ffn_hc_comb, out)
    return (
        kv_cache_out,
        attn_comp_kv_state_out,
        attn_comp_score_state_out,
        attn_comp_cache_out,
        idx_kv_cache_out,
        idx_comp_kv_state_out,
        idx_comp_score_state_out,
        out,
    )


@pl.jit
def block_hca_topk_prefill_fwd(
    x: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    attn_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
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
    comp_cos: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    comp_sin: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    comp_block_count: pl.Tensor[[1], pl.INT32],
    ffn_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    ffn_hc_scale: pl.Tensor[[3], pl.FP32],
    ffn_hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[HIDDEN], pl.BF16],
    gate_w_t: pl.Tensor[[HIDDEN, N_EXPERTS], pl.BF16],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    routed_w1_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    routed_w2_t: pl.Tensor[[N_EXPERTS, MOE_INTER_DIM, HIDDEN], pl.BF16],
    routed_w3_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w1_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w2_t: pl.Tensor[[MOE_INTER_DIM, HIDDEN], pl.BF16],
    shared_w3_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    attn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    attn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    attn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    attn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_x_mixed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    attn_hc_post: pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32],
    attn_hc_comb: pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_normed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    q_a: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    q_proj: pl.Tensor[[B, S_DYN, ATTN_Q_OUT], pl.BF16],
    kv_proj: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    kv_normed: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    q: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    comp_kv_proj: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.FP32],
    comp_score_proj: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.FP32],
    comp_pooled: pl.Tensor[[B, C_DYN, HEAD_DIM], pl.BF16],
    comp_normed: pl.Tensor[[B, C_DYN, HEAD_DIM], pl.BF16],
    compressed: pl.Tensor[[B, C_DYN, HEAD_DIM], pl.BF16],
    kv_pool: pl.Tensor[[B, K_DYN, HEAD_DIM], pl.BF16],
    kv_cache_out: pl.Out[pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16]],
    comp_kv_state_out: pl.Out[pl.Tensor[[B, COMPRESS_RATIO128, HEAD_DIM], pl.FP32]],
    comp_score_state_out: pl.Out[pl.Tensor[[B, COMPRESS_RATIO128, HEAD_DIM], pl.FP32]],
    comp_cache_out: pl.Out[pl.Tensor[[B, TOPK_HCA, HEAD_DIM], pl.BF16]],
    attn_o: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    o_inv: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    proj: pl.Tensor[[B, S_DYN, ATTN_OUT_IN], pl.BF16],
    attn_out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    attn_hc_out: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    ffn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    ffn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    ffn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    ffn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_hc_x_mixed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    ffn_hc_post: pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb: pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_normed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    logits: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    scores: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    indices: pl.Tensor[[B, S_DYN, TOPK], pl.INT32],
    weights: pl.Tensor[[B, S_DYN, TOPK], pl.FP32],
    route_y: pl.Tensor[[B, S_DYN, TOPK, HIDDEN], pl.BF16],
    shared_gate: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_up: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_hidden: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_y: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    moe_out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16]],
):
    """Run ordinary Block.forward for HCA attention + topk MoE, prefill path."""
    hc_pre_fwd(
        x,
        attn_hc_x_pad,
        attn_hc_fn,
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
    attention_hca_prefill_fwd(
        attn_normed,
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
        comp_block_count,
        q_a,
        q_proj,
        kv_proj,
        kv_normed,
        qr,
        q,
        kv,
        comp_kv_proj,
        comp_score_proj,
        comp_pooled,
        comp_normed,
        compressed,
        kv_pool,
        kv_cache_out,
        comp_kv_state_out,
        comp_score_state_out,
        comp_cache_out,
        attn_o,
        o_inv,
        proj,
        attn_out,
    )
    hc_post_fwd(attn_out, x, attn_hc_post, attn_hc_comb, attn_hc_out)

    hc_pre_fwd(
        attn_hc_out,
        ffn_hc_x_pad,
        ffn_hc_fn,
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
    moe_topk_fwd(
        ffn_normed,
        gate_w_t,
        gate_bias,
        routed_w1_t,
        routed_w2_t,
        routed_w3_t,
        shared_w1_t,
        shared_w2_t,
        shared_w3_t,
        logits,
        scores,
        indices,
        weights,
        route_y,
        shared_gate,
        shared_up,
        shared_hidden,
        shared_y,
        moe_out,
    )
    hc_post_fwd(moe_out, attn_hc_out, ffn_hc_post, ffn_hc_comb, out)
    return kv_cache_out, comp_kv_state_out, comp_score_state_out, comp_cache_out, out


@pl.jit
def block_hca_topk_decode_fwd(
    x: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    kv_cache: pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16],
    comp_kv_state: pl.Tensor[[B, COMPRESS_RATIO128, HEAD_DIM], pl.FP32],
    comp_score_state: pl.Tensor[[B, COMPRESS_RATIO128, HEAD_DIM], pl.FP32],
    comp_cache: pl.Tensor[[B, TOPK_HCA, HEAD_DIM], pl.BF16],
    cache_pos: pl.Tensor[[1], pl.INT32],
    comp_slot: pl.Tensor[[1], pl.INT32],
    comp_cache_slot: pl.Tensor[[1], pl.INT32],
    comp_should_compress: pl.Tensor[[1], pl.INT32],
    attn_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
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
    ffn_hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    ffn_hc_scale: pl.Tensor[[3], pl.FP32],
    ffn_hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    ffn_norm_w: pl.Tensor[[HIDDEN], pl.BF16],
    gate_w_t: pl.Tensor[[HIDDEN, N_EXPERTS], pl.BF16],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    routed_w1_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    routed_w2_t: pl.Tensor[[N_EXPERTS, MOE_INTER_DIM, HIDDEN], pl.BF16],
    routed_w3_t: pl.Tensor[[N_EXPERTS, HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w1_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    shared_w2_t: pl.Tensor[[MOE_INTER_DIM, HIDDEN], pl.BF16],
    shared_w3_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    attn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    attn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    attn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    attn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    attn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_hc_x_mixed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    attn_hc_post: pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32],
    attn_hc_comb: pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32],
    attn_normed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    q_a: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    q_proj: pl.Tensor[[B, S_DYN, ATTN_Q_OUT], pl.BF16],
    kv_proj: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    kv_normed: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    q: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    comp_kv_proj: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.FP32],
    comp_score_proj: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.FP32],
    comp_pooled: pl.Tensor[[B, 1, HEAD_DIM], pl.BF16],
    comp_normed: pl.Tensor[[B, 1, HEAD_DIM], pl.BF16],
    compressed: pl.Tensor[[B, 1, HEAD_DIM], pl.BF16],
    kv_pool: pl.Tensor[[B, WINDOW_SIZE + TOPK_HCA, HEAD_DIM], pl.BF16],
    kv_cache_out: pl.Out[pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16]],
    comp_kv_state_out: pl.Out[pl.Tensor[[B, COMPRESS_RATIO128, HEAD_DIM], pl.FP32]],
    comp_score_state_out: pl.Out[pl.Tensor[[B, COMPRESS_RATIO128, HEAD_DIM], pl.FP32]],
    comp_cache_out: pl.Out[pl.Tensor[[B, TOPK_HCA, HEAD_DIM], pl.BF16]],
    attn_o: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    o_inv: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    proj: pl.Tensor[[B, S_DYN, ATTN_OUT_IN], pl.BF16],
    attn_out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    attn_hc_out: pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16],
    ffn_hc_x_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT, HIDDEN], pl.BF16],
    ffn_hc_mixes: pl.Tensor[[B, S_PAD_DYN, MIX_PAD], pl.FP32],
    ffn_hc_pre: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_logits: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_hc_x_mixed_pad: pl.Tensor[[B, S_PAD_DYN, HIDDEN], pl.BF16],
    ffn_hc_post_pad: pl.Tensor[[B, S_PAD_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb_pad: pl.Tensor[[B, S_PAD_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_hc_x_mixed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    ffn_hc_post: pl.Tensor[[B, S_DYN, HC_PAD], pl.FP32],
    ffn_hc_comb: pl.Tensor[[B, S_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_normed: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    logits: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    scores: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    indices: pl.Tensor[[B, S_DYN, TOPK], pl.INT32],
    weights: pl.Tensor[[B, S_DYN, TOPK], pl.FP32],
    route_y: pl.Tensor[[B, S_DYN, TOPK, HIDDEN], pl.BF16],
    shared_gate: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_up: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_hidden: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    shared_y: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    moe_out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, HC_MULT, HIDDEN], pl.BF16]],
):
    """Run ordinary Block.forward for HCA attention + topk MoE, decode path."""
    hc_pre_fwd(
        x,
        attn_hc_x_pad,
        attn_hc_fn,
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
        q_a,
        q_proj,
        kv_proj,
        kv_normed,
        qr,
        q,
        kv,
        comp_kv_proj,
        comp_score_proj,
        comp_pooled,
        comp_normed,
        compressed,
        kv_pool,
        kv_cache_out,
        comp_kv_state_out,
        comp_score_state_out,
        comp_cache_out,
        attn_o,
        o_inv,
        proj,
        attn_out,
    )
    hc_post_fwd(attn_out, x, attn_hc_post, attn_hc_comb, attn_hc_out)

    hc_pre_fwd(
        attn_hc_out,
        ffn_hc_x_pad,
        ffn_hc_fn,
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
    moe_topk_fwd(
        ffn_normed,
        gate_w_t,
        gate_bias,
        routed_w1_t,
        routed_w2_t,
        routed_w3_t,
        shared_w1_t,
        shared_w2_t,
        shared_w3_t,
        logits,
        scores,
        indices,
        weights,
        route_y,
        shared_gate,
        shared_up,
        shared_hidden,
        shared_y,
        moe_out,
    )
    hc_post_fwd(moe_out, attn_hc_out, ffn_hc_post, ffn_hc_comb, out)
    return kv_cache_out, comp_kv_state_out, comp_score_state_out, comp_cache_out, out


def _prefixed(tensors: dict[str, torch.Tensor], prefix: str, key: str) -> torch.Tensor:
    return tensors[f"{prefix}_{key}"]


def _run_hc_pre(tensors: dict[str, torch.Tensor], *, prefix: str, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_tensors = {
        "x": x,
        "hc_fn": _prefixed(tensors, prefix, "hc_fn"),
        "hc_scale": _prefixed(tensors, prefix, "hc_scale"),
        "hc_base": _prefixed(tensors, prefix, "hc_base"),
        "x_mixed": _prefixed(tensors, prefix, "hc_x_mixed"),
        "post": _prefixed(tensors, prefix, "hc_post"),
        "comb": _prefixed(tensors, prefix, "hc_comb"),
    }
    golden_hc_pre(hc_tensors)
    return hc_tensors["x_mixed"], hc_tensors["post"], hc_tensors["comb"]


def _run_hc_post(
    tensors: dict[str, torch.Tensor],
    *,
    prefix: str,
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    out_key = f"{prefix}_hc_out"
    if out_key not in tensors and prefix == "ffn":
        out_key = "out"
    hc_tensors = {
        "x": x,
        "residual": residual,
        "post": post,
        "comb": comb,
        "out": tensors[out_key],
    }
    golden_hc_post(hc_tensors)
    return hc_tensors["out"]


def _run_hidden_rmsnorm(
    tensors: dict[str, torch.Tensor],
    *,
    x: torch.Tensor,
    norm_w_key: str,
    out_key: str,
) -> torch.Tensor:
    norm_tensors = {
        "x": x,
        "norm_w": tensors[norm_w_key],
        "out": tensors[out_key],
    }
    golden_rmsnorm(norm_tensors)
    return norm_tensors["out"]


def _run_attention(
    tensors: dict[str, torch.Tensor],
    *,
    x: torch.Tensor,
    start_pos: int,
    attention_kind: str,
) -> None:
    attention_tensors = dict(tensors)
    attention_tensors["x"] = x
    attention_tensors["out"] = tensors["attn_out"]

    if attention_kind == "swa":
        golden_attention_swa_forward(attention_tensors, start_pos=start_pos)
    elif attention_kind == "hca":
        golden_attention_hca_forward(attention_tensors, start_pos=start_pos)
    elif attention_kind == "csa":
        golden_attention_csa_forward(attention_tensors, start_pos=start_pos)
    else:
        raise ValueError(f"unsupported attention_kind: {attention_kind!r}")


def _run_moe(tensors: dict[str, torch.Tensor], *, x: torch.Tensor, hash_route: bool) -> None:
    moe_tensors = dict(tensors)
    moe_tensors["x"] = x
    moe_tensors["out"] = tensors["moe_out"]
    golden_moe_forward(moe_tensors, hash_route=hash_route)


def golden_block_forward(
    tensors: dict[str, torch.Tensor],
    *,
    start_pos: int,
    attention_kind: str,
    hash_route: bool,
) -> None:
    """Torch golden for official ``Block.forward``.

    The function mutates output tensors in ``tensors`` following the style used
    by the other module-level golden functions in this repository.
    """
    if start_pos < 0:
        raise ValueError(f"start_pos must be non-negative, got {start_pos}")

    x = tensors["x"]
    if start_pos > 0 and x.shape[1] != 1:
        raise ValueError(f"decode expects seq_len=1, got {x.shape[1]}")

    attn_residual = x
    attn_x, attn_post, attn_comb = _run_hc_pre(tensors, prefix="attn", x=x)
    attn_normed = _run_hidden_rmsnorm(tensors, x=attn_x, norm_w_key="attn_norm_w", out_key="attn_normed")
    _run_attention(tensors, x=attn_normed, start_pos=start_pos, attention_kind=attention_kind)
    attn_hc_out = _run_hc_post(
        tensors,
        prefix="attn",
        x=tensors["attn_out"],
        residual=attn_residual,
        post=attn_post,
        comb=attn_comb,
    )

    ffn_residual = attn_hc_out
    ffn_x, ffn_post, ffn_comb = _run_hc_pre(tensors, prefix="ffn", x=attn_hc_out)
    ffn_normed = _run_hidden_rmsnorm(tensors, x=ffn_x, norm_w_key="ffn_norm_w", out_key="ffn_normed")
    _run_moe(tensors, x=ffn_normed, hash_route=hash_route)
    out = _run_hc_post(
        tensors,
        prefix="ffn",
        x=tensors["moe_out"],
        residual=ffn_residual,
        post=ffn_post,
        comb=ffn_comb,
    )
    tensors["out"][:] = out


def golden_block_swa_hash_prefill(tensors: dict[str, torch.Tensor]) -> None:
    golden_block_forward(tensors, start_pos=0, attention_kind="swa", hash_route=True)


def golden_block_swa_hash_decode(tensors: dict[str, torch.Tensor], start_pos: int) -> None:
    golden_block_forward(tensors, start_pos=start_pos, attention_kind="swa", hash_route=True)


def golden_block_csa_hash_prefill(tensors: dict[str, torch.Tensor]) -> None:
    golden_block_forward(tensors, start_pos=0, attention_kind="csa", hash_route=True)


def golden_block_csa_hash_decode(tensors: dict[str, torch.Tensor], start_pos: int) -> None:
    golden_block_forward(tensors, start_pos=start_pos, attention_kind="csa", hash_route=True)


def golden_block_hca_topk_prefill(tensors: dict[str, torch.Tensor]) -> None:
    golden_block_forward(tensors, start_pos=0, attention_kind="hca", hash_route=False)


def golden_block_hca_topk_decode(tensors: dict[str, torch.Tensor], start_pos: int) -> None:
    golden_block_forward(tensors, start_pos=start_pos, attention_kind="hca", hash_route=False)


def golden_block_csa_topk_prefill(tensors: dict[str, torch.Tensor]) -> None:
    golden_block_forward(tensors, start_pos=0, attention_kind="csa", hash_route=False)


def golden_block_csa_topk_decode(tensors: dict[str, torch.Tensor], start_pos: int) -> None:
    golden_block_forward(tensors, start_pos=start_pos, attention_kind="csa", hash_route=False)


def _build_swa_hash_specs(seq_len: int, start_pos: int, *, decode: bool):
    from models.golden import TensorSpec

    seq_pad = ceil_div(seq_len, HC_T_TILE) * HC_T_TILE
    freqs_cos, freqs_sin = build_deepseek_v4_rope_tables(max_seq_len=start_pos + seq_len)
    local_cos, local_sin = materialize_rope_range(freqs_cos, freqs_sin, start_pos, seq_len)

    def init_x():
        return (torch.randn(B, seq_len, HC_MULT, HIDDEN, dtype=torch.float32) * 0.05).to(torch.bfloat16)

    def init_hc_fn():
        return torch.randn(MIX_HC, HC_DIM, dtype=torch.float32) * 0.01

    def init_hc_scale():
        return torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

    def init_hc_base():
        return torch.zeros(MIX_HC, dtype=torch.float32)

    def init_norm_w():
        return (torch.randn(HIDDEN, dtype=torch.float32) * 0.05 + 1.0).to(torch.bfloat16)

    def init_wq_a_t():
        return torch.randn(HIDDEN, Q_LORA_RANK, dtype=torch.float32) * 0.01

    def init_q_norm_w():
        return (torch.randn(Q_LORA_RANK, dtype=torch.float32) * 0.05 + 1.0).to(torch.bfloat16)

    def init_wq_b_t():
        return torch.randn(Q_LORA_RANK, ATTN_Q_OUT, dtype=torch.float32) * 0.005

    def init_wkv_t():
        return torch.randn(HIDDEN, HEAD_DIM, dtype=torch.float32) * 0.01

    def init_kv_norm_w():
        return (torch.randn(HEAD_DIM, dtype=torch.float32) * 0.05 + 1.0).to(torch.bfloat16)

    def init_attn_sink():
        return torch.randn(N_HEADS, dtype=torch.float32) * 0.1

    def init_wo_a_t():
        return torch.randn(O_GROUP_IN, ATTN_OUT_IN, dtype=torch.float32) * 0.005

    def init_wo_b_t():
        return torch.randn(ATTN_OUT_IN, HIDDEN, dtype=torch.float32) * 0.005

    def init_gate_w_t():
        return torch.randn(HIDDEN, N_EXPERTS, dtype=torch.float32) * 0.01

    def init_tid2eid():
        base = torch.arange(TOPK, dtype=torch.int32).view(1, TOPK)
        token_offsets = torch.arange(VOCAB, dtype=torch.int32).view(VOCAB, 1)
        return (base + token_offsets) % N_EXPERTS

    def init_input_ids():
        return torch.randint(0, VOCAB, (B, seq_len), dtype=torch.int64)

    def init_routed_w1_t():
        return torch.randn(N_EXPERTS, HIDDEN, MOE_INTER_DIM, dtype=torch.float32) * 0.005

    def init_routed_w2_t():
        return torch.randn(N_EXPERTS, MOE_INTER_DIM, HIDDEN, dtype=torch.float32) * 0.005

    def init_routed_w3_t():
        return torch.randn(N_EXPERTS, HIDDEN, MOE_INTER_DIM, dtype=torch.float32) * 0.005

    def init_shared_w1_t():
        return torch.randn(HIDDEN, MOE_INTER_DIM, dtype=torch.float32) * 0.005

    def init_shared_w2_t():
        return torch.randn(MOE_INTER_DIM, HIDDEN, dtype=torch.float32) * 0.005

    def init_shared_w3_t():
        return torch.randn(HIDDEN, MOE_INTER_DIM, dtype=torch.float32) * 0.005

    specs = [
        TensorSpec("x", [B, seq_len, HC_MULT, HIDDEN], torch.bfloat16, init_value=init_x),
    ]
    if decode:
        specs.extend(
            [
                TensorSpec(
                    "kv_cache",
                    [B, WINDOW_SIZE, HEAD_DIM],
                    torch.bfloat16,
                    init_value=lambda: torch.randn(B, WINDOW_SIZE, HEAD_DIM, dtype=torch.float32) * 0.05,
                ),
                TensorSpec("cache_pos", [1], torch.int32, init_value=torch.tensor([start_pos % WINDOW_SIZE], dtype=torch.int32)),
            ]
        )

    specs.extend(
        [
            TensorSpec("attn_hc_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_fn),
            TensorSpec("attn_hc_scale", [3], torch.float32, init_value=init_hc_scale),
            TensorSpec("attn_hc_base", [MIX_HC], torch.float32, init_value=init_hc_base),
            TensorSpec("attn_norm_w", [HIDDEN], torch.bfloat16, init_value=init_norm_w),
            TensorSpec("wq_a_t", [HIDDEN, Q_LORA_RANK], torch.bfloat16, init_value=init_wq_a_t),
            TensorSpec("q_norm_w", [Q_LORA_RANK], torch.bfloat16, init_value=init_q_norm_w),
            TensorSpec("wq_b_t", [Q_LORA_RANK, ATTN_Q_OUT], torch.bfloat16, init_value=init_wq_b_t),
            TensorSpec("wkv_t", [HIDDEN, HEAD_DIM], torch.bfloat16, init_value=init_wkv_t),
            TensorSpec("kv_norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_kv_norm_w),
            TensorSpec("attn_sink", [N_HEADS], torch.float32, init_value=init_attn_sink),
            TensorSpec(
                "topk_idxs",
                [B, seq_len, TOPK_SWA],
                torch.int32,
                init_value=lambda: build_window_topk_idxs(seq_len, start_pos=start_pos, topk_max=TOPK_SWA),
            ),
            TensorSpec("wo_a_t", [O_GROUP_IN, ATTN_OUT_IN], torch.bfloat16, init_value=init_wo_a_t),
            TensorSpec("wo_b_t", [ATTN_OUT_IN, HIDDEN], torch.bfloat16, init_value=init_wo_b_t),
            TensorSpec("cos", [seq_len, ROPE_HALF], torch.float32, init_value=local_cos),
            TensorSpec("sin", [seq_len, ROPE_HALF], torch.float32, init_value=local_sin),
            TensorSpec("ffn_hc_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_fn),
            TensorSpec("ffn_hc_scale", [3], torch.float32, init_value=init_hc_scale),
            TensorSpec("ffn_hc_base", [MIX_HC], torch.float32, init_value=init_hc_base),
            TensorSpec("ffn_norm_w", [HIDDEN], torch.bfloat16, init_value=init_norm_w),
            TensorSpec("gate_w_t", [HIDDEN, N_EXPERTS], torch.bfloat16, init_value=init_gate_w_t),
            TensorSpec("tid2eid", [VOCAB, TOPK], torch.int32, init_value=init_tid2eid),
            TensorSpec("input_ids", [B, seq_len], torch.int64, init_value=init_input_ids),
            TensorSpec("routed_w1_t", [N_EXPERTS, HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=init_routed_w1_t),
            TensorSpec("routed_w2_t", [N_EXPERTS, MOE_INTER_DIM, HIDDEN], torch.bfloat16, init_value=init_routed_w2_t),
            TensorSpec("routed_w3_t", [N_EXPERTS, HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=init_routed_w3_t),
            TensorSpec("shared_w1_t", [HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=init_shared_w1_t),
            TensorSpec("shared_w2_t", [MOE_INTER_DIM, HIDDEN], torch.bfloat16, init_value=init_shared_w2_t),
            TensorSpec("shared_w3_t", [HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=init_shared_w3_t),
            TensorSpec("attn_hc_x_pad", [B, seq_pad, HC_MULT, HIDDEN], torch.bfloat16),
            TensorSpec("attn_hc_mixes", [B, seq_pad, MIX_PAD], torch.float32),
            TensorSpec("attn_hc_pre", [B, seq_pad, HC_PAD], torch.float32),
            TensorSpec("attn_hc_comb_logits", [B, seq_pad, HC_MULT * HC_MULT], torch.float32),
            TensorSpec("attn_hc_x_mixed_pad", [B, seq_pad, HIDDEN], torch.bfloat16),
            TensorSpec("attn_hc_post_pad", [B, seq_pad, HC_PAD], torch.float32),
            TensorSpec("attn_hc_comb_pad", [B, seq_pad, HC_MULT * HC_MULT], torch.float32),
            TensorSpec("attn_hc_x_mixed", [B, seq_len, HIDDEN], torch.bfloat16),
            TensorSpec("attn_hc_post", [B, seq_len, HC_PAD], torch.float32),
            TensorSpec("attn_hc_comb", [B, seq_len, HC_MULT * HC_MULT], torch.float32),
            TensorSpec("attn_normed", [B, seq_len, HIDDEN], torch.bfloat16),
            TensorSpec("q_a", [B, seq_len, Q_LORA_RANK], torch.bfloat16),
            TensorSpec("q_proj", [B, seq_len, ATTN_Q_OUT], torch.bfloat16),
            TensorSpec("kv_proj", [B, seq_len, HEAD_DIM], torch.bfloat16),
            TensorSpec("kv_normed", [B, seq_len, HEAD_DIM], torch.bfloat16),
            TensorSpec("qr", [B, seq_len, Q_LORA_RANK], torch.bfloat16),
            TensorSpec("q", [B, seq_len, N_HEADS, HEAD_DIM], torch.bfloat16),
            TensorSpec("kv", [B, seq_len, HEAD_DIM], torch.bfloat16),
            TensorSpec("kv_cache_out", [B, WINDOW_SIZE, HEAD_DIM], torch.bfloat16, is_output=True, init_value=0.0),
            TensorSpec("attn_o", [B, seq_len, N_HEADS, HEAD_DIM], torch.bfloat16),
            TensorSpec("o_inv", [B, seq_len, N_HEADS, HEAD_DIM], torch.bfloat16),
            TensorSpec("proj", [B, seq_len, ATTN_OUT_IN], torch.bfloat16),
            TensorSpec("attn_out", [B, seq_len, HIDDEN], torch.bfloat16),
            TensorSpec("attn_hc_out", [B, seq_len, HC_MULT, HIDDEN], torch.bfloat16),
            TensorSpec("ffn_hc_x_pad", [B, seq_pad, HC_MULT, HIDDEN], torch.bfloat16),
            TensorSpec("ffn_hc_mixes", [B, seq_pad, MIX_PAD], torch.float32),
            TensorSpec("ffn_hc_pre", [B, seq_pad, HC_PAD], torch.float32),
            TensorSpec("ffn_hc_comb_logits", [B, seq_pad, HC_MULT * HC_MULT], torch.float32),
            TensorSpec("ffn_hc_x_mixed_pad", [B, seq_pad, HIDDEN], torch.bfloat16),
            TensorSpec("ffn_hc_post_pad", [B, seq_pad, HC_PAD], torch.float32),
            TensorSpec("ffn_hc_comb_pad", [B, seq_pad, HC_MULT * HC_MULT], torch.float32),
            TensorSpec("ffn_hc_x_mixed", [B, seq_len, HIDDEN], torch.bfloat16),
            TensorSpec("ffn_hc_post", [B, seq_len, HC_PAD], torch.float32),
            TensorSpec("ffn_hc_comb", [B, seq_len, HC_MULT * HC_MULT], torch.float32),
            TensorSpec("ffn_normed", [B, seq_len, HIDDEN], torch.bfloat16),
            TensorSpec("logits", [B, seq_len, N_EXPERTS], torch.float32),
            TensorSpec("scores", [B, seq_len, N_EXPERTS], torch.float32),
            TensorSpec("indices", [B, seq_len, TOPK], torch.int32),
            TensorSpec("weights", [B, seq_len, TOPK], torch.float32),
            TensorSpec("route_y", [B, seq_len, TOPK, HIDDEN], torch.bfloat16),
            TensorSpec("shared_gate", [B, seq_len, MOE_INTER_DIM], torch.bfloat16),
            TensorSpec("shared_up", [B, seq_len, MOE_INTER_DIM], torch.bfloat16),
            TensorSpec("shared_hidden", [B, seq_len, MOE_INTER_DIM], torch.bfloat16),
            TensorSpec("shared_y", [B, seq_len, HIDDEN], torch.bfloat16),
            TensorSpec("moe_out", [B, seq_len, HIDDEN], torch.bfloat16),
            TensorSpec("out", [B, seq_len, HC_MULT, HIDDEN], torch.bfloat16, is_output=True),
        ]
    )
    return specs


def build_swa_hash_prefill_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _build_swa_hash_specs(seq_len, start_pos=0, decode=False)


def build_swa_hash_decode_specs(start_pos: int = DEFAULT_DECODE_START_POS):
    if start_pos <= 0:
        raise ValueError(f"decode start_pos must be positive, got {start_pos}")
    return _build_swa_hash_specs(1, start_pos=start_pos, decode=True)


def _build_csa_specs(seq_len: int, start_pos: int, *, decode: bool, hash_route: bool):
    from models.golden import TensorSpec

    seq_pad = ceil_div(seq_len, HC_T_TILE) * HC_T_TILE
    freqs_cos, freqs_sin = build_deepseek_v4_rope_tables(
        compress_ratio=COMPRESS_RATIO4,
        max_seq_len=start_pos + seq_len,
    )
    local_cos, local_sin = materialize_rope_range(freqs_cos, freqs_sin, start_pos, seq_len)
    if decode:
        comp_slot = start_pos % COMPRESS_RATIO4
        comp_cache_slot = start_pos // COMPRESS_RATIO4
        comp_should_compress = int((start_pos + 1) % COMPRESS_RATIO4 == 0)
        if comp_should_compress:
            comp_rope_pos = start_pos + 1 - COMPRESS_RATIO4
            comp_cos = freqs_cos[comp_rope_pos : comp_rope_pos + 1].contiguous()
            comp_sin = freqs_sin[comp_rope_pos : comp_rope_pos + 1].contiguous()
        else:
            comp_cos = torch.zeros(1, ROPE_HALF, dtype=torch.float32)
            comp_sin = torch.zeros(1, ROPE_HALF, dtype=torch.float32)
        compressed_len = 1
        kv_pool_len = WINDOW_SIZE + TOPK_CSA_COMPRESSED
        block_count = 0
    else:
        block_count = seq_len // COMPRESS_RATIO4
        compressed_len = max(1, block_count)
        comp_cos, comp_sin = materialize_compressor_rope(freqs_cos, freqs_sin, seq_len, COMPRESS_RATIO4)
        if block_count == 0:
            comp_cos = freqs_cos[:1].contiguous()
            comp_sin = freqs_sin[:1].contiguous()
        kv_pool_len = seq_len + compressed_len

    def init_x():
        return (torch.randn(B, seq_len, HC_MULT, HIDDEN, dtype=torch.float32) * 0.05).to(torch.bfloat16)

    def init_hc_fn():
        return torch.randn(MIX_HC, HC_DIM, dtype=torch.float32) * 0.01

    def init_hc_scale():
        return torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

    def init_hc_base():
        return torch.zeros(MIX_HC, dtype=torch.float32)

    def init_norm_w():
        return (torch.randn(HIDDEN, dtype=torch.float32) * 0.05 + 1.0).to(torch.bfloat16)

    def init_wq_a_t():
        return torch.randn(HIDDEN, Q_LORA_RANK, dtype=torch.float32) * 0.01

    def init_q_norm_w():
        return (torch.randn(Q_LORA_RANK, dtype=torch.float32) * 0.05 + 1.0).to(torch.bfloat16)

    def init_wq_b_t():
        return torch.randn(Q_LORA_RANK, ATTN_Q_OUT, dtype=torch.float32) * 0.005

    def init_wkv_t():
        return torch.randn(HIDDEN, HEAD_DIM, dtype=torch.float32) * 0.01

    def init_kv_norm_w():
        return (torch.randn(HEAD_DIM, dtype=torch.float32) * 0.05 + 1.0).to(torch.bfloat16)

    def init_attn_sink():
        return torch.randn(N_HEADS, dtype=torch.float32) * 0.1

    def init_wo_a_t():
        return torch.randn(O_GROUP_IN, ATTN_OUT_IN, dtype=torch.float32) * 0.005

    def init_wo_b_t():
        return torch.randn(ATTN_OUT_IN, HIDDEN, dtype=torch.float32) * 0.005

    def init_attn_comp_w():
        return torch.randn(HIDDEN, ATTN_PROJ_DIM, dtype=torch.float32) * 0.01

    def init_attn_comp_ape():
        return torch.randn(COMPRESS_RATIO4, ATTN_PROJ_DIM, dtype=torch.float32) * 0.01

    def init_attn_comp_norm_w():
        return (torch.randn(HEAD_DIM, dtype=torch.float32) * 0.05 + 1.0).to(torch.bfloat16)

    def init_idx_wq_b_t():
        return torch.randn(Q_LORA_RANK, INDEX_Q_OUT, dtype=torch.float32) * 0.01

    def init_idx_weights_proj_t():
        return torch.randn(HIDDEN, INDEX_N_HEADS, dtype=torch.float32) * 0.01

    def init_idx_comp_w():
        return torch.randn(HIDDEN, INDEX_PROJ_DIM, dtype=torch.float32) * 0.01

    def init_idx_comp_ape():
        return torch.randn(COMPRESS_RATIO4, INDEX_PROJ_DIM, dtype=torch.float32) * 0.01

    def init_idx_comp_norm_w():
        return (torch.randn(INDEX_HEAD_DIM, dtype=torch.float32) * 0.05 + 1.0).to(torch.bfloat16)

    def init_gate_w_t():
        return torch.randn(HIDDEN, N_EXPERTS, dtype=torch.float32) * 0.01

    def init_gate_bias():
        return torch.randn(N_EXPERTS, dtype=torch.float32) * 0.1

    def init_tid2eid():
        base = torch.arange(TOPK, dtype=torch.int32).view(1, TOPK)
        token_offsets = torch.arange(VOCAB, dtype=torch.int32).view(VOCAB, 1)
        return (base + token_offsets) % N_EXPERTS

    def init_input_ids():
        return torch.randint(0, VOCAB, (B, seq_len), dtype=torch.int64)

    def init_routed_w1_t():
        return torch.randn(N_EXPERTS, HIDDEN, MOE_INTER_DIM, dtype=torch.float32) * 0.005

    def init_routed_w2_t():
        return torch.randn(N_EXPERTS, MOE_INTER_DIM, HIDDEN, dtype=torch.float32) * 0.005

    def init_routed_w3_t():
        return torch.randn(N_EXPERTS, HIDDEN, MOE_INTER_DIM, dtype=torch.float32) * 0.005

    def init_shared_w1_t():
        return torch.randn(HIDDEN, MOE_INTER_DIM, dtype=torch.float32) * 0.005

    def init_shared_w2_t():
        return torch.randn(MOE_INTER_DIM, HIDDEN, dtype=torch.float32) * 0.005

    def init_shared_w3_t():
        return torch.randn(HIDDEN, MOE_INTER_DIM, dtype=torch.float32) * 0.005

    specs = [TensorSpec("x", [B, seq_len, HC_MULT, HIDDEN], torch.bfloat16, init_value=init_x)]
    if decode:
        specs.extend(
            [
                TensorSpec(
                    "kv_cache",
                    [B, WINDOW_SIZE, HEAD_DIM],
                    torch.bfloat16,
                    init_value=lambda: torch.randn(B, WINDOW_SIZE, HEAD_DIM, dtype=torch.float32) * 0.05,
                ),
                TensorSpec(
                    "attn_comp_kv_state",
                    [B, RATIO4_STATE_ROWS, ATTN_PROJ_DIM],
                    torch.float32,
                    init_value=lambda: torch.randn(B, RATIO4_STATE_ROWS, ATTN_PROJ_DIM, dtype=torch.float32) * 0.05,
                ),
                TensorSpec(
                    "attn_comp_score_state",
                    [B, RATIO4_STATE_ROWS, ATTN_PROJ_DIM],
                    torch.float32,
                    init_value=lambda: torch.randn(B, RATIO4_STATE_ROWS, ATTN_PROJ_DIM, dtype=torch.float32) * 0.05,
                ),
                TensorSpec(
                    "attn_comp_cache",
                    [B, TOPK_CSA_COMPRESSED, HEAD_DIM],
                    torch.bfloat16,
                    init_value=lambda: torch.randn(B, TOPK_CSA_COMPRESSED, HEAD_DIM, dtype=torch.float32) * 0.05,
                ),
                TensorSpec(
                    "idx_kv_cache_in",
                    [B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM],
                    torch.bfloat16,
                    init_value=lambda: torch.randn(B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM, dtype=torch.float32) * 0.05,
                ),
                TensorSpec(
                    "idx_comp_kv_state",
                    [B, RATIO4_STATE_ROWS, INDEX_PROJ_DIM],
                    torch.float32,
                    init_value=lambda: torch.randn(B, RATIO4_STATE_ROWS, INDEX_PROJ_DIM, dtype=torch.float32) * 0.05,
                ),
                TensorSpec(
                    "idx_comp_score_state",
                    [B, RATIO4_STATE_ROWS, INDEX_PROJ_DIM],
                    torch.float32,
                    init_value=lambda: torch.randn(B, RATIO4_STATE_ROWS, INDEX_PROJ_DIM, dtype=torch.float32) * 0.05,
                ),
                TensorSpec("cache_pos", [1], torch.int32, init_value=torch.tensor([start_pos % WINDOW_SIZE], dtype=torch.int32)),
                TensorSpec("comp_slot", [1], torch.int32, init_value=torch.tensor([comp_slot], dtype=torch.int32)),
                TensorSpec("comp_cache_slot", [1], torch.int32, init_value=torch.tensor([comp_cache_slot], dtype=torch.int32)),
                TensorSpec("comp_should_compress", [1], torch.int32, init_value=torch.tensor([comp_should_compress], dtype=torch.int32)),
            ]
        )

    specs.extend(
        [
            TensorSpec("attn_hc_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_fn),
            TensorSpec("attn_hc_scale", [3], torch.float32, init_value=init_hc_scale),
            TensorSpec("attn_hc_base", [MIX_HC], torch.float32, init_value=init_hc_base),
            TensorSpec("attn_norm_w", [HIDDEN], torch.bfloat16, init_value=init_norm_w),
            TensorSpec("wq_a_t", [HIDDEN, Q_LORA_RANK], torch.bfloat16, init_value=init_wq_a_t),
            TensorSpec("q_norm_w", [Q_LORA_RANK], torch.bfloat16, init_value=init_q_norm_w),
            TensorSpec("wq_b_t", [Q_LORA_RANK, ATTN_Q_OUT], torch.bfloat16, init_value=init_wq_b_t),
            TensorSpec("wkv_t", [HIDDEN, HEAD_DIM], torch.bfloat16, init_value=init_wkv_t),
            TensorSpec("kv_norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_kv_norm_w),
            TensorSpec("attn_sink", [N_HEADS], torch.float32, init_value=init_attn_sink),
            TensorSpec(
                "window_topk_idxs",
                [B, seq_len, TOPK_SWA],
                torch.int32,
                init_value=lambda: build_window_topk_idxs(seq_len, start_pos=start_pos, topk_max=TOPK_SWA),
            ),
            TensorSpec("wo_a_t", [O_GROUP_IN, ATTN_OUT_IN], torch.bfloat16, init_value=init_wo_a_t),
            TensorSpec("wo_b_t", [ATTN_OUT_IN, HIDDEN], torch.bfloat16, init_value=init_wo_b_t),
            TensorSpec("cos", [seq_len, ROPE_HALF], torch.float32, init_value=local_cos),
            TensorSpec("sin", [seq_len, ROPE_HALF], torch.float32, init_value=local_sin),
            TensorSpec("attn_comp_wkv_t", [HIDDEN, ATTN_PROJ_DIM], torch.bfloat16, init_value=init_attn_comp_w),
            TensorSpec("attn_comp_wgate_t", [HIDDEN, ATTN_PROJ_DIM], torch.bfloat16, init_value=init_attn_comp_w),
            TensorSpec("attn_comp_ape", [COMPRESS_RATIO4, ATTN_PROJ_DIM], torch.float32, init_value=init_attn_comp_ape),
            TensorSpec("attn_comp_norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_attn_comp_norm_w),
            TensorSpec("attn_comp_cos", [compressed_len, ROPE_HALF], torch.float32, init_value=comp_cos),
            TensorSpec("attn_comp_sin", [compressed_len, ROPE_HALF], torch.float32, init_value=comp_sin),
        ]
    )
    if not decode:
        specs.append(TensorSpec("attn_comp_block_count", [1], torch.int32, init_value=torch.tensor([block_count], dtype=torch.int32)))
    specs.extend(
        [
            TensorSpec("idx_wq_b_t", [Q_LORA_RANK, INDEX_Q_OUT], torch.bfloat16, init_value=init_idx_wq_b_t),
            TensorSpec("idx_weights_proj_t", [HIDDEN, INDEX_N_HEADS], torch.bfloat16, init_value=init_idx_weights_proj_t),
            TensorSpec("idx_offset", [1], torch.int32, init_value=torch.tensor([WINDOW_SIZE if decode else seq_len], dtype=torch.int32)),
            TensorSpec("idx_comp_wkv_t", [HIDDEN, INDEX_PROJ_DIM], torch.bfloat16, init_value=init_idx_comp_w),
            TensorSpec("idx_comp_wgate_t", [HIDDEN, INDEX_PROJ_DIM], torch.bfloat16, init_value=init_idx_comp_w),
            TensorSpec("idx_comp_ape", [COMPRESS_RATIO4, INDEX_PROJ_DIM], torch.float32, init_value=init_idx_comp_ape),
            TensorSpec("idx_comp_norm_w", [INDEX_HEAD_DIM], torch.bfloat16, init_value=init_idx_comp_norm_w),
            TensorSpec("idx_comp_cos", [compressed_len, ROPE_HALF], torch.float32, init_value=comp_cos.clone()),
            TensorSpec("idx_comp_sin", [compressed_len, ROPE_HALF], torch.float32, init_value=comp_sin.clone()),
        ]
    )
    if not decode:
        specs.append(TensorSpec("idx_comp_block_count", [1], torch.int32, init_value=torch.tensor([block_count], dtype=torch.int32)))
    specs.extend(
        [
            TensorSpec("ffn_hc_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_fn),
            TensorSpec("ffn_hc_scale", [3], torch.float32, init_value=init_hc_scale),
            TensorSpec("ffn_hc_base", [MIX_HC], torch.float32, init_value=init_hc_base),
            TensorSpec("ffn_norm_w", [HIDDEN], torch.bfloat16, init_value=init_norm_w),
            TensorSpec("gate_w_t", [HIDDEN, N_EXPERTS], torch.bfloat16, init_value=init_gate_w_t),
        ]
    )
    if hash_route:
        specs.extend(
            [
                TensorSpec("tid2eid", [VOCAB, TOPK], torch.int32, init_value=init_tid2eid),
                TensorSpec("input_ids", [B, seq_len], torch.int64, init_value=init_input_ids),
            ]
        )
    else:
        specs.append(TensorSpec("gate_bias", [N_EXPERTS], torch.float32, init_value=init_gate_bias))
    specs.extend(
        [
            TensorSpec("routed_w1_t", [N_EXPERTS, HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=init_routed_w1_t),
            TensorSpec("routed_w2_t", [N_EXPERTS, MOE_INTER_DIM, HIDDEN], torch.bfloat16, init_value=init_routed_w2_t),
            TensorSpec("routed_w3_t", [N_EXPERTS, HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=init_routed_w3_t),
            TensorSpec("shared_w1_t", [HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=init_shared_w1_t),
            TensorSpec("shared_w2_t", [MOE_INTER_DIM, HIDDEN], torch.bfloat16, init_value=init_shared_w2_t),
            TensorSpec("shared_w3_t", [HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=init_shared_w3_t),
            TensorSpec("attn_hc_x_pad", [B, seq_pad, HC_MULT, HIDDEN], torch.bfloat16),
            TensorSpec("attn_hc_mixes", [B, seq_pad, MIX_PAD], torch.float32),
            TensorSpec("attn_hc_pre", [B, seq_pad, HC_PAD], torch.float32),
            TensorSpec("attn_hc_comb_logits", [B, seq_pad, HC_MULT * HC_MULT], torch.float32),
            TensorSpec("attn_hc_x_mixed_pad", [B, seq_pad, HIDDEN], torch.bfloat16),
            TensorSpec("attn_hc_post_pad", [B, seq_pad, HC_PAD], torch.float32),
            TensorSpec("attn_hc_comb_pad", [B, seq_pad, HC_MULT * HC_MULT], torch.float32),
            TensorSpec("attn_hc_x_mixed", [B, seq_len, HIDDEN], torch.bfloat16),
            TensorSpec("attn_hc_post", [B, seq_len, HC_PAD], torch.float32),
            TensorSpec("attn_hc_comb", [B, seq_len, HC_MULT * HC_MULT], torch.float32),
            TensorSpec("attn_normed", [B, seq_len, HIDDEN], torch.bfloat16),
            TensorSpec("q_a", [B, seq_len, Q_LORA_RANK], torch.bfloat16),
            TensorSpec("q_proj", [B, seq_len, ATTN_Q_OUT], torch.bfloat16),
            TensorSpec("kv_proj", [B, seq_len, HEAD_DIM], torch.bfloat16),
            TensorSpec("kv_normed", [B, seq_len, HEAD_DIM], torch.bfloat16),
            TensorSpec("qr", [B, seq_len, Q_LORA_RANK], torch.bfloat16),
            TensorSpec("q", [B, seq_len, N_HEADS, HEAD_DIM], torch.bfloat16),
            TensorSpec("kv", [B, seq_len, HEAD_DIM], torch.bfloat16),
            TensorSpec("attn_comp_kv_proj", [B, seq_len, ATTN_PROJ_DIM], torch.float32),
            TensorSpec("attn_comp_score_proj", [B, seq_len, ATTN_PROJ_DIM], torch.float32),
            TensorSpec("attn_comp_pooled", [B, compressed_len, HEAD_DIM], torch.bfloat16),
            TensorSpec("attn_comp_normed", [B, compressed_len, HEAD_DIM], torch.bfloat16),
            TensorSpec("attn_compressed", [B, compressed_len, HEAD_DIM], torch.bfloat16),
            TensorSpec("kv_pool", [B, kv_pool_len, HEAD_DIM], torch.bfloat16),
            TensorSpec("kv_cache_out", [B, WINDOW_SIZE, HEAD_DIM], torch.bfloat16, is_output=True, init_value=0.0),
            TensorSpec("attn_comp_kv_state_out", [B, RATIO4_STATE_ROWS, ATTN_PROJ_DIM], torch.float32, is_output=True),
            TensorSpec("attn_comp_score_state_out", [B, RATIO4_STATE_ROWS, ATTN_PROJ_DIM], torch.float32, is_output=True),
            TensorSpec("attn_comp_cache_out", [B, TOPK_CSA_COMPRESSED, HEAD_DIM], torch.bfloat16, is_output=True),
            TensorSpec("idx_q_proj", [B, seq_len, INDEX_Q_OUT], torch.bfloat16),
            TensorSpec("idx_q_rope", [B, seq_len, INDEX_N_HEADS, INDEX_HEAD_DIM], torch.bfloat16),
            TensorSpec("idx_weights", [B, seq_len, INDEX_N_HEADS], torch.bfloat16),
            TensorSpec("idx_comp_kv_proj", [B, seq_len, INDEX_PROJ_DIM], torch.float32),
            TensorSpec("idx_comp_score_proj", [B, seq_len, INDEX_PROJ_DIM], torch.float32),
            TensorSpec("idx_comp_pooled", [B, compressed_len, INDEX_HEAD_DIM], torch.bfloat16),
            TensorSpec("idx_comp_normed", [B, compressed_len, INDEX_HEAD_DIM], torch.bfloat16),
            TensorSpec("idx_score", [B, seq_len, INDEX_SCORE_LEN], torch.float32),
            TensorSpec("idx_topk_idxs", [B, seq_len, INDEX_TOPK], torch.int32, init_value=-1),
            TensorSpec("idx_kv_cache_out", [B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM], torch.bfloat16, is_output=True),
            TensorSpec("idx_comp_kv_state_out", [B, RATIO4_STATE_ROWS, INDEX_PROJ_DIM], torch.float32, is_output=True),
            TensorSpec("idx_comp_score_state_out", [B, RATIO4_STATE_ROWS, INDEX_PROJ_DIM], torch.float32, is_output=True),
            TensorSpec("csa_topk_idxs", [B, seq_len, TOPK_CSA_TOTAL], torch.int32, init_value=-1),
            TensorSpec("attn_o", [B, seq_len, N_HEADS, HEAD_DIM], torch.bfloat16),
            TensorSpec("o_inv", [B, seq_len, N_HEADS, HEAD_DIM], torch.bfloat16),
            TensorSpec("proj", [B, seq_len, ATTN_OUT_IN], torch.bfloat16),
            TensorSpec("attn_out", [B, seq_len, HIDDEN], torch.bfloat16),
            TensorSpec("attn_hc_out", [B, seq_len, HC_MULT, HIDDEN], torch.bfloat16),
            TensorSpec("ffn_hc_x_pad", [B, seq_pad, HC_MULT, HIDDEN], torch.bfloat16),
            TensorSpec("ffn_hc_mixes", [B, seq_pad, MIX_PAD], torch.float32),
            TensorSpec("ffn_hc_pre", [B, seq_pad, HC_PAD], torch.float32),
            TensorSpec("ffn_hc_comb_logits", [B, seq_pad, HC_MULT * HC_MULT], torch.float32),
            TensorSpec("ffn_hc_x_mixed_pad", [B, seq_pad, HIDDEN], torch.bfloat16),
            TensorSpec("ffn_hc_post_pad", [B, seq_pad, HC_PAD], torch.float32),
            TensorSpec("ffn_hc_comb_pad", [B, seq_pad, HC_MULT * HC_MULT], torch.float32),
            TensorSpec("ffn_hc_x_mixed", [B, seq_len, HIDDEN], torch.bfloat16),
            TensorSpec("ffn_hc_post", [B, seq_len, HC_PAD], torch.float32),
            TensorSpec("ffn_hc_comb", [B, seq_len, HC_MULT * HC_MULT], torch.float32),
            TensorSpec("ffn_normed", [B, seq_len, HIDDEN], torch.bfloat16),
            TensorSpec("logits", [B, seq_len, N_EXPERTS], torch.float32),
            TensorSpec("scores", [B, seq_len, N_EXPERTS], torch.float32),
            TensorSpec("indices", [B, seq_len, TOPK], torch.int32),
            TensorSpec("weights", [B, seq_len, TOPK], torch.float32),
            TensorSpec("route_y", [B, seq_len, TOPK, HIDDEN], torch.bfloat16),
            TensorSpec("shared_gate", [B, seq_len, MOE_INTER_DIM], torch.bfloat16),
            TensorSpec("shared_up", [B, seq_len, MOE_INTER_DIM], torch.bfloat16),
            TensorSpec("shared_hidden", [B, seq_len, MOE_INTER_DIM], torch.bfloat16),
            TensorSpec("shared_y", [B, seq_len, HIDDEN], torch.bfloat16),
            TensorSpec("moe_out", [B, seq_len, HIDDEN], torch.bfloat16),
            TensorSpec("out", [B, seq_len, HC_MULT, HIDDEN], torch.bfloat16, is_output=True),
        ]
    )
    return specs


def build_csa_hash_prefill_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _build_csa_specs(seq_len, start_pos=0, decode=False, hash_route=True)


def build_csa_hash_decode_specs(start_pos: int = DEFAULT_DECODE_START_POS):
    if start_pos <= 0:
        raise ValueError(f"decode start_pos must be positive, got {start_pos}")
    return _build_csa_specs(1, start_pos=start_pos, decode=True, hash_route=True)


def build_csa_topk_prefill_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _build_csa_specs(seq_len, start_pos=0, decode=False, hash_route=False)


def build_csa_topk_decode_specs(start_pos: int = DEFAULT_DECODE_START_POS):
    if start_pos <= 0:
        raise ValueError(f"decode start_pos must be positive, got {start_pos}")
    return _build_csa_specs(1, start_pos=start_pos, decode=True, hash_route=False)


def _build_hca_topk_specs(seq_len: int, start_pos: int, *, decode: bool):
    from models.golden import TensorSpec

    seq_pad = ceil_div(seq_len, HC_T_TILE) * HC_T_TILE
    attn_cos_all, attn_sin_all = build_deepseek_v4_rope_tables(max_seq_len=start_pos + seq_len)
    local_cos, local_sin = materialize_rope_range(attn_cos_all, attn_sin_all, start_pos, seq_len)
    if decode:
        comp_should_compress = int((start_pos + 1) % COMPRESS_RATIO128 == 0)
        if comp_should_compress:
            comp_rope_pos = start_pos + 1 - COMPRESS_RATIO128
            comp_cos_all, comp_sin_all = build_deepseek_v4_rope_tables(
                compress_ratio=COMPRESS_RATIO128,
                max_seq_len=max(start_pos + seq_len, comp_rope_pos + 1),
            )
            comp_cos = comp_cos_all[comp_rope_pos : comp_rope_pos + 1].contiguous()
            comp_sin = comp_sin_all[comp_rope_pos : comp_rope_pos + 1].contiguous()
        else:
            comp_cos = torch.zeros(1, ROPE_HALF, dtype=torch.float32)
            comp_sin = torch.zeros(1, ROPE_HALF, dtype=torch.float32)
        compressed_len = 1
        kv_pool_len = WINDOW_SIZE + TOPK_HCA
        block_count = 0
    else:
        block_count = seq_len // COMPRESS_RATIO128
        compressed_len = max(1, block_count)
        comp_cos_all, comp_sin_all = build_deepseek_v4_rope_tables(
            compress_ratio=COMPRESS_RATIO128,
            max_seq_len=seq_len,
        )
        comp_cos, comp_sin = materialize_compressor_rope(comp_cos_all, comp_sin_all, seq_len, COMPRESS_RATIO128)
        if block_count == 0:
            comp_cos = comp_cos_all[:1].contiguous()
            comp_sin = comp_sin_all[:1].contiguous()
        kv_pool_len = seq_len + compressed_len

    def init_x():
        return (torch.randn(B, seq_len, HC_MULT, HIDDEN, dtype=torch.float32) * 0.05).to(torch.bfloat16)

    def init_hc_fn():
        return torch.randn(MIX_HC, HC_DIM, dtype=torch.float32) * 0.01

    def init_hc_scale():
        return torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

    def init_hc_base():
        return torch.zeros(MIX_HC, dtype=torch.float32)

    def init_norm_w():
        return (torch.randn(HIDDEN, dtype=torch.float32) * 0.05 + 1.0).to(torch.bfloat16)

    def init_wq_a_t():
        return torch.randn(HIDDEN, Q_LORA_RANK, dtype=torch.float32) * 0.01

    def init_q_norm_w():
        return (torch.randn(Q_LORA_RANK, dtype=torch.float32) * 0.05 + 1.0).to(torch.bfloat16)

    def init_wq_b_t():
        return torch.randn(Q_LORA_RANK, ATTN_Q_OUT, dtype=torch.float32) * 0.005

    def init_wkv_t():
        return torch.randn(HIDDEN, HEAD_DIM, dtype=torch.float32) * 0.01

    def init_kv_norm_w():
        return (torch.randn(HEAD_DIM, dtype=torch.float32) * 0.05 + 1.0).to(torch.bfloat16)

    def init_attn_sink():
        return torch.randn(N_HEADS, dtype=torch.float32) * 0.1

    def init_topk():
        if decode:
            window_topk = build_window_topk_idxs(seq_len, start_pos=start_pos, topk_max=TOPK_SWA)
            compress_topk = build_compress_topk_idxs(
                COMPRESS_RATIO128,
                seq_len,
                start_pos=start_pos,
                offset=WINDOW_SIZE,
                topk_max=TOPK_HCA,
            )
        else:
            window_topk = build_window_topk_idxs(seq_len, start_pos=0, topk_max=TOPK_SWA)
            compress_topk = build_compress_topk_idxs(
                COMPRESS_RATIO128,
                seq_len,
                start_pos=0,
                offset=seq_len,
                topk_max=TOPK_HCA,
            )
        return torch.cat([window_topk, compress_topk], dim=-1)

    def init_wo_a_t():
        return torch.randn(O_GROUP_IN, ATTN_OUT_IN, dtype=torch.float32) * 0.005

    def init_wo_b_t():
        return torch.randn(ATTN_OUT_IN, HIDDEN, dtype=torch.float32) * 0.005

    def init_comp_ape():
        return torch.randn(COMPRESS_RATIO128, HEAD_DIM, dtype=torch.float32) * 0.01

    def init_comp_norm_w():
        return (torch.randn(HEAD_DIM, dtype=torch.float32) * 0.05 + 1.0).to(torch.bfloat16)

    def init_gate_w_t():
        return torch.randn(HIDDEN, N_EXPERTS, dtype=torch.float32) * 0.01

    def init_gate_bias():
        return torch.randn(N_EXPERTS, dtype=torch.float32) * 0.1

    def init_routed_w1_t():
        return torch.randn(N_EXPERTS, HIDDEN, MOE_INTER_DIM, dtype=torch.float32) * 0.005

    def init_routed_w2_t():
        return torch.randn(N_EXPERTS, MOE_INTER_DIM, HIDDEN, dtype=torch.float32) * 0.005

    def init_routed_w3_t():
        return torch.randn(N_EXPERTS, HIDDEN, MOE_INTER_DIM, dtype=torch.float32) * 0.005

    def init_shared_w1_t():
        return torch.randn(HIDDEN, MOE_INTER_DIM, dtype=torch.float32) * 0.005

    def init_shared_w2_t():
        return torch.randn(MOE_INTER_DIM, HIDDEN, dtype=torch.float32) * 0.005

    def init_shared_w3_t():
        return torch.randn(HIDDEN, MOE_INTER_DIM, dtype=torch.float32) * 0.005

    specs = [TensorSpec("x", [B, seq_len, HC_MULT, HIDDEN], torch.bfloat16, init_value=init_x)]
    if decode:
        specs.extend(
            [
                TensorSpec(
                    "kv_cache",
                    [B, WINDOW_SIZE, HEAD_DIM],
                    torch.bfloat16,
                    init_value=lambda: torch.randn(B, WINDOW_SIZE, HEAD_DIM, dtype=torch.float32) * 0.05,
                ),
                TensorSpec(
                    "comp_kv_state",
                    [B, COMPRESS_RATIO128, HEAD_DIM],
                    torch.float32,
                    init_value=lambda: torch.randn(B, COMPRESS_RATIO128, HEAD_DIM, dtype=torch.float32) * 0.05,
                ),
                TensorSpec(
                    "comp_score_state",
                    [B, COMPRESS_RATIO128, HEAD_DIM],
                    torch.float32,
                    init_value=lambda: torch.randn(B, COMPRESS_RATIO128, HEAD_DIM, dtype=torch.float32) * 0.05,
                ),
                TensorSpec(
                    "comp_cache",
                    [B, TOPK_HCA, HEAD_DIM],
                    torch.bfloat16,
                    init_value=lambda: torch.randn(B, TOPK_HCA, HEAD_DIM, dtype=torch.float32) * 0.05,
                ),
                TensorSpec("cache_pos", [1], torch.int32, init_value=torch.tensor([start_pos % WINDOW_SIZE], dtype=torch.int32)),
                TensorSpec("comp_slot", [1], torch.int32, init_value=torch.tensor([start_pos % COMPRESS_RATIO128], dtype=torch.int32)),
                TensorSpec("comp_cache_slot", [1], torch.int32, init_value=torch.tensor([start_pos // COMPRESS_RATIO128], dtype=torch.int32)),
                TensorSpec("comp_should_compress", [1], torch.int32, init_value=torch.tensor([comp_should_compress], dtype=torch.int32)),
            ]
        )

    specs.extend(
        [
            TensorSpec("attn_hc_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_fn),
            TensorSpec("attn_hc_scale", [3], torch.float32, init_value=init_hc_scale),
            TensorSpec("attn_hc_base", [MIX_HC], torch.float32, init_value=init_hc_base),
            TensorSpec("attn_norm_w", [HIDDEN], torch.bfloat16, init_value=init_norm_w),
            TensorSpec("wq_a_t", [HIDDEN, Q_LORA_RANK], torch.bfloat16, init_value=init_wq_a_t),
            TensorSpec("q_norm_w", [Q_LORA_RANK], torch.bfloat16, init_value=init_q_norm_w),
            TensorSpec("wq_b_t", [Q_LORA_RANK, ATTN_Q_OUT], torch.bfloat16, init_value=init_wq_b_t),
            TensorSpec("wkv_t", [HIDDEN, HEAD_DIM], torch.bfloat16, init_value=init_wkv_t),
            TensorSpec("kv_norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_kv_norm_w),
            TensorSpec("attn_sink", [N_HEADS], torch.float32, init_value=init_attn_sink),
            TensorSpec("topk_idxs", [B, seq_len, TOPK_HCA_TOTAL], torch.int32, init_value=init_topk),
            TensorSpec("wo_a_t", [O_GROUP_IN, ATTN_OUT_IN], torch.bfloat16, init_value=init_wo_a_t),
            TensorSpec("wo_b_t", [ATTN_OUT_IN, HIDDEN], torch.bfloat16, init_value=init_wo_b_t),
            TensorSpec("cos", [seq_len, ROPE_HALF], torch.float32, init_value=local_cos),
            TensorSpec("sin", [seq_len, ROPE_HALF], torch.float32, init_value=local_sin),
            TensorSpec("comp_wkv_t", [HIDDEN, HEAD_DIM], torch.bfloat16, init_value=init_wkv_t),
            TensorSpec("comp_wgate_t", [HIDDEN, HEAD_DIM], torch.bfloat16, init_value=init_wkv_t),
            TensorSpec("comp_ape", [COMPRESS_RATIO128, HEAD_DIM], torch.float32, init_value=init_comp_ape),
            TensorSpec("comp_norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_comp_norm_w),
            TensorSpec("comp_cos", [compressed_len, ROPE_HALF], torch.float32, init_value=comp_cos),
            TensorSpec("comp_sin", [compressed_len, ROPE_HALF], torch.float32, init_value=comp_sin),
        ]
    )
    if not decode:
        specs.append(TensorSpec("comp_block_count", [1], torch.int32, init_value=torch.tensor([block_count], dtype=torch.int32)))
    specs.extend(
        [
            TensorSpec("ffn_hc_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_fn),
            TensorSpec("ffn_hc_scale", [3], torch.float32, init_value=init_hc_scale),
            TensorSpec("ffn_hc_base", [MIX_HC], torch.float32, init_value=init_hc_base),
            TensorSpec("ffn_norm_w", [HIDDEN], torch.bfloat16, init_value=init_norm_w),
            TensorSpec("gate_w_t", [HIDDEN, N_EXPERTS], torch.bfloat16, init_value=init_gate_w_t),
            TensorSpec("gate_bias", [N_EXPERTS], torch.float32, init_value=init_gate_bias),
            TensorSpec("routed_w1_t", [N_EXPERTS, HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=init_routed_w1_t),
            TensorSpec("routed_w2_t", [N_EXPERTS, MOE_INTER_DIM, HIDDEN], torch.bfloat16, init_value=init_routed_w2_t),
            TensorSpec("routed_w3_t", [N_EXPERTS, HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=init_routed_w3_t),
            TensorSpec("shared_w1_t", [HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=init_shared_w1_t),
            TensorSpec("shared_w2_t", [MOE_INTER_DIM, HIDDEN], torch.bfloat16, init_value=init_shared_w2_t),
            TensorSpec("shared_w3_t", [HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=init_shared_w3_t),
            TensorSpec("attn_hc_x_pad", [B, seq_pad, HC_MULT, HIDDEN], torch.bfloat16),
            TensorSpec("attn_hc_mixes", [B, seq_pad, MIX_PAD], torch.float32),
            TensorSpec("attn_hc_pre", [B, seq_pad, HC_PAD], torch.float32),
            TensorSpec("attn_hc_comb_logits", [B, seq_pad, HC_MULT * HC_MULT], torch.float32),
            TensorSpec("attn_hc_x_mixed_pad", [B, seq_pad, HIDDEN], torch.bfloat16),
            TensorSpec("attn_hc_post_pad", [B, seq_pad, HC_PAD], torch.float32),
            TensorSpec("attn_hc_comb_pad", [B, seq_pad, HC_MULT * HC_MULT], torch.float32),
            TensorSpec("attn_hc_x_mixed", [B, seq_len, HIDDEN], torch.bfloat16),
            TensorSpec("attn_hc_post", [B, seq_len, HC_PAD], torch.float32),
            TensorSpec("attn_hc_comb", [B, seq_len, HC_MULT * HC_MULT], torch.float32),
            TensorSpec("attn_normed", [B, seq_len, HIDDEN], torch.bfloat16),
            TensorSpec("q_a", [B, seq_len, Q_LORA_RANK], torch.bfloat16),
            TensorSpec("q_proj", [B, seq_len, ATTN_Q_OUT], torch.bfloat16),
            TensorSpec("kv_proj", [B, seq_len, HEAD_DIM], torch.bfloat16),
            TensorSpec("kv_normed", [B, seq_len, HEAD_DIM], torch.bfloat16),
            TensorSpec("qr", [B, seq_len, Q_LORA_RANK], torch.bfloat16),
            TensorSpec("q", [B, seq_len, N_HEADS, HEAD_DIM], torch.bfloat16),
            TensorSpec("kv", [B, seq_len, HEAD_DIM], torch.bfloat16),
            TensorSpec("comp_kv_proj", [B, seq_len, HEAD_DIM], torch.float32),
            TensorSpec("comp_score_proj", [B, seq_len, HEAD_DIM], torch.float32),
            TensorSpec("comp_pooled", [B, compressed_len, HEAD_DIM], torch.bfloat16),
            TensorSpec("comp_normed", [B, compressed_len, HEAD_DIM], torch.bfloat16),
            TensorSpec("compressed", [B, compressed_len, HEAD_DIM], torch.bfloat16),
            TensorSpec("kv_pool", [B, kv_pool_len, HEAD_DIM], torch.bfloat16),
            TensorSpec("kv_cache_out", [B, WINDOW_SIZE, HEAD_DIM], torch.bfloat16, is_output=True, init_value=0.0),
            TensorSpec("comp_kv_state_out", [B, COMPRESS_RATIO128, HEAD_DIM], torch.float32, is_output=True),
            TensorSpec("comp_score_state_out", [B, COMPRESS_RATIO128, HEAD_DIM], torch.float32, is_output=True),
            TensorSpec("comp_cache_out", [B, TOPK_HCA, HEAD_DIM], torch.bfloat16, is_output=True),
            TensorSpec("attn_o", [B, seq_len, N_HEADS, HEAD_DIM], torch.bfloat16),
            TensorSpec("o_inv", [B, seq_len, N_HEADS, HEAD_DIM], torch.bfloat16),
            TensorSpec("proj", [B, seq_len, ATTN_OUT_IN], torch.bfloat16),
            TensorSpec("attn_out", [B, seq_len, HIDDEN], torch.bfloat16),
            TensorSpec("attn_hc_out", [B, seq_len, HC_MULT, HIDDEN], torch.bfloat16),
            TensorSpec("ffn_hc_x_pad", [B, seq_pad, HC_MULT, HIDDEN], torch.bfloat16),
            TensorSpec("ffn_hc_mixes", [B, seq_pad, MIX_PAD], torch.float32),
            TensorSpec("ffn_hc_pre", [B, seq_pad, HC_PAD], torch.float32),
            TensorSpec("ffn_hc_comb_logits", [B, seq_pad, HC_MULT * HC_MULT], torch.float32),
            TensorSpec("ffn_hc_x_mixed_pad", [B, seq_pad, HIDDEN], torch.bfloat16),
            TensorSpec("ffn_hc_post_pad", [B, seq_pad, HC_PAD], torch.float32),
            TensorSpec("ffn_hc_comb_pad", [B, seq_pad, HC_MULT * HC_MULT], torch.float32),
            TensorSpec("ffn_hc_x_mixed", [B, seq_len, HIDDEN], torch.bfloat16),
            TensorSpec("ffn_hc_post", [B, seq_len, HC_PAD], torch.float32),
            TensorSpec("ffn_hc_comb", [B, seq_len, HC_MULT * HC_MULT], torch.float32),
            TensorSpec("ffn_normed", [B, seq_len, HIDDEN], torch.bfloat16),
            TensorSpec("logits", [B, seq_len, N_EXPERTS], torch.float32),
            TensorSpec("scores", [B, seq_len, N_EXPERTS], torch.float32),
            TensorSpec("indices", [B, seq_len, TOPK], torch.int32),
            TensorSpec("weights", [B, seq_len, TOPK], torch.float32),
            TensorSpec("route_y", [B, seq_len, TOPK, HIDDEN], torch.bfloat16),
            TensorSpec("shared_gate", [B, seq_len, MOE_INTER_DIM], torch.bfloat16),
            TensorSpec("shared_up", [B, seq_len, MOE_INTER_DIM], torch.bfloat16),
            TensorSpec("shared_hidden", [B, seq_len, MOE_INTER_DIM], torch.bfloat16),
            TensorSpec("shared_y", [B, seq_len, HIDDEN], torch.bfloat16),
            TensorSpec("moe_out", [B, seq_len, HIDDEN], torch.bfloat16),
            TensorSpec("out", [B, seq_len, HC_MULT, HIDDEN], torch.bfloat16, is_output=True),
        ]
    )
    return specs


def build_hca_topk_prefill_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _build_hca_topk_specs(seq_len, start_pos=0, decode=False)


def build_hca_topk_decode_specs(start_pos: int = DEFAULT_DECODE_START_POS):
    if start_pos <= 0:
        raise ValueError(f"decode start_pos must be positive, got {start_pos}")
    return _build_hca_topk_specs(1, start_pos=start_pos, decode=True)


def main() -> int:
    import argparse
    import time

    from models.golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 Flash hash Block validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3sim", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-s", "--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--decode-start-pos", type=int, default=DEFAULT_DECODE_START_POS)
    parser.add_argument(
        "--case",
        choices=[
            "all",
            "prefill",
            "decode",
            "swa-hash-prefill",
            "swa-hash-decode",
            "csa-hash-prefill",
            "csa-hash-decode",
            "hca-topk-prefill",
            "hca-topk-decode",
            "csa-topk-prefill",
            "csa-topk-decode",
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
        "out": ratio_allclose(atol=1e-3, rtol=2.0 / 128, max_error_ratio=0.005),
    }

    cases = []
    if args.case in ("all", "prefill", "swa-hash-prefill"):
        cases.append(("swa-hash-prefill", block_swa_hash_prefill_fwd, lambda: build_swa_hash_prefill_specs(args.seq_len), golden_block_swa_hash_prefill))
    if args.case in ("all", "decode", "swa-hash-decode"):
        cases.append(
            (
                "swa-hash-decode",
                block_swa_hash_decode_fwd,
                lambda: build_swa_hash_decode_specs(args.decode_start_pos),
                lambda tensors: golden_block_swa_hash_decode(tensors, args.decode_start_pos),
            )
        )
    if args.case in ("all", "prefill", "csa-hash-prefill"):
        cases.append(("csa-hash-prefill", block_csa_hash_prefill_fwd, lambda: build_csa_hash_prefill_specs(args.seq_len), golden_block_csa_hash_prefill))
    if args.case in ("all", "decode", "csa-hash-decode"):
        cases.append(
            (
                "csa-hash-decode",
                block_csa_hash_decode_fwd,
                lambda: build_csa_hash_decode_specs(args.decode_start_pos),
                lambda tensors: golden_block_csa_hash_decode(tensors, args.decode_start_pos),
            )
        )
    if args.case in ("all", "prefill", "hca-topk-prefill"):
        cases.append(("hca-topk-prefill", block_hca_topk_prefill_fwd, lambda: build_hca_topk_prefill_specs(args.seq_len), golden_block_hca_topk_prefill))
    if args.case in ("all", "decode", "hca-topk-decode"):
        cases.append(
            (
                "hca-topk-decode",
                block_hca_topk_decode_fwd,
                lambda: build_hca_topk_decode_specs(args.decode_start_pos),
                lambda tensors: golden_block_hca_topk_decode(tensors, args.decode_start_pos),
            )
        )
    if args.case in ("all", "prefill", "csa-topk-prefill"):
        cases.append(("csa-topk-prefill", block_csa_topk_prefill_fwd, lambda: build_csa_topk_prefill_specs(args.seq_len), golden_block_csa_topk_prefill))
    if args.case in ("all", "decode", "csa-topk-decode"):
        cases.append(
            (
                "csa-topk-decode",
                block_csa_topk_decode_fwd,
                lambda: build_csa_topk_decode_specs(args.decode_start_pos),
                lambda tensors: golden_block_csa_topk_decode(tensors, args.decode_start_pos),
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
    "B",
    "S_DYN",
    "S_PAD_DYN",
    "HIDDEN",
    "HC_MULT",
    "block_swa_hash_prefill_fwd",
    "block_swa_hash_decode_fwd",
    "block_csa_hash_prefill_fwd",
    "block_csa_hash_decode_fwd",
    "block_hca_topk_prefill_fwd",
    "block_hca_topk_decode_fwd",
    "block_csa_topk_prefill_fwd",
    "block_csa_topk_decode_fwd",
    "golden_block_forward",
    "golden_block_swa_hash_prefill",
    "golden_block_swa_hash_decode",
    "golden_block_csa_hash_prefill",
    "golden_block_csa_hash_decode",
    "golden_block_hca_topk_prefill",
    "golden_block_hca_topk_decode",
    "golden_block_csa_topk_prefill",
    "golden_block_csa_topk_decode",
    "build_swa_hash_prefill_specs",
    "build_swa_hash_decode_specs",
    "build_csa_hash_prefill_specs",
    "build_csa_hash_decode_specs",
    "build_hca_topk_prefill_specs",
    "build_hca_topk_decode_specs",
    "build_csa_topk_prefill_specs",
    "build_csa_topk_decode_specs",
]
