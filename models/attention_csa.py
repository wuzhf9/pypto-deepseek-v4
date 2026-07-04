"""DeepSeek V4 Flash CSA attention PyPTO kernel."""

import pypto.language as pl

from models.attention_out import attention_out_fwd
from models.attention_qkv import attention_qkv_fwd
from models.attention_hca import build_decode_kv_pool, build_prefill_kv_pool
from models.attention_swa import update_decode_window_cache, update_prefill_window_cache
from models.compressor_ratio4 import (
    STATE_ROWS as C4_STATE_ROWS,
    compressor_ratio4_attention_decode_fwd,
    compressor_ratio4_attention_prefill_fwd,
    golden_compressor_ratio4_attention_forward,
)
from models.config import FLASH_CONFIG as M
from models.indexer import golden_indexer_decode, golden_indexer_prefill, indexer_decode_fwd, indexer_prefill_fwd
from models.rope import (
    _apply_rope_golden,
    build_deepseek_v4_rope_tables,
    materialize_compressor_rope,
    materialize_rope_range,
)
from models.sparse_attn import build_window_topk_idxs, golden_sparse_attn, sparse_attn_csa_fwd


B = 1
S_DYN = pl.dynamic("S_DYN")
C_DYN = pl.dynamic("C_DYN")
K_DYN = pl.dynamic("K_DYN")

HIDDEN = M.dim
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

INDEX_N_HEADS = M.index_n_heads
INDEX_HEAD_DIM = M.index_head_dim
INDEX_Q_OUT = INDEX_N_HEADS * INDEX_HEAD_DIM
INDEX_PROJ_DIM = 2 * INDEX_HEAD_DIM
INDEX_TOPK = M.index_topk
INDEX_MAX_POSITION_EMBEDDINGS = 4096
INDEX_SCORE_LEN = INDEX_MAX_POSITION_EMBEDDINGS // 4

COMPRESS_RATIO = 4
ATTN_PROJ_DIM = 2 * HEAD_DIM
TOPK_SWA = WINDOW_SIZE
TOPK_CSA = INDEX_TOPK
TOPK_CSA_TOTAL = TOPK_SWA + TOPK_CSA
TOPK_CSA_COMPRESSED = INDEX_SCORE_LEN
SOFTMAX_SCALE = HEAD_DIM**-0.5
EPS = M.rms_norm_eps
DEFAULT_SEQ_LEN = 16
DEFAULT_DECODE_START_POS = COMPRESS_RATIO - 1


@pl.jit.inline
def build_csa_prefill_topk(
    window_topk_idxs: pl.Tensor[[B, S_DYN, TOPK_SWA], pl.INT32],
    index_topk_idxs: pl.Tensor[[B, S_DYN, TOPK_CSA], pl.INT32],
    csa_topk_idxs: pl.Tensor[[B, S_DYN, TOPK_CSA_TOTAL], pl.INT32],
):
    """Build CSA sparse indices ``cat([window_topk, index_topk])``."""
    window_topk_idxs.bind_dynamic(1, S_DYN)
    index_topk_idxs.bind_dynamic(1, S_DYN)
    csa_topk_idxs.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(window_topk_idxs, 1)
    window_flat = pl.reshape(window_topk_idxs, [tokens, TOPK_SWA])
    index_flat = pl.reshape(index_topk_idxs, [tokens, TOPK_CSA])
    csa_flat = pl.reshape(csa_topk_idxs, [tokens, TOPK_CSA_TOTAL])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_prefill_topk"):
        for t in pl.range(tokens):
            csa_flat[t : t + 1, 0:TOPK_SWA] = window_flat[t : t + 1, 0:TOPK_SWA]
            csa_flat[t : t + 1, TOPK_SWA:TOPK_CSA_TOTAL] = index_flat[t : t + 1, 0:TOPK_CSA]

    return pl.reshape(csa_flat, [B, tokens, TOPK_CSA_TOTAL])


@pl.jit.inline
def attention_csa_prefill_fwd(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
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
    attn_comp_ape: pl.Tensor[[COMPRESS_RATIO, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_comp_cos: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    attn_comp_sin: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    attn_comp_block_count: pl.Tensor[[1], pl.INT32],
    idx_wq_b_t: pl.Tensor[[Q_LORA_RANK, INDEX_Q_OUT], pl.BF16],
    idx_weights_proj_t: pl.Tensor[[HIDDEN, INDEX_N_HEADS], pl.BF16],
    idx_offset: pl.Tensor[[1], pl.INT32],
    idx_comp_wkv_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    idx_comp_wgate_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    idx_comp_ape: pl.Tensor[[COMPRESS_RATIO, INDEX_PROJ_DIM], pl.FP32],
    idx_comp_norm_w: pl.Tensor[[INDEX_HEAD_DIM], pl.BF16],
    idx_comp_cos: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    idx_comp_sin: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    idx_comp_block_count: pl.Tensor[[1], pl.INT32],
    kv_pool: pl.Tensor[[B, K_DYN, HEAD_DIM], pl.BF16],
    kv_cache_out: pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16],
    attn_comp_kv_state_out: pl.Tensor[[B, C4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_score_state_out: pl.Tensor[[B, C4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_cache_out: pl.Tensor[[B, TOPK_CSA_COMPRESSED, HEAD_DIM], pl.BF16],
    idx_kv_cache_out: pl.Tensor[[B, INDEX_SCORE_LEN, INDEX_HEAD_DIM], pl.BF16],
    idx_comp_kv_state_out: pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    idx_comp_score_state_out: pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
):
    """Run ``Attention.forward`` for ``compress_ratio == 4, start_pos == 0``."""
    x.bind_dynamic(1, S_DYN)
    window_topk_idxs.bind_dynamic(1, S_DYN)
    cos.bind_dynamic(0, S_DYN)
    sin.bind_dynamic(0, S_DYN)
    attn_comp_cos.bind_dynamic(0, C_DYN)
    attn_comp_sin.bind_dynamic(0, C_DYN)
    idx_comp_cos.bind_dynamic(0, C_DYN)
    idx_comp_sin.bind_dynamic(0, C_DYN)
    kv_pool.bind_dynamic(1, K_DYN)

    tokens = pl.tensor.dim(x, 1)
    compressed_len = pl.tensor.dim(attn_comp_cos, 0)
    qr = pl.create_tensor([B, tokens, Q_LORA_RANK], dtype=pl.BF16)
    q = pl.create_tensor([B, tokens, N_HEADS, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([B, tokens, HEAD_DIM], dtype=pl.BF16)
    attn_compressed = pl.create_tensor([B, compressed_len, HEAD_DIM], dtype=pl.BF16)
    idx_topk_idxs = pl.create_tensor([B, tokens, INDEX_TOPK], dtype=pl.INT32)
    csa_topk_idxs = pl.create_tensor([B, tokens, TOPK_CSA_TOTAL], dtype=pl.INT32)
    attn_o = pl.create_tensor([B, tokens, N_HEADS, HEAD_DIM], dtype=pl.BF16)

    attention_qkv_fwd(
        x,
        wq_a_t,
        q_norm_w,
        wq_b_t,
        wkv_t,
        kv_norm_w,
        cos,
        sin,
        qr,
        q,
        kv,
    )
    kv_cache_out = update_prefill_window_cache(kv, kv_cache_out)

    indexer_prefill_fwd(
        x,
        qr,
        idx_wq_b_t,
        idx_weights_proj_t,
        cos,
        sin,
        idx_offset,
        idx_comp_wkv_t,
        idx_comp_wgate_t,
        idx_comp_ape,
        idx_comp_norm_w,
        idx_comp_cos,
        idx_comp_sin,
        idx_comp_block_count,
        idx_kv_cache_out,
        idx_comp_kv_state_out,
        idx_comp_score_state_out,
        idx_topk_idxs,
    )
    build_csa_prefill_topk(window_topk_idxs, idx_topk_idxs, csa_topk_idxs)

    compressor_ratio4_attention_prefill_fwd(
        x,
        attn_comp_wkv_t,
        attn_comp_wgate_t,
        attn_comp_ape,
        attn_comp_norm_w,
        attn_comp_cos,
        attn_comp_sin,
        attn_comp_block_count,
        attn_compressed,
        attn_comp_kv_state_out,
        attn_comp_score_state_out,
        attn_comp_cache_out,
    )
    build_prefill_kv_pool(kv, attn_compressed, kv_pool)
    sparse_attn_csa_fwd(q, kv_pool, attn_sink, csa_topk_idxs, attn_o)
    attention_out_fwd(attn_o, wo_a_t, wo_b_t, cos, sin, out)

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


@pl.jit.inline
def attention_csa_decode_fwd(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    kv_cache: pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16],
    attn_comp_kv_state: pl.Tensor[[B, C4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_score_state: pl.Tensor[[B, C4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_cache: pl.Tensor[[B, TOPK_CSA_COMPRESSED, HEAD_DIM], pl.BF16],
    idx_kv_cache_in: pl.Tensor[[B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM], pl.BF16],
    idx_comp_kv_state: pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    idx_comp_score_state: pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    cache_pos: pl.Tensor[[1], pl.INT32],
    comp_slot: pl.Tensor[[1], pl.INT32],
    comp_cache_slot: pl.Tensor[[1], pl.INT32],
    comp_should_compress: pl.Tensor[[1], pl.INT32],
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
    attn_comp_ape: pl.Tensor[[COMPRESS_RATIO, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_comp_cos: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    attn_comp_sin: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    idx_wq_b_t: pl.Tensor[[Q_LORA_RANK, INDEX_Q_OUT], pl.BF16],
    idx_weights_proj_t: pl.Tensor[[HIDDEN, INDEX_N_HEADS], pl.BF16],
    idx_offset: pl.Tensor[[1], pl.INT32],
    idx_comp_wkv_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    idx_comp_wgate_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    idx_comp_ape: pl.Tensor[[COMPRESS_RATIO, INDEX_PROJ_DIM], pl.FP32],
    idx_comp_norm_w: pl.Tensor[[INDEX_HEAD_DIM], pl.BF16],
    idx_comp_cos: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    idx_comp_sin: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    kv_cache_out: pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16],
    attn_comp_kv_state_out: pl.Tensor[[B, C4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_score_state_out: pl.Tensor[[B, C4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_cache_out: pl.Tensor[[B, TOPK_CSA_COMPRESSED, HEAD_DIM], pl.BF16],
    idx_kv_cache_out: pl.Tensor[[B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM], pl.BF16],
    idx_comp_kv_state_out: pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    idx_comp_score_state_out: pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
):
    """Run ``Attention.forward`` for ``compress_ratio == 4, start_pos > 0``."""
    x.bind_dynamic(1, S_DYN)
    window_topk_idxs.bind_dynamic(1, S_DYN)
    cos.bind_dynamic(0, S_DYN)
    sin.bind_dynamic(0, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    qr = pl.create_tensor([B, tokens, Q_LORA_RANK], dtype=pl.BF16)
    q = pl.create_tensor([B, tokens, N_HEADS, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([B, tokens, HEAD_DIM], dtype=pl.BF16)
    attn_compressed = pl.create_tensor([B, 1, HEAD_DIM], dtype=pl.BF16)
    kv_pool = pl.create_tensor([B, WINDOW_SIZE + TOPK_CSA_COMPRESSED, HEAD_DIM], dtype=pl.BF16)
    idx_topk_idxs = pl.create_tensor([B, tokens, INDEX_TOPK], dtype=pl.INT32)
    csa_topk_idxs = pl.create_tensor([B, tokens, TOPK_CSA_TOTAL], dtype=pl.INT32)
    attn_o = pl.create_tensor([B, tokens, N_HEADS, HEAD_DIM], dtype=pl.BF16)

    attention_qkv_fwd(
        x,
        wq_a_t,
        q_norm_w,
        wq_b_t,
        wkv_t,
        kv_norm_w,
        cos,
        sin,
        qr,
        q,
        kv,
    )
    kv_cache_out = update_decode_window_cache(kv_cache, kv, cache_pos, kv_cache_out)

    indexer_decode_fwd(
        x,
        qr,
        idx_wq_b_t,
        idx_weights_proj_t,
        cos,
        sin,
        idx_offset,
        idx_comp_kv_state,
        idx_comp_score_state,
        idx_kv_cache_in,
        comp_slot,
        comp_cache_slot,
        comp_should_compress,
        idx_comp_wkv_t,
        idx_comp_wgate_t,
        idx_comp_ape,
        idx_comp_norm_w,
        idx_comp_cos,
        idx_comp_sin,
        idx_kv_cache_out,
        idx_comp_kv_state_out,
        idx_comp_score_state_out,
        idx_topk_idxs,
    )
    build_csa_prefill_topk(window_topk_idxs, idx_topk_idxs, csa_topk_idxs)

    compressor_ratio4_attention_decode_fwd(
        x,
        attn_comp_kv_state,
        attn_comp_score_state,
        attn_comp_cache,
        comp_slot,
        comp_cache_slot,
        comp_should_compress,
        attn_comp_wkv_t,
        attn_comp_wgate_t,
        attn_comp_ape,
        attn_comp_norm_w,
        attn_comp_cos,
        attn_comp_sin,
        attn_compressed,
        attn_comp_kv_state_out,
        attn_comp_score_state_out,
        attn_comp_cache_out,
    )
    build_decode_kv_pool(kv_cache_out, attn_comp_cache_out, kv_pool)
    sparse_attn_csa_fwd(q, kv_pool, attn_sink, csa_topk_idxs, attn_o)
    attention_out_fwd(attn_o, wo_a_t, wo_b_t, cos, sin, out)

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
def attention_csa_prefill_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
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
    attn_comp_ape: pl.Tensor[[COMPRESS_RATIO, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_comp_cos: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    attn_comp_sin: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    attn_comp_block_count: pl.Tensor[[1], pl.INT32],
    idx_wq_b_t: pl.Tensor[[Q_LORA_RANK, INDEX_Q_OUT], pl.BF16],
    idx_weights_proj_t: pl.Tensor[[HIDDEN, INDEX_N_HEADS], pl.BF16],
    idx_offset: pl.Tensor[[1], pl.INT32],
    idx_comp_wkv_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    idx_comp_wgate_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    idx_comp_ape: pl.Tensor[[COMPRESS_RATIO, INDEX_PROJ_DIM], pl.FP32],
    idx_comp_norm_w: pl.Tensor[[INDEX_HEAD_DIM], pl.BF16],
    idx_comp_cos: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    idx_comp_sin: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    idx_comp_block_count: pl.Tensor[[1], pl.INT32],
    kv_pool: pl.Tensor[[B, K_DYN, HEAD_DIM], pl.BF16],
    kv_cache_out: pl.Out[pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16]],
    attn_comp_kv_state_out: pl.Out[pl.Tensor[[B, C4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32]],
    attn_comp_score_state_out: pl.Out[pl.Tensor[[B, C4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32]],
    attn_comp_cache_out: pl.Out[pl.Tensor[[B, TOPK_CSA_COMPRESSED, HEAD_DIM], pl.BF16]],
    idx_kv_cache_out: pl.Out[pl.Tensor[[B, INDEX_SCORE_LEN, INDEX_HEAD_DIM], pl.BF16]],
    idx_comp_kv_state_out: pl.Out[pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
    idx_comp_score_state_out: pl.Out[pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
    out: pl.Out[pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16]],
):
    return attention_csa_prefill_fwd(
        x,
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
        kv_pool,
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
def attention_csa_decode_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    kv_cache: pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16],
    attn_comp_kv_state: pl.Tensor[[B, C4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_score_state: pl.Tensor[[B, C4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_cache: pl.Tensor[[B, TOPK_CSA_COMPRESSED, HEAD_DIM], pl.BF16],
    idx_kv_cache_in: pl.Tensor[[B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM], pl.BF16],
    idx_comp_kv_state: pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    idx_comp_score_state: pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    cache_pos: pl.Tensor[[1], pl.INT32],
    comp_slot: pl.Tensor[[1], pl.INT32],
    comp_cache_slot: pl.Tensor[[1], pl.INT32],
    comp_should_compress: pl.Tensor[[1], pl.INT32],
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
    attn_comp_ape: pl.Tensor[[COMPRESS_RATIO, ATTN_PROJ_DIM], pl.FP32],
    attn_comp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_comp_cos: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    attn_comp_sin: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    idx_wq_b_t: pl.Tensor[[Q_LORA_RANK, INDEX_Q_OUT], pl.BF16],
    idx_weights_proj_t: pl.Tensor[[HIDDEN, INDEX_N_HEADS], pl.BF16],
    idx_offset: pl.Tensor[[1], pl.INT32],
    idx_comp_wkv_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    idx_comp_wgate_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    idx_comp_ape: pl.Tensor[[COMPRESS_RATIO, INDEX_PROJ_DIM], pl.FP32],
    idx_comp_norm_w: pl.Tensor[[INDEX_HEAD_DIM], pl.BF16],
    idx_comp_cos: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    idx_comp_sin: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    kv_cache_out: pl.Out[pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16]],
    attn_comp_kv_state_out: pl.Out[pl.Tensor[[B, C4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32]],
    attn_comp_score_state_out: pl.Out[pl.Tensor[[B, C4_STATE_ROWS, ATTN_PROJ_DIM], pl.FP32]],
    attn_comp_cache_out: pl.Out[pl.Tensor[[B, TOPK_CSA_COMPRESSED, HEAD_DIM], pl.BF16]],
    idx_kv_cache_out: pl.Out[pl.Tensor[[B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM], pl.BF16]],
    idx_comp_kv_state_out: pl.Out[pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
    idx_comp_score_state_out: pl.Out[pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
    out: pl.Out[pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16]],
):
    return attention_csa_decode_fwd(
        x,
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
        out,
    )


def _golden_sparse_attn(q, kv, attn_sink, topk_idxs):
    import torch

    tensors = {
        "q": q,
        "kv": kv,
        "attn_sink": attn_sink,
        "topk_idxs": topk_idxs,
        "softmax_scale": SOFTMAX_SCALE,
        "out": torch.empty_like(q),
    }
    golden_sparse_attn(tensors)
    return tensors["out"]


def golden_attention_csa_forward(tensors, start_pos: int):
    import torch

    if start_pos < 0:
        raise ValueError(f"start_pos must be non-negative, got {start_pos}")

    x = tensors["x"]
    seq_len = x.shape[1]
    if start_pos > 0 and seq_len != 1:
        raise ValueError(f"decode expects seq_len=1, got {seq_len}")
    if start_pos > 0 and int(tensors["cache_pos"][0].item()) != start_pos % WINDOW_SIZE:
        raise ValueError(
            f"decode cache_pos mismatch: expected {start_pos % WINDOW_SIZE}, "
            f"got {int(tensors['cache_pos'][0].item())}"
        )

    # q
    qr = q = torch.matmul(x.float(), tensors["wq_a_t"].float()).to(torch.bfloat16)
    q = q.float()
    q = (q * torch.rsqrt(q.square().mean(-1, keepdim=True) + EPS) * tensors["q_norm_w"].float()).to(torch.bfloat16)
    qr = q
    q = torch.matmul(q.float(), tensors["wq_b_t"].float()).to(torch.bfloat16)
    q = q.unflatten(-1, (N_HEADS, HEAD_DIM))
    q = (q.float() * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + EPS)).to(torch.bfloat16)
    q = _apply_rope_golden(q, tensors["cos"], tensors["sin"], inverse=False)

    # win kv
    kv = torch.matmul(x.float(), tensors["wkv_t"].float()).to(torch.bfloat16)
    kv = kv.float()
    kv = (kv * torch.rsqrt(kv.square().mean(-1, keepdim=True) + EPS) * tensors["kv_norm_w"].float()).to(torch.bfloat16)
    kv = _apply_rope_golden(kv, tensors["cos"], tensors["sin"], inverse=False)

    compressed_len = tensors["attn_comp_cos"].shape[0]

    if start_pos == 0:
        kv_cache_out = tensors["kv_cache_out"].clone()
        if seq_len <= WINDOW_SIZE:
            kv_cache_out[:, :seq_len] = kv
        else:
            cutoff = seq_len % WINDOW_SIZE
            kv_cache_out[:, cutoff:WINDOW_SIZE], kv_cache_out[:, :cutoff] = kv[:, -WINDOW_SIZE:].split(
                [WINDOW_SIZE - cutoff, cutoff],
                dim=1,
            )
    else:
        kv_cache_out = tensors["kv_cache"].clone()
        kv_cache_out[0, int(tensors["cache_pos"][0].item())] = kv[0, 0]

    idx_tensors = {
        "x": x,
        "qr": qr,
        "wq_b_t": tensors["idx_wq_b_t"],
        "weights_proj_t": tensors["idx_weights_proj_t"],
        "cos": tensors["cos"],
        "sin": tensors["sin"],
        "offset": tensors["idx_offset"],
        "comp_wkv_t": tensors["idx_comp_wkv_t"],
        "comp_wgate_t": tensors["idx_comp_wgate_t"],
        "comp_ape": tensors["idx_comp_ape"],
        "comp_norm_w": tensors["idx_comp_norm_w"],
        "comp_cos": tensors["idx_comp_cos"],
        "comp_sin": tensors["idx_comp_sin"],
        "index_kv_cache": tensors["idx_kv_cache_out"],
        "comp_kv_state_out": tensors["idx_comp_kv_state_out"],
        "comp_score_state_out": tensors["idx_comp_score_state_out"],
        "topk_idxs": torch.full((B, seq_len, INDEX_TOPK), -1, dtype=torch.int32),
    }
    if start_pos == 0:
        idx_tensors["comp_block_count"] = tensors["idx_comp_block_count"]
        golden_indexer_prefill(idx_tensors)
    else:
        idx_tensors.update(
            {
                "comp_kv_state": tensors["idx_comp_kv_state"],
                "comp_score_state": tensors["idx_comp_score_state"],
                "index_kv_cache_in": tensors["idx_kv_cache_in"],
                "comp_slot": tensors["comp_slot"],
                "comp_cache_slot": tensors["comp_cache_slot"],
                "comp_should_compress": tensors["comp_should_compress"],
            }
        )
        golden_indexer_decode(idx_tensors)

    csa_topk_idxs = torch.cat([tensors["window_topk_idxs"], idx_tensors["topk_idxs"]], dim=-1)

    comp_tensors = {
        "x": x,
        "wkv_t": tensors["attn_comp_wkv_t"],
        "wgate_t": tensors["attn_comp_wgate_t"],
        "ape": tensors["attn_comp_ape"],
        "norm_w": tensors["attn_comp_norm_w"],
        "cos": tensors["attn_comp_cos"],
        "sin": tensors["attn_comp_sin"],
        "compressed": torch.empty(B, compressed_len, HEAD_DIM, dtype=torch.bfloat16),
        "kv_state_out": tensors["attn_comp_kv_state_out"],
        "score_state_out": tensors["attn_comp_score_state_out"],
        "compressed_cache_out": tensors["attn_comp_cache_out"],
    }
    if start_pos == 0:
        comp_tensors["block_count"] = tensors["attn_comp_block_count"]
        golden_compressor_ratio4_attention_forward(comp_tensors, start_pos=0)
        blocks = int(tensors["attn_comp_block_count"][0].item())
        if blocks > 0:
            kv_pool = torch.cat([kv, comp_tensors["compressed"][:, :blocks]], dim=1)
        else:
            kv_pool = kv
    else:
        comp_tensors.update(
            {
                "kv_state": tensors["attn_comp_kv_state"],
                "score_state": tensors["attn_comp_score_state"],
                "compressed_cache": tensors["attn_comp_cache"],
                "slot": tensors["comp_slot"],
                "cache_slot": tensors["comp_cache_slot"],
                "should_compress": tensors["comp_should_compress"],
            }
        )
        golden_compressor_ratio4_attention_forward(comp_tensors, start_pos=start_pos)
        kv_pool = torch.cat([kv_cache_out, comp_tensors["compressed_cache_out"]], dim=1)

    attn_o = _golden_sparse_attn(q, kv_pool, tensors["attn_sink"], csa_topk_idxs)
    o_inv = _apply_rope_golden(attn_o, tensors["cos"], tensors["sin"], inverse=True)
    o = o_inv.view(B, seq_len, O_GROUPS, O_GROUP_IN)
    wo_a = tensors["wo_a_t"].transpose(0, 1).contiguous().view(O_GROUPS, O_LORA_RANK, O_GROUP_IN)
    proj = torch.einsum("bsgd,grd->bsgr", o.float(), wo_a.float()).to(torch.bfloat16)
    out = torch.matmul(proj.flatten(2).float(), tensors["wo_b_t"].float()).to(torch.bfloat16)

    tensors["kv_cache_out"][:] = kv_cache_out
    tensors["idx_kv_cache_out"][:] = idx_tensors["index_kv_cache"]
    tensors["idx_comp_kv_state_out"][:] = idx_tensors["comp_kv_state_out"]
    tensors["idx_comp_score_state_out"][:] = idx_tensors["comp_score_state_out"]
    if "kv_pool" in tensors:
        tensors["kv_pool"][:, : kv_pool.shape[1]] = kv_pool
    tensors["attn_comp_kv_state_out"][:] = comp_tensors["kv_state_out"]
    tensors["attn_comp_score_state_out"][:] = comp_tensors["score_state_out"]
    tensors["attn_comp_cache_out"][:] = comp_tensors["compressed_cache_out"]
    tensors["out"][:] = out


def golden_attention_csa_prefill(tensors):
    golden_attention_csa_forward(tensors, start_pos=0)


def golden_attention_csa_decode(tensors, start_pos: int = DEFAULT_DECODE_START_POS):
    if start_pos <= 0:
        raise ValueError(f"decode start_pos must be positive, got {start_pos}")
    golden_attention_csa_forward(tensors, start_pos=start_pos)


def _common_specs(seq_len: int, start_pos: int, *, decode: bool):
    import torch

    from models.golden import TensorSpec

    freqs_cos, freqs_sin = build_deepseek_v4_rope_tables(
        compress_ratio=COMPRESS_RATIO,
        max_seq_len=start_pos + seq_len,
    )
    local_cos, local_sin = materialize_rope_range(freqs_cos, freqs_sin, start_pos, seq_len)
    if decode:
        slot = start_pos % COMPRESS_RATIO
        cache_slot = start_pos // COMPRESS_RATIO
        should_compress = int((start_pos + 1) % COMPRESS_RATIO == 0)
        if should_compress:
            comp_rope_pos = start_pos + 1 - COMPRESS_RATIO
            comp_cos = freqs_cos[comp_rope_pos : comp_rope_pos + 1].contiguous()
            comp_sin = freqs_sin[comp_rope_pos : comp_rope_pos + 1].contiguous()
        else:
            comp_cos = torch.zeros(1, ROPE_HALF, dtype=torch.float32)
            comp_sin = torch.zeros(1, ROPE_HALF, dtype=torch.float32)
        compressed_len = 1
        kv_pool_len = WINDOW_SIZE + TOPK_CSA_COMPRESSED
    else:
        blocks = seq_len // COMPRESS_RATIO
        compressed_len = max(1, blocks)
        comp_cos, comp_sin = materialize_compressor_rope(freqs_cos, freqs_sin, seq_len, COMPRESS_RATIO)
        if blocks == 0:
            comp_cos = freqs_cos[:1].contiguous()
            comp_sin = freqs_sin[:1].contiguous()
        kv_pool_len = seq_len + compressed_len

    def init_x():
        return torch.randn(B, seq_len, HIDDEN) * 0.1

    def init_wq_a_t():
        return torch.randn(HIDDEN, Q_LORA_RANK) * 0.02

    def init_q_norm_w():
        return torch.randn(Q_LORA_RANK) * 0.1 + 1.0

    def init_wq_b_t():
        return torch.randn(Q_LORA_RANK, ATTN_Q_OUT) * 0.02

    def init_wkv_t():
        return torch.randn(HIDDEN, HEAD_DIM) * 0.02

    def init_kv_norm_w():
        return torch.randn(HEAD_DIM) * 0.1 + 1.0

    def init_attn_sink():
        return torch.randn(N_HEADS) * 0.1

    def init_wo_a_t():
        return torch.randn(O_GROUP_IN, ATTN_OUT_IN) * 0.02

    def init_wo_b_t():
        return torch.randn(ATTN_OUT_IN, HIDDEN) * 0.02

    def init_attn_comp_w():
        return torch.randn(HIDDEN, ATTN_PROJ_DIM) * 0.02

    def init_attn_comp_ape():
        return torch.randn(COMPRESS_RATIO, ATTN_PROJ_DIM) * 0.02

    def init_attn_comp_norm_w():
        return torch.randn(HEAD_DIM) * 0.1 + 1.0

    def init_idx_wq_b_t():
        return torch.randn(Q_LORA_RANK, INDEX_Q_OUT) * 0.02

    def init_idx_weights_proj_t():
        return torch.randn(HIDDEN, INDEX_N_HEADS) * 0.02

    def init_idx_comp_w():
        return torch.randn(HIDDEN, INDEX_PROJ_DIM) * 0.02

    def init_idx_comp_ape():
        return torch.randn(COMPRESS_RATIO, INDEX_PROJ_DIM) * 0.02

    def init_idx_comp_norm_w():
        return torch.randn(INDEX_HEAD_DIM) * 0.1 + 1.0

    x_spec = [TensorSpec("x", [B, seq_len, HIDDEN], torch.bfloat16, init_value=init_x)]
    if decode:
        prefix_specs = [
            *x_spec,
            TensorSpec("kv_cache", [B, WINDOW_SIZE, HEAD_DIM], torch.bfloat16, init_value=lambda: torch.randn(B, WINDOW_SIZE, HEAD_DIM) * 0.1),
            TensorSpec(
                "attn_comp_kv_state",
                [B, C4_STATE_ROWS, ATTN_PROJ_DIM],
                torch.float32,
                init_value=lambda: torch.randn(B, C4_STATE_ROWS, ATTN_PROJ_DIM) * 0.1,
            ),
            TensorSpec(
                "attn_comp_score_state",
                [B, C4_STATE_ROWS, ATTN_PROJ_DIM],
                torch.float32,
                init_value=lambda: torch.randn(B, C4_STATE_ROWS, ATTN_PROJ_DIM) * 0.1,
            ),
            TensorSpec(
                "attn_comp_cache",
                [B, TOPK_CSA_COMPRESSED, HEAD_DIM],
                torch.bfloat16,
                init_value=lambda: torch.randn(B, TOPK_CSA_COMPRESSED, HEAD_DIM) * 0.1,
            ),
            TensorSpec(
                "idx_kv_cache_in",
                [B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM],
                torch.bfloat16,
                init_value=lambda: torch.randn(B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM) * 0.1,
            ),
            TensorSpec(
                "idx_comp_kv_state",
                [B, C4_STATE_ROWS, INDEX_PROJ_DIM],
                torch.float32,
                init_value=lambda: torch.randn(B, C4_STATE_ROWS, INDEX_PROJ_DIM) * 0.1,
            ),
            TensorSpec(
                "idx_comp_score_state",
                [B, C4_STATE_ROWS, INDEX_PROJ_DIM],
                torch.float32,
                init_value=lambda: torch.randn(B, C4_STATE_ROWS, INDEX_PROJ_DIM) * 0.1,
            ),
            TensorSpec("cache_pos", [1], torch.int32, init_value=torch.tensor([start_pos % WINDOW_SIZE], dtype=torch.int32)),
            TensorSpec("comp_slot", [1], torch.int32, init_value=torch.tensor([slot], dtype=torch.int32)),
            TensorSpec("comp_cache_slot", [1], torch.int32, init_value=torch.tensor([cache_slot], dtype=torch.int32)),
            TensorSpec("comp_should_compress", [1], torch.int32, init_value=torch.tensor([should_compress], dtype=torch.int32)),
        ]
    else:
        prefix_specs = x_spec

    if not decode:
        attn_prefill_specs = [
            TensorSpec("attn_comp_block_count", [1], torch.int32, init_value=torch.tensor([blocks], dtype=torch.int32))
        ]
        index_prefill_specs = [
            TensorSpec("idx_comp_block_count", [1], torch.int32, init_value=torch.tensor([blocks], dtype=torch.int32)),
            TensorSpec("kv_pool", [B, kv_pool_len, HEAD_DIM], torch.bfloat16),
        ]
    else:
        attn_prefill_specs = []
        index_prefill_specs = []

    specs = [
        *prefix_specs,
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
        TensorSpec("attn_comp_ape", [COMPRESS_RATIO, ATTN_PROJ_DIM], torch.float32, init_value=init_attn_comp_ape),
        TensorSpec("attn_comp_norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_attn_comp_norm_w),
        TensorSpec("attn_comp_cos", [compressed_len, ROPE_HALF], torch.float32, init_value=comp_cos),
        TensorSpec("attn_comp_sin", [compressed_len, ROPE_HALF], torch.float32, init_value=comp_sin),
        *attn_prefill_specs,
        TensorSpec("idx_wq_b_t", [Q_LORA_RANK, INDEX_Q_OUT], torch.bfloat16, init_value=init_idx_wq_b_t),
        TensorSpec("idx_weights_proj_t", [HIDDEN, INDEX_N_HEADS], torch.bfloat16, init_value=init_idx_weights_proj_t),
        TensorSpec("idx_offset", [1], torch.int32, init_value=torch.tensor([WINDOW_SIZE if decode else seq_len], dtype=torch.int32)),
        TensorSpec("idx_comp_wkv_t", [HIDDEN, INDEX_PROJ_DIM], torch.bfloat16, init_value=init_idx_comp_w),
        TensorSpec("idx_comp_wgate_t", [HIDDEN, INDEX_PROJ_DIM], torch.bfloat16, init_value=init_idx_comp_w),
        TensorSpec("idx_comp_ape", [COMPRESS_RATIO, INDEX_PROJ_DIM], torch.float32, init_value=init_idx_comp_ape),
        TensorSpec("idx_comp_norm_w", [INDEX_HEAD_DIM], torch.bfloat16, init_value=init_idx_comp_norm_w),
        TensorSpec("idx_comp_cos", [compressed_len, ROPE_HALF], torch.float32, init_value=comp_cos.clone()),
        TensorSpec("idx_comp_sin", [compressed_len, ROPE_HALF], torch.float32, init_value=comp_sin.clone()),
        *index_prefill_specs,
        TensorSpec("kv_cache_out", [B, WINDOW_SIZE, HEAD_DIM], torch.bfloat16, is_output=True, init_value=0.0),
        TensorSpec("attn_comp_kv_state_out", [B, C4_STATE_ROWS, ATTN_PROJ_DIM], torch.float32, is_output=True),
        TensorSpec("attn_comp_score_state_out", [B, C4_STATE_ROWS, ATTN_PROJ_DIM], torch.float32, is_output=True),
        TensorSpec("attn_comp_cache_out", [B, TOPK_CSA_COMPRESSED, HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec("idx_kv_cache_out", [B, INDEX_SCORE_LEN, INDEX_HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec("idx_comp_kv_state_out", [B, C4_STATE_ROWS, INDEX_PROJ_DIM], torch.float32, is_output=True),
        TensorSpec("idx_comp_score_state_out", [B, C4_STATE_ROWS, INDEX_PROJ_DIM], torch.float32, is_output=True),
        TensorSpec("out", [B, seq_len, HIDDEN], torch.bfloat16, is_output=True),
    ]
    return specs


def build_csa_prefill_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _common_specs(seq_len, start_pos=0, decode=False)


def build_csa_decode_specs(start_pos: int = DEFAULT_DECODE_START_POS):
    if start_pos <= 0:
        raise ValueError(f"decode start_pos must be positive, got {start_pos}")
    return _common_specs(1, start_pos=start_pos, decode=True)


def main() -> int:
    import argparse

    from models.golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 Flash CSA attention validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3sim", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-s", "--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--decode-start-pos", type=int, default=DEFAULT_DECODE_START_POS)
    parser.add_argument("--case", choices=["all", "prefill", "decode"], default="all")
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
        "out": ratio_allclose(atol=1e-3, rtol=2.0 / 128, max_error_ratio=0.005),
    }

    cases = []
    if args.case in ("all", "prefill"):
        cases.append(
            (
                "csa-prefill",
                attention_csa_prefill_test,
                lambda: build_csa_prefill_specs(args.seq_len),
                golden_attention_csa_prefill,
            )
        )
    if args.case in ("all", "decode"):
        cases.append(
            (
                "csa-decode",
                attention_csa_decode_test,
                lambda: build_csa_decode_specs(args.decode_start_pos),
                lambda tensors: golden_attention_csa_forward(tensors, start_pos=args.decode_start_pos),
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

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "B",
    "S_DYN",
    "C_DYN",
    "K_DYN",
    "HIDDEN",
    "Q_LORA_RANK",
    "N_HEADS",
    "HEAD_DIM",
    "ATTN_Q_OUT",
    "ATTN_PROJ_DIM",
    "ATTN_OUT_IN",
    "INDEX_N_HEADS",
    "INDEX_HEAD_DIM",
    "INDEX_Q_OUT",
    "INDEX_PROJ_DIM",
    "INDEX_TOPK",
    "INDEX_SCORE_LEN",
    "COMPRESS_RATIO",
    "TOPK_SWA",
    "TOPK_CSA",
    "TOPK_CSA_TOTAL",
    "TOPK_CSA_COMPRESSED",
    "SOFTMAX_SCALE",
    "EPS",
    "DEFAULT_SEQ_LEN",
    "DEFAULT_DECODE_START_POS",
    "build_csa_prefill_topk",
    "attention_csa_prefill_fwd",
    "attention_csa_decode_fwd",
    "attention_csa_prefill_test",
    "attention_csa_decode_test",
    "golden_attention_csa_forward",
    "golden_attention_csa_prefill",
    "golden_attention_csa_decode",
    "build_csa_prefill_specs",
    "build_csa_decode_specs",
]
