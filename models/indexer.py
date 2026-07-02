"""DeepSeek V4 Flash bf16 Indexer PyPTO kernels."""

import pypto.language as pl

from models.common import assert_divisible
from models.compressor_ratio4 import (
    STATE_ROWS as C4_STATE_ROWS,
    compressor_ratio4_indexer_decode_fwd,
    compressor_ratio4_indexer_prefill_fwd,
    golden_compressor_ratio4_indexer_forward,
)
from models.config import FLASH_CONFIG as M
from models.linear import linear_1024_to_8192, linear_4096_to_64
from models.rope import (
    _apply_rope_golden,
    build_deepseek_v4_rope_tables,
    materialize_compressor_rope,
    materialize_rope_range,
    rope_4d_128_fwd,
)


B = 1
S_DYN = pl.dynamic("S_DYN")
C_DYN = pl.dynamic("C_DYN")

HIDDEN = M.dim
Q_LORA_RANK = M.q_lora_rank
INDEX_N_HEADS = M.index_n_heads
INDEX_HEAD_DIM = M.index_head_dim
INDEX_Q_OUT = INDEX_N_HEADS * INDEX_HEAD_DIM
INDEX_PROJ_DIM = 2 * INDEX_HEAD_DIM
INDEX_TOPK = M.index_topk
INDEX_MAX_POSITION_EMBEDDINGS = 4096
INDEX_SCORE_LEN = INDEX_MAX_POSITION_EMBEDDINGS // 4
COMPRESS_RATIO = 4
ROPE_HALF = M.rope_head_dim // 2
NEG_INF = -3.4028234663852886e38
INDEX_WEIGHTS_SCALE = (INDEX_HEAD_DIM**-0.5) * (INDEX_N_HEADS**-0.5)

CACHE_TILE = 32
ROW_ARGMAX_WORK_ROWS = 8
INDEX_SCORE_BLOCKS = INDEX_SCORE_LEN // CACHE_TILE
DEFAULT_SEQ_LEN = 16
DEFAULT_DECODE_START_POS = COMPRESS_RATIO - 1
WINDOW_SIZE = M.window_size

assert_divisible(INDEX_Q_OUT, INDEX_HEAD_DIM, "indexer q projection output")
assert_divisible(INDEX_SCORE_LEN, CACHE_TILE, "indexer score length")


@pl.jit.inline
def indexer_prefill_fwd(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    qr: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    wq_b_t: pl.Tensor[[Q_LORA_RANK, INDEX_Q_OUT], pl.BF16],
    weights_proj_t: pl.Tensor[[HIDDEN, INDEX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    offset: pl.Tensor[[1], pl.INT32],
    comp_wkv_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    comp_wgate_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    comp_ape: pl.Tensor[[COMPRESS_RATIO, INDEX_PROJ_DIM], pl.FP32],
    comp_norm_w: pl.Tensor[[INDEX_HEAD_DIM], pl.BF16],
    comp_cos: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    comp_sin: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    comp_block_count: pl.Tensor[[1], pl.INT32],
    q_proj: pl.Tensor[[B, S_DYN, INDEX_Q_OUT], pl.BF16],
    q_rope: pl.Tensor[[B, S_DYN, INDEX_N_HEADS, INDEX_HEAD_DIM], pl.BF16],
    weights: pl.Tensor[[B, S_DYN, INDEX_N_HEADS], pl.BF16],
    comp_kv_proj: pl.Tensor[[B, S_DYN, INDEX_PROJ_DIM], pl.FP32],
    comp_score_proj: pl.Tensor[[B, S_DYN, INDEX_PROJ_DIM], pl.FP32],
    comp_pooled: pl.Tensor[[B, C_DYN, INDEX_HEAD_DIM], pl.BF16],
    comp_normed: pl.Tensor[[B, C_DYN, INDEX_HEAD_DIM], pl.BF16],
    index_score: pl.Tensor[[B, S_DYN, INDEX_SCORE_LEN], pl.FP32],
    index_kv_cache: pl.Tensor[[B, INDEX_SCORE_LEN, INDEX_HEAD_DIM], pl.BF16],
    comp_kv_state_out: pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    comp_score_state_out: pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    topk_idxs: pl.Tensor[[B, S_DYN, INDEX_TOPK], pl.INT32],
):
    """Run official ``Indexer.forward`` prefill path with bf16 computation."""
    x.bind_dynamic(1, S_DYN)
    qr.bind_dynamic(1, S_DYN)
    cos.bind_dynamic(0, S_DYN)
    sin.bind_dynamic(0, S_DYN)
    comp_cos.bind_dynamic(0, C_DYN)
    comp_sin.bind_dynamic(0, C_DYN)
    q_proj.bind_dynamic(1, S_DYN)
    q_rope.bind_dynamic(1, S_DYN)
    weights.bind_dynamic(1, S_DYN)
    comp_kv_proj.bind_dynamic(1, S_DYN)
    comp_score_proj.bind_dynamic(1, S_DYN)
    comp_pooled.bind_dynamic(1, C_DYN)
    comp_normed.bind_dynamic(1, C_DYN)
    index_score.bind_dynamic(1, S_DYN)
    topk_idxs.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    blocks = pl.read(comp_block_count, [0])
    offset_i32 = pl.read(offset, [0])

    q_proj = linear_1024_to_8192(qr, wq_b_t, q_proj)
    q_unflat = pl.reshape(q_proj, [B, tokens, INDEX_N_HEADS, INDEX_HEAD_DIM])
    q_rope = rope_4d_128_fwd(q_unflat, cos, sin, q_rope)

    weights = linear_4096_to_64(x, weights_proj_t, weights)
    weights_flat = pl.reshape(weights, [tokens, INDEX_N_HEADS])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="indexer_weights_scale"):
        for t in pl.range(tokens):
            scaled = pl.mul(
                pl.cast(weights_flat[t : t + 1, 0:INDEX_N_HEADS], target_type=pl.FP32),
                INDEX_WEIGHTS_SCALE,
            )
            weights_flat[t : t + 1, 0:INDEX_N_HEADS] = pl.cast(scaled, target_type=pl.BF16, mode="rint")

    comp_pooled, comp_kv_state_out, comp_score_state_out, index_kv_cache = compressor_ratio4_indexer_prefill_fwd(
        x,
        comp_wkv_t,
        comp_wgate_t,
        comp_ape,
        comp_norm_w,
        comp_cos,
        comp_sin,
        comp_block_count,
        comp_kv_proj,
        comp_score_proj,
        comp_pooled,
        comp_normed,
        comp_pooled,
        comp_kv_state_out,
        comp_score_state_out,
        index_kv_cache,
    )

    q_flat = pl.reshape(q_rope, [tokens * INDEX_N_HEADS, INDEX_HEAD_DIM])
    cache_flat = pl.reshape(index_kv_cache, [INDEX_SCORE_LEN, INDEX_HEAD_DIM])
    score_flat = pl.reshape(index_score, [tokens, INDEX_SCORE_LEN])
    topk_flat = pl.reshape(topk_idxs, [tokens, INDEX_TOPK])

    for t in pl.range(tokens):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="indexer_prefill_score_init"):
            score_flat[t : t + 1, 0:INDEX_SCORE_LEN] = pl.full(
                [1, INDEX_SCORE_LEN], dtype=pl.FP32, value=NEG_INF
            )
            topk_flat[t : t + 1, 0:INDEX_TOPK] = pl.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)

        visible_len = pl.min((t + 1) // COMPRESS_RATIO, blocks)
        if visible_len > 0:
            q_row0 = t * INDEX_N_HEADS

            for cb in pl.range(INDEX_SCORE_BLOCKS):
                cache0 = cb * CACHE_TILE
                if visible_len > cache0:
                    valid_len = pl.min(CACHE_TILE, visible_len - cache0)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="indexer_prefill_score_chunk"):
                        q_tile = q_flat[q_row0 : q_row0 + INDEX_N_HEADS, 0:INDEX_HEAD_DIM]
                        kv_tile = pl.slice(
                            cache_flat,
                            [CACHE_TILE, INDEX_HEAD_DIM],
                            [cache0, 0],
                        )
                        score_tile_ch = pl.matmul(kv_tile, q_tile, b_trans=True, out_dtype=pl.FP32)
                        weights_row = pl.cast(weights_flat[t : t + 1, 0:INDEX_N_HEADS], target_type=pl.FP32)
                        relu_score = pl.maximum(score_tile_ch, pl.mul(score_tile_ch, 0.0))
                        weighted = pl.col_expand_mul(relu_score, weights_row)
                        score_chunk = pl.reshape(pl.row_sum(weighted), [1, CACHE_TILE])
                        score_valid = pl.fillpad(
                            pl.set_validshape(score_chunk, 1, valid_len),
                            pad_value=pl.PadValue.min,
                        )
                        score_valid = pl.maximum(
                            score_valid,
                            pl.full([1, CACHE_TILE], dtype=pl.FP32, value=NEG_INF),
                        )
                        score_flat[t : t + 1, cache0 : cache0 + CACHE_TILE] = score_valid

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="indexer_prefill_topk"):
                score_row = score_flat[t : t + 1, 0:INDEX_SCORE_LEN]
                score_work = pl.full([ROW_ARGMAX_WORK_ROWS, INDEX_SCORE_LEN], dtype=pl.FP32, value=NEG_INF)
                score_work[0:1, 0:INDEX_SCORE_LEN] = score_row
                pos_i32 = pl.arange(0, [1, INDEX_SCORE_LEN], dtype=pl.INT32)
                neg_inf_row = pl.full([1, INDEX_SCORE_LEN], dtype=pl.FP32, value=NEG_INF)
                valid_topk = pl.min(INDEX_TOPK, visible_len)
                for k in pl.range(INDEX_TOPK):
                    if k < valid_topk:
                        best_pos_tile = pl.row_argmax(score_work)
                        best_pos_i32 = pl.read(best_pos_tile, [0, 0])
                        pl.write(topk_flat, [t, k], best_pos_i32 + offset_i32)
                        selected_i32 = pl.cmp(pos_i32, best_pos_i32, cmp_type=0)
                        selected = pl.cast(selected_i32, target_type=pl.FP32)
                        keep = pl.sub(pl.full([1, INDEX_SCORE_LEN], dtype=pl.FP32, value=1.0), selected)
                        score_work_row = score_work[0:1, 0:INDEX_SCORE_LEN]
                        masked_score = pl.add(pl.mul(score_work_row, keep), pl.mul(neg_inf_row, selected))
                        score_work[0:1, 0:INDEX_SCORE_LEN] = masked_score

    return topk_idxs, index_kv_cache, comp_kv_state_out, comp_score_state_out


@pl.jit
def indexer_prefill_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    qr: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    wq_b_t: pl.Tensor[[Q_LORA_RANK, INDEX_Q_OUT], pl.BF16],
    weights_proj_t: pl.Tensor[[HIDDEN, INDEX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    offset: pl.Tensor[[1], pl.INT32],
    comp_wkv_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    comp_wgate_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    comp_ape: pl.Tensor[[COMPRESS_RATIO, INDEX_PROJ_DIM], pl.FP32],
    comp_norm_w: pl.Tensor[[INDEX_HEAD_DIM], pl.BF16],
    comp_cos: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    comp_sin: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    comp_block_count: pl.Tensor[[1], pl.INT32],
    q_proj: pl.Tensor[[B, S_DYN, INDEX_Q_OUT], pl.BF16],
    q_rope: pl.Tensor[[B, S_DYN, INDEX_N_HEADS, INDEX_HEAD_DIM], pl.BF16],
    weights: pl.Tensor[[B, S_DYN, INDEX_N_HEADS], pl.BF16],
    comp_kv_proj: pl.Tensor[[B, S_DYN, INDEX_PROJ_DIM], pl.FP32],
    comp_score_proj: pl.Tensor[[B, S_DYN, INDEX_PROJ_DIM], pl.FP32],
    comp_pooled: pl.Tensor[[B, C_DYN, INDEX_HEAD_DIM], pl.BF16],
    comp_normed: pl.Tensor[[B, C_DYN, INDEX_HEAD_DIM], pl.BF16],
    topk_idxs: pl.Out[pl.Tensor[[B, S_DYN, INDEX_TOPK], pl.INT32]],
    index_score: pl.Tensor[[B, S_DYN, INDEX_SCORE_LEN], pl.FP32],
    index_kv_cache: pl.Out[pl.Tensor[[B, INDEX_SCORE_LEN, INDEX_HEAD_DIM], pl.BF16]],
    comp_kv_state_out: pl.Out[pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
    comp_score_state_out: pl.Out[pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
):
    topk_idxs, index_kv_cache, comp_kv_state_out, comp_score_state_out = indexer_prefill_fwd(
        x,
        qr,
        wq_b_t,
        weights_proj_t,
        cos,
        sin,
        offset,
        comp_wkv_t,
        comp_wgate_t,
        comp_ape,
        comp_norm_w,
        comp_cos,
        comp_sin,
        comp_block_count,
        q_proj,
        q_rope,
        weights,
        comp_kv_proj,
        comp_score_proj,
        comp_pooled,
        comp_normed,
        index_score,
        index_kv_cache,
        comp_kv_state_out,
        comp_score_state_out,
        topk_idxs,
    )
    return topk_idxs, index_kv_cache, comp_kv_state_out, comp_score_state_out


@pl.jit.inline
def indexer_decode_fwd(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    qr: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    wq_b_t: pl.Tensor[[Q_LORA_RANK, INDEX_Q_OUT], pl.BF16],
    weights_proj_t: pl.Tensor[[HIDDEN, INDEX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    offset: pl.Tensor[[1], pl.INT32],
    comp_kv_state: pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    comp_score_state: pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    index_kv_cache_in: pl.Tensor[[B, INDEX_SCORE_LEN, INDEX_HEAD_DIM], pl.BF16],
    comp_slot: pl.Tensor[[1], pl.INT32],
    comp_cache_slot: pl.Tensor[[1], pl.INT32],
    comp_should_compress: pl.Tensor[[1], pl.INT32],
    comp_wkv_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    comp_wgate_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    comp_ape: pl.Tensor[[COMPRESS_RATIO, INDEX_PROJ_DIM], pl.FP32],
    comp_norm_w: pl.Tensor[[INDEX_HEAD_DIM], pl.BF16],
    comp_cos: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    comp_sin: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    q_proj: pl.Tensor[[B, S_DYN, INDEX_Q_OUT], pl.BF16],
    q_rope: pl.Tensor[[B, S_DYN, INDEX_N_HEADS, INDEX_HEAD_DIM], pl.BF16],
    weights: pl.Tensor[[B, S_DYN, INDEX_N_HEADS], pl.BF16],
    comp_kv_proj: pl.Tensor[[B, S_DYN, INDEX_PROJ_DIM], pl.FP32],
    comp_score_proj: pl.Tensor[[B, S_DYN, INDEX_PROJ_DIM], pl.FP32],
    comp_pooled: pl.Tensor[[B, 1, INDEX_HEAD_DIM], pl.BF16],
    comp_normed: pl.Tensor[[B, 1, INDEX_HEAD_DIM], pl.BF16],
    index_score: pl.Tensor[[B, S_DYN, INDEX_SCORE_LEN], pl.FP32],
    index_kv_cache: pl.Tensor[[B, INDEX_SCORE_LEN, INDEX_HEAD_DIM], pl.BF16],
    comp_kv_state_out: pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    comp_score_state_out: pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    topk_idxs: pl.Tensor[[B, S_DYN, INDEX_TOPK], pl.INT32],
):
    """Run official ``Indexer.forward`` decode path with bf16 computation."""
    x.bind_dynamic(1, S_DYN)
    qr.bind_dynamic(1, S_DYN)
    cos.bind_dynamic(0, S_DYN)
    sin.bind_dynamic(0, S_DYN)
    q_proj.bind_dynamic(1, S_DYN)
    q_rope.bind_dynamic(1, S_DYN)
    weights.bind_dynamic(1, S_DYN)
    comp_kv_proj.bind_dynamic(1, S_DYN)
    comp_score_proj.bind_dynamic(1, S_DYN)
    index_score.bind_dynamic(1, S_DYN)
    topk_idxs.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    offset_i32 = pl.read(offset, [0])
    cache_slot_i32 = pl.read(comp_cache_slot, [0])
    should_flag = pl.read(comp_should_compress, [0])
    cache_len = cache_slot_i32 + should_flag

    q_proj = linear_1024_to_8192(qr, wq_b_t, q_proj)
    q_unflat = pl.reshape(q_proj, [B, tokens, INDEX_N_HEADS, INDEX_HEAD_DIM])
    q_rope = rope_4d_128_fwd(q_unflat, cos, sin, q_rope)

    weights = linear_4096_to_64(x, weights_proj_t, weights)
    weights_flat = pl.reshape(weights, [tokens, INDEX_N_HEADS])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="indexer_decode_weights_scale"):
        scaled = pl.mul(
            pl.cast(weights_flat[0:1, 0:INDEX_N_HEADS], target_type=pl.FP32),
            INDEX_WEIGHTS_SCALE,
        )
        weights_flat[0:1, 0:INDEX_N_HEADS] = pl.cast(scaled, target_type=pl.BF16, mode="rint")

    comp_pooled, comp_kv_state_out, comp_score_state_out, index_kv_cache = compressor_ratio4_indexer_decode_fwd(
        x,
        comp_kv_state,
        comp_score_state,
        index_kv_cache_in,
        comp_slot,
        comp_cache_slot,
        comp_should_compress,
        comp_wkv_t,
        comp_wgate_t,
        comp_ape,
        comp_norm_w,
        comp_cos,
        comp_sin,
        comp_kv_proj,
        comp_score_proj,
        comp_pooled,
        comp_normed,
        comp_pooled,
        comp_kv_state_out,
        comp_score_state_out,
        index_kv_cache,
    )

    q_flat = pl.reshape(q_rope, [tokens * INDEX_N_HEADS, INDEX_HEAD_DIM])
    cache_flat = pl.reshape(index_kv_cache, [INDEX_SCORE_LEN, INDEX_HEAD_DIM])
    score_flat = pl.reshape(index_score, [tokens, INDEX_SCORE_LEN])
    topk_flat = pl.reshape(topk_idxs, [tokens, INDEX_TOPK])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="indexer_decode_score_init"):
        score_flat[0:1, 0:INDEX_SCORE_LEN] = pl.full([1, INDEX_SCORE_LEN], dtype=pl.FP32, value=NEG_INF)
        topk_flat[0:1, 0:INDEX_TOPK] = pl.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)

    if cache_len > 0:
        for cb in pl.range(INDEX_SCORE_BLOCKS):
            cache0 = cb * CACHE_TILE
            if cache_len > cache0:
                valid_len = pl.min(CACHE_TILE, cache_len - cache0)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="indexer_decode_score_chunk"):
                    q_tile = q_flat[0:INDEX_N_HEADS, 0:INDEX_HEAD_DIM]
                    kv_tile = pl.slice(cache_flat, [CACHE_TILE, INDEX_HEAD_DIM], [cache0, 0])
                    score_tile_ch = pl.matmul(kv_tile, q_tile, b_trans=True, out_dtype=pl.FP32)
                    weights_row = pl.cast(weights_flat[0:1, 0:INDEX_N_HEADS], target_type=pl.FP32)
                    relu_score = pl.maximum(score_tile_ch, pl.mul(score_tile_ch, 0.0))
                    weighted = pl.col_expand_mul(relu_score, weights_row)
                    score_chunk = pl.reshape(pl.row_sum(weighted), [1, CACHE_TILE])
                    score_valid = pl.fillpad(
                        pl.set_validshape(score_chunk, 1, valid_len),
                        pad_value=pl.PadValue.min,
                    )
                    score_valid = pl.maximum(
                        score_valid,
                        pl.full([1, CACHE_TILE], dtype=pl.FP32, value=NEG_INF),
                    )
                    score_flat[0:1, cache0 : cache0 + CACHE_TILE] = score_valid

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="indexer_decode_topk"):
            score_row = score_flat[0:1, 0:INDEX_SCORE_LEN]
            score_work = pl.full([ROW_ARGMAX_WORK_ROWS, INDEX_SCORE_LEN], dtype=pl.FP32, value=NEG_INF)
            score_work[0:1, 0:INDEX_SCORE_LEN] = score_row
            pos_i32 = pl.arange(0, [1, INDEX_SCORE_LEN], dtype=pl.INT32)
            neg_inf_row = pl.full([1, INDEX_SCORE_LEN], dtype=pl.FP32, value=NEG_INF)
            valid_topk = pl.min(INDEX_TOPK, cache_len)
            for k in pl.range(INDEX_TOPK):
                if k < valid_topk:
                    best_pos_tile = pl.row_argmax(score_work)
                    best_pos_i32 = pl.read(best_pos_tile, [0, 0])
                    pl.write(topk_flat, [0, k], best_pos_i32 + offset_i32)
                    selected_i32 = pl.cmp(pos_i32, best_pos_i32, cmp_type=0)
                    selected = pl.cast(selected_i32, target_type=pl.FP32)
                    keep = pl.sub(pl.full([1, INDEX_SCORE_LEN], dtype=pl.FP32, value=1.0), selected)
                    score_work_row = score_work[0:1, 0:INDEX_SCORE_LEN]
                    masked_score = pl.add(pl.mul(score_work_row, keep), pl.mul(neg_inf_row, selected))
                    score_work[0:1, 0:INDEX_SCORE_LEN] = masked_score

    return topk_idxs, index_kv_cache, comp_kv_state_out, comp_score_state_out


@pl.jit
def indexer_decode_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    qr: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    wq_b_t: pl.Tensor[[Q_LORA_RANK, INDEX_Q_OUT], pl.BF16],
    weights_proj_t: pl.Tensor[[HIDDEN, INDEX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    offset: pl.Tensor[[1], pl.INT32],
    comp_kv_state: pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    comp_score_state: pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    index_kv_cache_in: pl.Tensor[[B, INDEX_SCORE_LEN, INDEX_HEAD_DIM], pl.BF16],
    comp_slot: pl.Tensor[[1], pl.INT32],
    comp_cache_slot: pl.Tensor[[1], pl.INT32],
    comp_should_compress: pl.Tensor[[1], pl.INT32],
    comp_wkv_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    comp_wgate_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    comp_ape: pl.Tensor[[COMPRESS_RATIO, INDEX_PROJ_DIM], pl.FP32],
    comp_norm_w: pl.Tensor[[INDEX_HEAD_DIM], pl.BF16],
    comp_cos: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    comp_sin: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    q_proj: pl.Tensor[[B, S_DYN, INDEX_Q_OUT], pl.BF16],
    q_rope: pl.Tensor[[B, S_DYN, INDEX_N_HEADS, INDEX_HEAD_DIM], pl.BF16],
    weights: pl.Tensor[[B, S_DYN, INDEX_N_HEADS], pl.BF16],
    comp_kv_proj: pl.Tensor[[B, S_DYN, INDEX_PROJ_DIM], pl.FP32],
    comp_score_proj: pl.Tensor[[B, S_DYN, INDEX_PROJ_DIM], pl.FP32],
    comp_pooled: pl.Tensor[[B, 1, INDEX_HEAD_DIM], pl.BF16],
    comp_normed: pl.Tensor[[B, 1, INDEX_HEAD_DIM], pl.BF16],
    topk_idxs: pl.Out[pl.Tensor[[B, S_DYN, INDEX_TOPK], pl.INT32]],
    index_score: pl.Tensor[[B, S_DYN, INDEX_SCORE_LEN], pl.FP32],
    index_kv_cache: pl.Out[pl.Tensor[[B, INDEX_SCORE_LEN, INDEX_HEAD_DIM], pl.BF16]],
    comp_kv_state_out: pl.Out[pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
    comp_score_state_out: pl.Out[pl.Tensor[[B, C4_STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
):
    topk_idxs, index_kv_cache, comp_kv_state_out, comp_score_state_out = indexer_decode_fwd(
        x,
        qr,
        wq_b_t,
        weights_proj_t,
        cos,
        sin,
        offset,
        comp_kv_state,
        comp_score_state,
        index_kv_cache_in,
        comp_slot,
        comp_cache_slot,
        comp_should_compress,
        comp_wkv_t,
        comp_wgate_t,
        comp_ape,
        comp_norm_w,
        comp_cos,
        comp_sin,
        q_proj,
        q_rope,
        weights,
        comp_kv_proj,
        comp_score_proj,
        comp_pooled,
        comp_normed,
        index_score,
        index_kv_cache,
        comp_kv_state_out,
        comp_score_state_out,
        topk_idxs,
    )
    return topk_idxs, index_kv_cache, comp_kv_state_out, comp_score_state_out


def golden_indexer_forward(tensors, start_pos: int):
    import torch

    seq_len = tensors["x"].shape[1]
    offset = int(tensors["offset"][0].item())
    if start_pos < 0:
        raise ValueError(f"start_pos must be non-negative, got {start_pos}")
    if start_pos > 0 and seq_len != 1:
        raise ValueError(f"decode expects seq_len=1, got {seq_len}")
    end_pos = start_pos + seq_len
    cache_len = end_pos // COMPRESS_RATIO

    q_proj = torch.matmul(tensors["qr"].float(), tensors["wq_b_t"].float()).to(torch.bfloat16)
    q = q_proj.unflatten(-1, (INDEX_N_HEADS, INDEX_HEAD_DIM))
    q = _apply_rope_golden(q, tensors["cos"], tensors["sin"], inverse=False)

    if start_pos == 0:
        blocks = seq_len // COMPRESS_RATIO
        comp_tensors = {
            "x": tensors["x"],
            "wkv_t": tensors["comp_wkv_t"],
            "wgate_t": tensors["comp_wgate_t"],
            "ape": tensors["comp_ape"],
            "norm_w": tensors["comp_norm_w"],
            "cos": tensors["comp_cos"],
            "sin": tensors["comp_sin"],
            "block_count": tensors["comp_block_count"],
            "kv_proj": tensors["comp_kv_proj"],
            "score_proj": tensors["comp_score_proj"],
            "pooled": tensors["comp_pooled"],
            "normed": tensors["comp_normed"],
            "compressed": torch.zeros(B, max(1, blocks), INDEX_HEAD_DIM, dtype=torch.bfloat16),
            "kv_state_out": tensors["comp_kv_state_out"],
            "score_state_out": tensors["comp_score_state_out"],
            "compressed_cache_out": tensors["index_kv_cache"],
        }
    else:
        comp_tensors = {
            "x": tensors["x"],
            "kv_state": tensors["comp_kv_state"],
            "score_state": tensors["comp_score_state"],
            "compressed_cache": tensors.get("index_kv_cache_in", tensors["index_kv_cache"]),
            "slot": tensors["comp_slot"],
            "cache_slot": tensors["comp_cache_slot"],
            "should_compress": tensors["comp_should_compress"],
            "wkv_t": tensors["comp_wkv_t"],
            "wgate_t": tensors["comp_wgate_t"],
            "ape": tensors["comp_ape"],
            "norm_w": tensors["comp_norm_w"],
            "cos": tensors["comp_cos"],
            "sin": tensors["comp_sin"],
            "kv_proj": tensors["comp_kv_proj"],
            "score_proj": tensors["comp_score_proj"],
            "pooled": tensors["comp_pooled"],
            "normed": tensors["comp_normed"],
            "compressed": torch.zeros(B, 1, INDEX_HEAD_DIM, dtype=torch.bfloat16),
            "kv_state_out": tensors["comp_kv_state_out"],
            "score_state_out": tensors["comp_score_state_out"],
            "compressed_cache_out": tensors["index_kv_cache"],
        }
    golden_compressor_ratio4_indexer_forward(comp_tensors, start_pos=start_pos)

    weights = torch.matmul(tensors["x"].float(), tensors["weights_proj_t"].float()).to(torch.bfloat16)
    weights = weights.float() * INDEX_WEIGHTS_SCALE

    score_full = torch.full((B, seq_len, INDEX_SCORE_LEN), NEG_INF, dtype=torch.float32)
    topk = torch.full((B, seq_len, INDEX_TOPK), -1, dtype=torch.int32)
    if cache_len > 0:
        kv_cache = tensors["index_kv_cache"][:, :cache_len]
        score = torch.einsum("bshd,btd->bsht", q.float(), kv_cache.float())
        score = (score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if start_pos == 0:
            visible = torch.arange(1, seq_len + 1, dtype=torch.int64).unsqueeze(1) // COMPRESS_RATIO
            pos = torch.arange(cache_len, dtype=torch.int64).unsqueeze(0)
            score = score.masked_fill(pos >= visible, NEG_INF)
        else:
            visible = torch.full((seq_len, 1), cache_len, dtype=torch.int64)
        score_full[:, :, :cache_len] = score

        k = min(INDEX_TOPK, cache_len)
        idx = score.topk(k, dim=-1)[1]
        if start_pos == 0:
            invalid = idx >= visible.view(1, seq_len, 1)
            idx = torch.where(invalid, torch.full_like(idx, -1), idx + offset)
        else:
            idx = idx + offset
        topk[:, :, :k] = idx.to(torch.int32)

    tensors["topk_idxs"][:] = topk
    tensors["index_score"][:] = score_full
    tensors["index_kv_cache"][:] = comp_tensors["compressed_cache_out"]
    tensors["comp_kv_state_out"][:] = comp_tensors["kv_state_out"]
    tensors["comp_score_state_out"][:] = comp_tensors["score_state_out"]


def golden_indexer_prefill(tensors):
    golden_indexer_forward(tensors, start_pos=0)


def golden_indexer_decode(tensors):
    if "start_pos" in tensors:
        start_pos = int(tensors["start_pos"][0].item())
    else:
        start_pos = int(tensors["comp_cache_slot"][0].item()) * COMPRESS_RATIO + int(tensors["comp_slot"][0].item())
    if start_pos <= 0:
        raise ValueError("decode start_pos must be greater than 0; start_pos=0 is prefill")
    golden_indexer_forward(tensors, start_pos=start_pos)


def build_indexer_prefill_specs(seq_len: int = DEFAULT_SEQ_LEN):
    import torch

    from models.golden import TensorSpec

    blocks = seq_len // COMPRESS_RATIO
    compressed_len = max(1, blocks)
    freqs_cos, freqs_sin = build_deepseek_v4_rope_tables(
        compress_ratio=COMPRESS_RATIO,
        max_seq_len=max(seq_len, 1),
    )
    cos, sin = materialize_rope_range(freqs_cos, freqs_sin, 0, seq_len)
    comp_cos, comp_sin = materialize_compressor_rope(freqs_cos, freqs_sin, seq_len, COMPRESS_RATIO)
    if blocks == 0:
        comp_cos = freqs_cos[:1].contiguous()
        comp_sin = freqs_sin[:1].contiguous()

    def init_x():
        return torch.randn(B, seq_len, HIDDEN) * 0.1

    def init_qr():
        return torch.randn(B, seq_len, Q_LORA_RANK) * 0.1

    def init_wq_b_t():
        return torch.randn(Q_LORA_RANK, INDEX_Q_OUT) * 0.02

    def init_weights_proj_t():
        return torch.randn(HIDDEN, INDEX_N_HEADS) * 0.02

    def init_comp_w():
        return torch.randn(HIDDEN, INDEX_PROJ_DIM) * 0.02

    def init_comp_ape():
        return torch.randn(COMPRESS_RATIO, INDEX_PROJ_DIM) * 0.02

    def init_comp_norm_w():
        return torch.randn(INDEX_HEAD_DIM) * 0.1 + 1.0

    return [
        TensorSpec("x", [B, seq_len, HIDDEN], torch.bfloat16, init_value=init_x),
        TensorSpec("qr", [B, seq_len, Q_LORA_RANK], torch.bfloat16, init_value=init_qr),
        TensorSpec("wq_b_t", [Q_LORA_RANK, INDEX_Q_OUT], torch.bfloat16, init_value=init_wq_b_t),
        TensorSpec("weights_proj_t", [HIDDEN, INDEX_N_HEADS], torch.bfloat16, init_value=init_weights_proj_t),
        TensorSpec("cos", [seq_len, ROPE_HALF], torch.float32, init_value=cos),
        TensorSpec("sin", [seq_len, ROPE_HALF], torch.float32, init_value=sin),
        TensorSpec("offset", [1], torch.int32, init_value=torch.tensor([seq_len], dtype=torch.int32)),
        TensorSpec("comp_wkv_t", [HIDDEN, INDEX_PROJ_DIM], torch.bfloat16, init_value=init_comp_w),
        TensorSpec("comp_wgate_t", [HIDDEN, INDEX_PROJ_DIM], torch.bfloat16, init_value=init_comp_w),
        TensorSpec("comp_ape", [COMPRESS_RATIO, INDEX_PROJ_DIM], torch.float32, init_value=init_comp_ape),
        TensorSpec("comp_norm_w", [INDEX_HEAD_DIM], torch.bfloat16, init_value=init_comp_norm_w),
        TensorSpec("comp_cos", [compressed_len, ROPE_HALF], torch.float32, init_value=comp_cos),
        TensorSpec("comp_sin", [compressed_len, ROPE_HALF], torch.float32, init_value=comp_sin),
        TensorSpec("comp_block_count", [1], torch.int32, init_value=torch.tensor([blocks], dtype=torch.int32)),
        TensorSpec("q_proj", [B, seq_len, INDEX_Q_OUT], torch.bfloat16, init_value=0.0),
        TensorSpec("q_rope", [B, seq_len, INDEX_N_HEADS, INDEX_HEAD_DIM], torch.bfloat16, init_value=0.0),
        TensorSpec("weights", [B, seq_len, INDEX_N_HEADS], torch.bfloat16, init_value=0.0),
        TensorSpec("comp_kv_proj", [B, seq_len, INDEX_PROJ_DIM], torch.float32, init_value=0.0),
        TensorSpec("comp_score_proj", [B, seq_len, INDEX_PROJ_DIM], torch.float32, init_value=0.0),
        TensorSpec("comp_pooled", [B, compressed_len, INDEX_HEAD_DIM], torch.bfloat16, init_value=0.0),
        TensorSpec("comp_normed", [B, compressed_len, INDEX_HEAD_DIM], torch.bfloat16, init_value=0.0),
        TensorSpec("topk_idxs", [B, seq_len, INDEX_TOPK], torch.int32, is_output=True),
        TensorSpec("index_score", [B, seq_len, INDEX_SCORE_LEN], torch.float32, init_value=0.0, is_output=True),
        TensorSpec("index_kv_cache", [B, INDEX_SCORE_LEN, INDEX_HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec("comp_kv_state_out", [B, C4_STATE_ROWS, INDEX_PROJ_DIM], torch.float32, is_output=True),
        TensorSpec("comp_score_state_out", [B, C4_STATE_ROWS, INDEX_PROJ_DIM], torch.float32, is_output=True),
    ]


def build_indexer_decode_specs(start_pos: int = DEFAULT_DECODE_START_POS):
    import torch

    from models.golden import TensorSpec

    if start_pos <= 0:
        raise ValueError(f"decode start_pos must be greater than 0, got {start_pos}")

    seq_len = 1
    slot = start_pos % COMPRESS_RATIO
    cache_slot = start_pos // COMPRESS_RATIO
    should_compress = int((start_pos + 1) % COMPRESS_RATIO == 0)
    max_seq_len = max(start_pos + seq_len, 1)
    freqs_cos, freqs_sin = build_deepseek_v4_rope_tables(
        compress_ratio=COMPRESS_RATIO,
        max_seq_len=max_seq_len,
    )
    cos, sin = materialize_rope_range(freqs_cos, freqs_sin, start_pos, seq_len)
    if should_compress:
        rope_pos = start_pos + 1 - COMPRESS_RATIO
        comp_cos = freqs_cos[rope_pos : rope_pos + 1].contiguous()
        comp_sin = freqs_sin[rope_pos : rope_pos + 1].contiguous()
    else:
        comp_cos = torch.zeros(1, ROPE_HALF, dtype=torch.float32)
        comp_sin = torch.zeros(1, ROPE_HALF, dtype=torch.float32)

    def init_x():
        return torch.randn(B, seq_len, HIDDEN) * 0.1

    def init_qr():
        return torch.randn(B, seq_len, Q_LORA_RANK) * 0.1

    def init_wq_b_t():
        return torch.randn(Q_LORA_RANK, INDEX_Q_OUT) * 0.02

    def init_weights_proj_t():
        return torch.randn(HIDDEN, INDEX_N_HEADS) * 0.02

    def init_comp_w():
        return torch.randn(HIDDEN, INDEX_PROJ_DIM) * 0.02

    def init_comp_ape():
        return torch.randn(COMPRESS_RATIO, INDEX_PROJ_DIM) * 0.02

    def init_comp_norm_w():
        return torch.randn(INDEX_HEAD_DIM) * 0.1 + 1.0

    def init_comp_kv_state():
        return torch.randn(B, C4_STATE_ROWS, INDEX_PROJ_DIM) * 0.1

    def init_comp_score_state():
        return torch.randn(B, C4_STATE_ROWS, INDEX_PROJ_DIM) * 0.1

    def init_index_kv_cache():
        return torch.randn(B, INDEX_SCORE_LEN, INDEX_HEAD_DIM) * 0.1

    return [
        TensorSpec("x", [B, seq_len, HIDDEN], torch.bfloat16, init_value=init_x),
        TensorSpec("qr", [B, seq_len, Q_LORA_RANK], torch.bfloat16, init_value=init_qr),
        TensorSpec("wq_b_t", [Q_LORA_RANK, INDEX_Q_OUT], torch.bfloat16, init_value=init_wq_b_t),
        TensorSpec("weights_proj_t", [HIDDEN, INDEX_N_HEADS], torch.bfloat16, init_value=init_weights_proj_t),
        TensorSpec("cos", [seq_len, ROPE_HALF], torch.float32, init_value=cos),
        TensorSpec("sin", [seq_len, ROPE_HALF], torch.float32, init_value=sin),
        TensorSpec("offset", [1], torch.int32, init_value=torch.tensor([WINDOW_SIZE], dtype=torch.int32)),
        TensorSpec("comp_kv_state", [B, C4_STATE_ROWS, INDEX_PROJ_DIM], torch.float32, init_value=init_comp_kv_state),
        TensorSpec(
            "comp_score_state",
            [B, C4_STATE_ROWS, INDEX_PROJ_DIM],
            torch.float32,
            init_value=init_comp_score_state,
        ),
        TensorSpec("index_kv_cache_in", [B, INDEX_SCORE_LEN, INDEX_HEAD_DIM], torch.bfloat16, init_value=init_index_kv_cache),
        TensorSpec("comp_slot", [1], torch.int32, init_value=torch.tensor([slot], dtype=torch.int32)),
        TensorSpec("comp_cache_slot", [1], torch.int32, init_value=torch.tensor([cache_slot], dtype=torch.int32)),
        TensorSpec(
            "comp_should_compress",
            [1],
            torch.int32,
            init_value=torch.tensor([should_compress], dtype=torch.int32),
        ),
        TensorSpec("comp_wkv_t", [HIDDEN, INDEX_PROJ_DIM], torch.bfloat16, init_value=init_comp_w),
        TensorSpec("comp_wgate_t", [HIDDEN, INDEX_PROJ_DIM], torch.bfloat16, init_value=init_comp_w),
        TensorSpec("comp_ape", [COMPRESS_RATIO, INDEX_PROJ_DIM], torch.float32, init_value=init_comp_ape),
        TensorSpec("comp_norm_w", [INDEX_HEAD_DIM], torch.bfloat16, init_value=init_comp_norm_w),
        TensorSpec("comp_cos", [1, ROPE_HALF], torch.float32, init_value=comp_cos),
        TensorSpec("comp_sin", [1, ROPE_HALF], torch.float32, init_value=comp_sin),
        TensorSpec("q_proj", [B, seq_len, INDEX_Q_OUT], torch.bfloat16, init_value=0.0),
        TensorSpec("q_rope", [B, seq_len, INDEX_N_HEADS, INDEX_HEAD_DIM], torch.bfloat16, init_value=0.0),
        TensorSpec("weights", [B, seq_len, INDEX_N_HEADS], torch.bfloat16, init_value=0.0),
        TensorSpec("comp_kv_proj", [B, seq_len, INDEX_PROJ_DIM], torch.float32, init_value=0.0),
        TensorSpec("comp_score_proj", [B, seq_len, INDEX_PROJ_DIM], torch.float32, init_value=0.0),
        TensorSpec("comp_pooled", [B, 1, INDEX_HEAD_DIM], torch.bfloat16, init_value=0.0),
        TensorSpec("comp_normed", [B, 1, INDEX_HEAD_DIM], torch.bfloat16, init_value=0.0),
        TensorSpec("topk_idxs", [B, seq_len, INDEX_TOPK], torch.int32, is_output=True),
        TensorSpec("index_score", [B, seq_len, INDEX_SCORE_LEN], torch.float32, init_value=0.0, is_output=True),
        TensorSpec("index_kv_cache", [B, INDEX_SCORE_LEN, INDEX_HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec("comp_kv_state_out", [B, C4_STATE_ROWS, INDEX_PROJ_DIM], torch.float32, is_output=True),
        TensorSpec("comp_score_state_out", [B, C4_STATE_ROWS, INDEX_PROJ_DIM], torch.float32, is_output=True),
    ]


def main() -> int:
    import argparse

    from models.golden import ratio_allclose, run_jit, topk_indices_by_score

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 Flash Indexer validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3sim", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-s", "--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--case", choices=["all", "prefill", "decode"], default="all")
    parser.add_argument("--decode-start-pos", type=int, default=DEFAULT_DECODE_START_POS)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    runtime_cfg = {
        "platform": args.platform,
        "device_id": args.device,
        "enable_l2_swimlane": args.enable_l2_swimlane,
    }
    compare_fn = {
        "topk_idxs": topk_indices_by_score(
            "index_score",
            index_offset_name="offset",
            invalid_index=-1,
            atol=1e-4,
            rtol=1.0 / 128,
        ),
        "index_score": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "index_kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "comp_kv_state_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "comp_score_state_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
    }

    cases = []
    if args.case in ("all", "prefill"):
        cases.append(
            (
                "indexer-prefill",
                indexer_prefill_test,
                lambda: build_indexer_prefill_specs(args.seq_len),
                golden_indexer_prefill,
            )
        )
    if args.case in ("all", "decode"):
        cases.append(
            (
                "indexer-decode",
                indexer_decode_test,
                lambda: build_indexer_decode_specs(args.decode_start_pos),
                golden_indexer_decode,
            )
        )

    passed = True
    for name, fn, specs_fn, golden_fn in cases:
        print(f"[CASE] {name}", flush=True)
        result = run_jit(
            fn=fn,
            specs=specs_fn(),
            golden_fn=golden_fn,
            runtime_cfg=runtime_cfg,
            compile_only=args.compile_only,
            compare_fn=compare_fn,
        )
        if not result.passed:
            passed = False
            if result.error:
                print(result.error)
            if not args.compile_only:
                break
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "B",
    "S_DYN",
    "C_DYN",
    "HIDDEN",
    "Q_LORA_RANK",
    "INDEX_N_HEADS",
    "INDEX_HEAD_DIM",
    "INDEX_Q_OUT",
    "INDEX_TOPK",
    "INDEX_SCORE_LEN",
    "INDEX_WEIGHTS_SCALE",
    "indexer_prefill_fwd",
    "indexer_prefill_test",
    "indexer_decode_fwd",
    "indexer_decode_test",
    "golden_indexer_forward",
    "golden_indexer_prefill",
    "golden_indexer_decode",
    "build_indexer_prefill_specs",
    "build_indexer_decode_specs",
]
