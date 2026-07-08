"""DeepSeek V4 Flash ratio-4 compressor PyPTO kernels."""

import pypto.language as pl

from models.common import assert_divisible
from models.config import FLASH_CONFIG as M
from models.linear import linear_4096_to_1024_fp32, linear_4096_to_256_fp32
from models.rope import (
    _apply_rope_golden,
    build_deepseek_v4_rope_tables,
    materialize_compressor_rope,
)


B = 1
S_DYN = pl.dynamic("S_DYN")
C_DYN = pl.dynamic("C_DYN")

HIDDEN = M.dim
ATTN_HEAD_DIM = M.head_dim
INDEX_HEAD_DIM = M.index_head_dim
ATTN_PROJ_DIM = 2 * ATTN_HEAD_DIM
INDEX_PROJ_DIM = 2 * INDEX_HEAD_DIM
ROPE_DIM = M.rope_head_dim
ROPE_HALF = M.rope_head_dim // 2
ATTN_TAIL_OFFSET = ATTN_HEAD_DIM - ROPE_DIM
INDEX_TAIL_OFFSET = INDEX_HEAD_DIM - ROPE_DIM
COMPRESS_RATIO = 4
STATE_ROWS = 2 * COMPRESS_RATIO
CSA_MAX_POSITION_EMBEDDINGS = 4096
TOPK_CSA_COMPRESSED = CSA_MAX_POSITION_EMBEDDINGS // COMPRESS_RATIO
NEG_INF = -3.4028234663852886e38
EPS = M.rms_norm_eps
INDEX_INV_HEAD_DIM = 1.0 / INDEX_HEAD_DIM
ATTN_INV_HEAD_DIM = 1.0 / ATTN_HEAD_DIM

HEAD_CHUNK = 64
ATTN_HEAD_CHUNKS = ATTN_HEAD_DIM // HEAD_CHUNK
ATTN_PROJ_CHUNKS = ATTN_PROJ_DIM // HEAD_CHUNK
INDEX_HEAD_CHUNKS = INDEX_HEAD_DIM // HEAD_CHUNK
INDEX_PROJ_CHUNKS = INDEX_PROJ_DIM // HEAD_CHUNK
RMS_T_TILE = 8
RMS_D_TILE = 128
ATTN_RMS_BLOCKS = ATTN_HEAD_DIM // RMS_D_TILE
INDEX_RMS_BLOCKS = INDEX_HEAD_DIM // RMS_D_TILE
ROPE_T_TILE = 16
ROPE_PREFIX_TILE = 64
DEFAULT_SEQ_LEN = 16
DEFAULT_DECODE_START_POS = COMPRESS_RATIO - 1

assert_divisible(ATTN_HEAD_DIM, HEAD_CHUNK, "compressor ratio4 attention head chunks")
assert_divisible(ATTN_PROJ_DIM, HEAD_CHUNK, "compressor ratio4 attention projection chunks")
assert_divisible(ATTN_HEAD_DIM, RMS_D_TILE, "compressor ratio4 attention RMSNorm size")
assert_divisible(ATTN_TAIL_OFFSET, ROPE_PREFIX_TILE, "compressor ratio4 attention RoPE prefix")
assert_divisible(INDEX_HEAD_DIM, HEAD_CHUNK, "compressor ratio4 index head chunks")
assert_divisible(INDEX_PROJ_DIM, HEAD_CHUNK, "compressor ratio4 projection chunks")
assert_divisible(INDEX_HEAD_DIM, RMS_D_TILE, "compressor ratio4 RMSNorm size")
assert_divisible(INDEX_TAIL_OFFSET, ROPE_PREFIX_TILE, "compressor ratio4 RoPE prefix")


@pl.jit.inline
def compressor_ratio4_indexer_prefill_fwd(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    wkv_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    wgate_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, INDEX_PROJ_DIM], pl.FP32],
    norm_w: pl.Tensor[[INDEX_HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    block_count: pl.Tensor[[1], pl.INT32],
    compressed: pl.Tensor[[B, C_DYN, INDEX_HEAD_DIM], pl.BF16],
    kv_state_out: pl.Tensor[[B, STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    score_state_out: pl.Tensor[[B, STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    compressed_cache_out: pl.Tensor[[B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM], pl.BF16],
):
    """Run official Indexer ``Compressor.forward`` for ``compress_ratio == 4`` prefill."""
    x.bind_dynamic(1, S_DYN)
    cos.bind_dynamic(0, C_DYN)
    sin.bind_dynamic(0, C_DYN)
    compressed.bind_dynamic(1, C_DYN)

    tokens = pl.tensor.dim(x, 1)
    padded_blocks = pl.tensor.dim(cos, 0)
    blocks = pl.read(block_count, [0])
    should_compress = blocks > 0
    cutoff = blocks * COMPRESS_RATIO
    remainder = tokens - cutoff

    kv_proj = pl.create_tensor([B, tokens, INDEX_PROJ_DIM], dtype=pl.FP32)
    score_proj = pl.create_tensor([B, tokens, INDEX_PROJ_DIM], dtype=pl.FP32)
    pooled = pl.create_tensor([B, padded_blocks, INDEX_HEAD_DIM], dtype=pl.BF16)
    normed = pl.create_tensor([B, padded_blocks, INDEX_HEAD_DIM], dtype=pl.BF16)

    kv_proj = linear_4096_to_256_fp32(x, wkv_t, kv_proj)
    score_proj = linear_4096_to_256_fp32(x, wgate_t, score_proj)

    kv_proj_flat = pl.reshape(kv_proj, [tokens, INDEX_PROJ_DIM])
    score_proj_flat = pl.reshape(score_proj, [tokens, INDEX_PROJ_DIM])
    pooled_flat = pl.reshape(pooled, [padded_blocks, INDEX_HEAD_DIM])
    kv_state_flat = pl.reshape(kv_state_out, [STATE_ROWS, INDEX_PROJ_DIM])
    score_state_flat = pl.reshape(score_state_out, [STATE_ROWS, INDEX_PROJ_DIM])
    cache_flat = pl.reshape(compressed_cache_out, [TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM])

    for row in pl.spmd(STATE_ROWS, name_hint="compressor_c4_index_state_write"):
        for hb in pl.pipeline(INDEX_PROJ_CHUNKS, stage=2):
            h0 = hb * HEAD_CHUNK
            if should_compress:
                if row < COMPRESS_RATIO:
                    src = cutoff - COMPRESS_RATIO + row
                    kv_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = kv_proj_flat[
                        src : src + 1, h0 : h0 + HEAD_CHUNK
                    ]
                    score_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.add(
                        score_proj_flat[src : src + 1, h0 : h0 + HEAD_CHUNK],
                        ape[row : row + 1, h0 : h0 + HEAD_CHUNK],
                    )
                elif row < COMPRESS_RATIO + remainder:
                    rem = row - COMPRESS_RATIO
                    src = cutoff + rem
                    kv_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = kv_proj_flat[
                        src : src + 1, h0 : h0 + HEAD_CHUNK
                    ]
                    score_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.add(
                        score_proj_flat[src : src + 1, h0 : h0 + HEAD_CHUNK],
                        ape[rem : rem + 1, h0 : h0 + HEAD_CHUNK],
                    )
                else:
                    kv_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.full(
                        [1, HEAD_CHUNK], dtype=pl.FP32, value=0.0
                    )
                    score_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.full(
                        [1, HEAD_CHUNK], dtype=pl.FP32, value=NEG_INF
                    )
            elif row >= COMPRESS_RATIO:
                rem = row - COMPRESS_RATIO
                if rem < remainder:
                    src = cutoff + rem
                    kv_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = kv_proj_flat[
                        src : src + 1, h0 : h0 + HEAD_CHUNK
                    ]
                    score_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.add(
                        score_proj_flat[src : src + 1, h0 : h0 + HEAD_CHUNK],
                        ape[rem : rem + 1, h0 : h0 + HEAD_CHUNK],
                    )
                else:
                    kv_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.full(
                        [1, HEAD_CHUNK], dtype=pl.FP32, value=0.0
                    )
                    score_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.full(
                        [1, HEAD_CHUNK], dtype=pl.FP32, value=NEG_INF
                    )
            else:
                kv_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.full(
                    [1, HEAD_CHUNK], dtype=pl.FP32, value=0.0
                )
                score_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.full(
                    [1, HEAD_CHUNK], dtype=pl.FP32, value=NEG_INF
                )

    for cache_row in pl.spmd(TOPK_CSA_COMPRESSED, name_hint="compressor_c4_index_cache_init"):
        for hb in pl.pipeline(INDEX_HEAD_CHUNKS, stage=2):
            h0 = hb * HEAD_CHUNK
            cache_flat[cache_row : cache_row + 1, h0 : h0 + HEAD_CHUNK] = pl.full(
                [1, HEAD_CHUNK], dtype=pl.BF16, value=0.0
            )

    if should_compress:
        for block in pl.range(blocks):
            t0 = block * COMPRESS_RATIO
            for hb in pl.spmd(INDEX_HEAD_CHUNKS, name_hint="compressor_c4_index_softmax_pool"):
                h0 = hb * HEAD_CHUNK
                kv_tile = pl.full([STATE_ROWS, HEAD_CHUNK], dtype=pl.FP32, value=0.0)
                score_tile = pl.full([STATE_ROWS, HEAD_CHUNK], dtype=pl.FP32, value=NEG_INF)

                if block > 0:
                    prev_t0 = t0 - COMPRESS_RATIO
                    for r in pl.range(COMPRESS_RATIO):
                        kv_tile[r : r + 1, 0:HEAD_CHUNK] = kv_proj_flat[
                            prev_t0 + r : prev_t0 + r + 1,
                            h0 : h0 + HEAD_CHUNK,
                        ]
                        score_tile[r : r + 1, 0:HEAD_CHUNK] = pl.add(
                            score_proj_flat[
                                prev_t0 + r : prev_t0 + r + 1,
                                h0 : h0 + HEAD_CHUNK,
                            ],
                            ape[r : r + 1, h0 : h0 + HEAD_CHUNK],
                        )

                for r in pl.range(COMPRESS_RATIO):
                    dst = COMPRESS_RATIO + r
                    kv_tile[dst : dst + 1, 0:HEAD_CHUNK] = kv_proj_flat[
                        t0 + r : t0 + r + 1,
                        INDEX_HEAD_DIM + h0 : INDEX_HEAD_DIM + h0 + HEAD_CHUNK,
                    ]
                    score_tile[dst : dst + 1, 0:HEAD_CHUNK] = pl.add(
                        score_proj_flat[
                            t0 + r : t0 + r + 1,
                            INDEX_HEAD_DIM + h0 : INDEX_HEAD_DIM + h0 + HEAD_CHUNK,
                        ],
                        ape[r : r + 1, INDEX_HEAD_DIM + h0 : INDEX_HEAD_DIM + h0 + HEAD_CHUNK],
                    )

                score_t = pl.transpose(score_tile, axis1=0, axis2=1)
                kv_t = pl.transpose(kv_tile, axis1=0, axis2=1)
                score_max = pl.row_max(score_t)
                score_exp = pl.exp(pl.row_expand_sub(score_t, score_max))
                score_sum = pl.row_sum(score_exp)
                score_prob = pl.row_expand_div(score_exp, score_sum)
                pooled_t = pl.row_sum(pl.mul(kv_t, score_prob))
                pooled_chunk = pl.reshape(pooled_t, [1, HEAD_CHUNK])
                pooled_flat[block : block + 1, h0 : h0 + HEAD_CHUNK] = pl.cast(
                    pooled_chunk, target_type=pl.BF16, mode="rint"
                )

        normed_flat = pl.reshape(normed, [padded_blocks, INDEX_HEAD_DIM])
        rms_row_blocks = (blocks + RMS_T_TILE - 1) // RMS_T_TILE
        for rb in pl.range(rms_row_blocks):
            r0 = rb * RMS_T_TILE
            valid_rows = pl.min(RMS_T_TILE, blocks - r0)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="compressor_c4_index_inline_rmsnorm"):
                partial_sq = pl.full([1, RMS_T_TILE], dtype=pl.FP32, value=0.0)
                for db in pl.range(INDEX_RMS_BLOCKS):
                    d0 = db * RMS_D_TILE
                    pooled_bf16 = pl.slice(
                        pooled_flat,
                        [RMS_T_TILE, RMS_D_TILE],
                        [r0, d0],
                        valid_shape=[valid_rows, RMS_D_TILE],
                    )
                    pooled_fp32 = pl.cast(pooled_bf16, target_type=pl.FP32)
                    partial_sq = pl.add(
                        partial_sq,
                        pl.reshape(pl.row_sum(pl.mul(pooled_fp32, pooled_fp32)), [1, RMS_T_TILE]),
                    )

                variance = pl.reshape(pl.add(pl.mul(partial_sq, INDEX_INV_HEAD_DIM), EPS), [RMS_T_TILE, 1])
                inv_rms = pl.recip(pl.sqrt(variance))

                for db in pl.range(INDEX_RMS_BLOCKS):
                    d0 = db * RMS_D_TILE
                    pooled_bf16 = pl.slice(
                        pooled_flat,
                        [RMS_T_TILE, RMS_D_TILE],
                        [r0, d0],
                        valid_shape=[valid_rows, RMS_D_TILE],
                    )
                    pooled_fp32 = pl.cast(pooled_bf16, target_type=pl.FP32)
                    weight_fp32 = pl.cast(
                        pl.reshape(norm_w[d0 : d0 + RMS_D_TILE], [1, RMS_D_TILE]),
                        target_type=pl.FP32,
                    )
                    normed_tile = pl.col_expand_mul(pl.row_expand_mul(pooled_fp32, inv_rms), weight_fp32)
                    normed_bf16 = pl.cast(normed_tile, target_type=pl.BF16, mode="rint")
                    for row in pl.range(valid_rows):
                        inline_rms_out_row = pl.slice(
                            normed_bf16,
                            [1, RMS_D_TILE],
                            [row, 0],
                            valid_shape=[1, RMS_D_TILE],
                        )
                        normed_flat = pl.assemble(normed_flat, inline_rms_out_row, [r0 + row, d0])

        compressed_flat = pl.reshape(compressed, [padded_blocks, INDEX_HEAD_DIM])
        rope_row_blocks = (blocks + ROPE_T_TILE - 1) // ROPE_T_TILE
        for rb in pl.range(rope_row_blocks):
            r0 = rb * ROPE_T_TILE
            valid_rows = pl.min(ROPE_T_TILE, blocks - r0)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="compressor_c4_index_inline_rope"):
                for pb in pl.range(INDEX_TAIL_OFFSET // ROPE_PREFIX_TILE):
                    p0 = pb * ROPE_PREFIX_TILE
                    prefix_tile = pl.slice(
                        normed_flat,
                        [ROPE_T_TILE, ROPE_PREFIX_TILE],
                        [r0, p0],
                        valid_shape=[valid_rows, ROPE_PREFIX_TILE],
                    )
                    for row in pl.range(valid_rows):
                        prefix_row = pl.slice(
                            prefix_tile,
                            [1, ROPE_PREFIX_TILE],
                            [row, 0],
                            valid_shape=[1, ROPE_PREFIX_TILE],
                        )
                        compressed_flat = pl.assemble(compressed_flat, prefix_row, [r0 + row, p0])

                ones = pl.full([ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
                col = pl.col_expand_mul(
                    ones,
                    pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32),
                )
                dup_f = pl.cast(pl.cast(pl.mul(col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
                dup_idx = pl.cast(dup_f, target_type=pl.INT32)
                lane = pl.sub(col, pl.mul(dup_f, 2.0))
                swap_idx = pl.cast(pl.sub(pl.add(col, 1.0), pl.mul(lane, 2.0)), target_type=pl.INT32)
                sign = pl.sub(pl.mul(lane, 2.0), 1.0)

                x_tile = pl.cast(
                    pl.slice(
                        normed_flat,
                        [ROPE_T_TILE, ROPE_DIM],
                        [r0, INDEX_TAIL_OFFSET],
                        valid_shape=[valid_rows, ROPE_DIM],
                    ),
                    target_type=pl.FP32,
                )
                cos_tile = pl.slice(cos, [ROPE_T_TILE, ROPE_HALF], [r0, 0], valid_shape=[valid_rows, ROPE_HALF])
                sin_tile = pl.slice(sin, [ROPE_T_TILE, ROPE_HALF], [r0, 0], valid_shape=[valid_rows, ROPE_HALF])
                cos_il = pl.gather(cos_tile, dim=-1, index=dup_idx)
                sin_il = pl.gather(sin_tile, dim=-1, index=dup_idx)
                swapped = pl.gather(x_tile, dim=-1, index=swap_idx)
                rotated = pl.add(pl.mul(x_tile, cos_il), pl.mul(pl.mul(swapped, sign), sin_il))
                rotated_bf16 = pl.cast(rotated, target_type=pl.BF16, mode="rint")

                for row in pl.range(valid_rows):
                    inline_rope_out_row = pl.slice(
                        rotated_bf16,
                        [1, ROPE_DIM],
                        [row, 0],
                        valid_shape=[1, ROPE_DIM],
                    )
                    compressed_flat = pl.assemble(compressed_flat, inline_rope_out_row, [r0 + row, INDEX_TAIL_OFFSET])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="compressor_c4_index_cache_write"):
            for block in pl.range(blocks):
                for hb in pl.range(INDEX_HEAD_CHUNKS):
                    h0 = hb * HEAD_CHUNK
                    cache_flat[block : block + 1, h0 : h0 + HEAD_CHUNK] = compressed_flat[
                        block : block + 1, h0 : h0 + HEAD_CHUNK
                    ]

    return compressed, kv_state_out, score_state_out, compressed_cache_out


@pl.jit
def compressor_ratio4_indexer_prefill_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    wkv_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    wgate_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, INDEX_PROJ_DIM], pl.FP32],
    norm_w: pl.Tensor[[INDEX_HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    block_count: pl.Tensor[[1], pl.INT32],
    compressed: pl.Out[pl.Tensor[[B, C_DYN, INDEX_HEAD_DIM], pl.BF16]],
    kv_state_out: pl.Out[pl.Tensor[[B, STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
    score_state_out: pl.Out[pl.Tensor[[B, STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
    compressed_cache_out: pl.Out[pl.Tensor[[B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM], pl.BF16]],
):
    compressed, kv_state_out, score_state_out, compressed_cache_out = compressor_ratio4_indexer_prefill_fwd(
        x,
        wkv_t,
        wgate_t,
        ape,
        norm_w,
        cos,
        sin,
        block_count,
        compressed,
        kv_state_out,
        score_state_out,
        compressed_cache_out,
    )
    return compressed, kv_state_out, score_state_out, compressed_cache_out


@pl.jit.inline
def compressor_ratio4_indexer_decode_fwd(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    kv_state: pl.Tensor[[B, STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    score_state: pl.Tensor[[B, STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    compressed_cache: pl.Tensor[[B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM], pl.BF16],
    slot: pl.Tensor[[1], pl.INT32],
    cache_slot: pl.Tensor[[1], pl.INT32],
    should_compress: pl.Tensor[[1], pl.INT32],
    wkv_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    wgate_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, INDEX_PROJ_DIM], pl.FP32],
    norm_w: pl.Tensor[[INDEX_HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    compressed: pl.Tensor[[B, 1, INDEX_HEAD_DIM], pl.BF16],
    kv_state_out: pl.Tensor[[B, STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    score_state_out: pl.Tensor[[B, STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    compressed_cache_out: pl.Tensor[[B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM], pl.BF16],
):
    """Run official Indexer ``Compressor.forward`` for ``compress_ratio == 4`` decode."""
    x.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    kv_proj = pl.create_tensor([B, tokens, INDEX_PROJ_DIM], dtype=pl.FP32)
    score_proj = pl.create_tensor([B, tokens, INDEX_PROJ_DIM], dtype=pl.FP32)
    pooled = pl.create_tensor([B, 1, INDEX_HEAD_DIM], dtype=pl.BF16)
    normed = pl.create_tensor([B, 1, INDEX_HEAD_DIM], dtype=pl.BF16)

    kv_proj = linear_4096_to_256_fp32(x, wkv_t, kv_proj)
    score_proj = linear_4096_to_256_fp32(x, wgate_t, score_proj)

    kv_proj_flat = pl.reshape(kv_proj, [tokens, INDEX_PROJ_DIM])
    score_proj_flat = pl.reshape(score_proj, [tokens, INDEX_PROJ_DIM])
    kv_state_flat = pl.reshape(kv_state, [STATE_ROWS, INDEX_PROJ_DIM])
    score_state_flat = pl.reshape(score_state, [STATE_ROWS, INDEX_PROJ_DIM])
    kv_state_out_flat = pl.reshape(kv_state_out, [STATE_ROWS, INDEX_PROJ_DIM])
    score_state_out_flat = pl.reshape(score_state_out, [STATE_ROWS, INDEX_PROJ_DIM])
    cache_flat = pl.reshape(compressed_cache, [TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM])
    cache_out_flat = pl.reshape(compressed_cache_out, [TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM])

    raw_slot = pl.read(slot, [0])
    slot_idx = pl.cast(raw_slot, pl.INDEX)
    current_idx = slot_idx + COMPRESS_RATIO
    should_flag = pl.read(should_compress, [0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="compressor_c4_index_decode_state_update"):
        for row in pl.range(STATE_ROWS):
            for hb in pl.range(INDEX_PROJ_CHUNKS):
                h0 = hb * HEAD_CHUNK
                if should_flag != 0:
                    if row < COMPRESS_RATIO:
                        src_row = row + COMPRESS_RATIO
                        if src_row == current_idx:
                            kv_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = kv_proj_flat[
                                0:1, h0 : h0 + HEAD_CHUNK
                            ]
                            score_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.add(
                                score_proj_flat[0:1, h0 : h0 + HEAD_CHUNK],
                                ape[slot_idx : slot_idx + 1, h0 : h0 + HEAD_CHUNK],
                            )
                        else:
                            kv_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = kv_state_flat[
                                src_row : src_row + 1, h0 : h0 + HEAD_CHUNK
                            ]
                            score_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = score_state_flat[
                                src_row : src_row + 1, h0 : h0 + HEAD_CHUNK
                            ]
                    else:
                        if row == current_idx:
                            kv_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = kv_proj_flat[
                                0:1, h0 : h0 + HEAD_CHUNK
                            ]
                            score_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.add(
                                score_proj_flat[0:1, h0 : h0 + HEAD_CHUNK],
                                ape[slot_idx : slot_idx + 1, h0 : h0 + HEAD_CHUNK],
                            )
                        else:
                            kv_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = kv_state_flat[
                                row : row + 1, h0 : h0 + HEAD_CHUNK
                            ]
                            score_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = score_state_flat[
                                row : row + 1, h0 : h0 + HEAD_CHUNK
                            ]
                else:
                    if row == current_idx:
                        kv_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = kv_proj_flat[
                            0:1, h0 : h0 + HEAD_CHUNK
                        ]
                        score_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.add(
                            score_proj_flat[0:1, h0 : h0 + HEAD_CHUNK],
                            ape[slot_idx : slot_idx + 1, h0 : h0 + HEAD_CHUNK],
                        )
                    else:
                        kv_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = kv_state_flat[
                            row : row + 1, h0 : h0 + HEAD_CHUNK
                        ]
                        score_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = score_state_flat[
                            row : row + 1, h0 : h0 + HEAD_CHUNK
                        ]

    raw_cache_slot = pl.read(cache_slot, [0])
    cache_slot_idx = pl.cast(raw_cache_slot, pl.INDEX)

    for cache_row in pl.spmd(TOPK_CSA_COMPRESSED, name_hint="compressor_c4_index_decode_cache_copy"):
        for hb in pl.range(INDEX_HEAD_CHUNKS):
            h0 = hb * HEAD_CHUNK
            if should_flag != 0:
                if cache_row != cache_slot_idx:
                    cache_out_flat[cache_row : cache_row + 1, h0 : h0 + HEAD_CHUNK] = cache_flat[
                        cache_row : cache_row + 1, h0 : h0 + HEAD_CHUNK
                    ]
            else:
                cache_out_flat[cache_row : cache_row + 1, h0 : h0 + HEAD_CHUNK] = cache_flat[
                    cache_row : cache_row + 1, h0 : h0 + HEAD_CHUNK
                ]

    if should_flag != 0:
        pooled_flat = pl.reshape(pooled, [1, INDEX_HEAD_DIM])
        for hb in pl.spmd(INDEX_HEAD_CHUNKS, name_hint="compressor_c4_index_decode_softmax_pool"):
            h0 = hb * HEAD_CHUNK
            kv_tile = pl.full([STATE_ROWS, HEAD_CHUNK], dtype=pl.FP32, value=0.0)
            score_tile = pl.full([STATE_ROWS, HEAD_CHUNK], dtype=pl.FP32, value=NEG_INF)

            for row in pl.range(COMPRESS_RATIO):
                kv_tile[row : row + 1, 0:HEAD_CHUNK] = kv_state_flat[
                    row : row + 1,
                    h0 : h0 + HEAD_CHUNK,
                ]
                score_tile[row : row + 1, 0:HEAD_CHUNK] = score_state_flat[
                    row : row + 1,
                    h0 : h0 + HEAD_CHUNK,
                ]

            for row in pl.range(COMPRESS_RATIO):
                src_row = COMPRESS_RATIO + row
                dst_row = COMPRESS_RATIO + row
                if src_row == current_idx:
                    kv_tile[dst_row : dst_row + 1, 0:HEAD_CHUNK] = kv_proj_flat[
                        0:1,
                        INDEX_HEAD_DIM + h0 : INDEX_HEAD_DIM + h0 + HEAD_CHUNK,
                    ]
                    score_tile[dst_row : dst_row + 1, 0:HEAD_CHUNK] = pl.add(
                        score_proj_flat[
                            0:1,
                            INDEX_HEAD_DIM + h0 : INDEX_HEAD_DIM + h0 + HEAD_CHUNK,
                        ],
                        ape[slot_idx : slot_idx + 1, INDEX_HEAD_DIM + h0 : INDEX_HEAD_DIM + h0 + HEAD_CHUNK],
                    )
                else:
                    kv_tile[dst_row : dst_row + 1, 0:HEAD_CHUNK] = kv_state_flat[
                        src_row : src_row + 1,
                        INDEX_HEAD_DIM + h0 : INDEX_HEAD_DIM + h0 + HEAD_CHUNK,
                    ]
                    score_tile[dst_row : dst_row + 1, 0:HEAD_CHUNK] = score_state_flat[
                        src_row : src_row + 1,
                        INDEX_HEAD_DIM + h0 : INDEX_HEAD_DIM + h0 + HEAD_CHUNK,
                    ]

            score_t = pl.transpose(score_tile, axis1=0, axis2=1)
            kv_t = pl.transpose(kv_tile, axis1=0, axis2=1)
            score_max = pl.row_max(score_t)
            score_exp = pl.exp(pl.row_expand_sub(score_t, score_max))
            score_sum = pl.row_sum(score_exp)
            score_prob = pl.row_expand_div(score_exp, score_sum)
            pooled_t = pl.row_sum(pl.mul(kv_t, score_prob))
            pooled_chunk = pl.reshape(pooled_t, [1, HEAD_CHUNK])
            pooled_flat[0:1, h0 : h0 + HEAD_CHUNK] = pl.cast(
                pooled_chunk, target_type=pl.BF16, mode="rint"
            )

        normed_flat = pl.reshape(normed, [1, INDEX_HEAD_DIM])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="compressor_c4_index_decode_rmsnorm"):
            partial_sq = pl.full([1, RMS_T_TILE], dtype=pl.FP32, value=0.0)
            for db in pl.range(INDEX_RMS_BLOCKS):
                d0 = db * RMS_D_TILE
                pooled_bf16 = pl.slice(pooled_flat, [RMS_T_TILE, RMS_D_TILE], [0, d0], valid_shape=[1, RMS_D_TILE])
                pooled_fp32 = pl.cast(pooled_bf16, target_type=pl.FP32)
                partial_sq = pl.add(
                    partial_sq,
                    pl.reshape(pl.row_sum(pl.mul(pooled_fp32, pooled_fp32)), [1, RMS_T_TILE]),
                )

            variance = pl.reshape(pl.add(pl.mul(partial_sq, INDEX_INV_HEAD_DIM), EPS), [RMS_T_TILE, 1])
            inv_rms = pl.recip(pl.sqrt(variance))

            for db in pl.range(INDEX_RMS_BLOCKS):
                d0 = db * RMS_D_TILE
                pooled_bf16 = pl.slice(pooled_flat, [RMS_T_TILE, RMS_D_TILE], [0, d0], valid_shape=[1, RMS_D_TILE])
                pooled_fp32 = pl.cast(pooled_bf16, target_type=pl.FP32)
                weight_fp32 = pl.cast(
                    pl.reshape(norm_w[d0 : d0 + RMS_D_TILE], [1, RMS_D_TILE]),
                    target_type=pl.FP32,
                )
                normed_tile = pl.col_expand_mul(pl.row_expand_mul(pooled_fp32, inv_rms), weight_fp32)
                normed_bf16 = pl.cast(normed_tile, target_type=pl.BF16, mode="rint")
                decode_rms_out_row = pl.slice(normed_bf16, [1, RMS_D_TILE], [0, 0], valid_shape=[1, RMS_D_TILE])
                normed_flat = pl.assemble(normed_flat, decode_rms_out_row, [0, d0])

        compressed_flat = pl.reshape(compressed, [1, INDEX_HEAD_DIM])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="compressor_c4_index_decode_rope"):
            for pb in pl.range(INDEX_TAIL_OFFSET // ROPE_PREFIX_TILE):
                p0 = pb * ROPE_PREFIX_TILE
                prefix_tile = pl.slice(normed_flat, [1, ROPE_PREFIX_TILE], [0, p0], valid_shape=[1, ROPE_PREFIX_TILE])
                compressed_flat = pl.assemble(compressed_flat, prefix_tile, [0, p0])

            ones = pl.full([ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
            col = pl.col_expand_mul(
                ones,
                pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32),
            )
            dup_f = pl.cast(pl.cast(pl.mul(col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
            dup_idx = pl.cast(dup_f, target_type=pl.INT32)
            lane = pl.sub(col, pl.mul(dup_f, 2.0))
            swap_idx = pl.cast(pl.sub(pl.add(col, 1.0), pl.mul(lane, 2.0)), target_type=pl.INT32)
            sign = pl.sub(pl.mul(lane, 2.0), 1.0)

            x_tile = pl.cast(
                pl.slice(normed_flat, [ROPE_T_TILE, ROPE_DIM], [0, INDEX_TAIL_OFFSET], valid_shape=[1, ROPE_DIM]),
                target_type=pl.FP32,
            )
            cos_tile = pl.slice(cos, [ROPE_T_TILE, ROPE_HALF], [0, 0], valid_shape=[1, ROPE_HALF])
            sin_tile = pl.slice(sin, [ROPE_T_TILE, ROPE_HALF], [0, 0], valid_shape=[1, ROPE_HALF])
            cos_il = pl.gather(cos_tile, dim=-1, index=dup_idx)
            sin_il = pl.gather(sin_tile, dim=-1, index=dup_idx)
            swapped = pl.gather(x_tile, dim=-1, index=swap_idx)
            rotated = pl.add(pl.mul(x_tile, cos_il), pl.mul(pl.mul(swapped, sign), sin_il))
            rotated_bf16 = pl.cast(rotated, target_type=pl.BF16, mode="rint")
            decode_rope_out_row = pl.slice(rotated_bf16, [1, ROPE_DIM], [0, 0], valid_shape=[1, ROPE_DIM])
            compressed_flat = pl.assemble(compressed_flat, decode_rope_out_row, [0, INDEX_TAIL_OFFSET])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="compressor_c4_index_decode_cache_write"):
            for hb in pl.range(INDEX_HEAD_CHUNKS):
                h0 = hb * HEAD_CHUNK
                cache_out_flat[cache_slot_idx : cache_slot_idx + 1, h0 : h0 + HEAD_CHUNK] = compressed_flat[
                    0:1, h0 : h0 + HEAD_CHUNK
                ]

    return compressed, kv_state_out, score_state_out, compressed_cache_out


@pl.jit
def compressor_ratio4_indexer_decode_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    kv_state: pl.Tensor[[B, STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    score_state: pl.Tensor[[B, STATE_ROWS, INDEX_PROJ_DIM], pl.FP32],
    compressed_cache: pl.Tensor[[B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM], pl.BF16],
    slot: pl.Tensor[[1], pl.INT32],
    cache_slot: pl.Tensor[[1], pl.INT32],
    should_compress: pl.Tensor[[1], pl.INT32],
    wkv_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    wgate_t: pl.Tensor[[HIDDEN, INDEX_PROJ_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, INDEX_PROJ_DIM], pl.FP32],
    norm_w: pl.Tensor[[INDEX_HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    compressed: pl.Out[pl.Tensor[[B, 1, INDEX_HEAD_DIM], pl.BF16]],
    kv_state_out: pl.Out[pl.Tensor[[B, STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
    score_state_out: pl.Out[pl.Tensor[[B, STATE_ROWS, INDEX_PROJ_DIM], pl.FP32]],
    compressed_cache_out: pl.Out[pl.Tensor[[B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM], pl.BF16]],
):
    compressed, kv_state_out, score_state_out, compressed_cache_out = compressor_ratio4_indexer_decode_fwd(
        x,
        kv_state,
        score_state,
        compressed_cache,
        slot,
        cache_slot,
        should_compress,
        wkv_t,
        wgate_t,
        ape,
        norm_w,
        cos,
        sin,
        compressed,
        kv_state_out,
        score_state_out,
        compressed_cache_out,
    )
    return compressed, kv_state_out, score_state_out, compressed_cache_out


@pl.jit.inline
def compressor_ratio4_attention_prefill_fwd(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    wkv_t: pl.Tensor[[HIDDEN, ATTN_PROJ_DIM], pl.BF16],
    wgate_t: pl.Tensor[[HIDDEN, ATTN_PROJ_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, ATTN_PROJ_DIM], pl.FP32],
    norm_w: pl.Tensor[[ATTN_HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    block_count: pl.Tensor[[1], pl.INT32],
    compressed: pl.Tensor[[B, C_DYN, ATTN_HEAD_DIM], pl.BF16],
    kv_state_out: pl.Tensor[[B, STATE_ROWS, ATTN_PROJ_DIM], pl.FP32],
    score_state_out: pl.Tensor[[B, STATE_ROWS, ATTN_PROJ_DIM], pl.FP32],
    compressed_cache_out: pl.Tensor[[B, TOPK_CSA_COMPRESSED, ATTN_HEAD_DIM], pl.BF16],
):
    """Run official Attention ``Compressor.forward`` for ``compress_ratio == 4`` prefill."""
    x.bind_dynamic(1, S_DYN)
    cos.bind_dynamic(0, C_DYN)
    sin.bind_dynamic(0, C_DYN)
    compressed.bind_dynamic(1, C_DYN)

    tokens = pl.tensor.dim(x, 1)
    padded_blocks = pl.tensor.dim(cos, 0)
    blocks = pl.read(block_count, [0])
    should_compress = blocks > 0
    cutoff = blocks * COMPRESS_RATIO
    remainder = tokens - cutoff

    kv_proj = pl.create_tensor([B, tokens, ATTN_PROJ_DIM], dtype=pl.FP32)
    score_proj = pl.create_tensor([B, tokens, ATTN_PROJ_DIM], dtype=pl.FP32)
    pooled = pl.create_tensor([B, padded_blocks, ATTN_HEAD_DIM], dtype=pl.BF16)
    normed = pl.create_tensor([B, padded_blocks, ATTN_HEAD_DIM], dtype=pl.BF16)

    kv_proj = linear_4096_to_1024_fp32(x, wkv_t, kv_proj)
    score_proj = linear_4096_to_1024_fp32(x, wgate_t, score_proj)

    kv_proj_flat = pl.reshape(kv_proj, [tokens, ATTN_PROJ_DIM])
    score_proj_flat = pl.reshape(score_proj, [tokens, ATTN_PROJ_DIM])
    pooled_flat = pl.reshape(pooled, [padded_blocks, ATTN_HEAD_DIM])
    kv_state_flat = pl.reshape(kv_state_out, [STATE_ROWS, ATTN_PROJ_DIM])
    score_state_flat = pl.reshape(score_state_out, [STATE_ROWS, ATTN_PROJ_DIM])
    cache_flat = pl.reshape(compressed_cache_out, [TOPK_CSA_COMPRESSED, ATTN_HEAD_DIM])

    for row in pl.spmd(STATE_ROWS, name_hint="compressor_c4_attn_state_write"):
        for hb in pl.pipeline(ATTN_PROJ_CHUNKS, stage=2):
            h0 = hb * HEAD_CHUNK
            if should_compress:
                if row < COMPRESS_RATIO:
                    src = cutoff - COMPRESS_RATIO + row
                    kv_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = kv_proj_flat[
                        src : src + 1, h0 : h0 + HEAD_CHUNK
                    ]
                    score_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.add(
                        score_proj_flat[src : src + 1, h0 : h0 + HEAD_CHUNK],
                        ape[row : row + 1, h0 : h0 + HEAD_CHUNK],
                    )
                elif row < COMPRESS_RATIO + remainder:
                    rem = row - COMPRESS_RATIO
                    src = cutoff + rem
                    kv_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = kv_proj_flat[
                        src : src + 1, h0 : h0 + HEAD_CHUNK
                    ]
                    score_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.add(
                        score_proj_flat[src : src + 1, h0 : h0 + HEAD_CHUNK],
                        ape[rem : rem + 1, h0 : h0 + HEAD_CHUNK],
                    )
                else:
                    kv_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.full(
                        [1, HEAD_CHUNK], dtype=pl.FP32, value=0.0
                    )
                    score_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.full(
                        [1, HEAD_CHUNK], dtype=pl.FP32, value=NEG_INF
                    )
            elif row >= COMPRESS_RATIO:
                rem = row - COMPRESS_RATIO
                if rem < remainder:
                    src = cutoff + rem
                    kv_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = kv_proj_flat[
                        src : src + 1, h0 : h0 + HEAD_CHUNK
                    ]
                    score_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.add(
                        score_proj_flat[src : src + 1, h0 : h0 + HEAD_CHUNK],
                        ape[rem : rem + 1, h0 : h0 + HEAD_CHUNK],
                    )
                else:
                    kv_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.full(
                        [1, HEAD_CHUNK], dtype=pl.FP32, value=0.0
                    )
                    score_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.full(
                        [1, HEAD_CHUNK], dtype=pl.FP32, value=NEG_INF
                    )
            else:
                kv_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.full(
                    [1, HEAD_CHUNK], dtype=pl.FP32, value=0.0
                )
                score_state_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.full(
                    [1, HEAD_CHUNK], dtype=pl.FP32, value=NEG_INF
                )

    for cache_row in pl.spmd(TOPK_CSA_COMPRESSED, name_hint="compressor_c4_attn_cache_init"):
        for hb in pl.pipeline(ATTN_HEAD_CHUNKS, stage=2):
            h0 = hb * HEAD_CHUNK
            cache_flat[cache_row : cache_row + 1, h0 : h0 + HEAD_CHUNK] = pl.full(
                [1, HEAD_CHUNK], dtype=pl.BF16, value=0.0
            )

    if should_compress:
        for block in pl.range(blocks):
            t0 = block * COMPRESS_RATIO
            for hb in pl.spmd(ATTN_HEAD_CHUNKS, name_hint="compressor_c4_attn_softmax_pool"):
                h0 = hb * HEAD_CHUNK
                kv_tile = pl.full([STATE_ROWS, HEAD_CHUNK], dtype=pl.FP32, value=0.0)
                score_tile = pl.full([STATE_ROWS, HEAD_CHUNK], dtype=pl.FP32, value=NEG_INF)

                if block > 0:
                    prev_t0 = t0 - COMPRESS_RATIO
                    for r in pl.range(COMPRESS_RATIO):
                        kv_tile[r : r + 1, 0:HEAD_CHUNK] = kv_proj_flat[
                            prev_t0 + r : prev_t0 + r + 1,
                            h0 : h0 + HEAD_CHUNK,
                        ]
                        score_tile[r : r + 1, 0:HEAD_CHUNK] = pl.add(
                            score_proj_flat[
                                prev_t0 + r : prev_t0 + r + 1,
                                h0 : h0 + HEAD_CHUNK,
                            ],
                            ape[r : r + 1, h0 : h0 + HEAD_CHUNK],
                        )

                for r in pl.range(COMPRESS_RATIO):
                    dst = COMPRESS_RATIO + r
                    kv_tile[dst : dst + 1, 0:HEAD_CHUNK] = kv_proj_flat[
                        t0 + r : t0 + r + 1,
                        ATTN_HEAD_DIM + h0 : ATTN_HEAD_DIM + h0 + HEAD_CHUNK,
                    ]
                    score_tile[dst : dst + 1, 0:HEAD_CHUNK] = pl.add(
                        score_proj_flat[
                            t0 + r : t0 + r + 1,
                            ATTN_HEAD_DIM + h0 : ATTN_HEAD_DIM + h0 + HEAD_CHUNK,
                        ],
                        ape[r : r + 1, ATTN_HEAD_DIM + h0 : ATTN_HEAD_DIM + h0 + HEAD_CHUNK],
                    )

                score_t = pl.transpose(score_tile, axis1=0, axis2=1)
                kv_t = pl.transpose(kv_tile, axis1=0, axis2=1)
                score_max = pl.row_max(score_t)
                score_exp = pl.exp(pl.row_expand_sub(score_t, score_max))
                score_sum = pl.row_sum(score_exp)
                score_prob = pl.row_expand_div(score_exp, score_sum)
                pooled_t = pl.row_sum(pl.mul(kv_t, score_prob))
                pooled_chunk = pl.reshape(pooled_t, [1, HEAD_CHUNK])
                pooled_flat[block : block + 1, h0 : h0 + HEAD_CHUNK] = pl.cast(
                    pooled_chunk, target_type=pl.BF16, mode="rint"
                )

        normed_flat = pl.reshape(normed, [padded_blocks, ATTN_HEAD_DIM])
        rms_row_blocks = (blocks + RMS_T_TILE - 1) // RMS_T_TILE
        for rb in pl.range(rms_row_blocks):
            r0 = rb * RMS_T_TILE
            valid_rows = pl.min(RMS_T_TILE, blocks - r0)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="compressor_c4_attn_inline_rmsnorm"):
                partial_sq = pl.full([1, RMS_T_TILE], dtype=pl.FP32, value=0.0)
                for db in pl.range(ATTN_RMS_BLOCKS):
                    d0 = db * RMS_D_TILE
                    pooled_bf16 = pl.slice(
                        pooled_flat,
                        [RMS_T_TILE, RMS_D_TILE],
                        [r0, d0],
                        valid_shape=[valid_rows, RMS_D_TILE],
                    )
                    pooled_fp32 = pl.cast(pooled_bf16, target_type=pl.FP32)
                    partial_sq = pl.add(
                        partial_sq,
                        pl.reshape(pl.row_sum(pl.mul(pooled_fp32, pooled_fp32)), [1, RMS_T_TILE]),
                    )

                variance = pl.reshape(pl.add(pl.mul(partial_sq, ATTN_INV_HEAD_DIM), EPS), [RMS_T_TILE, 1])
                inv_rms = pl.recip(pl.sqrt(variance))

                for db in pl.range(ATTN_RMS_BLOCKS):
                    d0 = db * RMS_D_TILE
                    pooled_bf16 = pl.slice(
                        pooled_flat,
                        [RMS_T_TILE, RMS_D_TILE],
                        [r0, d0],
                        valid_shape=[valid_rows, RMS_D_TILE],
                    )
                    pooled_fp32 = pl.cast(pooled_bf16, target_type=pl.FP32)
                    weight_fp32 = pl.cast(
                        pl.reshape(norm_w[d0 : d0 + RMS_D_TILE], [1, RMS_D_TILE]),
                        target_type=pl.FP32,
                    )
                    normed_tile = pl.col_expand_mul(pl.row_expand_mul(pooled_fp32, inv_rms), weight_fp32)
                    normed_bf16 = pl.cast(normed_tile, target_type=pl.BF16, mode="rint")
                    for row in pl.range(valid_rows):
                        inline_rms_out_row = pl.slice(
                            normed_bf16,
                            [1, RMS_D_TILE],
                            [row, 0],
                            valid_shape=[1, RMS_D_TILE],
                        )
                        normed_flat = pl.assemble(normed_flat, inline_rms_out_row, [r0 + row, d0])

        compressed_flat = pl.reshape(compressed, [padded_blocks, ATTN_HEAD_DIM])
        rope_row_blocks = (blocks + ROPE_T_TILE - 1) // ROPE_T_TILE
        for rb in pl.range(rope_row_blocks):
            r0 = rb * ROPE_T_TILE
            valid_rows = pl.min(ROPE_T_TILE, blocks - r0)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="compressor_c4_attn_inline_rope"):
                for pb in pl.range(ATTN_TAIL_OFFSET // ROPE_PREFIX_TILE):
                    p0 = pb * ROPE_PREFIX_TILE
                    prefix_tile = pl.slice(
                        normed_flat,
                        [ROPE_T_TILE, ROPE_PREFIX_TILE],
                        [r0, p0],
                        valid_shape=[valid_rows, ROPE_PREFIX_TILE],
                    )
                    for row in pl.range(valid_rows):
                        prefix_row = pl.slice(
                            prefix_tile,
                            [1, ROPE_PREFIX_TILE],
                            [row, 0],
                            valid_shape=[1, ROPE_PREFIX_TILE],
                        )
                        compressed_flat = pl.assemble(compressed_flat, prefix_row, [r0 + row, p0])

                ones = pl.full([ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
                col = pl.col_expand_mul(
                    ones,
                    pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32),
                )
                dup_f = pl.cast(pl.cast(pl.mul(col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
                dup_idx = pl.cast(dup_f, target_type=pl.INT32)
                lane = pl.sub(col, pl.mul(dup_f, 2.0))
                swap_idx = pl.cast(pl.sub(pl.add(col, 1.0), pl.mul(lane, 2.0)), target_type=pl.INT32)
                sign = pl.sub(pl.mul(lane, 2.0), 1.0)

                x_tile = pl.cast(
                    pl.slice(
                        normed_flat,
                        [ROPE_T_TILE, ROPE_DIM],
                        [r0, ATTN_TAIL_OFFSET],
                        valid_shape=[valid_rows, ROPE_DIM],
                    ),
                    target_type=pl.FP32,
                )
                cos_tile = pl.slice(cos, [ROPE_T_TILE, ROPE_HALF], [r0, 0], valid_shape=[valid_rows, ROPE_HALF])
                sin_tile = pl.slice(sin, [ROPE_T_TILE, ROPE_HALF], [r0, 0], valid_shape=[valid_rows, ROPE_HALF])
                cos_il = pl.gather(cos_tile, dim=-1, index=dup_idx)
                sin_il = pl.gather(sin_tile, dim=-1, index=dup_idx)
                swapped = pl.gather(x_tile, dim=-1, index=swap_idx)
                rotated = pl.add(pl.mul(x_tile, cos_il), pl.mul(pl.mul(swapped, sign), sin_il))
                rotated_bf16 = pl.cast(rotated, target_type=pl.BF16, mode="rint")

                for row in pl.range(valid_rows):
                    inline_rope_out_row = pl.slice(
                        rotated_bf16,
                        [1, ROPE_DIM],
                        [row, 0],
                        valid_shape=[1, ROPE_DIM],
                    )
                    compressed_flat = pl.assemble(compressed_flat, inline_rope_out_row, [r0 + row, ATTN_TAIL_OFFSET])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="compressor_c4_attn_cache_write"):
            for block in pl.range(blocks):
                for hb in pl.range(ATTN_HEAD_CHUNKS):
                    h0 = hb * HEAD_CHUNK
                    cache_flat[block : block + 1, h0 : h0 + HEAD_CHUNK] = compressed_flat[
                        block : block + 1, h0 : h0 + HEAD_CHUNK
                    ]

    return compressed, kv_state_out, score_state_out, compressed_cache_out


@pl.jit
def compressor_ratio4_attention_prefill_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    wkv_t: pl.Tensor[[HIDDEN, ATTN_PROJ_DIM], pl.BF16],
    wgate_t: pl.Tensor[[HIDDEN, ATTN_PROJ_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, ATTN_PROJ_DIM], pl.FP32],
    norm_w: pl.Tensor[[ATTN_HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    block_count: pl.Tensor[[1], pl.INT32],
    compressed: pl.Out[pl.Tensor[[B, C_DYN, ATTN_HEAD_DIM], pl.BF16]],
    kv_state_out: pl.Out[pl.Tensor[[B, STATE_ROWS, ATTN_PROJ_DIM], pl.FP32]],
    score_state_out: pl.Out[pl.Tensor[[B, STATE_ROWS, ATTN_PROJ_DIM], pl.FP32]],
    compressed_cache_out: pl.Out[pl.Tensor[[B, TOPK_CSA_COMPRESSED, ATTN_HEAD_DIM], pl.BF16]],
):
    compressed, kv_state_out, score_state_out, compressed_cache_out = compressor_ratio4_attention_prefill_fwd(
        x,
        wkv_t,
        wgate_t,
        ape,
        norm_w,
        cos,
        sin,
        block_count,
        compressed,
        kv_state_out,
        score_state_out,
        compressed_cache_out,
    )
    return compressed, kv_state_out, score_state_out, compressed_cache_out


@pl.jit.inline
def compressor_ratio4_attention_decode_fwd(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    kv_state: pl.Tensor[[B, STATE_ROWS, ATTN_PROJ_DIM], pl.FP32],
    score_state: pl.Tensor[[B, STATE_ROWS, ATTN_PROJ_DIM], pl.FP32],
    compressed_cache: pl.Tensor[[B, TOPK_CSA_COMPRESSED, ATTN_HEAD_DIM], pl.BF16],
    slot: pl.Tensor[[1], pl.INT32],
    cache_slot: pl.Tensor[[1], pl.INT32],
    should_compress: pl.Tensor[[1], pl.INT32],
    wkv_t: pl.Tensor[[HIDDEN, ATTN_PROJ_DIM], pl.BF16],
    wgate_t: pl.Tensor[[HIDDEN, ATTN_PROJ_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, ATTN_PROJ_DIM], pl.FP32],
    norm_w: pl.Tensor[[ATTN_HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    compressed: pl.Tensor[[B, 1, ATTN_HEAD_DIM], pl.BF16],
    kv_state_out: pl.Tensor[[B, STATE_ROWS, ATTN_PROJ_DIM], pl.FP32],
    score_state_out: pl.Tensor[[B, STATE_ROWS, ATTN_PROJ_DIM], pl.FP32],
    compressed_cache_out: pl.Tensor[[B, TOPK_CSA_COMPRESSED, ATTN_HEAD_DIM], pl.BF16],
):
    """Run official Attention ``Compressor.forward`` for ``compress_ratio == 4`` decode."""
    x.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    kv_proj = pl.create_tensor([B, tokens, ATTN_PROJ_DIM], dtype=pl.FP32)
    score_proj = pl.create_tensor([B, tokens, ATTN_PROJ_DIM], dtype=pl.FP32)
    pooled = pl.create_tensor([B, 1, ATTN_HEAD_DIM], dtype=pl.BF16)
    normed = pl.create_tensor([B, 1, ATTN_HEAD_DIM], dtype=pl.BF16)

    kv_proj = linear_4096_to_1024_fp32(x, wkv_t, kv_proj)
    score_proj = linear_4096_to_1024_fp32(x, wgate_t, score_proj)

    kv_proj_flat = pl.reshape(kv_proj, [tokens, ATTN_PROJ_DIM])
    score_proj_flat = pl.reshape(score_proj, [tokens, ATTN_PROJ_DIM])
    kv_state_flat = pl.reshape(kv_state, [STATE_ROWS, ATTN_PROJ_DIM])
    score_state_flat = pl.reshape(score_state, [STATE_ROWS, ATTN_PROJ_DIM])
    kv_state_out_flat = pl.reshape(kv_state_out, [STATE_ROWS, ATTN_PROJ_DIM])
    score_state_out_flat = pl.reshape(score_state_out, [STATE_ROWS, ATTN_PROJ_DIM])
    cache_flat = pl.reshape(compressed_cache, [TOPK_CSA_COMPRESSED, ATTN_HEAD_DIM])
    cache_out_flat = pl.reshape(compressed_cache_out, [TOPK_CSA_COMPRESSED, ATTN_HEAD_DIM])

    raw_slot = pl.read(slot, [0])
    slot_idx = pl.cast(raw_slot, pl.INDEX)
    current_idx = slot_idx + COMPRESS_RATIO
    should_flag = pl.read(should_compress, [0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="compressor_c4_attn_decode_state_update"):
        for row in pl.range(STATE_ROWS):
            for hb in pl.range(ATTN_PROJ_CHUNKS):
                h0 = hb * HEAD_CHUNK
                if should_flag != 0:
                    if row < COMPRESS_RATIO:
                        src_row = row + COMPRESS_RATIO
                        if src_row == current_idx:
                            kv_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = kv_proj_flat[
                                0:1, h0 : h0 + HEAD_CHUNK
                            ]
                            score_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.add(
                                score_proj_flat[0:1, h0 : h0 + HEAD_CHUNK],
                                ape[slot_idx : slot_idx + 1, h0 : h0 + HEAD_CHUNK],
                            )
                        else:
                            kv_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = kv_state_flat[
                                src_row : src_row + 1, h0 : h0 + HEAD_CHUNK
                            ]
                            score_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = score_state_flat[
                                src_row : src_row + 1, h0 : h0 + HEAD_CHUNK
                            ]
                    else:
                        if row == current_idx:
                            kv_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = kv_proj_flat[
                                0:1, h0 : h0 + HEAD_CHUNK
                            ]
                            score_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.add(
                                score_proj_flat[0:1, h0 : h0 + HEAD_CHUNK],
                                ape[slot_idx : slot_idx + 1, h0 : h0 + HEAD_CHUNK],
                            )
                        else:
                            kv_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = kv_state_flat[
                                row : row + 1, h0 : h0 + HEAD_CHUNK
                            ]
                            score_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = score_state_flat[
                                row : row + 1, h0 : h0 + HEAD_CHUNK
                            ]
                else:
                    if row == current_idx:
                        kv_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = kv_proj_flat[
                            0:1, h0 : h0 + HEAD_CHUNK
                        ]
                        score_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = pl.add(
                            score_proj_flat[0:1, h0 : h0 + HEAD_CHUNK],
                            ape[slot_idx : slot_idx + 1, h0 : h0 + HEAD_CHUNK],
                        )
                    else:
                        kv_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = kv_state_flat[
                            row : row + 1, h0 : h0 + HEAD_CHUNK
                        ]
                        score_state_out_flat[row : row + 1, h0 : h0 + HEAD_CHUNK] = score_state_flat[
                            row : row + 1, h0 : h0 + HEAD_CHUNK
                        ]

    raw_cache_slot = pl.read(cache_slot, [0])
    cache_slot_idx = pl.cast(raw_cache_slot, pl.INDEX)

    for cache_row in pl.spmd(TOPK_CSA_COMPRESSED, name_hint="compressor_c4_attn_decode_cache_copy"):
        for hb in pl.range(ATTN_HEAD_CHUNKS):
            h0 = hb * HEAD_CHUNK
            if should_flag != 0:
                if cache_row != cache_slot_idx:
                    cache_out_flat[cache_row : cache_row + 1, h0 : h0 + HEAD_CHUNK] = cache_flat[
                        cache_row : cache_row + 1, h0 : h0 + HEAD_CHUNK
                    ]
            else:
                cache_out_flat[cache_row : cache_row + 1, h0 : h0 + HEAD_CHUNK] = cache_flat[
                    cache_row : cache_row + 1, h0 : h0 + HEAD_CHUNK
                ]

    if should_flag != 0:
        pooled_flat = pl.reshape(pooled, [1, ATTN_HEAD_DIM])
        for hb in pl.spmd(ATTN_HEAD_CHUNKS, name_hint="compressor_c4_attn_decode_softmax_pool"):
            h0 = hb * HEAD_CHUNK
            kv_tile = pl.full([STATE_ROWS, HEAD_CHUNK], dtype=pl.FP32, value=0.0)
            score_tile = pl.full([STATE_ROWS, HEAD_CHUNK], dtype=pl.FP32, value=NEG_INF)

            for row in pl.range(COMPRESS_RATIO):
                kv_tile[row : row + 1, 0:HEAD_CHUNK] = kv_state_flat[
                    row : row + 1,
                    h0 : h0 + HEAD_CHUNK,
                ]
                score_tile[row : row + 1, 0:HEAD_CHUNK] = score_state_flat[
                    row : row + 1,
                    h0 : h0 + HEAD_CHUNK,
                ]

            for row in pl.range(COMPRESS_RATIO):
                src_row = COMPRESS_RATIO + row
                dst_row = COMPRESS_RATIO + row
                if src_row == current_idx:
                    kv_tile[dst_row : dst_row + 1, 0:HEAD_CHUNK] = kv_proj_flat[
                        0:1,
                        ATTN_HEAD_DIM + h0 : ATTN_HEAD_DIM + h0 + HEAD_CHUNK,
                    ]
                    score_tile[dst_row : dst_row + 1, 0:HEAD_CHUNK] = pl.add(
                        score_proj_flat[
                            0:1,
                            ATTN_HEAD_DIM + h0 : ATTN_HEAD_DIM + h0 + HEAD_CHUNK,
                        ],
                        ape[slot_idx : slot_idx + 1, ATTN_HEAD_DIM + h0 : ATTN_HEAD_DIM + h0 + HEAD_CHUNK],
                    )
                else:
                    kv_tile[dst_row : dst_row + 1, 0:HEAD_CHUNK] = kv_state_flat[
                        src_row : src_row + 1,
                        ATTN_HEAD_DIM + h0 : ATTN_HEAD_DIM + h0 + HEAD_CHUNK,
                    ]
                    score_tile[dst_row : dst_row + 1, 0:HEAD_CHUNK] = score_state_flat[
                        src_row : src_row + 1,
                        ATTN_HEAD_DIM + h0 : ATTN_HEAD_DIM + h0 + HEAD_CHUNK,
                    ]

            score_t = pl.transpose(score_tile, axis1=0, axis2=1)
            kv_t = pl.transpose(kv_tile, axis1=0, axis2=1)
            score_max = pl.row_max(score_t)
            score_exp = pl.exp(pl.row_expand_sub(score_t, score_max))
            score_sum = pl.row_sum(score_exp)
            score_prob = pl.row_expand_div(score_exp, score_sum)
            pooled_t = pl.row_sum(pl.mul(kv_t, score_prob))
            pooled_chunk = pl.reshape(pooled_t, [1, HEAD_CHUNK])
            pooled_flat[0:1, h0 : h0 + HEAD_CHUNK] = pl.cast(
                pooled_chunk, target_type=pl.BF16, mode="rint"
            )

        normed_flat = pl.reshape(normed, [1, ATTN_HEAD_DIM])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="compressor_c4_attn_decode_rmsnorm"):
            partial_sq = pl.full([1, RMS_T_TILE], dtype=pl.FP32, value=0.0)
            for db in pl.range(ATTN_RMS_BLOCKS):
                d0 = db * RMS_D_TILE
                pooled_bf16 = pl.slice(pooled_flat, [RMS_T_TILE, RMS_D_TILE], [0, d0], valid_shape=[1, RMS_D_TILE])
                pooled_fp32 = pl.cast(pooled_bf16, target_type=pl.FP32)
                partial_sq = pl.add(
                    partial_sq,
                    pl.reshape(pl.row_sum(pl.mul(pooled_fp32, pooled_fp32)), [1, RMS_T_TILE]),
                )

            variance = pl.reshape(pl.add(pl.mul(partial_sq, ATTN_INV_HEAD_DIM), EPS), [RMS_T_TILE, 1])
            inv_rms = pl.recip(pl.sqrt(variance))

            for db in pl.range(ATTN_RMS_BLOCKS):
                d0 = db * RMS_D_TILE
                pooled_bf16 = pl.slice(pooled_flat, [RMS_T_TILE, RMS_D_TILE], [0, d0], valid_shape=[1, RMS_D_TILE])
                pooled_fp32 = pl.cast(pooled_bf16, target_type=pl.FP32)
                weight_fp32 = pl.cast(
                    pl.reshape(norm_w[d0 : d0 + RMS_D_TILE], [1, RMS_D_TILE]),
                    target_type=pl.FP32,
                )
                normed_tile = pl.col_expand_mul(pl.row_expand_mul(pooled_fp32, inv_rms), weight_fp32)
                normed_bf16 = pl.cast(normed_tile, target_type=pl.BF16, mode="rint")
                decode_rms_out_row = pl.slice(normed_bf16, [1, RMS_D_TILE], [0, 0], valid_shape=[1, RMS_D_TILE])
                normed_flat = pl.assemble(normed_flat, decode_rms_out_row, [0, d0])

        compressed_flat = pl.reshape(compressed, [1, ATTN_HEAD_DIM])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="compressor_c4_attn_decode_rope"):
            for pb in pl.range(ATTN_TAIL_OFFSET // ROPE_PREFIX_TILE):
                p0 = pb * ROPE_PREFIX_TILE
                prefix_tile = pl.slice(normed_flat, [1, ROPE_PREFIX_TILE], [0, p0], valid_shape=[1, ROPE_PREFIX_TILE])
                compressed_flat = pl.assemble(compressed_flat, prefix_tile, [0, p0])

            ones = pl.full([ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
            col = pl.col_expand_mul(
                ones,
                pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32),
            )
            dup_f = pl.cast(pl.cast(pl.mul(col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
            dup_idx = pl.cast(dup_f, target_type=pl.INT32)
            lane = pl.sub(col, pl.mul(dup_f, 2.0))
            swap_idx = pl.cast(pl.sub(pl.add(col, 1.0), pl.mul(lane, 2.0)), target_type=pl.INT32)
            sign = pl.sub(pl.mul(lane, 2.0), 1.0)

            x_tile = pl.cast(
                pl.slice(normed_flat, [ROPE_T_TILE, ROPE_DIM], [0, ATTN_TAIL_OFFSET], valid_shape=[1, ROPE_DIM]),
                target_type=pl.FP32,
            )
            cos_tile = pl.slice(cos, [ROPE_T_TILE, ROPE_HALF], [0, 0], valid_shape=[1, ROPE_HALF])
            sin_tile = pl.slice(sin, [ROPE_T_TILE, ROPE_HALF], [0, 0], valid_shape=[1, ROPE_HALF])
            cos_il = pl.gather(cos_tile, dim=-1, index=dup_idx)
            sin_il = pl.gather(sin_tile, dim=-1, index=dup_idx)
            swapped = pl.gather(x_tile, dim=-1, index=swap_idx)
            rotated = pl.add(pl.mul(x_tile, cos_il), pl.mul(pl.mul(swapped, sign), sin_il))
            rotated_bf16 = pl.cast(rotated, target_type=pl.BF16, mode="rint")
            decode_rope_out_row = pl.slice(rotated_bf16, [1, ROPE_DIM], [0, 0], valid_shape=[1, ROPE_DIM])
            compressed_flat = pl.assemble(compressed_flat, decode_rope_out_row, [0, ATTN_TAIL_OFFSET])

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="compressor_c4_attn_decode_cache_write"):
            for hb in pl.range(ATTN_HEAD_CHUNKS):
                h0 = hb * HEAD_CHUNK
                cache_out_flat[cache_slot_idx : cache_slot_idx + 1, h0 : h0 + HEAD_CHUNK] = compressed_flat[
                    0:1, h0 : h0 + HEAD_CHUNK
                ]

    return compressed, kv_state_out, score_state_out, compressed_cache_out


@pl.jit
def compressor_ratio4_attention_decode_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    kv_state: pl.Tensor[[B, STATE_ROWS, ATTN_PROJ_DIM], pl.FP32],
    score_state: pl.Tensor[[B, STATE_ROWS, ATTN_PROJ_DIM], pl.FP32],
    compressed_cache: pl.Tensor[[B, TOPK_CSA_COMPRESSED, ATTN_HEAD_DIM], pl.BF16],
    slot: pl.Tensor[[1], pl.INT32],
    cache_slot: pl.Tensor[[1], pl.INT32],
    should_compress: pl.Tensor[[1], pl.INT32],
    wkv_t: pl.Tensor[[HIDDEN, ATTN_PROJ_DIM], pl.BF16],
    wgate_t: pl.Tensor[[HIDDEN, ATTN_PROJ_DIM], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, ATTN_PROJ_DIM], pl.FP32],
    norm_w: pl.Tensor[[ATTN_HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    compressed: pl.Out[pl.Tensor[[B, 1, ATTN_HEAD_DIM], pl.BF16]],
    kv_state_out: pl.Out[pl.Tensor[[B, STATE_ROWS, ATTN_PROJ_DIM], pl.FP32]],
    score_state_out: pl.Out[pl.Tensor[[B, STATE_ROWS, ATTN_PROJ_DIM], pl.FP32]],
    compressed_cache_out: pl.Out[pl.Tensor[[B, TOPK_CSA_COMPRESSED, ATTN_HEAD_DIM], pl.BF16]],
):
    compressed, kv_state_out, score_state_out, compressed_cache_out = compressor_ratio4_attention_decode_fwd(
        x,
        kv_state,
        score_state,
        compressed_cache,
        slot,
        cache_slot,
        should_compress,
        wkv_t,
        wgate_t,
        ape,
        norm_w,
        cos,
        sin,
        compressed,
        kv_state_out,
        score_state_out,
        compressed_cache_out,
    )
    return compressed, kv_state_out, score_state_out, compressed_cache_out


def golden_compressor_ratio4_forward(tensors, start_pos: int, *, head_dim: int, proj_dim: int, module_name: str):
    import torch

    x = tensors["x"]
    bsz, seq_len, _ = x.shape
    if bsz != B:
        raise ValueError(f"ratio4 {module_name} compressor golden expects batch={B}, got {bsz}")

    kv_proj = torch.matmul(x.float(), tensors["wkv_t"].float())
    score_proj = torch.matmul(x.float(), tensors["wgate_t"].float())

    if start_pos == 0:
        blocks = seq_len // COMPRESS_RATIO
        if "block_count" in tensors:
            actual_blocks = int(tensors["block_count"][0].item())
            if actual_blocks != blocks:
                raise ValueError(f"prefill block_count mismatch: expected {blocks}, got {actual_blocks}")

        should_compress = seq_len >= COMPRESS_RATIO
        remainder = seq_len % COMPRESS_RATIO
        cutoff = seq_len - remainder
        cache_slot = 0

        kv_state = torch.zeros(B, STATE_ROWS, proj_dim, dtype=torch.float32)
        score_state = torch.full((B, STATE_ROWS, proj_dim), NEG_INF, dtype=torch.float32)
        compressed_cache = torch.zeros_like(tensors["compressed_cache_out"])

        if should_compress:
            kv_state[:, :COMPRESS_RATIO] = kv_proj[:, cutoff - COMPRESS_RATIO : cutoff]
            score_state[:, :COMPRESS_RATIO] = (
                score_proj[:, cutoff - COMPRESS_RATIO : cutoff]
                + tensors["ape"].view(1, COMPRESS_RATIO, proj_dim)
            )

        if remainder > 0:
            kv_state[:, COMPRESS_RATIO : COMPRESS_RATIO + remainder] = kv_proj[:, cutoff:]
            score_state[:, COMPRESS_RATIO : COMPRESS_RATIO + remainder] = (
                score_proj[:, cutoff:] + tensors["ape"][:remainder].view(1, remainder, proj_dim)
            )

        if should_compress:
            kv_blocks = kv_proj[:, :cutoff].unflatten(1, (-1, COMPRESS_RATIO))
            score_blocks = score_proj[:, :cutoff].unflatten(1, (-1, COMPRESS_RATIO))
            score_blocks = score_blocks + tensors["ape"].view(1, 1, COMPRESS_RATIO, proj_dim)

            kv_for_pool = torch.zeros(B, blocks, STATE_ROWS, head_dim, dtype=torch.float32)
            score_for_pool = torch.full((B, blocks, STATE_ROWS, head_dim), NEG_INF, dtype=torch.float32)
            kv_for_pool[:, :, COMPRESS_RATIO:] = kv_blocks[:, :, :, head_dim:]
            score_for_pool[:, :, COMPRESS_RATIO:] = score_blocks[:, :, :, head_dim:]
            if blocks > 1:
                kv_for_pool[:, 1:, :COMPRESS_RATIO] = kv_blocks[:, :-1, :, :head_dim]
                score_for_pool[:, 1:, :COMPRESS_RATIO] = score_blocks[:, :-1, :, :head_dim]
            kv_for_norm = (kv_for_pool * score_for_pool.softmax(dim=2)).sum(dim=2)
        else:
            kv_for_norm = None
    else:
        if seq_len != 1:
            raise ValueError(f"decode expects seq_len=1, got {seq_len}")

        slot = int(tensors["slot"][0].item())
        cache_slot = int(tensors["cache_slot"][0].item())
        should_compress = bool(int(tensors["should_compress"][0].item()))
        expected_slot = start_pos % COMPRESS_RATIO
        expected_cache_slot = start_pos // COMPRESS_RATIO
        expected_should_compress = (start_pos + 1) % COMPRESS_RATIO == 0
        if slot != expected_slot:
            raise ValueError(f"decode slot mismatch: expected {expected_slot}, got {slot}")
        if cache_slot != expected_cache_slot:
            raise ValueError(f"decode cache_slot mismatch: expected {expected_cache_slot}, got {cache_slot}")
        if should_compress != expected_should_compress:
            raise ValueError(
                "decode should_compress must match (start_pos + 1) % compress_ratio == 0; "
                f"got start_pos={start_pos}, should_compress={should_compress}"
            )

        kv_state = tensors["kv_state"].clone()
        score_state = tensors["score_state"].clone()
        compressed_cache = tensors["compressed_cache"].clone()

        score_proj += tensors["ape"][slot]
        kv_state[:, COMPRESS_RATIO + slot] = kv_proj.squeeze(1)
        score_state[:, COMPRESS_RATIO + slot] = score_proj.squeeze(1)
        if should_compress:
            kv_pool_state = torch.cat(
                [
                    kv_state[:, :COMPRESS_RATIO, :head_dim],
                    kv_state[:, COMPRESS_RATIO:, head_dim:],
                ],
                dim=1,
            )
            score_pool_state = torch.cat(
                [
                    score_state[:, :COMPRESS_RATIO, :head_dim],
                    score_state[:, COMPRESS_RATIO:, head_dim:],
                ],
                dim=1,
            )
            kv_for_norm = (kv_pool_state * score_pool_state.softmax(dim=1)).sum(dim=1, keepdim=True)
            kv_state[:, :COMPRESS_RATIO] = kv_state[:, COMPRESS_RATIO:]
            score_state[:, :COMPRESS_RATIO] = score_state[:, COMPRESS_RATIO:]
        else:
            kv_for_norm = None

    compressed = torch.zeros_like(tensors["compressed"])
    if should_compress:
        pooled = kv_for_norm.to(torch.bfloat16)
        inv_rms = torch.rsqrt(pooled.float().square().mean(-1, keepdim=True) + M.rms_norm_eps)
        normed = (pooled.float() * inv_rms * tensors["norm_w"].float()).to(torch.bfloat16)
        compressed = _apply_rope_golden(normed, tensors["cos"], tensors["sin"], inverse=False)
        if start_pos == 0:
            compressed_cache[:, : compressed.shape[1]] = compressed
        else:
            compressed_cache[:, cache_slot] = compressed[:, 0]

    tensors["compressed"][:] = compressed
    tensors["kv_state_out"][:] = kv_state
    tensors["score_state_out"][:] = score_state
    tensors["compressed_cache_out"][:] = compressed_cache


def golden_compressor_ratio4_indexer_forward(tensors, start_pos: int):
    golden_compressor_ratio4_forward(
        tensors,
        start_pos=start_pos,
        head_dim=INDEX_HEAD_DIM,
        proj_dim=INDEX_PROJ_DIM,
        module_name="Indexer",
    )


def golden_compressor_ratio4_attention_forward(tensors, start_pos: int):
    golden_compressor_ratio4_forward(
        tensors,
        start_pos=start_pos,
        head_dim=ATTN_HEAD_DIM,
        proj_dim=ATTN_PROJ_DIM,
        module_name="Attention",
    )


def golden_compressor_ratio4_indexer_prefill(tensors):
    golden_compressor_ratio4_indexer_forward(tensors, start_pos=0)


def golden_compressor_ratio4_indexer_decode(tensors):
    slot = int(tensors["slot"][0].item())
    cache_slot = int(tensors["cache_slot"][0].item())
    start_pos = cache_slot * COMPRESS_RATIO + slot
    if start_pos <= 0:
        raise ValueError("decode start_pos must be greater than 0; start_pos=0 is prefill")
    golden_compressor_ratio4_indexer_forward(tensors, start_pos=start_pos)


def golden_compressor_ratio4_attention_prefill(tensors):
    golden_compressor_ratio4_attention_forward(tensors, start_pos=0)


def golden_compressor_ratio4_attention_decode(tensors):
    slot = int(tensors["slot"][0].item())
    cache_slot = int(tensors["cache_slot"][0].item())
    start_pos = cache_slot * COMPRESS_RATIO + slot
    if start_pos <= 0:
        raise ValueError("decode start_pos must be greater than 0; start_pos=0 is prefill")
    golden_compressor_ratio4_attention_forward(tensors, start_pos=start_pos)


def build_indexer_prefill_specs(seq_len: int = DEFAULT_SEQ_LEN):
    import torch

    from models.golden import TensorSpec

    actual_compressed_len = seq_len // COMPRESS_RATIO
    compressed_len = max(1, actual_compressed_len)
    freqs_cos, freqs_sin = build_deepseek_v4_rope_tables(compress_ratio=COMPRESS_RATIO, max_seq_len=seq_len)
    local_cos, local_sin = materialize_compressor_rope(freqs_cos, freqs_sin, seq_len, COMPRESS_RATIO)
    if actual_compressed_len == 0:
        local_cos = freqs_cos[:1].contiguous()
        local_sin = freqs_sin[:1].contiguous()

    def init_x():
        return torch.randn(B, seq_len, HIDDEN) * 0.1

    def init_w():
        return torch.randn(HIDDEN, INDEX_PROJ_DIM) * 0.02

    def init_ape():
        return torch.randn(COMPRESS_RATIO, INDEX_PROJ_DIM) * 0.02

    def init_norm_w():
        return torch.randn(INDEX_HEAD_DIM) * 0.1 + 1.0

    return [
        TensorSpec("x", [B, seq_len, HIDDEN], torch.bfloat16, init_value=init_x),
        TensorSpec("wkv_t", [HIDDEN, INDEX_PROJ_DIM], torch.bfloat16, init_value=init_w),
        TensorSpec("wgate_t", [HIDDEN, INDEX_PROJ_DIM], torch.bfloat16, init_value=init_w),
        TensorSpec("ape", [COMPRESS_RATIO, INDEX_PROJ_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [INDEX_HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("cos", [compressed_len, ROPE_HALF], torch.float32, init_value=local_cos),
        TensorSpec("sin", [compressed_len, ROPE_HALF], torch.float32, init_value=local_sin),
        TensorSpec(
            "block_count",
            [1],
            torch.int32,
            init_value=torch.tensor([actual_compressed_len], dtype=torch.int32),
        ),
        TensorSpec("compressed", [B, compressed_len, INDEX_HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec("kv_state_out", [B, STATE_ROWS, INDEX_PROJ_DIM], torch.float32, is_output=True),
        TensorSpec("score_state_out", [B, STATE_ROWS, INDEX_PROJ_DIM], torch.float32, is_output=True),
        TensorSpec(
            "compressed_cache_out",
            [B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM],
            torch.bfloat16,
            is_output=True,
        ),
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
    if should_compress:
        rope_pos = start_pos + 1 - COMPRESS_RATIO
        max_seq_len = max(start_pos + seq_len, rope_pos + 1)
        freqs_cos, freqs_sin = build_deepseek_v4_rope_tables(
            compress_ratio=COMPRESS_RATIO,
            max_seq_len=max_seq_len,
        )
        local_cos = freqs_cos[rope_pos : rope_pos + 1].contiguous()
        local_sin = freqs_sin[rope_pos : rope_pos + 1].contiguous()
    else:
        local_cos = torch.zeros(1, ROPE_HALF, dtype=torch.float32)
        local_sin = torch.zeros(1, ROPE_HALF, dtype=torch.float32)

    def init_x():
        return torch.randn(B, seq_len, HIDDEN) * 0.1

    def init_w():
        return torch.randn(HIDDEN, INDEX_PROJ_DIM) * 0.02

    def init_ape():
        return torch.randn(COMPRESS_RATIO, INDEX_PROJ_DIM) * 0.02

    def init_norm_w():
        return torch.randn(INDEX_HEAD_DIM) * 0.1 + 1.0

    def init_kv_state():
        return torch.randn(B, STATE_ROWS, INDEX_PROJ_DIM) * 0.1

    def init_score_state():
        return torch.randn(B, STATE_ROWS, INDEX_PROJ_DIM) * 0.1

    def init_compressed_cache():
        return torch.randn(B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM) * 0.1

    return [
        TensorSpec("x", [B, seq_len, HIDDEN], torch.bfloat16, init_value=init_x),
        TensorSpec("kv_state", [B, STATE_ROWS, INDEX_PROJ_DIM], torch.float32, init_value=init_kv_state),
        TensorSpec("score_state", [B, STATE_ROWS, INDEX_PROJ_DIM], torch.float32, init_value=init_score_state),
        TensorSpec(
            "compressed_cache",
            [B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM],
            torch.bfloat16,
            init_value=init_compressed_cache,
        ),
        TensorSpec("slot", [1], torch.int32, init_value=torch.tensor([slot], dtype=torch.int32)),
        TensorSpec("cache_slot", [1], torch.int32, init_value=torch.tensor([cache_slot], dtype=torch.int32)),
        TensorSpec(
            "should_compress",
            [1],
            torch.int32,
            init_value=torch.tensor([should_compress], dtype=torch.int32),
        ),
        TensorSpec("wkv_t", [HIDDEN, INDEX_PROJ_DIM], torch.bfloat16, init_value=init_w),
        TensorSpec("wgate_t", [HIDDEN, INDEX_PROJ_DIM], torch.bfloat16, init_value=init_w),
        TensorSpec("ape", [COMPRESS_RATIO, INDEX_PROJ_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [INDEX_HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("cos", [1, ROPE_HALF], torch.float32, init_value=local_cos),
        TensorSpec("sin", [1, ROPE_HALF], torch.float32, init_value=local_sin),
        TensorSpec("compressed", [B, 1, INDEX_HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec("kv_state_out", [B, STATE_ROWS, INDEX_PROJ_DIM], torch.float32, is_output=True),
        TensorSpec("score_state_out", [B, STATE_ROWS, INDEX_PROJ_DIM], torch.float32, is_output=True),
        TensorSpec(
            "compressed_cache_out",
            [B, TOPK_CSA_COMPRESSED, INDEX_HEAD_DIM],
            torch.bfloat16,
            is_output=True,
        ),
    ]


def build_attention_prefill_specs(seq_len: int = DEFAULT_SEQ_LEN):
    import torch

    from models.golden import TensorSpec

    actual_compressed_len = seq_len // COMPRESS_RATIO
    compressed_len = max(1, actual_compressed_len)
    freqs_cos, freqs_sin = build_deepseek_v4_rope_tables(compress_ratio=COMPRESS_RATIO, max_seq_len=seq_len)
    local_cos, local_sin = materialize_compressor_rope(freqs_cos, freqs_sin, seq_len, COMPRESS_RATIO)
    if actual_compressed_len == 0:
        local_cos = freqs_cos[:1].contiguous()
        local_sin = freqs_sin[:1].contiguous()

    def init_x():
        return torch.randn(B, seq_len, HIDDEN) * 0.1

    def init_w():
        return torch.randn(HIDDEN, ATTN_PROJ_DIM) * 0.02

    def init_ape():
        return torch.randn(COMPRESS_RATIO, ATTN_PROJ_DIM) * 0.02

    def init_norm_w():
        return torch.randn(ATTN_HEAD_DIM) * 0.1 + 1.0

    return [
        TensorSpec("x", [B, seq_len, HIDDEN], torch.bfloat16, init_value=init_x),
        TensorSpec("wkv_t", [HIDDEN, ATTN_PROJ_DIM], torch.bfloat16, init_value=init_w),
        TensorSpec("wgate_t", [HIDDEN, ATTN_PROJ_DIM], torch.bfloat16, init_value=init_w),
        TensorSpec("ape", [COMPRESS_RATIO, ATTN_PROJ_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [ATTN_HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("cos", [compressed_len, ROPE_HALF], torch.float32, init_value=local_cos),
        TensorSpec("sin", [compressed_len, ROPE_HALF], torch.float32, init_value=local_sin),
        TensorSpec(
            "block_count",
            [1],
            torch.int32,
            init_value=torch.tensor([actual_compressed_len], dtype=torch.int32),
        ),
        TensorSpec("compressed", [B, compressed_len, ATTN_HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec("kv_state_out", [B, STATE_ROWS, ATTN_PROJ_DIM], torch.float32, is_output=True),
        TensorSpec("score_state_out", [B, STATE_ROWS, ATTN_PROJ_DIM], torch.float32, is_output=True),
        TensorSpec(
            "compressed_cache_out",
            [B, TOPK_CSA_COMPRESSED, ATTN_HEAD_DIM],
            torch.bfloat16,
            is_output=True,
        ),
    ]


def build_attention_decode_specs(start_pos: int = DEFAULT_DECODE_START_POS):
    import torch

    from models.golden import TensorSpec

    if start_pos <= 0:
        raise ValueError(f"decode start_pos must be greater than 0, got {start_pos}")

    seq_len = 1
    slot = start_pos % COMPRESS_RATIO
    cache_slot = start_pos // COMPRESS_RATIO
    should_compress = int((start_pos + 1) % COMPRESS_RATIO == 0)
    if should_compress:
        rope_pos = start_pos + 1 - COMPRESS_RATIO
        max_seq_len = max(start_pos + seq_len, rope_pos + 1)
        freqs_cos, freqs_sin = build_deepseek_v4_rope_tables(
            compress_ratio=COMPRESS_RATIO,
            max_seq_len=max_seq_len,
        )
        local_cos = freqs_cos[rope_pos : rope_pos + 1].contiguous()
        local_sin = freqs_sin[rope_pos : rope_pos + 1].contiguous()
    else:
        local_cos = torch.zeros(1, ROPE_HALF, dtype=torch.float32)
        local_sin = torch.zeros(1, ROPE_HALF, dtype=torch.float32)

    def init_x():
        return torch.randn(B, seq_len, HIDDEN) * 0.1

    def init_w():
        return torch.randn(HIDDEN, ATTN_PROJ_DIM) * 0.02

    def init_ape():
        return torch.randn(COMPRESS_RATIO, ATTN_PROJ_DIM) * 0.02

    def init_norm_w():
        return torch.randn(ATTN_HEAD_DIM) * 0.1 + 1.0

    def init_kv_state():
        return torch.randn(B, STATE_ROWS, ATTN_PROJ_DIM) * 0.1

    def init_score_state():
        return torch.randn(B, STATE_ROWS, ATTN_PROJ_DIM) * 0.1

    def init_compressed_cache():
        return torch.randn(B, TOPK_CSA_COMPRESSED, ATTN_HEAD_DIM) * 0.1

    return [
        TensorSpec("x", [B, seq_len, HIDDEN], torch.bfloat16, init_value=init_x),
        TensorSpec("kv_state", [B, STATE_ROWS, ATTN_PROJ_DIM], torch.float32, init_value=init_kv_state),
        TensorSpec("score_state", [B, STATE_ROWS, ATTN_PROJ_DIM], torch.float32, init_value=init_score_state),
        TensorSpec(
            "compressed_cache",
            [B, TOPK_CSA_COMPRESSED, ATTN_HEAD_DIM],
            torch.bfloat16,
            init_value=init_compressed_cache,
        ),
        TensorSpec("slot", [1], torch.int32, init_value=torch.tensor([slot], dtype=torch.int32)),
        TensorSpec("cache_slot", [1], torch.int32, init_value=torch.tensor([cache_slot], dtype=torch.int32)),
        TensorSpec(
            "should_compress",
            [1],
            torch.int32,
            init_value=torch.tensor([should_compress], dtype=torch.int32),
        ),
        TensorSpec("wkv_t", [HIDDEN, ATTN_PROJ_DIM], torch.bfloat16, init_value=init_w),
        TensorSpec("wgate_t", [HIDDEN, ATTN_PROJ_DIM], torch.bfloat16, init_value=init_w),
        TensorSpec("ape", [COMPRESS_RATIO, ATTN_PROJ_DIM], torch.float32, init_value=init_ape),
        TensorSpec("norm_w", [ATTN_HEAD_DIM], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("cos", [1, ROPE_HALF], torch.float32, init_value=local_cos),
        TensorSpec("sin", [1, ROPE_HALF], torch.float32, init_value=local_sin),
        TensorSpec("compressed", [B, 1, ATTN_HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec("kv_state_out", [B, STATE_ROWS, ATTN_PROJ_DIM], torch.float32, is_output=True),
        TensorSpec("score_state_out", [B, STATE_ROWS, ATTN_PROJ_DIM], torch.float32, is_output=True),
        TensorSpec(
            "compressed_cache_out",
            [B, TOPK_CSA_COMPRESSED, ATTN_HEAD_DIM],
            torch.bfloat16,
            is_output=True,
        ),
    ]


def main() -> int:
    import argparse

    from models.golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 Flash ratio-4 Indexer compressor validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3sim", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-s", "--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--decode-start-pos", type=int, default=DEFAULT_DECODE_START_POS)
    parser.add_argument("--case", choices=["all", "prefill", "decode"], default="all")
    parser.add_argument("--target", choices=["all", "indexer", "attention"], default="all")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    runtime_cfg = {
        "platform": args.platform,
        "device_id": args.device,
        "enable_l2_swimlane": args.enable_l2_swimlane,
    }
    compare_fn = {
        "compressed": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "kv_state_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "score_state_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "compressed_cache_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
    }

    cases = []
    if args.target in ("all", "indexer") and args.case in ("all", "prefill"):
        cases.append(
            (
                "compressor-ratio4-indexer-prefill",
                compressor_ratio4_indexer_prefill_test,
                lambda: build_indexer_prefill_specs(args.seq_len),
                golden_compressor_ratio4_indexer_prefill,
            )
        )
    if args.target in ("all", "indexer") and args.case in ("all", "decode"):
        cases.append(
            (
                "compressor-ratio4-indexer-decode",
                compressor_ratio4_indexer_decode_test,
                lambda: build_indexer_decode_specs(args.decode_start_pos),
                golden_compressor_ratio4_indexer_decode,
            )
        )
    if args.target in ("all", "attention") and args.case in ("all", "prefill"):
        cases.append(
            (
                "compressor-ratio4-attention-prefill",
                compressor_ratio4_attention_prefill_test,
                lambda: build_attention_prefill_specs(args.seq_len),
                golden_compressor_ratio4_attention_prefill,
            )
        )
    if args.target in ("all", "attention") and args.case in ("all", "decode"):
        cases.append(
            (
                "compressor-ratio4-attention-decode",
                compressor_ratio4_attention_decode_test,
                lambda: build_attention_decode_specs(args.decode_start_pos),
                golden_compressor_ratio4_attention_decode,
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
    "HIDDEN",
    "ATTN_HEAD_DIM",
    "INDEX_HEAD_DIM",
    "ATTN_PROJ_DIM",
    "INDEX_PROJ_DIM",
    "ROPE_HALF",
    "COMPRESS_RATIO",
    "STATE_ROWS",
    "CSA_MAX_POSITION_EMBEDDINGS",
    "TOPK_CSA_COMPRESSED",
    "NEG_INF",
    "DEFAULT_SEQ_LEN",
    "DEFAULT_DECODE_START_POS",
    "compressor_ratio4_indexer_prefill_fwd",
    "compressor_ratio4_indexer_prefill_test",
    "compressor_ratio4_indexer_decode_fwd",
    "compressor_ratio4_indexer_decode_test",
    "compressor_ratio4_attention_prefill_fwd",
    "compressor_ratio4_attention_prefill_test",
    "compressor_ratio4_attention_decode_fwd",
    "compressor_ratio4_attention_decode_test",
    "golden_compressor_ratio4_forward",
    "golden_compressor_ratio4_indexer_forward",
    "golden_compressor_ratio4_indexer_prefill",
    "golden_compressor_ratio4_indexer_decode",
    "golden_compressor_ratio4_attention_forward",
    "golden_compressor_ratio4_attention_prefill",
    "golden_compressor_ratio4_attention_decode",
    "build_indexer_prefill_specs",
    "build_indexer_decode_specs",
    "build_attention_prefill_specs",
    "build_attention_decode_specs",
]
