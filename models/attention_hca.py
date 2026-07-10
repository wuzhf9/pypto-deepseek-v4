"""DeepSeek V4 Flash HCA attention PyPTO kernel."""

import pypto.language as pl

from models.attention_out import attention_out_fwd
from models.attention_qkv import attention_qkv_fwd
from models.attention_swa import update_decode_window_cache, update_prefill_window_cache
from models.compressor_ratio128 import (
    COMPRESS_RATIO,
    compressor_ratio128_decode_fwd,
    compressor_ratio128_prefill_fwd,
    golden_compressor_ratio128_forward,
)
from models.config import FLASH_CONFIG as M
from models.rope import _apply_rope_golden, build_deepseek_v4_rope_tables, materialize_compressor_rope, materialize_rope_range
from models.sparse_attn import (
    build_compress_topk_idxs,
    build_window_topk_idxs,
    golden_sparse_attn,
    sparse_attn_hca_fwd,
)


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
HCA_MAX_POSITION_EMBEDDINGS = 4096
TOPK_HCA = HCA_MAX_POSITION_EMBEDDINGS // COMPRESS_RATIO
TOPK_SWA = WINDOW_SIZE
TOPK_HCA_TOTAL = TOPK_SWA + TOPK_HCA
SOFTMAX_SCALE = HEAD_DIM**-0.5
EPS = M.rms_norm_eps

DEFAULT_SEQ_LEN = 256
DEFAULT_DECODE_START_POS = COMPRESS_RATIO - 1
DECODE_KV_POOL_ROW_TILE = 16


@pl.jit.inline
def build_prefill_kv_pool(
    kv: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    compressed: pl.Tensor[[B, C_DYN, HEAD_DIM], pl.BF16],
    kv_pool: pl.Tensor[[B, K_DYN, HEAD_DIM], pl.BF16],
):
    """Build sparse-attention KV pool ``cat([window_kv, compressed_kv])``."""
    kv.bind_dynamic(1, S_DYN)
    compressed.bind_dynamic(1, C_DYN)
    kv_pool.bind_dynamic(1, K_DYN)

    tokens = pl.tensor.dim(kv, 1)
    compressed_len = pl.tensor.dim(compressed, 1)
    kv_flat = pl.reshape(kv, [tokens, HEAD_DIM])
    compressed_flat = pl.reshape(compressed, [compressed_len, HEAD_DIM])
    pool_flat = pl.reshape(kv_pool, [tokens + compressed_len, HEAD_DIM])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_kv_pool"):
        for t in pl.range(tokens):
            pool_flat[t : t + 1, 0:HEAD_DIM] = kv_flat[t : t + 1, 0:HEAD_DIM]
        for c in pl.range(compressed_len):
            dst = tokens + c
            pool_flat[dst : dst + 1, 0:HEAD_DIM] = compressed_flat[c : c + 1, 0:HEAD_DIM]

    return pl.reshape(pool_flat, [B, tokens + compressed_len, HEAD_DIM])


@pl.jit.inline
def build_decode_kv_pool(
    kv_cache: pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16],
    compressed_cache: pl.Tensor[[B, C_DYN, HEAD_DIM], pl.BF16],
    kv_pool: pl.Tensor[[B, K_DYN, HEAD_DIM], pl.BF16],
):
    """Build decode KV pool ``cat([window_cache, compressed_cache])``."""
    compressed_cache.bind_dynamic(1, C_DYN)
    kv_pool.bind_dynamic(1, K_DYN)

    compressed_len = pl.tensor.dim(compressed_cache, 1)
    cache_flat = pl.reshape(kv_cache, [WINDOW_SIZE, HEAD_DIM])
    compressed_flat = pl.reshape(compressed_cache, [compressed_len, HEAD_DIM])
    pool_rows = WINDOW_SIZE + compressed_len
    pool_flat = pl.reshape(kv_pool, [pool_rows, HEAD_DIM])

    pool_blocks = (pool_rows + DECODE_KV_POOL_ROW_TILE - 1) // DECODE_KV_POOL_ROW_TILE
    for block in pl.spmd(pool_blocks, name_hint="decode_kv_pool"):
        row0 = block * DECODE_KV_POOL_ROW_TILE
        for offset in pl.range(DECODE_KV_POOL_ROW_TILE):
            row = row0 + offset
            if row < pool_rows:
                if row < WINDOW_SIZE:
                    pool_flat[row : row + 1, 0:HEAD_DIM] = cache_flat[row : row + 1, 0:HEAD_DIM]
                else:
                    src = row - WINDOW_SIZE
                    pool_flat[row : row + 1, 0:HEAD_DIM] = compressed_flat[src : src + 1, 0:HEAD_DIM]

    return pl.reshape(pool_flat, [B, pool_rows, HEAD_DIM])


@pl.jit.inline
def attention_hca_prefill_fwd(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
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
    comp_ape: pl.Tensor[[COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    comp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    comp_cos: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    comp_sin: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    comp_block_count: pl.Tensor[[1], pl.INT32],
    kv_pool: pl.Tensor[[B, K_DYN, HEAD_DIM], pl.BF16],
    kv_cache_out: pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16],
    comp_kv_state_out: pl.Tensor[[B, COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    comp_score_state_out: pl.Tensor[[B, COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    comp_cache_out: pl.Tensor[[B, TOPK_HCA, HEAD_DIM], pl.BF16],
    out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
):
    """Run ``Attention.forward`` for ``compress_ratio == 128, start_pos == 0``."""
    x.bind_dynamic(1, S_DYN)
    topk_idxs.bind_dynamic(1, S_DYN)
    cos.bind_dynamic(0, S_DYN)
    sin.bind_dynamic(0, S_DYN)
    comp_cos.bind_dynamic(0, C_DYN)
    comp_sin.bind_dynamic(0, C_DYN)
    kv_pool.bind_dynamic(1, K_DYN)

    tokens = pl.tensor.dim(x, 1)
    compressed_len = pl.tensor.dim(comp_cos, 0)
    qr = pl.create_tensor([B, tokens, Q_LORA_RANK], dtype=pl.BF16)
    q = pl.create_tensor([B, tokens, N_HEADS, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([B, tokens, HEAD_DIM], dtype=pl.BF16)
    compressed = pl.create_tensor([B, compressed_len, HEAD_DIM], dtype=pl.BF16)
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
    compressor_ratio128_prefill_fwd(
        x,
        comp_wkv_t,
        comp_wgate_t,
        comp_ape,
        comp_norm_w,
        comp_cos,
        comp_sin,
        comp_block_count,
        compressed,
        comp_kv_state_out,
        comp_score_state_out,
        comp_cache_out,
    )
    build_prefill_kv_pool(kv, compressed, kv_pool)
    sparse_attn_hca_fwd(q, kv_pool, attn_sink, topk_idxs, attn_o)
    attention_out_fwd(attn_o, wo_a_t, wo_b_t, cos, sin, out)
    return kv_cache_out, comp_kv_state_out, comp_score_state_out, comp_cache_out, out


@pl.jit.inline
def attention_hca_decode_fwd(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    kv_cache: pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16],
    comp_kv_state: pl.Tensor[[B, COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    comp_score_state: pl.Tensor[[B, COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    comp_cache: pl.Tensor[[B, TOPK_HCA, HEAD_DIM], pl.BF16],
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
    topk_idxs: pl.Tensor[[B, S_DYN, TOPK_HCA_TOTAL], pl.INT32],
    wo_a_t: pl.Tensor[[O_GROUP_IN, ATTN_OUT_IN], pl.BF16],
    wo_b_t: pl.Tensor[[ATTN_OUT_IN, HIDDEN], pl.BF16],
    cos: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    comp_wkv_t: pl.Tensor[[HIDDEN, HEAD_DIM], pl.BF16],
    comp_wgate_t: pl.Tensor[[HIDDEN, HEAD_DIM], pl.BF16],
    comp_ape: pl.Tensor[[COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    comp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    comp_cos: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    comp_sin: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    kv_cache_out: pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16],
    comp_kv_state_out: pl.Tensor[[B, COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    comp_score_state_out: pl.Tensor[[B, COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    comp_cache_out: pl.Tensor[[B, TOPK_HCA, HEAD_DIM], pl.BF16],
    out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
):
    """Run ``Attention.forward`` for ``compress_ratio == 128, start_pos > 0``."""
    x.bind_dynamic(1, S_DYN)
    topk_idxs.bind_dynamic(1, S_DYN)
    cos.bind_dynamic(0, S_DYN)
    sin.bind_dynamic(0, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    qr = pl.create_tensor([B, tokens, Q_LORA_RANK], dtype=pl.BF16)
    q = pl.create_tensor([B, tokens, N_HEADS, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([B, tokens, HEAD_DIM], dtype=pl.BF16)
    compressed = pl.create_tensor([B, 1, HEAD_DIM], dtype=pl.BF16)
    kv_pool = pl.create_tensor([B, WINDOW_SIZE + TOPK_HCA, HEAD_DIM], dtype=pl.BF16)
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
    compressor_ratio128_decode_fwd(
        x,
        comp_kv_state,
        comp_score_state,
        comp_cache,
        comp_slot,
        comp_cache_slot,
        comp_should_compress,
        comp_wkv_t,
        comp_wgate_t,
        comp_ape,
        comp_norm_w,
        comp_cos,
        comp_sin,
        compressed,
        comp_kv_state_out,
        comp_score_state_out,
        comp_cache_out,
    )
    build_decode_kv_pool(kv_cache_out, comp_cache_out, kv_pool)
    sparse_attn_hca_fwd(q, kv_pool, attn_sink, topk_idxs, attn_o)
    attention_out_fwd(attn_o, wo_a_t, wo_b_t, cos, sin, out)
    return kv_cache_out, comp_kv_state_out, comp_score_state_out, comp_cache_out, out


@pl.jit
def attention_hca_prefill_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
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
    comp_ape: pl.Tensor[[COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    comp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    comp_cos: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    comp_sin: pl.Tensor[[C_DYN, ROPE_HALF], pl.FP32],
    comp_block_count: pl.Tensor[[1], pl.INT32],
    kv_pool: pl.Tensor[[B, K_DYN, HEAD_DIM], pl.BF16],
    kv_cache_out: pl.Out[pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16]],
    comp_kv_state_out: pl.Out[pl.Tensor[[B, COMPRESS_RATIO, HEAD_DIM], pl.FP32]],
    comp_score_state_out: pl.Out[pl.Tensor[[B, COMPRESS_RATIO, HEAD_DIM], pl.FP32]],
    comp_cache_out: pl.Out[pl.Tensor[[B, TOPK_HCA, HEAD_DIM], pl.BF16]],
    out: pl.Out[pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16]],
):
    return attention_hca_prefill_fwd(
        x,
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
        kv_pool,
        kv_cache_out,
        comp_kv_state_out,
        comp_score_state_out,
        comp_cache_out,
        out,
    )


@pl.jit
def attention_hca_decode_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    kv_cache: pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16],
    comp_kv_state: pl.Tensor[[B, COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    comp_score_state: pl.Tensor[[B, COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    comp_cache: pl.Tensor[[B, TOPK_HCA, HEAD_DIM], pl.BF16],
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
    topk_idxs: pl.Tensor[[B, S_DYN, TOPK_HCA_TOTAL], pl.INT32],
    wo_a_t: pl.Tensor[[O_GROUP_IN, ATTN_OUT_IN], pl.BF16],
    wo_b_t: pl.Tensor[[ATTN_OUT_IN, HIDDEN], pl.BF16],
    cos: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    comp_wkv_t: pl.Tensor[[HIDDEN, HEAD_DIM], pl.BF16],
    comp_wgate_t: pl.Tensor[[HIDDEN, HEAD_DIM], pl.BF16],
    comp_ape: pl.Tensor[[COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    comp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    comp_cos: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    comp_sin: pl.Tensor[[1, ROPE_HALF], pl.FP32],
    kv_cache_out: pl.Out[pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16]],
    comp_kv_state_out: pl.Out[pl.Tensor[[B, COMPRESS_RATIO, HEAD_DIM], pl.FP32]],
    comp_score_state_out: pl.Out[pl.Tensor[[B, COMPRESS_RATIO, HEAD_DIM], pl.FP32]],
    comp_cache_out: pl.Out[pl.Tensor[[B, TOPK_HCA, HEAD_DIM], pl.BF16]],
    out: pl.Out[pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16]],
):
    return attention_hca_decode_fwd(
        x,
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


def golden_attention_hca_forward(tensors, start_pos: int):
    import torch

    if start_pos < 0:
        raise ValueError(f"start_pos must be non-negative, got {start_pos}")

    x = tensors["x"]
    if start_pos > 0 and x.shape[1] != 1:
        raise ValueError(f"decode expects seq_len=1, got {x.shape[1]}")
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

    comp_len = tensors["comp_cos"].shape[0]

    comp_tensors = {
        "x": x,
        "wkv_t": tensors["comp_wkv_t"],
        "wgate_t": tensors["comp_wgate_t"],
        "ape": tensors["comp_ape"],
        "norm_w": tensors["comp_norm_w"],
        "cos": tensors["comp_cos"],
        "sin": tensors["comp_sin"],
        "kv_proj": torch.empty(B, x.shape[1], HEAD_DIM, dtype=torch.float32),
        "score_proj": torch.empty(B, x.shape[1], HEAD_DIM, dtype=torch.float32),
        "pooled": torch.empty(B, comp_len, HEAD_DIM, dtype=torch.bfloat16),
        "normed": torch.empty(B, comp_len, HEAD_DIM, dtype=torch.bfloat16),
        "compressed": torch.empty(B, comp_len, HEAD_DIM, dtype=torch.bfloat16),
        "kv_state_out": tensors["comp_kv_state_out"].clone(),
        "score_state_out": tensors["comp_score_state_out"].clone(),
        "compressed_cache_out": tensors["comp_cache_out"].clone(),
    }

    if start_pos == 0:
        seqlen = kv.shape[1]
        kv_cache_out = tensors["kv_cache_out"].clone()
        if seqlen <= WINDOW_SIZE:
            kv_cache_out[:, :seqlen] = kv
        else:
            cutoff = seqlen % WINDOW_SIZE
            kv_cache_out[:, cutoff:WINDOW_SIZE], kv_cache_out[:, :cutoff] = kv[:, -WINDOW_SIZE:].split(
                [WINDOW_SIZE - cutoff, cutoff],
                dim=1,
            )

        comp_tensors["block_count"] = tensors["comp_block_count"]
        golden_compressor_ratio128_forward(comp_tensors, start_pos=0)
        blocks = int(tensors["comp_block_count"][0].item())
        compressed = comp_tensors["compressed"]
        if blocks > 0:
            kv_pool = torch.cat([kv, compressed[:, :blocks]], dim=1)
        else:
            kv_pool = kv
    else:
        kv_cache_out = tensors["kv_cache"].clone()
        kv_cache_out[0, int(tensors["cache_pos"][0].item())] = kv[0, 0]

        comp_tensors.update(
            {
                "kv_state": tensors["comp_kv_state"],
                "score_state": tensors["comp_score_state"],
                "compressed_cache": tensors["comp_cache"],
                "slot": tensors["comp_slot"],
                "cache_slot": tensors["comp_cache_slot"],
                "should_compress": tensors["comp_should_compress"],
            }
        )
        golden_compressor_ratio128_forward(comp_tensors, start_pos=start_pos)
        kv_pool = torch.cat([kv_cache_out, comp_tensors["compressed_cache_out"]], dim=1)

    attn_o = _golden_sparse_attn(q, kv_pool, tensors["attn_sink"], tensors["topk_idxs"])
    o_inv = _apply_rope_golden(attn_o, tensors["cos"], tensors["sin"], inverse=True)
    o = o_inv.view(B, x.shape[1], O_GROUPS, O_GROUP_IN)
    wo_a = tensors["wo_a_t"].transpose(0, 1).contiguous().view(O_GROUPS, O_LORA_RANK, O_GROUP_IN)
    proj = torch.einsum("bsgd,grd->bsgr", o.float(), wo_a.float()).to(torch.bfloat16)
    out = torch.matmul(proj.flatten(2).float(), tensors["wo_b_t"].float()).to(torch.bfloat16)

    if "kv_pool" in tensors:
        tensors["kv_pool"][:, : kv_pool.shape[1]] = kv_pool
    tensors["kv_cache_out"][:] = kv_cache_out
    tensors["comp_kv_state_out"][:] = comp_tensors["kv_state_out"]
    tensors["comp_score_state_out"][:] = comp_tensors["score_state_out"]
    tensors["comp_cache_out"][:] = comp_tensors["compressed_cache_out"]
    tensors["out"][:] = out


def golden_attention_hca_prefill(tensors):
    golden_attention_hca_forward(tensors, start_pos=0)


def golden_attention_hca_decode(tensors, start_pos: int = DEFAULT_DECODE_START_POS):
    if start_pos <= 0:
        raise ValueError(f"decode start_pos must be positive, got {start_pos}")
    golden_attention_hca_forward(tensors, start_pos=start_pos)


def _common_specs(seq_len: int, start_pos: int, *, decode: bool):
    import torch

    from models.golden import TensorSpec

    attn_cos_all, attn_sin_all = build_deepseek_v4_rope_tables(
        compress=True,
        max_seq_len=start_pos + seq_len,
    )
    local_cos, local_sin = materialize_rope_range(attn_cos_all, attn_sin_all, start_pos, seq_len)
    if decode:
        comp_should = int((start_pos + 1) % COMPRESS_RATIO == 0)
        if comp_should:
            comp_rope_pos = start_pos + 1 - COMPRESS_RATIO
            comp_cos_all, comp_sin_all = build_deepseek_v4_rope_tables(
                compress=True,
                max_seq_len=max(start_pos + seq_len, comp_rope_pos + 1),
            )
            comp_cos = comp_cos_all[comp_rope_pos : comp_rope_pos + 1].contiguous()
            comp_sin = comp_sin_all[comp_rope_pos : comp_rope_pos + 1].contiguous()
        else:
            comp_cos = torch.zeros(1, ROPE_HALF, dtype=torch.float32)
            comp_sin = torch.zeros(1, ROPE_HALF, dtype=torch.float32)
        compressed_len = 1
        kv_pool_len = WINDOW_SIZE + TOPK_HCA
    else:
        blocks = seq_len // COMPRESS_RATIO
        compressed_len = max(1, blocks)
        comp_cos_all, comp_sin_all = build_deepseek_v4_rope_tables(
            compress=True,
            max_seq_len=seq_len,
        )
        comp_cos, comp_sin = materialize_compressor_rope(comp_cos_all, comp_sin_all, seq_len, COMPRESS_RATIO)
        if blocks == 0:
            comp_cos = comp_cos_all[:1].contiguous()
            comp_sin = comp_sin_all[:1].contiguous()
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

    def init_comp_ape():
        return torch.randn(COMPRESS_RATIO, HEAD_DIM) * 0.02

    def init_comp_norm_w():
        return torch.randn(HEAD_DIM) * 0.1 + 1.0

    def init_topk():
        if decode:
            window_topk = build_window_topk_idxs(seq_len, start_pos=start_pos, topk_max=TOPK_SWA)
            compress_topk = build_compress_topk_idxs(
                COMPRESS_RATIO,
                seq_len,
                start_pos=start_pos,
                offset=WINDOW_SIZE,
                topk_max=TOPK_HCA,
            )
        else:
            window_topk = build_window_topk_idxs(seq_len, start_pos=0, topk_max=TOPK_SWA)
            compress_topk = build_compress_topk_idxs(
                COMPRESS_RATIO,
                seq_len,
                start_pos=0,
                offset=seq_len,
                topk_max=TOPK_HCA,
            )
        return torch.cat([window_topk, compress_topk], dim=-1)

    x_spec = [TensorSpec("x", [B, seq_len, HIDDEN], torch.bfloat16, init_value=init_x)]
    if decode:
        prefix_specs = [
            *x_spec,
            TensorSpec("kv_cache", [B, WINDOW_SIZE, HEAD_DIM], torch.bfloat16, init_value=lambda: torch.randn(B, WINDOW_SIZE, HEAD_DIM) * 0.1),
            TensorSpec("comp_kv_state", [B, COMPRESS_RATIO, HEAD_DIM], torch.float32, init_value=lambda: torch.randn(B, COMPRESS_RATIO, HEAD_DIM) * 0.1),
            TensorSpec("comp_score_state", [B, COMPRESS_RATIO, HEAD_DIM], torch.float32, init_value=lambda: torch.randn(B, COMPRESS_RATIO, HEAD_DIM) * 0.1),
            TensorSpec("comp_cache", [B, TOPK_HCA, HEAD_DIM], torch.bfloat16, init_value=lambda: torch.randn(B, TOPK_HCA, HEAD_DIM) * 0.1),
            TensorSpec("cache_pos", [1], torch.int32, init_value=torch.tensor([start_pos % WINDOW_SIZE], dtype=torch.int32)),
            TensorSpec("comp_slot", [1], torch.int32, init_value=torch.tensor([start_pos % COMPRESS_RATIO], dtype=torch.int32)),
            TensorSpec("comp_cache_slot", [1], torch.int32, init_value=torch.tensor([start_pos // COMPRESS_RATIO], dtype=torch.int32)),
            TensorSpec("comp_should_compress", [1], torch.int32, init_value=torch.tensor([comp_should], dtype=torch.int32)),
        ]
    else:
        prefix_specs = x_spec

    specs = [
        *prefix_specs,
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
        TensorSpec("comp_ape", [COMPRESS_RATIO, HEAD_DIM], torch.float32, init_value=init_comp_ape),
        TensorSpec("comp_norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_comp_norm_w),
        TensorSpec("comp_cos", [compressed_len, ROPE_HALF], torch.float32, init_value=comp_cos),
        TensorSpec("comp_sin", [compressed_len, ROPE_HALF], torch.float32, init_value=comp_sin),
    ]
    if not decode:
        specs.extend(
            [
                TensorSpec(
                    "comp_block_count",
                    [1],
                    torch.int32,
                    init_value=torch.tensor([seq_len // COMPRESS_RATIO], dtype=torch.int32),
                ),
                TensorSpec("kv_pool", [B, kv_pool_len, HEAD_DIM], torch.bfloat16),
            ]
        )

    specs.extend(
        [
            TensorSpec("kv_cache_out", [B, WINDOW_SIZE, HEAD_DIM], torch.bfloat16, is_output=True, init_value=0.0),
            TensorSpec("comp_kv_state_out", [B, COMPRESS_RATIO, HEAD_DIM], torch.float32, is_output=True),
            TensorSpec("comp_score_state_out", [B, COMPRESS_RATIO, HEAD_DIM], torch.float32, is_output=True),
            TensorSpec("comp_cache_out", [B, TOPK_HCA, HEAD_DIM], torch.bfloat16, is_output=True),
            TensorSpec("out", [B, seq_len, HIDDEN], torch.bfloat16, is_output=True),
        ]
    )
    return specs


def build_hca_prefill_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _common_specs(seq_len, start_pos=0, decode=False)


def build_hca_decode_specs(start_pos: int = DEFAULT_DECODE_START_POS):
    if start_pos <= 0:
        raise ValueError(f"decode start_pos must be positive, got {start_pos}")
    return _common_specs(1, start_pos=start_pos, decode=True)


def main() -> int:
    import argparse

    from models.golden import ignore_output, ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 Flash HCA attention validation.")
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
        "comp_kv_state_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "comp_score_state_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "comp_cache_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "out": ratio_allclose(atol=1e-3, rtol=2.0 / 128, max_error_ratio=0.005),
    }

    cases = []
    if args.case in ("all", "prefill"):
        cases.append(("hca-prefill", attention_hca_prefill_test, lambda: build_hca_prefill_specs(args.seq_len), golden_attention_hca_prefill))
    if args.case in ("all", "decode"):
        cases.append(
            (
                "hca-decode",
                attention_hca_decode_test,
                lambda: build_hca_decode_specs(args.decode_start_pos),
                lambda tensors: golden_attention_hca_forward(tensors, start_pos=args.decode_start_pos),
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
    "O_GROUPS",
    "O_LORA_RANK",
    "O_GROUP_IN",
    "ATTN_OUT_IN",
    "ROPE_HALF",
    "WINDOW_SIZE",
    "TOPK_HCA",
    "TOPK_HCA_TOTAL",
    "DEFAULT_SEQ_LEN",
    "DEFAULT_DECODE_START_POS",
    "build_prefill_kv_pool",
    "build_decode_kv_pool",
    "attention_hca_prefill_fwd",
    "attention_hca_decode_fwd",
    "attention_hca_prefill_test",
    "attention_hca_decode_test",
    "golden_attention_hca_forward",
    "golden_attention_hca_prefill",
    "golden_attention_hca_decode",
    "build_hca_prefill_specs",
    "build_hca_decode_specs",
]
