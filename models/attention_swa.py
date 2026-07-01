"""DeepSeek V4 Flash SWA attention PyPTO kernel."""

import pypto.language as pl

from models.attention_out import attention_out_fwd
from models.attention_qkv import attention_qkv_fwd
from models.config import FLASH_CONFIG as M
from models.rope import _apply_rope_golden, build_deepseek_v4_rope_tables, materialize_rope_range
from models.sparse_attn import build_window_topk_idxs, golden_sparse_attn, sparse_attn_swa_fwd


B = 1
S_DYN = pl.dynamic("S_DYN")

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
TOPK_SWA = WINDOW_SIZE
SOFTMAX_SCALE = HEAD_DIM**-0.5
EPS = M.rms_norm_eps

DEFAULT_SEQ_LEN = 8
DEFAULT_DECODE_START_POS = 1


@pl.jit.inline
def update_swa_prefill_cache(
    kv: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    kv_cache_out: pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16],
):
    """Mirror the prefill window-cache update in ``Attention.forward``."""
    kv.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(kv, 1)
    kv_flat = pl.reshape(kv, [tokens, HEAD_DIM])
    cache_flat = pl.reshape(kv_cache_out, [WINDOW_SIZE, HEAD_DIM])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_prefill_cache_write"):
        if tokens <= WINDOW_SIZE:
            for t in pl.range(tokens):
                cache_flat[t : t + 1, 0:HEAD_DIM] = kv_flat[t : t + 1, 0:HEAD_DIM]
        else:
            cutoff = tokens % WINDOW_SIZE
            for c in pl.range(WINDOW_SIZE):
                if c < cutoff:
                    src = tokens - cutoff + c
                else:
                    src = tokens - WINDOW_SIZE + c - cutoff
                src_idx = pl.cast(src, pl.INDEX)
                cache_flat[c : c + 1, 0:HEAD_DIM] = kv_flat[src_idx : src_idx + 1, 0:HEAD_DIM]

    return pl.reshape(cache_flat, [B, WINDOW_SIZE, HEAD_DIM])


@pl.jit.inline
def update_swa_decode_cache(
    kv_cache: pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    cache_pos: pl.Tensor[[1], pl.INT32],
    kv_cache_out: pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16],
):
    """Mirror ``kv_cache[:, start_pos % win] = kv.squeeze(1)``."""
    kv.bind_dynamic(1, S_DYN)

    cache_in_flat = pl.reshape(kv_cache, [WINDOW_SIZE, HEAD_DIM])
    cache_out_flat = pl.reshape(kv_cache_out, [WINDOW_SIZE, HEAD_DIM])
    kv_flat = pl.reshape(kv, [1, HEAD_DIM])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_decode_cache_copy"):
        cache_out_flat[0:WINDOW_SIZE, 0:HEAD_DIM] = cache_in_flat[0:WINDOW_SIZE, 0:HEAD_DIM]
        raw_pos = pl.read(cache_pos, [0])
        pos = pl.cast(raw_pos, pl.INDEX)
        cache_out_flat[pos : pos + 1, 0:HEAD_DIM] = kv_flat[0:1, 0:HEAD_DIM]

    return pl.reshape(cache_out_flat, [B, WINDOW_SIZE, HEAD_DIM])


@pl.jit.inline
def attention_swa_prefill_fwd(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
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
    q_a: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    q_proj: pl.Tensor[[B, S_DYN, ATTN_Q_OUT], pl.BF16],
    kv_proj: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    kv_normed: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    q: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    kv_cache_out: pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16],
    attn_o: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    o_inv: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    proj: pl.Tensor[[B, S_DYN, ATTN_OUT_IN], pl.BF16],
    out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
):
    """Run ``Attention.forward`` for ``compress_ratio == 0, start_pos == 0``."""
    x.bind_dynamic(1, S_DYN)
    topk_idxs.bind_dynamic(1, S_DYN)
    cos.bind_dynamic(0, S_DYN)
    sin.bind_dynamic(0, S_DYN)

    qr, q, kv = attention_qkv_fwd(
        x,
        wq_a_t,
        q_norm_w,
        wq_b_t,
        wkv_t,
        kv_norm_w,
        cos,
        sin,
        q_a,
        q_proj,
        kv_proj,
        kv_normed,
        qr,
        q,
        kv,
    )
    kv_cache_out = update_swa_prefill_cache(kv, kv_cache_out)
    attn_o = sparse_attn_swa_fwd(q, kv, attn_sink, topk_idxs, attn_o)
    out_final = attention_out_fwd(attn_o, wo_a_t, wo_b_t, cos, sin, o_inv, proj, out)
    return kv_cache_out, out_final


@pl.jit.inline
def attention_swa_decode_fwd(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    kv_cache: pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16],
    cache_pos: pl.Tensor[[1], pl.INT32],
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
    q_a: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    q_proj: pl.Tensor[[B, S_DYN, ATTN_Q_OUT], pl.BF16],
    kv_proj: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    kv_normed: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    q: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    kv_cache_out: pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16],
    attn_o: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    o_inv: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    proj: pl.Tensor[[B, S_DYN, ATTN_OUT_IN], pl.BF16],
    out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
):
    """Run ``Attention.forward`` for ``compress_ratio == 0, start_pos > 0``."""
    qr, q, kv = attention_qkv_fwd(
        x,
        wq_a_t,
        q_norm_w,
        wq_b_t,
        wkv_t,
        kv_norm_w,
        cos,
        sin,
        q_a,
        q_proj,
        kv_proj,
        kv_normed,
        qr,
        q,
        kv,
    )
    kv_cache_out = update_swa_decode_cache(kv_cache, kv, cache_pos, kv_cache_out)
    attn_o = sparse_attn_swa_fwd(q, kv_cache_out, attn_sink, topk_idxs, attn_o)
    out_final = attention_out_fwd(attn_o, wo_a_t, wo_b_t, cos, sin, o_inv, proj, out)
    return kv_cache_out, out_final


@pl.jit
def attention_swa_prefill_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
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
    out: pl.Out[pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16]],
):
    return attention_swa_prefill_fwd(
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
        out,
    )


@pl.jit
def attention_swa_decode_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    kv_cache: pl.Tensor[[B, WINDOW_SIZE, HEAD_DIM], pl.BF16],
    cache_pos: pl.Tensor[[1], pl.INT32],
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
    out: pl.Out[pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16]],
):
    return attention_swa_decode_fwd(
        x,
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


def golden_attention_swa_forward(tensors, start_pos: int):
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
    q_a = q
    q = q.float()
    q = (q * torch.rsqrt(q.square().mean(-1, keepdim=True) + EPS) * tensors["q_norm_w"].float()).to(torch.bfloat16)
    qr = q
    q = torch.matmul(q.float(), tensors["wq_b_t"].float()).to(torch.bfloat16)
    q_proj = q
    q = q.unflatten(-1, (N_HEADS, HEAD_DIM))
    q = (q.float() * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + EPS)).to(torch.bfloat16)
    q = _apply_rope_golden(q, tensors["cos"], tensors["sin"], inverse=False)

    # win kv
    kv = torch.matmul(x.float(), tensors["wkv_t"].float()).to(torch.bfloat16)
    kv_proj = kv
    kv = kv.float()
    kv = (kv * torch.rsqrt(kv.square().mean(-1, keepdim=True) + EPS) * tensors["kv_norm_w"].float()).to(torch.bfloat16)
    kv_normed = kv
    kv = _apply_rope_golden(kv, tensors["cos"], tensors["sin"], inverse=False)

    # FP8 act_quant on non-RoPE dims is intentionally removed in this bf16 path.
    if start_pos > 0:
        kv_cache_out = tensors["kv_cache"].clone()
        kv_cache_out[0, int(tensors["cache_pos"][0].item())] = kv[0, 0]
        attn_kv = kv_cache_out
    else:
        kv_cache_out = tensors["kv_cache_out"].clone()
        seqlen = kv.shape[1]
        if seqlen <= WINDOW_SIZE:
            kv_cache_out[:, :seqlen] = kv
        else:
            cutoff = seqlen % WINDOW_SIZE
            kv_cache_out[:, cutoff:WINDOW_SIZE], kv_cache_out[:, :cutoff] = kv[:, -WINDOW_SIZE:].split(
                [WINDOW_SIZE - cutoff, cutoff],
                dim=1,
            )
        attn_kv = kv

    attn_o = _golden_sparse_attn(q, attn_kv, tensors["attn_sink"], tensors["topk_idxs"])
    o_inv = _apply_rope_golden(attn_o, tensors["cos"], tensors["sin"], inverse=True)
    o = o_inv.view(B, x.shape[1], O_GROUPS, O_GROUP_IN)
    wo_a = tensors["wo_a_t"].transpose(0, 1).contiguous().view(O_GROUPS, O_LORA_RANK, O_GROUP_IN)
    proj = torch.einsum("bsgd,grd->bsgr", o.float(), wo_a.float()).to(torch.bfloat16)
    out = torch.matmul(proj.flatten(2).float(), tensors["wo_b_t"].float()).to(torch.bfloat16)

    tensors["q_a"][:] = q_a
    tensors["q_proj"][:] = q_proj
    tensors["kv_proj"][:] = kv_proj
    tensors["kv_normed"][:] = kv_normed
    tensors["qr"][:] = qr
    tensors["q"][:] = q
    tensors["kv"][:] = kv
    tensors["kv_cache_out"][:] = kv_cache_out
    tensors["attn_o"][:] = attn_o
    tensors["o_inv"][:] = o_inv
    tensors["proj"][:] = proj.flatten(2)
    tensors["out"][:] = out


def golden_attention_swa_prefill(tensors):
    golden_attention_swa_forward(tensors, start_pos=0)


def golden_attention_swa_decode(tensors, start_pos: int = DEFAULT_DECODE_START_POS):
    if start_pos <= 0:
        raise ValueError(f"decode start_pos must be positive, got {start_pos}")
    golden_attention_swa_forward(tensors, start_pos=start_pos)


def _common_specs(seq_len: int, start_pos: int, *, decode: bool):
    import torch

    from models.golden import TensorSpec

    freqs_cos, freqs_sin = build_deepseek_v4_rope_tables(max_seq_len=start_pos + seq_len)
    local_cos, local_sin = materialize_rope_range(freqs_cos, freqs_sin, start_pos, seq_len)

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

    x_spec = TensorSpec("x", [B, seq_len, HIDDEN], torch.bfloat16, init_value=init_x)
    weight_specs = [
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
    ]
    if decode:
        specs = [
            x_spec,
            TensorSpec(
                "kv_cache",
                [B, WINDOW_SIZE, HEAD_DIM],
                torch.bfloat16,
                init_value=lambda: torch.randn(B, WINDOW_SIZE, HEAD_DIM) * 0.1,
            ),
            TensorSpec("cache_pos", [1], torch.int32, init_value=torch.tensor([start_pos % WINDOW_SIZE], dtype=torch.int32)),
            *weight_specs,
        ]
    else:
        specs = [x_spec, *weight_specs]
    specs.extend(
        [
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
            TensorSpec("out", [B, seq_len, HIDDEN], torch.bfloat16, is_output=True),
        ]
    )
    return specs


def build_swa_prefill_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _common_specs(seq_len, start_pos=0, decode=False)


def build_swa_decode_specs(start_pos: int = DEFAULT_DECODE_START_POS):
    if start_pos <= 0:
        raise ValueError(f"decode start_pos must be positive, got {start_pos}")
    return _common_specs(1, start_pos=start_pos, decode=True)


def main() -> int:
    import argparse

    from models.golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 Flash SWA attention validation.")
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
        "out": ratio_allclose(atol=1e-3, rtol=2.0 / 128, max_error_ratio=0.005),
    }

    cases = []
    if args.case in ("all", "prefill"):
        cases.append(("swa-prefill", attention_swa_prefill_test, lambda: build_swa_prefill_specs(args.seq_len), golden_attention_swa_prefill))
    if args.case in ("all", "decode"):
        cases.append(
            (
                "swa-decode",
                attention_swa_decode_test,
                lambda: build_swa_decode_specs(args.decode_start_pos),
                lambda tensors: golden_attention_swa_forward(tensors, start_pos=args.decode_start_pos),
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
    "TOPK_SWA",
    "attention_swa_prefill_fwd",
    "attention_swa_decode_fwd",
    "attention_swa_prefill_test",
    "attention_swa_decode_test",
    "golden_attention_swa_forward",
    "golden_attention_swa_prefill",
    "golden_attention_swa_decode",
    "build_swa_prefill_specs",
    "build_swa_decode_specs",
]
