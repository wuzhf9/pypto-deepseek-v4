"""DeepSeek V4 Flash attention Q/KV projection and RoPE PyPTO kernel."""

import pypto.language as pl

from models.config import FLASH_CONFIG as M
from models.linear import (
    linear_1024_to_32768,
    linear_4096_to_1024,
    linear_4096_to_512,
)
from models.rmsnorm import rmsnorm_1024, rmsnorm_512
from models.rope import (
    _apply_rope_golden,
    build_deepseek_v4_rope_tables,
    materialize_rope_range,
    rope_3d_512_fwd,
    rope_4d_512_fwd,
)


B = 1
S_DYN = pl.dynamic("S_DYN")

HIDDEN = M.dim
Q_LORA_RANK = M.q_lora_rank
N_HEADS = M.n_heads
HEAD_DIM = M.head_dim
ATTN_Q_OUT = N_HEADS * HEAD_DIM
ROPE_HALF = M.rope_head_dim // 2
EPS = M.rms_norm_eps
INV_HEAD_DIM = 1.0 / HEAD_DIM

Q_HEAD_T_TILE = 8
Q_HEAD_D_TILE = 128
Q_HEAD_D_BLOCKS = HEAD_DIM // Q_HEAD_D_TILE
DEFAULT_SEQ_LEN = 8


@pl.jit.inline
def q_head_rms_scale(
    q: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    out: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
):
    """Apply official ``q *= rsqrt(mean(q^2) + eps)`` per attention head."""
    q.bind_dynamic(1, S_DYN)
    out.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(q, 1)
    q_flat = pl.reshape(q, [tokens, N_HEADS * HEAD_DIM])
    out_flat = pl.reshape(out, [tokens, N_HEADS * HEAD_DIM])
    token_blocks = (tokens + Q_HEAD_T_TILE - 1) // Q_HEAD_T_TILE

    for tb in pl.range(token_blocks):
        t0 = tb * Q_HEAD_T_TILE
        valid_tok = pl.min(Q_HEAD_T_TILE, tokens - t0)

        for h in pl.spmd(N_HEADS, name_hint="q_head_rms_scale"):
            h0 = h * HEAD_DIM

            partial_sq = pl.full([1, Q_HEAD_T_TILE], dtype=pl.FP32, value=0.0)
            for db in pl.range(Q_HEAD_D_BLOCKS):
                d0 = h0 + db * Q_HEAD_D_TILE
                q_bf16 = pl.slice(
                    q_flat,
                    [Q_HEAD_T_TILE, Q_HEAD_D_TILE],
                    [t0, d0],
                    valid_shape=[valid_tok, Q_HEAD_D_TILE],
                )
                q_fp32 = pl.cast(q_bf16, target_type=pl.FP32)
                partial_sq = pl.add(
                    partial_sq,
                    pl.reshape(pl.row_sum(pl.mul(q_fp32, q_fp32)), [1, Q_HEAD_T_TILE]),
                )

            variance = pl.reshape(pl.add(pl.mul(partial_sq, INV_HEAD_DIM), EPS), [Q_HEAD_T_TILE, 1])
            inv_rms = pl.recip(pl.sqrt(variance))

            for db in pl.range(Q_HEAD_D_BLOCKS):
                d0 = h0 + db * Q_HEAD_D_TILE
                q_bf16 = pl.slice(
                    q_flat,
                    [Q_HEAD_T_TILE, Q_HEAD_D_TILE],
                    [t0, d0],
                    valid_shape=[valid_tok, Q_HEAD_D_TILE],
                )
                q_fp32 = pl.cast(q_bf16, target_type=pl.FP32)
                scaled = pl.row_expand_mul(q_fp32, inv_rms)
                scaled_bf16 = pl.cast(scaled, target_type=pl.BF16, mode="rint")
                for row in pl.range(valid_tok):
                    out_row = pl.slice(
                        scaled_bf16,
                        [1, Q_HEAD_D_TILE],
                        [row, 0],
                        valid_shape=[1, Q_HEAD_D_TILE],
                    )
                    out_flat = pl.assemble(out_flat, out_row, [t0 + row, d0])

    return pl.reshape(out_flat, [B, tokens, N_HEADS, HEAD_DIM])


@pl.jit.inline
def attention_qkv_fwd(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    wq_a_t: pl.Tensor[[HIDDEN, Q_LORA_RANK], pl.BF16],
    q_norm_w: pl.Tensor[[Q_LORA_RANK], pl.BF16],
    wq_b_t: pl.Tensor[[Q_LORA_RANK, ATTN_Q_OUT], pl.BF16],
    wkv_t: pl.Tensor[[HIDDEN, HEAD_DIM], pl.BF16],
    kv_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    q_a: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    q_proj: pl.Tensor[[B, S_DYN, ATTN_Q_OUT], pl.BF16],
    q_scaled: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    kv_proj: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    kv_normed: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16],
    q: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16],
):
    """Compute official attention q/kv projection, q scale, and RoPE."""
    x.bind_dynamic(1, S_DYN)
    cos.bind_dynamic(0, S_DYN)
    sin.bind_dynamic(0, S_DYN)
    q_a.bind_dynamic(1, S_DYN)
    q_proj.bind_dynamic(1, S_DYN)
    q_scaled.bind_dynamic(1, S_DYN)
    kv_proj.bind_dynamic(1, S_DYN)
    kv_normed.bind_dynamic(1, S_DYN)
    qr.bind_dynamic(1, S_DYN)
    q.bind_dynamic(1, S_DYN)
    kv.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(x, 1)
    q_a = linear_4096_to_1024(x, wq_a_t, q_a)
    qr = rmsnorm_1024(q_a, q_norm_w, qr)
    q_proj = linear_1024_to_32768(qr, wq_b_t, q_proj)
    q_unflat = pl.reshape(q_proj, [B, tokens, N_HEADS, HEAD_DIM])
    q_scaled = q_head_rms_scale(q_unflat, q_scaled)
    q = rope_4d_512_fwd(q_scaled, cos, sin, q)

    kv_proj = linear_4096_to_512(x, wkv_t, kv_proj)
    kv_normed = rmsnorm_512(kv_proj, kv_norm_w, kv_normed)
    kv = rope_3d_512_fwd(kv_normed, cos, sin, kv)

    return q_a, q_proj, q_scaled, kv_proj, kv_normed, qr, q, kv


@pl.jit
def attention_qkv_fwd_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    wq_a_t: pl.Tensor[[HIDDEN, Q_LORA_RANK], pl.BF16],
    q_norm_w: pl.Tensor[[Q_LORA_RANK], pl.BF16],
    wq_b_t: pl.Tensor[[Q_LORA_RANK, ATTN_Q_OUT], pl.BF16],
    wkv_t: pl.Tensor[[HIDDEN, HEAD_DIM], pl.BF16],
    kv_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cos: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    q_a: pl.Out[pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16]],
    q_proj: pl.Out[pl.Tensor[[B, S_DYN, ATTN_Q_OUT], pl.BF16]],
    q_scaled: pl.Out[pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16]],
    kv_proj: pl.Out[pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16]],
    kv_normed: pl.Out[pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16]],
    qr: pl.Out[pl.Tensor[[B, S_DYN, Q_LORA_RANK], pl.BF16]],
    q: pl.Out[pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16]],
    kv: pl.Out[pl.Tensor[[B, S_DYN, HEAD_DIM], pl.BF16]],
):
    q_a, q_proj, q_scaled, kv_proj, kv_normed, qr, q, kv = attention_qkv_fwd(
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
        q_scaled,
        kv_proj,
        kv_normed,
        qr,
        q,
        kv,
    )
    return q_a, q_proj, q_scaled, kv_proj, kv_normed, qr, q, kv


def golden_attention_qkv(tensors):
    import torch

    x = tensors["x"]

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
    q_scaled = q
    q = _apply_rope_golden(q, tensors["cos"], tensors["sin"], inverse=False)

    # win kv
    kv = torch.matmul(x.float(), tensors["wkv_t"].float()).to(torch.bfloat16)
    kv_proj = kv
    kv = kv.float()
    kv = (kv * torch.rsqrt(kv.square().mean(-1, keepdim=True) + EPS) * tensors["kv_norm_w"].float()).to(torch.bfloat16)
    kv_normed = kv
    kv = _apply_rope_golden(kv, tensors["cos"], tensors["sin"], inverse=False)

    tensors["q_a"][:] = q_a
    tensors["q_proj"][:] = q_proj
    tensors["q_scaled"][:] = q_scaled
    tensors["kv_proj"][:] = kv_proj
    tensors["kv_normed"][:] = kv_normed
    tensors["qr"][:] = qr
    tensors["q"][:] = q
    tensors["kv"][:] = kv


def build_attention_qkv_specs(seq_len: int = DEFAULT_SEQ_LEN, start_pos: int = 0):
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

    return [
        TensorSpec("x", [B, seq_len, HIDDEN], torch.bfloat16, init_value=init_x),
        TensorSpec("wq_a_t", [HIDDEN, Q_LORA_RANK], torch.bfloat16, init_value=init_wq_a_t),
        TensorSpec("q_norm_w", [Q_LORA_RANK], torch.bfloat16, init_value=init_q_norm_w),
        TensorSpec("wq_b_t", [Q_LORA_RANK, ATTN_Q_OUT], torch.bfloat16, init_value=init_wq_b_t),
        TensorSpec("wkv_t", [HIDDEN, HEAD_DIM], torch.bfloat16, init_value=init_wkv_t),
        TensorSpec("kv_norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_kv_norm_w),
        TensorSpec("cos", [seq_len, ROPE_HALF], torch.float32, init_value=local_cos),
        TensorSpec("sin", [seq_len, ROPE_HALF], torch.float32, init_value=local_sin),
        TensorSpec("q_a", [B, seq_len, Q_LORA_RANK], torch.bfloat16, is_output=True),
        TensorSpec("q_proj", [B, seq_len, ATTN_Q_OUT], torch.bfloat16, is_output=True),
        TensorSpec("q_scaled", [B, seq_len, N_HEADS, HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec("kv_proj", [B, seq_len, HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec("kv_normed", [B, seq_len, HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec("qr", [B, seq_len, Q_LORA_RANK], torch.bfloat16, is_output=True),
        TensorSpec("q", [B, seq_len, N_HEADS, HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec("kv", [B, seq_len, HEAD_DIM], torch.bfloat16, is_output=True),
    ]


def main() -> int:
    import argparse

    from models.golden import ignore_output, ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 Flash attention QKV validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3sim", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-s", "--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--start-pos", type=int, default=7)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    runtime_cfg = {
        "platform": args.platform,
        "device_id": args.device,
        "enable_l2_swimlane": args.enable_l2_swimlane,
    }
    compare_fn = {
        "q_a": ignore_output,
        "q_proj": ignore_output,
        "q_scaled": ignore_output,
        "kv_proj": ignore_output,
        "kv_normed": ignore_output,
        "qr": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "q": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
        "kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
    }

    result = run_jit(
        fn=attention_qkv_fwd_test,
        specs=build_attention_qkv_specs(args.seq_len, args.start_pos),
        golden_fn=golden_attention_qkv,
        runtime_cfg=runtime_cfg,
        compile_only=args.compile_only,
        compare_fn=compare_fn,
    )
    if not result.passed and result.error:
        print(result.error)
    return 0 if result.passed else 1


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
    "ROPE_HALF",
    "EPS",
    "INV_HEAD_DIM",
    "Q_HEAD_T_TILE",
    "Q_HEAD_D_TILE",
    "Q_HEAD_D_BLOCKS",
    "DEFAULT_SEQ_LEN",
    "q_head_rms_scale",
    "attention_qkv_fwd",
    "attention_qkv_fwd_test",
    "golden_attention_qkv",
    "build_attention_qkv_specs",
]
