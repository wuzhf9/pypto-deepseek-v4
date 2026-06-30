"""DeepSeek V4 Flash bf16 sparse attention PyPTO kernels."""

import pypto.language as pl

from models.config import FLASH_CONFIG as M


B = 1
S_DYN = pl.dynamic("S_DYN")
K_DYN = pl.dynamic("K_DYN")

N_HEADS = M.n_heads
HEAD_DIM = M.head_dim
WINDOW_SIZE = M.window_size
TOPK_SWA = WINDOW_SIZE
SOFTMAX_SCALE = HEAD_DIM**-0.5
NEG_INF = -3.4028234663852886e38

H_TILE = 16
DEFAULT_SEQ_LEN = 8
DEFAULT_DECODE_START_POS = 1


@pl.jit.inline
def sparse_attn_swa_fwd(
    q: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[B, K_DYN, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[N_HEADS], pl.FP32],
    topk_idxs: pl.Tensor[[B, S_DYN, TOPK_SWA], pl.INT32],
    out: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
):
    """SWA sparse attention with ``TOPK_MAX=128``."""
    q.bind_dynamic(1, S_DYN)
    kv.bind_dynamic(1, K_DYN)
    topk_idxs.bind_dynamic(1, S_DYN)
    out.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(q, 1)
    kv_len = pl.tensor.dim(kv, 1)
    q_flat = pl.reshape(q, [tokens * N_HEADS, HEAD_DIM])
    kv_flat = pl.reshape(kv, [kv_len, HEAD_DIM])
    topk_flat = pl.reshape(topk_idxs, [tokens, TOPK_SWA])
    out_flat = pl.reshape(out, [tokens * N_HEADS, HEAD_DIM])

    for t in pl.range(tokens):
        sparse_kv = pl.create_tensor([TOPK_SWA, HEAD_DIM], dtype=pl.BF16)
        sparse_bias = pl.create_tensor([1, TOPK_SWA], dtype=pl.FP32)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="sparse_attn_gather"):
            sparse_kv[0:TOPK_SWA, 0:HEAD_DIM] = pl.full([TOPK_SWA, HEAD_DIM], dtype=pl.BF16, value=0.0)
            topk_row = topk_flat[t : t + 1, 0:TOPK_SWA]
            topk_fp32 = pl.cast(topk_row, target_type=pl.FP32)
            valid_flag = pl.minimum(pl.maximum(pl.add(topk_fp32, 1.0), 0.0), 1.0)
            sparse_bias[0:1, 0:TOPK_SWA] = pl.mul(pl.sub(valid_flag, 1.0), -NEG_INF)

            for ki in pl.range(TOPK_SWA):
                raw = pl.read(topk_flat, [t, ki])
                if raw >= 0:
                    if raw < kv_len:
                        src = pl.cast(raw, pl.INDEX)
                        sparse_kv[ki : ki + 1, 0:HEAD_DIM] = kv_flat[src : src + 1, 0:HEAD_DIM]

        for hb in pl.spmd(N_HEADS // H_TILE, name_hint="sparse_attn_topk128"):
            h0 = hb * H_TILE
            out_row = t * N_HEADS + h0

            q_tile = q_flat[out_row : out_row + H_TILE, 0:HEAD_DIM]
            qk_raw = pl.matmul(q_tile, sparse_kv, b_trans=True, out_dtype=pl.FP32)
            qk_scaled = pl.mul(qk_raw, SOFTMAX_SCALE)
            qk_scores = pl.add(
                qk_scaled,
                pl.col_expand(pl.full([H_TILE, TOPK_SWA], dtype=pl.FP32, value=0.0), sparse_bias),
            )
            qk_mi = pl.row_max(qk_scores)
            qk_exp = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))
            qk_li = pl.row_sum(qk_exp)
            qk_exp_bf16 = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
            qk_oi = pl.matmul(qk_exp_bf16, sparse_kv, out_dtype=pl.FP32)

            sink_bias = pl.reshape(attn_sink[h0 : h0 + H_TILE], [H_TILE, 1])
            denom = pl.add(qk_li, pl.exp(pl.sub(sink_bias, qk_mi)))
            out_fp32 = pl.row_expand_div(qk_oi, denom)
            out_bf16 = pl.cast(out_fp32, target_type=pl.BF16, mode="rint")

            out_flat[out_row : out_row + H_TILE, 0:HEAD_DIM] = out_bf16

    return pl.reshape(out_flat, [B, tokens, N_HEADS, HEAD_DIM])


@pl.jit
def sparse_attn_swa_test(
    q: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[B, K_DYN, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[N_HEADS], pl.FP32],
    topk_idxs: pl.Tensor[[B, S_DYN, TOPK_SWA], pl.INT32],
    out: pl.Out[pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16]],
):
    out = sparse_attn_swa_fwd(q, kv, attn_sink, topk_idxs, out)
    return out


def golden_sparse_attn(tensors):
    """Torch reference matching official ``low_vram_kernels.sparse_attn_torch``."""
    import torch

    q = tensors["q"]
    kv = tensors["kv"]
    attn_sink = tensors["attn_sink"]
    topk_idxs = tensors["topk_idxs"]

    bsz, seqlen, n_heads, _ = q.shape
    out = torch.zeros_like(q)
    for batch_id in range(bsz):
        for seq_id in range(seqlen):
            idxs = topk_idxs[batch_id, seq_id]
            idxs = idxs[idxs >= 0].long()
            if idxs.numel() == 0:
                continue

            selected = kv[batch_id, idxs].float()
            scores = torch.einsum("hd,td->ht", q[batch_id, seq_id].float(), selected) * SOFTMAX_SCALE
            scores = torch.cat([scores, attn_sink.float().view(n_heads, 1)], dim=1)
            probs = torch.softmax(scores, dim=1)[:, :-1]
            out[batch_id, seq_id] = torch.einsum("ht,td->hd", probs, selected).to(q.dtype)

    tensors["out"][:] = out


def build_window_topk_idxs(seq_len: int, start_pos: int = 0, topk_max: int = TOPK_SWA):
    import torch

    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if start_pos < 0:
        raise ValueError(f"start_pos must be non-negative, got {start_pos}")
    if start_pos > 0 and seq_len != 1:
        raise ValueError(f"decode-style window topk expects seq_len=1, got {seq_len}")
    if topk_max < WINDOW_SIZE:
        raise ValueError(f"topk_max must be at least {WINDOW_SIZE}, got {topk_max}")

    topk = torch.full((B, seq_len, topk_max), -1, dtype=torch.int32)
    if start_pos >= WINDOW_SIZE - 1:
        pos = start_pos % WINDOW_SIZE
        idxs = torch.cat(
            [
                torch.arange(pos + 1, WINDOW_SIZE, dtype=torch.int32),
                torch.arange(0, pos + 1, dtype=torch.int32),
            ]
        )
        topk[0, 0, : idxs.numel()] = idxs
    elif start_pos > 0:
        idxs = torch.arange(0, start_pos + 1, dtype=torch.int32)
        topk[0, 0, : idxs.numel()] = idxs
    else:
        for t in range(seq_len):
            start = max(0, t - WINDOW_SIZE + 1)
            idxs = torch.arange(start, t + 1, dtype=torch.int32)
            topk[0, t, : idxs.numel()] = idxs
    return topk


def _build_tensor_specs(seq_len: int, kv_len: int, topk_max: int, topk_init):
    import torch

    from models.golden import TensorSpec

    def init_q():
        return torch.randn(B, seq_len, N_HEADS, HEAD_DIM) * 0.1

    def init_kv():
        return torch.randn(B, kv_len, HEAD_DIM) * 0.1

    def init_attn_sink():
        return torch.randn(N_HEADS) * 0.1

    return [
        TensorSpec("q", [B, seq_len, N_HEADS, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("kv", [B, kv_len, HEAD_DIM], torch.bfloat16, init_value=init_kv),
        TensorSpec("attn_sink", [N_HEADS], torch.float32, init_value=init_attn_sink),
        TensorSpec("topk_idxs", [B, seq_len, topk_max], torch.int32, init_value=topk_init),
        TensorSpec("out", [B, seq_len, N_HEADS, HEAD_DIM], torch.bfloat16, is_output=True),
    ]


def build_swa_prefill_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _build_tensor_specs(
        seq_len,
        seq_len,
        TOPK_SWA,
        lambda: build_window_topk_idxs(seq_len, start_pos=0, topk_max=TOPK_SWA),
    )


def build_swa_decode_specs(start_pos: int = DEFAULT_DECODE_START_POS):
    return _build_tensor_specs(
        1,
        WINDOW_SIZE,
        TOPK_SWA,
        lambda: build_window_topk_idxs(1, start_pos=start_pos, topk_max=TOPK_SWA),
    )


def main() -> int:
    import argparse

    from models.golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 Flash bf16 sparse attention validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3sim", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-s", "--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--decode-start-pos", type=int, default=DEFAULT_DECODE_START_POS)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    cases = [
        ("swa-prefill", sparse_attn_swa_test, lambda: build_swa_prefill_specs(args.seq_len)),
        ("swa-decode", sparse_attn_swa_test, lambda: build_swa_decode_specs(args.decode_start_pos)),
    ]
    runtime_cfg = {
        "platform": args.platform,
        "device_id": args.device,
        "enable_l2_swimlane": args.enable_l2_swimlane,
    }
    compare_fn = {
        "out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
    }

    failed = False
    for name, fn, build_specs in cases:
        print(f"[CASE] {name}", flush=True)
        result = run_jit(
            fn=fn,
            specs=build_specs(),
            golden_fn=golden_sparse_attn,
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
    "K_DYN",
    "N_HEADS",
    "HEAD_DIM",
    "WINDOW_SIZE",
    "TOPK_SWA",
    "SOFTMAX_SCALE",
    "NEG_INF",
    "H_TILE",
    "DEFAULT_SEQ_LEN",
    "sparse_attn_swa_fwd",
    "sparse_attn_swa_test",
    "golden_sparse_attn",
    "build_window_topk_idxs",
    "build_swa_prefill_specs",
    "build_swa_decode_specs",
]
