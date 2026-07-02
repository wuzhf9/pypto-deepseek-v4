"""DeepSeek V4 Flash MoE gate PyPTO kernels."""

import pypto.language as pl

from models.config import FLASH_CONFIG as M
from models.linear import linear_4096_to_256_fp32


B = 1
S_DYN = pl.dynamic("S_DYN")

HIDDEN = M.dim
N_EXPERTS = M.n_routed_experts
TOPK = M.n_activated_experts
VOCAB = M.vocab_size
ROUTE_SCALE = M.route_scale
DEFAULT_SEQ_LEN = 8

T_TILE = 16
# ``ALIGN_ROWS`` and ``TOPK_PAD`` keep temporary tile rows/columns at >=32B.
ALIGN_ROWS = 8
TOPK_PAD = 8
NEG_INF = -3.4028234663852886e38

assert TOPK <= TOPK_PAD


@pl.jit.inline
def _sqrt_softplus_scores(
    logits: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    scores: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
):
    logits.bind_dynamic(1, S_DYN)
    scores.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(logits, 1)
    logits_flat = pl.reshape(logits, [tokens, N_EXPERTS])
    scores_flat = pl.reshape(scores, [tokens, N_EXPERTS])
    token_blocks = (tokens + T_TILE - 1) // T_TILE

    for tb in pl.range(token_blocks):
        t0 = tb * T_TILE
        valid_tok = pl.min(T_TILE, tokens - t0)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_sqrt_softplus"):
            logits_tile = pl.slice(
                logits_flat,
                [T_TILE, N_EXPERTS],
                [t0, 0],
                valid_shape=[valid_tok, N_EXPERTS],
            )
            relu = pl.maximum(logits_tile, 0.0)
            abs_logits = pl.maximum(logits_tile, pl.neg(logits_tile))
            softplus = pl.add(relu, pl.log(pl.add(pl.exp(pl.neg(abs_logits)), 1.0)))
            score_tile = pl.sqrt(softplus)
            # Dynamic tail rows must be materialized before taking row subviews.
            score_tile = pl.fillpad(score_tile, pad_value=pl.PadValue.zero)
            for row in pl.range(valid_tok):
                score_row = pl.slice(score_tile, [1, N_EXPERTS], [row, 0])
                scores_flat = pl.assemble(scores_flat, score_row, [t0 + row, 0])

    return pl.reshape(scores_flat, [B, tokens, N_EXPERTS])


@pl.jit.inline
def gate_hash_fwd(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    gate_w_t: pl.Tensor[[HIDDEN, N_EXPERTS], pl.BF16],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[B, S_DYN], pl.INT64],
    logits: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    scores: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    indices: pl.Tensor[[B, S_DYN, TOPK], pl.INT32],
    weights: pl.Tensor[[B, S_DYN, TOPK], pl.FP32],
):
    """Run official ``Gate.forward`` hash-routing path."""
    x.bind_dynamic(1, S_DYN)
    input_ids.bind_dynamic(1, S_DYN)
    logits.bind_dynamic(1, S_DYN)
    scores.bind_dynamic(1, S_DYN)
    indices.bind_dynamic(1, S_DYN)
    weights.bind_dynamic(1, S_DYN)

    logits = linear_4096_to_256_fp32(x, gate_w_t, logits)
    scores = _sqrt_softplus_scores(logits, scores)

    tokens = pl.tensor.dim(x, 1)
    input_flat = pl.reshape(input_ids, [tokens])
    scores_flat = pl.reshape(scores, [tokens, N_EXPERTS])
    indices_flat = pl.reshape(indices, [tokens, TOPK])
    weights_flat = pl.reshape(weights, [tokens, TOPK])

    for t in pl.range(tokens):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_hash_route"):
            token_id = pl.cast(pl.read(input_flat, [t]), pl.INDEX)
            picked_scores = pl.full([ALIGN_ROWS, TOPK_PAD], dtype=pl.FP32, value=0.0)
            for k in pl.range(TOPK):
                eid = pl.read(tid2eid, [token_id, k])
                eid_idx = pl.cast(eid, pl.INDEX)
                score = pl.read(scores_flat, [t, eid_idx])
                pl.write(picked_scores, [0, k], score)
                pl.write(indices_flat, [t, k], eid)
            denom_tile = pl.row_sum(picked_scores)
            denom = pl.read(denom_tile, [0, 0])
            for k in pl.range(TOPK):
                weight = pl.mul(pl.div(pl.read(picked_scores, [0, k]), denom), ROUTE_SCALE)
                pl.write(weights_flat, [t, k], weight)

    return indices, weights


@pl.jit.inline
def gate_topk_fwd(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    gate_w_t: pl.Tensor[[HIDDEN, N_EXPERTS], pl.BF16],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    logits: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    scores: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    indices: pl.Tensor[[B, S_DYN, TOPK], pl.INT32],
    weights: pl.Tensor[[B, S_DYN, TOPK], pl.FP32],
):
    """Run official ``Gate.forward`` score-routing path."""
    x.bind_dynamic(1, S_DYN)
    logits.bind_dynamic(1, S_DYN)
    scores.bind_dynamic(1, S_DYN)
    indices.bind_dynamic(1, S_DYN)
    weights.bind_dynamic(1, S_DYN)

    logits = linear_4096_to_256_fp32(x, gate_w_t, logits)
    scores = _sqrt_softplus_scores(logits, scores)

    tokens = pl.tensor.dim(x, 1)
    scores_flat = pl.reshape(scores, [tokens, N_EXPERTS])
    indices_flat = pl.reshape(indices, [tokens, TOPK])
    weights_flat = pl.reshape(weights, [tokens, TOPK])

    for t in pl.range(tokens):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_topk_route"):
            score_row = scores_flat[t : t + 1, 0:N_EXPERTS]
            bias_row = pl.reshape(gate_bias, [1, N_EXPERTS])
            score_work = pl.full([ALIGN_ROWS, N_EXPERTS], dtype=pl.FP32, value=NEG_INF)
            score_work[0:1, 0:N_EXPERTS] = pl.add(score_row, bias_row)
            pos_i32 = pl.arange(0, [1, N_EXPERTS], dtype=pl.INT32)
            neg_inf_row = pl.full([1, N_EXPERTS], dtype=pl.FP32, value=NEG_INF)
            picked_scores = pl.full([ALIGN_ROWS, TOPK_PAD], dtype=pl.FP32, value=0.0)

            for k in pl.range(TOPK):
                best_pos_tile = pl.row_argmax(score_work)
                best_pos_i32 = pl.read(best_pos_tile, [0, 0])
                best_pos_idx = pl.cast(best_pos_i32, pl.INDEX)
                picked_score = pl.read(scores_flat, [t, best_pos_idx])
                pl.write(picked_scores, [0, k], picked_score)
                pl.write(indices_flat, [t, k], best_pos_i32)

                selected_i32 = pl.cmp(pos_i32, best_pos_i32, cmp_type=0)
                selected = pl.cast(selected_i32, target_type=pl.FP32)
                keep = pl.sub(pl.full([1, N_EXPERTS], dtype=pl.FP32, value=1.0), selected)
                score_work_row = score_work[0:1, 0:N_EXPERTS]
                masked_score = pl.add(pl.mul(score_work_row, keep), pl.mul(neg_inf_row, selected))
                score_work[0:1, 0:N_EXPERTS] = masked_score

            denom_tile = pl.row_sum(picked_scores)
            denom = pl.read(denom_tile, [0, 0])
            for k in pl.range(TOPK):
                weight = pl.mul(pl.div(pl.read(picked_scores, [0, k]), denom), ROUTE_SCALE)
                pl.write(weights_flat, [t, k], weight)

    return indices, weights


@pl.jit
def gate_hash_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    gate_w_t: pl.Tensor[[HIDDEN, N_EXPERTS], pl.BF16],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[B, S_DYN], pl.INT64],
    logits: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    scores: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    indices: pl.Out[pl.Tensor[[B, S_DYN, TOPK], pl.INT32]],
    weights: pl.Out[pl.Tensor[[B, S_DYN, TOPK], pl.FP32]],
):
    indices, weights = gate_hash_fwd(x, gate_w_t, tid2eid, input_ids, logits, scores, indices, weights)
    return indices, weights


@pl.jit
def gate_topk_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    gate_w_t: pl.Tensor[[HIDDEN, N_EXPERTS], pl.BF16],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    logits: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    scores: pl.Tensor[[B, S_DYN, N_EXPERTS], pl.FP32],
    indices: pl.Out[pl.Tensor[[B, S_DYN, TOPK], pl.INT32]],
    weights: pl.Out[pl.Tensor[[B, S_DYN, TOPK], pl.FP32]],
):
    indices, weights = gate_topk_fwd(x, gate_w_t, gate_bias, logits, scores, indices, weights)
    return indices, weights


def golden_gate_forward(tensors, *, hash_route: bool):
    import torch
    import torch.nn.functional as F

    x = tensors["x"]
    logits = torch.matmul(x.float(), tensors["gate_w_t"].float())
    scores = torch.sqrt(F.softplus(logits))

    if hash_route:
        indices = tensors["tid2eid"][tensors["input_ids"].long()]
    else:
        biased_scores = scores + tensors["gate_bias"].view(1, 1, N_EXPERTS)
        indices = biased_scores.topk(TOPK, dim=-1)[1]

    weights = scores.gather(-1, indices.long())
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * ROUTE_SCALE

    tensors["indices"][:] = indices.to(torch.int32)
    tensors["weights"][:] = weights


def golden_gate_hash(tensors):
    golden_gate_forward(tensors, hash_route=True)


def golden_gate_topk(tensors):
    golden_gate_forward(tensors, hash_route=False)


def _build_tensor_specs(seq_len: int, *, hash_route: bool):
    import torch

    from models.golden import TensorSpec

    def init_x():
        return torch.randn(B, seq_len, HIDDEN) * 0.2

    def init_gate_w_t():
        return torch.randn(HIDDEN, N_EXPERTS) * 0.02

    def init_gate_bias():
        return torch.randn(N_EXPERTS) * 0.02

    def init_input_ids():
        return torch.randint(0, VOCAB, (B, seq_len), dtype=torch.int64)

    def init_tid2eid():
        base = torch.arange(TOPK, dtype=torch.int32).view(1, TOPK)
        token_offsets = torch.arange(VOCAB, dtype=torch.int32).view(VOCAB, 1)
        return (base + token_offsets) % N_EXPERTS

    specs = [
        TensorSpec("x", [B, seq_len, HIDDEN], torch.bfloat16, init_value=init_x),
        TensorSpec("gate_w_t", [HIDDEN, N_EXPERTS], torch.bfloat16, init_value=init_gate_w_t),
    ]
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
            TensorSpec("logits", [B, seq_len, N_EXPERTS], torch.float32),
            TensorSpec("scores", [B, seq_len, N_EXPERTS], torch.float32),
            TensorSpec("indices", [B, seq_len, TOPK], torch.int32, is_output=True),
            TensorSpec("weights", [B, seq_len, TOPK], torch.float32, is_output=True),
        ]
    )
    return specs


def build_gate_hash_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _build_tensor_specs(seq_len, hash_route=True)


def build_gate_topk_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _build_tensor_specs(seq_len, hash_route=False)


def main() -> int:
    import argparse

    from models.golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 Flash MoE gate validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3sim", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-s", "--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    runtime_cfg = {
        "platform": args.platform,
        "device_id": args.device,
        "enable_l2_swimlane": args.enable_l2_swimlane,
    }
    compare_fn = {
        "weights": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
    }
    cases = [
        ("gate-hash", gate_hash_test, build_gate_hash_specs, golden_gate_hash),
        ("gate-topk", gate_topk_test, build_gate_topk_specs, golden_gate_topk),
    ]

    failed = False
    for name, fn, build_specs, golden_fn in cases:
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
    "HIDDEN",
    "N_EXPERTS",
    "TOPK",
    "VOCAB",
    "ROUTE_SCALE",
    "ALIGN_ROWS",
    "TOPK_PAD",
    "DEFAULT_SEQ_LEN",
    "gate_hash_fwd",
    "gate_topk_fwd",
    "gate_hash_test",
    "gate_topk_test",
    "golden_gate_forward",
    "golden_gate_hash",
    "golden_gate_topk",
    "build_gate_hash_specs",
    "build_gate_topk_specs",
]
