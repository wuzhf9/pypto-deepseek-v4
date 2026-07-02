"""DeepSeek V4 Flash MoE expert PyPTO kernels."""

import pypto.language as pl

from models.config import FLASH_CONFIG as M
from models.linear import linear_2048_to_4096, linear_4096_to_2048


B = 1
S_DYN = pl.dynamic("S_DYN")

HIDDEN = M.dim
MOE_INTER_DIM = M.moe_inter_dim
SWIGLU_LIMIT = M.swiglu_limit
T_TILE = 16
O_TILE = 32
MOE_INTER_O_BLOCKS = MOE_INTER_DIM // O_TILE
DEFAULT_SEQ_LEN = 8


@pl.jit.inline
def expert_shared_fwd(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    w1_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    w2_t: pl.Tensor[[MOE_INTER_DIM, HIDDEN], pl.BF16],
    w3_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    gate: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    up: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    hidden: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
):
    """Run ``Expert.forward(x, weights=None)``."""
    x.bind_dynamic(1, S_DYN)
    gate.bind_dynamic(1, S_DYN)
    up.bind_dynamic(1, S_DYN)
    hidden.bind_dynamic(1, S_DYN)
    out.bind_dynamic(1, S_DYN)

    gate = linear_4096_to_2048(x, w1_t, gate)
    up = linear_4096_to_2048(x, w3_t, up)

    tokens = pl.tensor.dim(x, 1)
    gate_flat = pl.reshape(gate, [tokens, MOE_INTER_DIM])
    up_flat = pl.reshape(up, [tokens, MOE_INTER_DIM])
    hidden_flat = pl.reshape(hidden, [tokens, MOE_INTER_DIM])
    token_blocks = (tokens + T_TILE - 1) // T_TILE

    for tb in pl.range(token_blocks):
        t0 = tb * T_TILE
        valid_tok = pl.min(T_TILE, tokens - t0)
        hidden_tile_full = pl.create_tensor([T_TILE, MOE_INTER_DIM], dtype=pl.BF16)

        for ob in pl.spmd(MOE_INTER_O_BLOCKS, name_hint="expert_shared_swiglu"):
            o0 = ob * O_TILE
            gate_tile = pl.cast(
                pl.slice(gate_flat, [T_TILE, O_TILE], [t0, o0], valid_shape=[valid_tok, O_TILE]),
                target_type=pl.FP32,
            )
            up_tile = pl.cast(
                pl.slice(up_flat, [T_TILE, O_TILE], [t0, o0], valid_shape=[valid_tok, O_TILE]),
                target_type=pl.FP32,
            )
            limit = pl.full([T_TILE, O_TILE], dtype=pl.FP32, value=SWIGLU_LIMIT)
            neg_limit = pl.full([T_TILE, O_TILE], dtype=pl.FP32, value=-SWIGLU_LIMIT)
            gate_clamped = pl.minimum(gate_tile, limit)
            up_clamped = pl.minimum(pl.maximum(up_tile, neg_limit), limit)
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_clamped)), 1.0))
            hidden_fp32 = pl.mul(pl.mul(gate_clamped, sigmoid), up_clamped)
            hidden_tile = pl.cast(hidden_fp32, target_type=pl.BF16, mode="rint")
            hidden_tile_full[:, o0 : o0 + O_TILE] = pl.fillpad(hidden_tile, pad_value=pl.PadValue.zero)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="expert_shared_hidden_write"):
            for ob in pl.range(MOE_INTER_O_BLOCKS):
                o0 = ob * O_TILE
                for row in pl.range(valid_tok):
                    hidden_row = pl.slice(hidden_tile_full, [1, O_TILE], [row, o0])
                    hidden_flat = pl.assemble(hidden_flat, hidden_row, [t0 + row, o0])

    out = linear_2048_to_4096(hidden, w2_t, out)
    return out


@pl.jit.inline
def expert_routed_fwd(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    weights: pl.Tensor[[B, S_DYN, 1], pl.FP32],
    w1_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    w2_t: pl.Tensor[[MOE_INTER_DIM, HIDDEN], pl.BF16],
    w3_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    gate: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    up: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    hidden: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
):
    """Run ``Expert.forward(x, weights=weights)``."""
    x.bind_dynamic(1, S_DYN)
    weights.bind_dynamic(1, S_DYN)
    gate.bind_dynamic(1, S_DYN)
    up.bind_dynamic(1, S_DYN)
    hidden.bind_dynamic(1, S_DYN)
    out.bind_dynamic(1, S_DYN)

    gate = linear_4096_to_2048(x, w1_t, gate)
    up = linear_4096_to_2048(x, w3_t, up)

    tokens = pl.tensor.dim(x, 1)
    weights_flat = pl.reshape(weights, [tokens, 1])
    gate_flat = pl.reshape(gate, [tokens, MOE_INTER_DIM])
    up_flat = pl.reshape(up, [tokens, MOE_INTER_DIM])
    hidden_flat = pl.reshape(hidden, [tokens, MOE_INTER_DIM])
    token_blocks = (tokens + T_TILE - 1) // T_TILE

    for tb in pl.range(token_blocks):
        t0 = tb * T_TILE
        valid_tok = pl.min(T_TILE, tokens - t0)
        weight_tile = pl.slice(weights_flat, [T_TILE, 1], [t0, 0], valid_shape=[valid_tok, 1])
        hidden_tile_full = pl.create_tensor([T_TILE, MOE_INTER_DIM], dtype=pl.BF16)

        for ob in pl.spmd(MOE_INTER_O_BLOCKS, name_hint="expert_routed_swiglu"):
            o0 = ob * O_TILE
            gate_tile = pl.cast(
                pl.slice(gate_flat, [T_TILE, O_TILE], [t0, o0], valid_shape=[valid_tok, O_TILE]),
                target_type=pl.FP32,
            )
            up_tile = pl.cast(
                pl.slice(up_flat, [T_TILE, O_TILE], [t0, o0], valid_shape=[valid_tok, O_TILE]),
                target_type=pl.FP32,
            )
            limit = pl.full([T_TILE, O_TILE], dtype=pl.FP32, value=SWIGLU_LIMIT)
            neg_limit = pl.full([T_TILE, O_TILE], dtype=pl.FP32, value=-SWIGLU_LIMIT)
            gate_clamped = pl.minimum(gate_tile, limit)
            up_clamped = pl.minimum(pl.maximum(up_tile, neg_limit), limit)
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_clamped)), 1.0))
            hidden_fp32 = pl.mul(pl.mul(gate_clamped, sigmoid), up_clamped)
            hidden_fp32 = pl.row_expand_mul(hidden_fp32, weight_tile)
            hidden_tile = pl.cast(hidden_fp32, target_type=pl.BF16, mode="rint")
            hidden_tile_full[:, o0 : o0 + O_TILE] = pl.fillpad(hidden_tile, pad_value=pl.PadValue.zero)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="expert_routed_hidden_write"):
            for ob in pl.range(MOE_INTER_O_BLOCKS):
                o0 = ob * O_TILE
                for row in pl.range(valid_tok):
                    hidden_row = pl.slice(hidden_tile_full, [1, O_TILE], [row, o0])
                    hidden_flat = pl.assemble(hidden_flat, hidden_row, [t0 + row, o0])

    out = linear_2048_to_4096(hidden, w2_t, out)
    return out


@pl.jit
def expert_shared_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    w1_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    w2_t: pl.Tensor[[MOE_INTER_DIM, HIDDEN], pl.BF16],
    w3_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    gate: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    up: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    hidden: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16]],
):
    return expert_shared_fwd(x, w1_t, w2_t, w3_t, gate, up, hidden, out)


@pl.jit
def expert_routed_test(
    x: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
    weights: pl.Tensor[[B, S_DYN, 1], pl.FP32],
    w1_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    w2_t: pl.Tensor[[MOE_INTER_DIM, HIDDEN], pl.BF16],
    w3_t: pl.Tensor[[HIDDEN, MOE_INTER_DIM], pl.BF16],
    gate: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    up: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    hidden: pl.Tensor[[B, S_DYN, MOE_INTER_DIM], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16]],
):
    return expert_routed_fwd(x, weights, w1_t, w2_t, w3_t, gate, up, hidden, out)


def _golden_expert(tensors, *, routed: bool):
    import torch
    import torch.nn.functional as F

    x = tensors["x"]
    gate = torch.matmul(x.float(), tensors["w1_t"].float()).to(torch.bfloat16).float()
    up = torch.matmul(x.float(), tensors["w3_t"].float()).to(torch.bfloat16).float()
    up = torch.clamp(up, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    gate = torch.clamp(gate, max=SWIGLU_LIMIT)
    hidden = F.silu(gate) * up
    if routed:
        hidden = hidden * tensors["weights"].float()
    hidden = hidden.to(torch.bfloat16)
    out = torch.matmul(hidden.float(), tensors["w2_t"].float()).to(torch.bfloat16)

    tensors["gate"][:] = gate.to(torch.bfloat16)
    tensors["up"][:] = up.to(torch.bfloat16)
    tensors["hidden"][:] = hidden
    tensors["out"][:] = out


def golden_expert_shared(tensors):
    _golden_expert(tensors, routed=False)


def golden_expert_routed(tensors):
    _golden_expert(tensors, routed=True)


def _build_tensor_specs(seq_len: int, *, routed: bool):
    import torch

    from models.golden import TensorSpec

    def init_x():
        return torch.randn(B, seq_len, HIDDEN) * 0.2

    def init_weights():
        return torch.rand(B, seq_len, 1) * 0.9 + 0.1

    def init_w1_t():
        return torch.randn(HIDDEN, MOE_INTER_DIM) * 0.02

    def init_w2_t():
        return torch.randn(MOE_INTER_DIM, HIDDEN) * 0.02

    def init_w3_t():
        return torch.randn(HIDDEN, MOE_INTER_DIM) * 0.02

    specs = [TensorSpec("x", [B, seq_len, HIDDEN], torch.bfloat16, init_value=init_x)]
    if routed:
        specs.append(TensorSpec("weights", [B, seq_len, 1], torch.float32, init_value=init_weights))
    specs.extend(
        [
            TensorSpec("w1_t", [HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=init_w1_t),
            TensorSpec("w2_t", [MOE_INTER_DIM, HIDDEN], torch.bfloat16, init_value=init_w2_t),
            TensorSpec("w3_t", [HIDDEN, MOE_INTER_DIM], torch.bfloat16, init_value=init_w3_t),
            TensorSpec("gate", [B, seq_len, MOE_INTER_DIM], torch.bfloat16),
            TensorSpec("up", [B, seq_len, MOE_INTER_DIM], torch.bfloat16),
            TensorSpec("hidden", [B, seq_len, MOE_INTER_DIM], torch.bfloat16),
            TensorSpec("out", [B, seq_len, HIDDEN], torch.bfloat16, is_output=True),
        ]
    )
    return specs


def build_expert_shared_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _build_tensor_specs(seq_len, routed=False)


def build_expert_routed_specs(seq_len: int = DEFAULT_SEQ_LEN):
    return _build_tensor_specs(seq_len, routed=True)


def main() -> int:
    import argparse

    from models.golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 Flash MoE expert validation.")
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
        "out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
    }

    cases = [
        ("expert-shared", expert_shared_test, build_expert_shared_specs, golden_expert_shared),
        ("expert-routed", expert_routed_test, build_expert_routed_specs, golden_expert_routed),
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
    "MOE_INTER_DIM",
    "SWIGLU_LIMIT",
    "DEFAULT_SEQ_LEN",
    "expert_shared_fwd",
    "expert_routed_fwd",
    "expert_shared_test",
    "expert_routed_test",
    "golden_expert_shared",
    "golden_expert_routed",
    "build_expert_shared_specs",
    "build_expert_routed_specs",
]
