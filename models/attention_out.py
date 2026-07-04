"""DeepSeek V4 Flash attention output projection PyPTO kernel."""

import pypto.language as pl

from models.config import FLASH_CONFIG as M
from models.linear import linear_8192_to_4096
from models.rope import (
    _apply_rope_golden,
    build_deepseek_v4_rope_tables,
    materialize_rope_range,
    rope_4d_512_inv,
)


B = 1
S_DYN = pl.dynamic("S_DYN")

HIDDEN = M.dim
N_HEADS = M.n_heads
HEAD_DIM = M.head_dim
O_GROUPS = M.o_groups
O_LORA_RANK = M.o_lora_rank
HEADS_PER_GROUP = M.heads_per_o_group
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
ATTN_OUT_IN = O_GROUPS * O_LORA_RANK
ROPE_HALF = M.rope_head_dim // 2

T_TILE = 16
K_TILE = 128
O_TILE = 32
OUT_GROUP = 2
GROUP_K_BLOCKS = O_GROUP_IN // K_TILE
O_LORA_O_BLOCKS = O_LORA_RANK // O_TILE
DEFAULT_SEQ_LEN = 8


@pl.jit.inline
def attention_out_fwd(
    o: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    wo_a_t: pl.Tensor[[O_GROUP_IN, ATTN_OUT_IN], pl.BF16],
    wo_b_t: pl.Tensor[[ATTN_OUT_IN, HIDDEN], pl.BF16],
    cos: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
):
    """Compute official inverse RoPE and attention output projection."""
    o.bind_dynamic(1, S_DYN)
    cos.bind_dynamic(0, S_DYN)
    sin.bind_dynamic(0, S_DYN)
    out.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(o, 1)
    o_inv = pl.create_tensor([B, tokens, N_HEADS, HEAD_DIM], dtype=pl.BF16)
    proj = pl.create_tensor([B, tokens, ATTN_OUT_IN], dtype=pl.BF16)

    o_inv = rope_4d_512_inv(o, cos, sin, o_inv)
    o_flat = pl.reshape(o_inv, [tokens, N_HEADS * HEAD_DIM])
    proj_flat = pl.reshape(proj, [tokens, ATTN_OUT_IN])
    token_blocks = (tokens + T_TILE - 1) // T_TILE

    for tb in pl.range(token_blocks):
        t0 = tb * T_TILE
        valid_tok = pl.min(T_TILE, tokens - t0)

        for task in pl.spmd(O_GROUPS * (O_LORA_O_BLOCKS // OUT_GROUP), name_hint="attention_out_wo_a"):
            g = task // (O_LORA_O_BLOCKS // OUT_GROUP)
            og_idx = task - g * (O_LORA_O_BLOCKS // OUT_GROUP)
            group_in0 = g * O_GROUP_IN
            group_out0 = g * O_LORA_RANK
            og = og_idx * OUT_GROUP

            for o_inner in pl.pipeline(OUT_GROUP, stage=2):
                r0 = (og + o_inner) * O_TILE
                x0 = pl.slice(o_flat, [T_TILE, K_TILE], [t0, group_in0], valid_shape=[valid_tok, K_TILE])
                w0 = pl.slice(wo_a_t, [K_TILE, O_TILE], [0, group_out0 + r0])
                acc = pl.matmul(x0, w0, out_dtype=pl.FP32)
                for kb in pl.pipeline(1, GROUP_K_BLOCKS, stage=2):
                    k0 = kb * K_TILE
                    xk = pl.slice(
                        o_flat,
                        [T_TILE, K_TILE],
                        [t0, group_in0 + k0],
                        valid_shape=[valid_tok, K_TILE],
                    )
                    wk = pl.slice(wo_a_t, [K_TILE, O_TILE], [k0, group_out0 + r0])
                    acc = pl.matmul_acc(acc, xk, wk)

                acc_bf16 = pl.cast(acc, target_type=pl.BF16, mode="rint")
                for row in pl.range(valid_tok):
                    out_row = pl.slice(acc_bf16, [1, O_TILE], [row, 0])
                    proj_flat = pl.assemble(proj_flat, out_row, [t0 + row, group_out0 + r0])

    proj = pl.reshape(proj_flat, [B, tokens, ATTN_OUT_IN])
    out = linear_8192_to_4096(proj, wo_b_t, out)
    return out


@pl.jit
def attention_out_fwd_test(
    o: pl.Tensor[[B, S_DYN, N_HEADS, HEAD_DIM], pl.BF16],
    wo_a_t: pl.Tensor[[O_GROUP_IN, ATTN_OUT_IN], pl.BF16],
    wo_b_t: pl.Tensor[[ATTN_OUT_IN, HIDDEN], pl.BF16],
    cos: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    sin: pl.Tensor[[S_DYN, ROPE_HALF], pl.FP32],
    out: pl.Out[pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16]],
):
    out = attention_out_fwd(o, wo_a_t, wo_b_t, cos, sin, out)
    return out


def golden_attention_out(tensors):
    import torch

    o = _apply_rope_golden(tensors["o"], tensors["cos"], tensors["sin"], inverse=True)
    o = o.view(B, tensors["o"].shape[1], O_GROUPS, O_GROUP_IN)
    wo_a = tensors["wo_a_t"].transpose(0, 1).contiguous().view(O_GROUPS, O_LORA_RANK, O_GROUP_IN)
    proj = torch.einsum("bsgd,grd->bsgr", o.float(), wo_a.float()).to(torch.bfloat16)
    out = torch.matmul(proj.flatten(2).float(), tensors["wo_b_t"].float()).to(torch.bfloat16)

    tensors["out"][:] = out


def build_tensor_specs(seq_len: int = DEFAULT_SEQ_LEN, start_pos: int = 0):
    import torch

    from models.golden import TensorSpec

    freqs_cos, freqs_sin = build_deepseek_v4_rope_tables(max_seq_len=start_pos + seq_len)
    local_cos, local_sin = materialize_rope_range(freqs_cos, freqs_sin, start_pos, seq_len)

    def init_o():
        return torch.randn(B, seq_len, N_HEADS, HEAD_DIM) * 0.1

    def init_wo_a():
        return torch.randn(O_GROUP_IN, ATTN_OUT_IN) * 0.02

    def init_wo_b_t():
        return torch.randn(ATTN_OUT_IN, HIDDEN) * 0.02

    return [
        TensorSpec("o", [B, seq_len, N_HEADS, HEAD_DIM], torch.bfloat16, init_value=init_o),
        TensorSpec("wo_a_t", [O_GROUP_IN, ATTN_OUT_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b_t", [ATTN_OUT_IN, HIDDEN], torch.bfloat16, init_value=init_wo_b_t),
        TensorSpec("cos", [seq_len, ROPE_HALF], torch.float32, init_value=local_cos),
        TensorSpec("sin", [seq_len, ROPE_HALF], torch.float32, init_value=local_sin),
        TensorSpec("out", [B, seq_len, HIDDEN], torch.bfloat16, is_output=True),
    ]


def main() -> int:
    import argparse

    from models.golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 Flash attention output validation.")
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
        "out": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.001),
    }

    result = run_jit(
        fn=attention_out_fwd_test,
        specs=build_tensor_specs(args.seq_len, args.start_pos),
        golden_fn=golden_attention_out,
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
    "N_HEADS",
    "HEAD_DIM",
    "O_GROUPS",
    "O_LORA_RANK",
    "HEADS_PER_GROUP",
    "O_GROUP_IN",
    "ATTN_OUT_IN",
    "ROPE_HALF",
    "T_TILE",
    "K_TILE",
    "O_TILE",
    "OUT_GROUP",
    "DEFAULT_SEQ_LEN",
    "attention_out_fwd",
    "attention_out_fwd_test",
    "golden_attention_out",
    "build_tensor_specs",
]
