"""DeepSeek V4 Flash embedding PyPTO kernel."""

import pypto.language as pl

from models.common import assert_divisible
from models.config import FLASH_CONFIG as M


B = 1
S_DYN = pl.dynamic("S_DYN")
VOCAB = M.vocab_size
HIDDEN = M.dim
DEFAULT_SEQ_LEN = 8

D_TILE = 128
assert_divisible(HIDDEN, D_TILE, "embedding hidden size")
H_BLOCKS = HIDDEN // D_TILE


@pl.jit.inline
def embedding_fwd(
    input_ids: pl.Tensor[[B, S_DYN], pl.INT64],
    weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    out: pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16],
):
    """Run official single-card ``ParallelEmbedding.forward``."""
    input_ids.bind_dynamic(1, S_DYN)
    out.bind_dynamic(1, S_DYN)

    tokens = pl.tensor.dim(input_ids, 1)
    input_flat = pl.reshape(input_ids, [tokens])
    out_flat = pl.reshape(out, [tokens, HIDDEN])

    for t in pl.range(tokens):
        for hb in pl.spmd(H_BLOCKS, name_hint="embedding"):
            token_id = pl.cast(pl.read(input_flat, [t]), pl.INDEX)
            h0 = hb * D_TILE
            out_flat[t : t + 1, h0 : h0 + D_TILE] = weight[token_id : token_id + 1, h0 : h0 + D_TILE]

    return pl.reshape(out_flat, [B, tokens, HIDDEN])


@pl.jit
def embedding_test(
    input_ids: pl.Tensor[[B, S_DYN], pl.INT64],
    weight: pl.Tensor[[VOCAB, HIDDEN], pl.BF16],
    out: pl.Out[pl.Tensor[[B, S_DYN, HIDDEN], pl.BF16]],
):
    out = embedding_fwd(input_ids, weight, out)
    return out


def golden_embedding(tensors):
    import torch.nn.functional as F

    input_ids = tensors["input_ids"].long()
    weight = tensors["weight"]
    tensors["out"][:] = F.embedding(input_ids, weight)


def build_embedding_specs(seq_len: int = DEFAULT_SEQ_LEN):
    import torch

    from models.golden import TensorSpec

    def init_input_ids():
        return torch.randint(0, VOCAB, (B, seq_len), dtype=torch.int64)

    def init_weight():
        return torch.randn(VOCAB, HIDDEN, dtype=torch.float32) * 0.02

    return [
        TensorSpec("input_ids", [B, seq_len], torch.int64, init_value=init_input_ids),
        TensorSpec("weight", [VOCAB, HIDDEN], torch.bfloat16, init_value=init_weight),
        TensorSpec("out", [B, seq_len, HIDDEN], torch.bfloat16, is_output=True),
    ]


def main() -> int:
    import argparse

    from models.golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 Flash embedding validation.")
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
        "out": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
    }

    print("[CASE] embedding", flush=True)
    result = run_jit(
        fn=embedding_test,
        specs=build_embedding_specs(args.seq_len),
        golden_fn=golden_embedding,
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
    "VOCAB",
    "HIDDEN",
    "DEFAULT_SEQ_LEN",
    "D_TILE",
    "H_BLOCKS",
    "embedding_fwd",
    "embedding_test",
    "golden_embedding",
    "build_embedding_specs",
]
