"""Tests for Block golden logic against official ``model.py``."""

import importlib

import pytest
import torch
from conftest import make_einsum_reference, make_linear_reference, make_square_reference, torch_sparse_attn
from conftest import compressor_cos_sin, pad_last_dim, rope_cos_sin

import models.attention_csa as attention_csa
import models.attention_hca as attention_hca
import models.attention_swa as attention_swa
import models.block as block_golden
import models.compressor_ratio128 as compressor_ratio128
import models.compressor_ratio4 as compressor_ratio4
import models.expert as expert
import models.gate as gate
import models.hc as hc_model
import models.indexer as indexer
import models.moe as moe
import models.rope as rope

official_model = importlib.import_module("official.model")

DIM = 16
MOE_INTER_DIM = 8
N_HEADS = 2
HEAD_DIM = 8
ROPE_HEAD_DIM = 2
Q_LORA_RANK = 4
O_GROUPS = 1
O_LORA_RANK = 4
WINDOW_SIZE = 4
INDEX_N_HEADS = 2
INDEX_HEAD_DIM = 4
INDEX_TOPK = 2
N_EXPERTS = 8
TOPK = 3
VOCAB = 32
MAX_SEQ_LEN = 512
HC_MULT = 4

PREFILL_SEQ_LEN = 5
DECODE_START_POS = 3

CASES = [
    ("swa_hash", 0, "swa", True),
    ("csa_hash", 2, "csa", True),
    ("hca_topk", 3, "hca", False),
    ("csa_topk", 4, "csa", False),
]


@pytest.fixture()
def tiny_args(monkeypatch):
    args = official_model.ModelArgs(
        max_batch_size=1,
        max_seq_len=MAX_SEQ_LEN,
        dtype="bf16",
        scale_dtype="fp32",
        expert_dtype=None,
        vocab_size=VOCAB,
        dim=DIM,
        moe_inter_dim=MOE_INTER_DIM,
        n_layers=5,
        n_hash_layers=3,
        n_heads=N_HEADS,
        n_routed_experts=N_EXPERTS,
        n_shared_experts=1,
        n_activated_experts=TOPK,
        score_func="sqrtsoftplus",
        route_scale=1.5,
        swiglu_limit=1.5,
        q_lora_rank=Q_LORA_RANK,
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        norm_eps=1e-6,
        o_groups=O_GROUPS,
        o_lora_rank=O_LORA_RANK,
        window_size=WINDOW_SIZE,
        compress_ratios=(0, 0, 4, 128, 4),
        rope_factor=1,
        beta_fast=32,
        beta_slow=1,
        index_n_heads=INDEX_N_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        index_topk=INDEX_TOPK,
        hc_mult=HC_MULT,
        hc_sinkhorn_iters=20,
    )

    _patch_attention_constants(monkeypatch, args)
    _patch_moe_constants(monkeypatch, args)

    monkeypatch.setattr(official_model, "world_size", 1)
    monkeypatch.setattr(official_model, "rank", 0)
    monkeypatch.setattr(official_model, "rotate_activation", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(official_model, "sparse_attn", torch_sparse_attn)
    monkeypatch.setattr(official_model, "act_quant", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(official_model, "fp4_act_quant", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(official_model, "linear", make_linear_reference())
    monkeypatch.setattr(
        official_model,
        "hc_split_sinkhorn",
        lambda mixes, scale, base, hc, iters, eps: hc_model.split_sinkhorn_golden(
            mixes,
            scale,
            base,
            hc_mult=hc,
            sinkhorn_iters=iters,
            eps=eps,
        ),
    )
    monkeypatch.setattr(torch, "einsum", make_einsum_reference(torch.einsum))
    monkeypatch.setattr(torch.Tensor, "square", make_square_reference(torch.Tensor.square))
    return args


def _patch_attention_constants(monkeypatch, args) -> None:
    attn_q_out = args.n_heads * args.head_dim
    attn_proj_dim = 2 * args.head_dim
    index_q_out = args.index_n_heads * args.index_head_dim
    index_proj_dim = 2 * args.index_head_dim
    score_len = args.max_seq_len // 4
    hca_slots = args.max_seq_len // 128

    for module in (attention_swa, attention_hca, attention_csa):
        monkeypatch.setattr(module, "HIDDEN", args.dim)
        monkeypatch.setattr(module, "Q_LORA_RANK", args.q_lora_rank)
        monkeypatch.setattr(module, "N_HEADS", args.n_heads)
        monkeypatch.setattr(module, "HEAD_DIM", args.head_dim)
        monkeypatch.setattr(module, "ATTN_Q_OUT", attn_q_out)
        monkeypatch.setattr(module, "O_GROUPS", args.o_groups)
        monkeypatch.setattr(module, "O_LORA_RANK", args.o_lora_rank)
        monkeypatch.setattr(module, "HEADS_PER_GROUP", args.n_heads // args.o_groups)
        monkeypatch.setattr(module, "O_GROUP_IN", attn_q_out // args.o_groups)
        monkeypatch.setattr(module, "ATTN_OUT_IN", args.o_groups * args.o_lora_rank)
        monkeypatch.setattr(module, "ROPE_HALF", args.rope_head_dim // 2)
        monkeypatch.setattr(module, "WINDOW_SIZE", args.window_size)
        monkeypatch.setattr(module, "TOPK_SWA", args.window_size)
        monkeypatch.setattr(module, "SOFTMAX_SCALE", args.head_dim**-0.5)
        monkeypatch.setattr(module, "EPS", args.norm_eps)

    monkeypatch.setattr(attention_hca, "TOPK_HCA", hca_slots)
    monkeypatch.setattr(attention_hca, "TOPK_HCA_TOTAL", args.window_size + hca_slots)

    monkeypatch.setattr(attention_csa, "INDEX_N_HEADS", args.index_n_heads)
    monkeypatch.setattr(attention_csa, "INDEX_HEAD_DIM", args.index_head_dim)
    monkeypatch.setattr(attention_csa, "INDEX_Q_OUT", index_q_out)
    monkeypatch.setattr(attention_csa, "INDEX_PROJ_DIM", index_proj_dim)
    monkeypatch.setattr(attention_csa, "INDEX_TOPK", args.index_topk)
    monkeypatch.setattr(attention_csa, "INDEX_SCORE_LEN", score_len)
    monkeypatch.setattr(attention_csa, "ATTN_PROJ_DIM", attn_proj_dim)
    monkeypatch.setattr(attention_csa, "TOPK_CSA", args.index_topk)
    monkeypatch.setattr(attention_csa, "TOPK_CSA_TOTAL", args.window_size + args.index_topk)
    monkeypatch.setattr(attention_csa, "TOPK_CSA_COMPRESSED", score_len)

    monkeypatch.setattr(compressor_ratio128, "HIDDEN", args.dim)
    monkeypatch.setattr(compressor_ratio128, "HEAD_DIM", args.head_dim)
    monkeypatch.setattr(compressor_ratio128, "ROPE_HALF", args.rope_head_dim // 2)
    monkeypatch.setattr(compressor_ratio128, "TOPK_HCA", hca_slots)

    monkeypatch.setattr(compressor_ratio4, "HIDDEN", args.dim)
    monkeypatch.setattr(compressor_ratio4, "ATTN_HEAD_DIM", args.head_dim)
    monkeypatch.setattr(compressor_ratio4, "INDEX_HEAD_DIM", args.index_head_dim)
    monkeypatch.setattr(compressor_ratio4, "ATTN_PROJ_DIM", attn_proj_dim)
    monkeypatch.setattr(compressor_ratio4, "INDEX_PROJ_DIM", index_proj_dim)
    monkeypatch.setattr(compressor_ratio4, "ROPE_DIM", args.rope_head_dim)
    monkeypatch.setattr(compressor_ratio4, "ROPE_HALF", args.rope_head_dim // 2)
    monkeypatch.setattr(compressor_ratio4, "ATTN_TAIL_OFFSET", args.head_dim - args.rope_head_dim)
    monkeypatch.setattr(compressor_ratio4, "INDEX_TAIL_OFFSET", args.index_head_dim - args.rope_head_dim)
    monkeypatch.setattr(compressor_ratio4, "TOPK_CSA_COMPRESSED", score_len)
    monkeypatch.setattr(compressor_ratio4, "ATTN_INV_HEAD_DIM", 1.0 / args.head_dim)
    monkeypatch.setattr(compressor_ratio4, "INDEX_INV_HEAD_DIM", 1.0 / args.index_head_dim)

    monkeypatch.setattr(indexer, "HIDDEN", args.dim)
    monkeypatch.setattr(indexer, "Q_LORA_RANK", args.q_lora_rank)
    monkeypatch.setattr(indexer, "INDEX_N_HEADS", args.index_n_heads)
    monkeypatch.setattr(indexer, "INDEX_HEAD_DIM", args.index_head_dim)
    monkeypatch.setattr(indexer, "INDEX_Q_OUT", index_q_out)
    monkeypatch.setattr(indexer, "INDEX_PROJ_DIM", index_proj_dim)
    monkeypatch.setattr(indexer, "INDEX_TOPK", args.index_topk)
    monkeypatch.setattr(indexer, "INDEX_SCORE_LEN", score_len)
    monkeypatch.setattr(indexer, "ROPE_HALF", args.rope_head_dim // 2)
    monkeypatch.setattr(indexer, "INDEX_WEIGHTS_SCALE", (args.index_head_dim**-0.5) * (args.index_n_heads**-0.5))

    monkeypatch.setattr(rope, "ROPE_DIM", args.rope_head_dim)
    monkeypatch.setattr(rope, "ROPE_HALF", args.rope_head_dim // 2)
    monkeypatch.setattr(rope, "N_HEADS", args.n_heads)
    monkeypatch.setattr(rope, "HEAD_DIM", args.head_dim)
    monkeypatch.setattr(rope, "INDEX_HEAD_DIM", args.index_head_dim)
    monkeypatch.setattr(rope, "HEAD_TAIL_OFFSET", args.head_dim - args.rope_head_dim)
    monkeypatch.setattr(rope, "INDEX_TAIL_OFFSET", args.index_head_dim - args.rope_head_dim)


def _patch_moe_constants(monkeypatch, args) -> None:
    for module in (moe, gate):
        monkeypatch.setattr(module, "HIDDEN", args.dim)
        monkeypatch.setattr(module, "N_EXPERTS", args.n_routed_experts)
        monkeypatch.setattr(module, "TOPK", args.n_activated_experts)
        monkeypatch.setattr(module, "VOCAB", args.vocab_size)
        monkeypatch.setattr(module, "ROUTE_SCALE", args.route_scale)
    monkeypatch.setattr(moe, "MOE_INTER_DIM", args.moe_inter_dim)
    monkeypatch.setattr(moe, "SWIGLU_LIMIT", args.swiglu_limit)
    monkeypatch.setattr(expert, "HIDDEN", args.dim)
    monkeypatch.setattr(expert, "MOE_INTER_DIM", args.moe_inter_dim)
    monkeypatch.setattr(expert, "SWIGLU_LIMIT", args.swiglu_limit)


def _make_block(args, layer_id: int) -> torch.nn.Module:
    torch.manual_seed(20260704 + layer_id)
    with official_model.set_dtype(torch.bfloat16):
        module = official_model.Block(layer_id, args)
    _init_block_parameters(module, seed=20260714 + layer_id)
    return module


def _randn_like_param(param: torch.nn.Parameter, gen: torch.Generator, scale: float) -> torch.Tensor:
    return torch.randn(param.shape, generator=gen, dtype=torch.float32, device=param.device) * scale


def _init_block_parameters(module: torch.nn.Module, *, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if not torch.is_floating_point(param):
                continue
            if name.endswith("norm.weight"):
                value = torch.rand(param.shape, generator=gen, dtype=torch.float32) + 0.5
            elif name.endswith("hc_attn_scale") or name.endswith("hc_ffn_scale"):
                value = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
            elif name.endswith("hc_attn_base") or name.endswith("hc_ffn_base"):
                value = torch.zeros(param.shape, dtype=torch.float32)
            elif name.endswith("hc_attn_fn") or name.endswith("hc_ffn_fn"):
                value = _randn_like_param(param, gen, 0.05)
            elif name.endswith("ape"):
                value = _randn_like_param(param, gen, 0.02)
            elif name.endswith("attn_sink"):
                value = _randn_like_param(param, gen, 0.1)
            elif name.endswith("bias"):
                value = _randn_like_param(param, gen, 0.2)
            else:
                value = (_randn_like_param(param, gen, 0.1)).to(torch.bfloat16)
            param.copy_(value.to(param.dtype))

        if hasattr(module.ffn.gate, "tid2eid") and module.ffn.gate.tid2eid is not None:
            base = torch.arange(TOPK, dtype=torch.int32).view(1, TOPK)
            token_offsets = torch.arange(VOCAB, dtype=torch.int32).view(VOCAB, 1)
            module.ffn.gate.tid2eid.copy_((base + token_offsets) % N_EXPERTS)


def _topk_hca(attn: torch.nn.Module, seq_len: int, start_pos: int) -> torch.Tensor:
    window_topk = official_model.get_window_topk_idxs(attn.window_size, 1, seq_len, start_pos).int()
    offset = seq_len if start_pos == 0 else attn.window_size
    compress_topk = official_model.get_compress_topk_idxs(128, 1, seq_len, start_pos, offset).int()
    compressed_slots = attn.kv_cache.shape[1] - attn.window_size
    return torch.cat(
        [
            pad_last_dim(window_topk, attn.window_size),
            pad_last_dim(compress_topk, compressed_slots),
        ],
        dim=-1,
    )


def _add_common_attention_tensors(tensors: dict[str, torch.Tensor], attn: torch.nn.Module, x: torch.Tensor, start_pos: int) -> None:
    seq_len = x.shape[1]
    cos, sin = rope_cos_sin(attn, start_pos, seq_len)
    tensors.update(
        {
            "wq_a_t": attn.wq_a.weight.detach().t().contiguous().to(torch.bfloat16),
            "q_norm_w": attn.q_norm.weight.detach().clone(),
            "wq_b_t": attn.wq_b.weight.detach().t().contiguous().to(torch.bfloat16),
            "wkv_t": attn.wkv.weight.detach().t().contiguous().to(torch.bfloat16),
            "kv_norm_w": attn.kv_norm.weight.detach().clone(),
            "attn_sink": attn.attn_sink.detach().clone(),
            "wo_a_t": attn.wo_a.weight.detach().t().contiguous().to(torch.bfloat16),
            "wo_b_t": attn.wo_b.weight.detach().t().contiguous().to(torch.bfloat16),
            "cos": cos,
            "sin": sin,
        }
    )


def _add_attention_tensors(
    tensors: dict[str, torch.Tensor],
    attn: torch.nn.Module,
    args,
    x: torch.Tensor,
    start_pos: int,
    attention_kind: str,
    *,
    decode: bool,
    state: dict[str, torch.Tensor] | None,
) -> None:
    seq_len = x.shape[1]
    _add_common_attention_tensors(tensors, attn, x, start_pos)

    if attention_kind == "swa":
        tensors["topk_idxs"] = pad_last_dim(
            official_model.get_window_topk_idxs(args.window_size, 1, seq_len, start_pos),
            args.window_size,
        )
    elif attention_kind == "hca":
        comp_cos, comp_sin = compressor_cos_sin(attn, 128, seq_len, start_pos)
        tensors.update(
            {
                "topk_idxs": _topk_hca(attn, seq_len, start_pos),
                "comp_wkv_t": attn.compressor.wkv.weight.detach().t().contiguous().to(torch.bfloat16),
                "comp_wgate_t": attn.compressor.wgate.weight.detach().t().contiguous().to(torch.bfloat16),
                "comp_ape": attn.compressor.ape.detach().clone(),
                "comp_norm_w": attn.compressor.norm.weight.detach().clone(),
                "comp_cos": comp_cos,
                "comp_sin": comp_sin,
                "comp_block_count": torch.tensor([seq_len // 128], dtype=torch.int32),
            }
        )
    elif attention_kind == "csa":
        score_len = args.max_seq_len // 4
        comp_cos, comp_sin = compressor_cos_sin(attn, 4, seq_len, start_pos)
        window_topk = official_model.get_window_topk_idxs(args.window_size, 1, seq_len, start_pos)
        offset = seq_len if start_pos == 0 else args.window_size
        tensors.update(
            {
                "window_topk_idxs": pad_last_dim(window_topk, args.window_size),
                "attn_comp_wkv_t": attn.compressor.wkv.weight.detach().t().contiguous().to(torch.bfloat16),
                "attn_comp_wgate_t": attn.compressor.wgate.weight.detach().t().contiguous().to(torch.bfloat16),
                "attn_comp_ape": attn.compressor.ape.detach().clone(),
                "attn_comp_norm_w": attn.compressor.norm.weight.detach().clone(),
                "attn_comp_cos": comp_cos,
                "attn_comp_sin": comp_sin,
                "attn_comp_block_count": torch.tensor([seq_len // 4], dtype=torch.int32),
                "idx_wq_b_t": attn.indexer.wq_b.weight.detach().t().contiguous().to(torch.bfloat16),
                "idx_weights_proj_t": attn.indexer.weights_proj.weight.detach().t().contiguous().to(torch.bfloat16),
                "idx_offset": torch.tensor([offset], dtype=torch.int32),
                "idx_comp_wkv_t": attn.indexer.compressor.wkv.weight.detach().t().contiguous().to(torch.bfloat16),
                "idx_comp_wgate_t": attn.indexer.compressor.wgate.weight.detach().t().contiguous().to(torch.bfloat16),
                "idx_comp_ape": attn.indexer.compressor.ape.detach().clone(),
                "idx_comp_norm_w": attn.indexer.compressor.norm.weight.detach().clone(),
                "idx_comp_cos": comp_cos.clone(),
                "idx_comp_sin": comp_sin.clone(),
                "idx_comp_block_count": torch.tensor([seq_len // 4], dtype=torch.int32),
            }
        )
        tensors["idx_kv_cache_out"] = torch.zeros(1, score_len, args.index_head_dim, dtype=torch.bfloat16)
    else:
        raise ValueError(f"unsupported attention_kind: {attention_kind!r}")

    if decode:
        assert state is not None
        tensors["kv_cache"] = state["kv_cache"]
        tensors["cache_pos"] = torch.tensor([start_pos % args.window_size], dtype=torch.int32)
        if attention_kind == "hca":
            tensors.update(
                {
                    "comp_kv_state": state["comp_kv_state"],
                    "comp_score_state": state["comp_score_state"],
                    "comp_cache": state["comp_cache"],
                    "comp_slot": torch.tensor([start_pos % 128], dtype=torch.int32),
                    "comp_cache_slot": torch.tensor([start_pos // 128], dtype=torch.int32),
                    "comp_should_compress": torch.tensor([int((start_pos + 1) % 128 == 0)], dtype=torch.int32),
                }
            )
        elif attention_kind == "csa":
            tensors.update(
                {
                    "attn_comp_kv_state": state["attn_comp_kv_state"],
                    "attn_comp_score_state": state["attn_comp_score_state"],
                    "attn_comp_cache": state["attn_comp_cache"],
                    "idx_kv_cache_in": state["idx_kv_cache"],
                    "idx_comp_kv_state": state["idx_comp_kv_state"],
                    "idx_comp_score_state": state["idx_comp_score_state"],
                    "comp_slot": torch.tensor([start_pos % 4], dtype=torch.int32),
                    "comp_cache_slot": torch.tensor([start_pos // 4], dtype=torch.int32),
                    "comp_should_compress": torch.tensor([int((start_pos + 1) % 4 == 0)], dtype=torch.int32),
                }
            )

    _add_attention_outputs(tensors, args, seq_len, attention_kind, decode=decode)


def _add_attention_outputs(tensors: dict[str, torch.Tensor], args, seq_len: int, attention_kind: str, *, decode: bool) -> None:
    tensors.update(
        {
            "q_a": torch.zeros(1, seq_len, args.q_lora_rank, dtype=torch.bfloat16),
            "q_proj": torch.zeros(1, seq_len, args.n_heads * args.head_dim, dtype=torch.bfloat16),
            "kv_proj": torch.zeros(1, seq_len, args.head_dim, dtype=torch.bfloat16),
            "kv_normed": torch.zeros(1, seq_len, args.head_dim, dtype=torch.bfloat16),
            "qr": torch.zeros(1, seq_len, args.q_lora_rank, dtype=torch.bfloat16),
            "q": torch.zeros(1, seq_len, args.n_heads, args.head_dim, dtype=torch.bfloat16),
            "kv": torch.zeros(1, seq_len, args.head_dim, dtype=torch.bfloat16),
            "kv_cache_out": torch.zeros(1, args.window_size, args.head_dim, dtype=torch.bfloat16),
            "attn_o": torch.zeros(1, seq_len, args.n_heads, args.head_dim, dtype=torch.bfloat16),
            "o_inv": torch.zeros(1, seq_len, args.n_heads, args.head_dim, dtype=torch.bfloat16),
            "proj": torch.zeros(1, seq_len, args.o_groups * args.o_lora_rank, dtype=torch.bfloat16),
            "attn_out": torch.zeros(1, seq_len, args.dim, dtype=torch.bfloat16),
        }
    )
    if attention_kind == "hca":
        compressed_len = 1 if decode else max(1, seq_len // 128)
        kv_pool_len = args.window_size + args.max_seq_len // 128 if decode else seq_len + compressed_len
        tensors.update(
            {
                "comp_kv_proj": torch.zeros(1, seq_len, args.head_dim, dtype=torch.float32),
                "comp_score_proj": torch.zeros(1, seq_len, args.head_dim, dtype=torch.float32),
                "comp_pooled": torch.zeros(1, compressed_len, args.head_dim, dtype=torch.bfloat16),
                "comp_normed": torch.zeros(1, compressed_len, args.head_dim, dtype=torch.bfloat16),
                "compressed": torch.zeros(1, compressed_len, args.head_dim, dtype=torch.bfloat16),
                "kv_pool": torch.zeros(1, kv_pool_len, args.head_dim, dtype=torch.bfloat16),
                "comp_kv_state_out": torch.zeros(1, 128, args.head_dim, dtype=torch.float32),
                "comp_score_state_out": torch.zeros(1, 128, args.head_dim, dtype=torch.float32),
                "comp_cache_out": torch.zeros(1, args.max_seq_len // 128, args.head_dim, dtype=torch.bfloat16),
            }
        )
    elif attention_kind == "csa":
        compressed_len = 1 if decode else max(1, seq_len // 4)
        score_len = args.max_seq_len // 4
        kv_pool_len = args.window_size + score_len if decode else seq_len + compressed_len
        tensors.update(
            {
                "attn_comp_kv_proj": torch.zeros(1, seq_len, 2 * args.head_dim, dtype=torch.float32),
                "attn_comp_score_proj": torch.zeros(1, seq_len, 2 * args.head_dim, dtype=torch.float32),
                "attn_comp_pooled": torch.zeros(1, compressed_len, args.head_dim, dtype=torch.bfloat16),
                "attn_comp_normed": torch.zeros(1, compressed_len, args.head_dim, dtype=torch.bfloat16),
                "attn_compressed": torch.zeros(1, compressed_len, args.head_dim, dtype=torch.bfloat16),
                "kv_pool": torch.zeros(1, kv_pool_len, args.head_dim, dtype=torch.bfloat16),
                "attn_comp_kv_state_out": torch.zeros(1, 8, 2 * args.head_dim, dtype=torch.float32),
                "attn_comp_score_state_out": torch.zeros(1, 8, 2 * args.head_dim, dtype=torch.float32),
                "attn_comp_cache_out": torch.zeros(1, score_len, args.head_dim, dtype=torch.bfloat16),
                "idx_q_proj": torch.zeros(1, seq_len, args.index_n_heads * args.index_head_dim, dtype=torch.bfloat16),
                "idx_q_rope": torch.zeros(1, seq_len, args.index_n_heads, args.index_head_dim, dtype=torch.bfloat16),
                "idx_weights": torch.zeros(1, seq_len, args.index_n_heads, dtype=torch.bfloat16),
                "idx_comp_kv_proj": torch.zeros(1, seq_len, 2 * args.index_head_dim, dtype=torch.float32),
                "idx_comp_score_proj": torch.zeros(1, seq_len, 2 * args.index_head_dim, dtype=torch.float32),
                "idx_comp_pooled": torch.zeros(1, compressed_len, args.index_head_dim, dtype=torch.bfloat16),
                "idx_comp_normed": torch.zeros(1, compressed_len, args.index_head_dim, dtype=torch.bfloat16),
                "idx_score": torch.zeros(1, seq_len, score_len, dtype=torch.float32),
                "idx_topk_idxs": torch.full((1, seq_len, args.index_topk), -1, dtype=torch.int32),
                "idx_comp_kv_state_out": torch.zeros(1, 8, 2 * args.index_head_dim, dtype=torch.float32),
                "idx_comp_score_state_out": torch.zeros(1, 8, 2 * args.index_head_dim, dtype=torch.float32),
                "csa_topk_idxs": torch.full((1, seq_len, args.window_size + args.index_topk), -1, dtype=torch.int32),
            }
        )


def _add_moe_tensors(tensors: dict[str, torch.Tensor], module: torch.nn.Module, input_ids: torch.Tensor, *, hash_route: bool) -> None:
    bsz, seq_len = input_ids.shape
    routed_w1_t = torch.stack([expert_module.w1.weight.detach().t().contiguous().to(torch.bfloat16) for expert_module in module.experts])
    routed_w2_t = torch.stack([expert_module.w2.weight.detach().t().contiguous().to(torch.bfloat16) for expert_module in module.experts])
    routed_w3_t = torch.stack([expert_module.w3.weight.detach().t().contiguous().to(torch.bfloat16) for expert_module in module.experts])
    tensors.update(
        {
            "gate_w_t": module.gate.weight.detach().t().contiguous().to(torch.bfloat16),
            "routed_w1_t": routed_w1_t,
            "routed_w2_t": routed_w2_t,
            "routed_w3_t": routed_w3_t,
            "shared_w1_t": module.shared_experts.w1.weight.detach().t().contiguous().to(torch.bfloat16),
            "shared_w2_t": module.shared_experts.w2.weight.detach().t().contiguous().to(torch.bfloat16),
            "shared_w3_t": module.shared_experts.w3.weight.detach().t().contiguous().to(torch.bfloat16),
            "logits": torch.zeros(bsz, seq_len, N_EXPERTS, dtype=torch.float32),
            "scores": torch.zeros(bsz, seq_len, N_EXPERTS, dtype=torch.float32),
            "indices": torch.zeros(bsz, seq_len, TOPK, dtype=torch.int32),
            "weights": torch.zeros(bsz, seq_len, TOPK, dtype=torch.float32),
            "route_y": torch.zeros(bsz, seq_len, TOPK, DIM, dtype=torch.bfloat16),
            "shared_gate": torch.zeros(bsz, seq_len, MOE_INTER_DIM, dtype=torch.bfloat16),
            "shared_up": torch.zeros(bsz, seq_len, MOE_INTER_DIM, dtype=torch.bfloat16),
            "shared_hidden": torch.zeros(bsz, seq_len, MOE_INTER_DIM, dtype=torch.bfloat16),
            "shared_y": torch.zeros(bsz, seq_len, DIM, dtype=torch.bfloat16),
            "moe_out": torch.zeros(bsz, seq_len, DIM, dtype=torch.bfloat16),
        }
    )
    if hash_route:
        tensors["tid2eid"] = module.gate.tid2eid.detach().clone()
        tensors["input_ids"] = input_ids.clone()
    else:
        tensors["gate_bias"] = module.gate.bias.detach().clone()


def _block_tensors(
    module: torch.nn.Module,
    x: torch.Tensor,
    input_ids: torch.Tensor,
    args,
    start_pos: int,
    attention_kind: str,
    *,
    hash_route: bool,
    decode: bool,
    state: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    seq_len = x.shape[1]
    tensors = {
        "x": x.clone(),
        "out": torch.zeros_like(x),
        "attn_hc_fn_t": module.hc_attn_fn.detach().t().contiguous(),
        "attn_hc_scale": module.hc_attn_scale.detach().clone(),
        "attn_hc_base": module.hc_attn_base.detach().clone(),
        "ffn_hc_fn_t": module.hc_ffn_fn.detach().t().contiguous(),
        "ffn_hc_scale": module.hc_ffn_scale.detach().clone(),
        "ffn_hc_base": module.hc_ffn_base.detach().clone(),
        "attn_norm_w": module.attn_norm.weight.detach().clone(),
        "ffn_norm_w": module.ffn_norm.weight.detach().clone(),
        "attn_hc_x_mixed": torch.zeros(1, seq_len, args.dim, dtype=torch.bfloat16),
        "attn_hc_post": torch.zeros(1, seq_len, HC_MULT, dtype=torch.float32),
        "attn_hc_comb": torch.zeros(1, seq_len, HC_MULT * HC_MULT, dtype=torch.float32),
        "attn_hc_out": torch.zeros_like(x),
        "ffn_hc_x_mixed": torch.zeros(1, seq_len, args.dim, dtype=torch.bfloat16),
        "ffn_hc_post": torch.zeros(1, seq_len, HC_MULT, dtype=torch.float32),
        "ffn_hc_comb": torch.zeros(1, seq_len, HC_MULT * HC_MULT, dtype=torch.float32),
        "ffn_hc_out": torch.zeros_like(x),
        "attn_normed": torch.zeros(1, seq_len, args.dim, dtype=torch.bfloat16),
        "ffn_normed": torch.zeros(1, seq_len, args.dim, dtype=torch.bfloat16),
    }
    _add_attention_tensors(tensors, module.attn, args, x, start_pos, attention_kind, decode=decode, state=state)
    _add_moe_tensors(tensors, module.ffn, input_ids, hash_route=hash_route)
    return tensors


def _capture_attention_state(attn: torch.nn.Module, attention_kind: str, args) -> dict[str, torch.Tensor]:
    state = {"kv_cache": attn.kv_cache[:, : args.window_size].detach().clone()}
    if attention_kind == "hca":
        state.update(
            {
                "comp_cache": attn.kv_cache[:, args.window_size :].detach().clone(),
                "comp_kv_state": attn.compressor.kv_state.detach().clone(),
                "comp_score_state": attn.compressor.score_state.detach().clone(),
            }
        )
    elif attention_kind == "csa":
        state.update(
            {
                "attn_comp_cache": attn.kv_cache[:, args.window_size :].detach().clone(),
                "attn_comp_kv_state": attn.compressor.kv_state.detach().clone(),
                "attn_comp_score_state": attn.compressor.score_state.detach().clone(),
                "idx_kv_cache": attn.indexer.kv_cache.detach().clone(),
                "idx_comp_kv_state": attn.indexer.compressor.kv_state.detach().clone(),
                "idx_comp_score_state": attn.indexer.compressor.score_state.detach().clone(),
            }
        )
    return state


def _assert_score_state_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], rtol=0, atol=0)


def _assert_attention_state(tensors: dict[str, torch.Tensor], attn: torch.nn.Module, attention_kind: str, args) -> None:
    torch.testing.assert_close(tensors["kv_cache_out"], attn.kv_cache[:, : args.window_size], rtol=0, atol=0)
    if attention_kind == "hca":
        torch.testing.assert_close(tensors["comp_cache_out"], attn.kv_cache[:, args.window_size :], rtol=0, atol=0)
        torch.testing.assert_close(tensors["comp_kv_state_out"], attn.compressor.kv_state, rtol=0, atol=0)
        _assert_score_state_close(tensors["comp_score_state_out"], attn.compressor.score_state)
    elif attention_kind == "csa":
        torch.testing.assert_close(tensors["attn_comp_cache_out"], attn.kv_cache[:, args.window_size :], rtol=0, atol=0)
        torch.testing.assert_close(tensors["attn_comp_kv_state_out"], attn.compressor.kv_state, rtol=0, atol=0)
        _assert_score_state_close(tensors["attn_comp_score_state_out"], attn.compressor.score_state)
        torch.testing.assert_close(tensors["idx_kv_cache_out"], attn.indexer.kv_cache, rtol=0, atol=0)
        torch.testing.assert_close(tensors["idx_comp_kv_state_out"], attn.indexer.compressor.kv_state, rtol=0, atol=0)
        _assert_score_state_close(tensors["idx_comp_score_state_out"], attn.indexer.compressor.score_state)


def _run_prefill_wrapper(tensors: dict[str, torch.Tensor], case_name: str) -> None:
    {
        "swa_hash": block_golden.golden_block_swa_hash_prefill,
        "csa_hash": block_golden.golden_block_csa_hash_prefill,
        "hca_topk": block_golden.golden_block_hca_topk_prefill,
        "csa_topk": block_golden.golden_block_csa_topk_prefill,
    }[case_name](tensors)


def _run_decode_wrapper(tensors: dict[str, torch.Tensor], case_name: str, start_pos: int) -> None:
    {
        "swa_hash": block_golden.golden_block_swa_hash_decode,
        "csa_hash": block_golden.golden_block_csa_hash_decode,
        "hca_topk": block_golden.golden_block_hca_topk_decode,
        "csa_topk": block_golden.golden_block_csa_topk_decode,
    }[case_name](tensors, start_pos)


@pytest.mark.parametrize(("case_name", "layer_id", "attention_kind", "hash_route"), CASES)
def test_block_prefill_golden_matches_official_model(
    tiny_args,
    case_name: str,
    layer_id: int,
    attention_kind: str,
    hash_route: bool,
) -> None:
    module = _make_block(tiny_args, layer_id)
    x = (torch.randn(1, PREFILL_SEQ_LEN, HC_MULT, DIM, dtype=torch.float32) * 0.2).to(torch.bfloat16)
    input_ids = torch.randint(0, VOCAB, (1, PREFILL_SEQ_LEN), dtype=torch.int64)

    with torch.no_grad():
        expected = module(x.clone(), start_pos=0, input_ids=input_ids.clone())

    tensors = _block_tensors(
        module,
        x,
        input_ids,
        tiny_args,
        start_pos=0,
        attention_kind=attention_kind,
        hash_route=hash_route,
        decode=False,
    )
    _run_prefill_wrapper(tensors, case_name)

    torch.testing.assert_close(tensors["out"], expected, rtol=0, atol=0)
    _assert_attention_state(tensors, module.attn, attention_kind, tiny_args)


@pytest.mark.parametrize(("case_name", "layer_id", "attention_kind", "hash_route"), CASES)
def test_block_decode_golden_matches_official_model(
    tiny_args,
    case_name: str,
    layer_id: int,
    attention_kind: str,
    hash_route: bool,
) -> None:
    module = _make_block(tiny_args, layer_id)
    prompt = (torch.randn(1, DECODE_START_POS, HC_MULT, DIM, dtype=torch.float32) * 0.2).to(torch.bfloat16)
    prompt_ids = torch.randint(0, VOCAB, (1, DECODE_START_POS), dtype=torch.int64)
    token = (torch.randn(1, 1, HC_MULT, DIM, dtype=torch.float32) * 0.2).to(torch.bfloat16)
    token_ids = torch.randint(0, VOCAB, (1, 1), dtype=torch.int64)

    with torch.no_grad():
        module(prompt.clone(), start_pos=0, input_ids=prompt_ids.clone())

    state = _capture_attention_state(module.attn, attention_kind, tiny_args)

    with torch.no_grad():
        expected = module(token.clone(), start_pos=DECODE_START_POS, input_ids=token_ids.clone())

    tensors = _block_tensors(
        module,
        token,
        token_ids,
        tiny_args,
        start_pos=DECODE_START_POS,
        attention_kind=attention_kind,
        hash_route=hash_route,
        decode=True,
        state=state,
    )
    _run_decode_wrapper(tensors, case_name, DECODE_START_POS)

    torch.testing.assert_close(tensors["out"], expected, rtol=0, atol=0)
    _assert_attention_state(tensors, module.attn, attention_kind, tiny_args)
