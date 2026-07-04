"""Tests for SWA attention golden logic against official ``model.py``."""

import importlib

import pytest
import torch
from conftest import make_einsum_reference, make_linear_reference, make_square_reference, rope_cos_sin, torch_sparse_attn

import models.attention_swa as attention_swa  # noqa: E402
import models.rope as rope  # noqa: E402

official_model = importlib.import_module("official.model")

PREFILL_SEQ_LENS = [3, 4, 7, 8, 13]
DECODE_START_POSITIONS = [1, 3, 4, 7, 8, 13]

@pytest.fixture()
def tiny_args(monkeypatch):
    args = official_model.ModelArgs(
        max_batch_size=1,
        max_seq_len=32,
        dtype="bf16",
        scale_dtype="fp32",
        expert_dtype=None,
        vocab_size=16,
        dim=8,
        moe_inter_dim=8,
        n_layers=1,
        n_hash_layers=0,
        n_heads=2,
        n_routed_experts=2,
        n_shared_experts=1,
        n_activated_experts=1,
        q_lora_rank=4,
        head_dim=4,
        rope_head_dim=2,
        norm_eps=1e-6,
        o_groups=1,
        o_lora_rank=4,
        window_size=4,
        compress_ratios=(0,),
        rope_factor=1,
        beta_fast=32,
        beta_slow=1,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=1,
        hc_mult=2,
        hc_sinkhorn_iters=2,
    )

    monkeypatch.setattr(attention_swa, "HIDDEN", args.dim)
    monkeypatch.setattr(attention_swa, "Q_LORA_RANK", args.q_lora_rank)
    monkeypatch.setattr(attention_swa, "N_HEADS", args.n_heads)
    monkeypatch.setattr(attention_swa, "HEAD_DIM", args.head_dim)
    monkeypatch.setattr(attention_swa, "ATTN_Q_OUT", args.n_heads * args.head_dim)
    monkeypatch.setattr(attention_swa, "O_GROUPS", args.o_groups)
    monkeypatch.setattr(attention_swa, "O_LORA_RANK", args.o_lora_rank)
    monkeypatch.setattr(attention_swa, "HEADS_PER_GROUP", args.n_heads // args.o_groups)
    monkeypatch.setattr(attention_swa, "O_GROUP_IN", args.n_heads * args.head_dim // args.o_groups)
    monkeypatch.setattr(attention_swa, "ATTN_OUT_IN", args.o_groups * args.o_lora_rank)
    monkeypatch.setattr(attention_swa, "ROPE_HALF", args.rope_head_dim // 2)
    monkeypatch.setattr(attention_swa, "WINDOW_SIZE", args.window_size)
    monkeypatch.setattr(attention_swa, "TOPK_SWA", args.window_size)
    monkeypatch.setattr(attention_swa, "SOFTMAX_SCALE", args.head_dim**-0.5)
    monkeypatch.setattr(attention_swa, "EPS", args.norm_eps)

    monkeypatch.setattr(rope, "ROPE_DIM", args.rope_head_dim)
    monkeypatch.setattr(rope, "ROPE_HALF", args.rope_head_dim // 2)
    monkeypatch.setattr(rope, "N_HEADS", args.n_heads)
    monkeypatch.setattr(rope, "HEAD_DIM", args.head_dim)
    monkeypatch.setattr(rope, "HEAD_TAIL_OFFSET", args.head_dim - args.rope_head_dim)

    monkeypatch.setattr(official_model, "sparse_attn", torch_sparse_attn)
    monkeypatch.setattr(official_model, "act_quant", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(official_model, "fp4_act_quant", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(official_model, "linear", make_linear_reference())
    monkeypatch.setattr(torch, "einsum", make_einsum_reference(torch.einsum))
    monkeypatch.setattr(torch.Tensor, "square", make_square_reference(torch.Tensor.square))
    return args

def _make_official_attention(args) -> torch.nn.Module:
    torch.manual_seed(20260630)
    with official_model.set_dtype(torch.bfloat16):
        attn = official_model.Attention(0, args)

    with torch.no_grad():
        attn.attn_sink.copy_(torch.randn(args.n_heads, dtype=torch.float32) * 0.1)
        attn.wq_a.weight.copy_(torch.randn(args.q_lora_rank, args.dim, dtype=torch.bfloat16) * 0.1)
        attn.q_norm.weight.copy_(torch.rand(args.q_lora_rank, dtype=torch.float32) + 0.5)
        attn.wq_b.weight.copy_(torch.randn(args.n_heads * args.head_dim, args.q_lora_rank, dtype=torch.bfloat16) * 0.1)
        attn.wkv.weight.copy_(torch.randn(args.head_dim, args.dim, dtype=torch.bfloat16) * 0.1)
        attn.kv_norm.weight.copy_(torch.rand(args.head_dim, dtype=torch.float32) + 0.5)
        attn.wo_a.weight.copy_(
            torch.randn(args.o_groups * args.o_lora_rank, args.n_heads * args.head_dim // args.o_groups, dtype=torch.bfloat16) * 0.1
        )
        attn.wo_b.weight.copy_(torch.randn(args.dim, args.o_groups * args.o_lora_rank, dtype=torch.bfloat16) * 0.1)
    return attn

def _base_tensors(attn: torch.nn.Module, x: torch.Tensor, start_pos: int) -> dict[str, torch.Tensor]:
    seq_len = x.shape[1]
    cos, sin = rope_cos_sin(attn, start_pos, seq_len)
    topk_idxs = official_model.get_window_topk_idxs(attn.window_size, x.shape[0], seq_len, start_pos).int()
    return {
        "x": x.clone(),
        "wq_a_t": attn.wq_a.weight.t().contiguous(),
        "q_norm_w": attn.q_norm.weight.detach().clone(),
        "wq_b_t": attn.wq_b.weight.t().contiguous(),
        "wkv_t": attn.wkv.weight.t().contiguous(),
        "kv_norm_w": attn.kv_norm.weight.detach().clone(),
        "attn_sink": attn.attn_sink.detach().clone(),
        "topk_idxs": topk_idxs,
        "wo_a_t": attn.wo_a.weight.t().contiguous(),
        "wo_b_t": attn.wo_b.weight.t().contiguous(),
        "cos": cos,
        "sin": sin,
    }

def _add_output_tensors(tensors: dict[str, torch.Tensor], args, seq_len: int) -> None:
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
            "out": torch.zeros(1, seq_len, args.dim, dtype=torch.bfloat16),
        }
    )

@pytest.mark.parametrize("seq_len", PREFILL_SEQ_LENS)
def test_attention_swa_prefill_golden_matches_official_model(tiny_args, seq_len: int) -> None:
    attn = _make_official_attention(tiny_args)
    x = torch.randn(1, seq_len, tiny_args.dim, dtype=torch.bfloat16)

    expected = attn(x.clone(), start_pos=0)
    tensors = _base_tensors(attn, x, start_pos=0)
    _add_output_tensors(tensors, tiny_args, seq_len=x.shape[1])

    attention_swa.golden_attention_swa_forward(tensors, start_pos=0)

    torch.testing.assert_close(tensors["out"], expected, rtol=0, atol=0)
    torch.testing.assert_close(tensors["kv_cache_out"], attn.kv_cache, rtol=0, atol=0)

@pytest.mark.parametrize("start_pos", DECODE_START_POSITIONS)
def test_attention_swa_decode_golden_matches_official_model(tiny_args, start_pos: int) -> None:
    attn = _make_official_attention(tiny_args)
    prompt = torch.randn(1, start_pos, tiny_args.dim, dtype=torch.bfloat16)
    token = torch.randn(1, 1, tiny_args.dim, dtype=torch.bfloat16)

    attn(prompt.clone(), start_pos=0)
    kv_cache_before = attn.kv_cache.clone()
    expected = attn(token.clone(), start_pos=start_pos)

    tensors = _base_tensors(attn, token, start_pos=start_pos)
    tensors["kv_cache"] = kv_cache_before
    tensors["cache_pos"] = torch.tensor([start_pos % tiny_args.window_size], dtype=torch.int32)
    _add_output_tensors(tensors, tiny_args, seq_len=1)

    attention_swa.golden_attention_swa_forward(tensors, start_pos=start_pos)

    torch.testing.assert_close(tensors["out"], expected, rtol=0, atol=0)
    torch.testing.assert_close(tensors["kv_cache_out"], attn.kv_cache, rtol=0, atol=0)
