"""Tests for CSA attention golden logic against official ``model.py``."""

import importlib

import pytest
import torch
from conftest import (
    compressor_cos_sin,
    make_einsum_reference,
    make_linear_reference,
    make_square_reference,
    pad_last_dim,
    rope_cos_sin,
    torch_sparse_attn,
)

import models.attention_csa as attention_csa  # noqa: E402
import models.compressor_ratio4 as compressor_ratio4  # noqa: E402
import models.indexer as indexer  # noqa: E402
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
        dim=16,
        moe_inter_dim=16,
        n_layers=1,
        n_hash_layers=0,
        n_heads=2,
        n_routed_experts=2,
        n_shared_experts=1,
        n_activated_experts=1,
        q_lora_rank=4,
        head_dim=8,
        rope_head_dim=2,
        norm_eps=1e-6,
        o_groups=1,
        o_lora_rank=4,
        window_size=4,
        compress_ratios=(4,),
        rope_factor=1,
        beta_fast=32,
        beta_slow=1,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        hc_mult=2,
        hc_sinkhorn_iters=2,
    )

    score_len = args.max_seq_len // 4
    attn_q_out = args.n_heads * args.head_dim
    attn_proj_dim = 2 * args.head_dim
    index_q_out = args.index_n_heads * args.index_head_dim
    index_proj_dim = 2 * args.index_head_dim

    monkeypatch.setattr(attention_csa, "HIDDEN", args.dim)
    monkeypatch.setattr(attention_csa, "Q_LORA_RANK", args.q_lora_rank)
    monkeypatch.setattr(attention_csa, "N_HEADS", args.n_heads)
    monkeypatch.setattr(attention_csa, "HEAD_DIM", args.head_dim)
    monkeypatch.setattr(attention_csa, "ATTN_Q_OUT", attn_q_out)
    monkeypatch.setattr(attention_csa, "O_GROUPS", args.o_groups)
    monkeypatch.setattr(attention_csa, "O_LORA_RANK", args.o_lora_rank)
    monkeypatch.setattr(attention_csa, "HEADS_PER_GROUP", args.n_heads // args.o_groups)
    monkeypatch.setattr(attention_csa, "O_GROUP_IN", attn_q_out // args.o_groups)
    monkeypatch.setattr(attention_csa, "ATTN_OUT_IN", args.o_groups * args.o_lora_rank)
    monkeypatch.setattr(attention_csa, "ROPE_HALF", args.rope_head_dim // 2)
    monkeypatch.setattr(attention_csa, "WINDOW_SIZE", args.window_size)
    monkeypatch.setattr(attention_csa, "INDEX_N_HEADS", args.index_n_heads)
    monkeypatch.setattr(attention_csa, "INDEX_HEAD_DIM", args.index_head_dim)
    monkeypatch.setattr(attention_csa, "INDEX_Q_OUT", index_q_out)
    monkeypatch.setattr(attention_csa, "INDEX_PROJ_DIM", index_proj_dim)
    monkeypatch.setattr(attention_csa, "INDEX_TOPK", args.index_topk)
    monkeypatch.setattr(attention_csa, "INDEX_SCORE_LEN", score_len)
    monkeypatch.setattr(attention_csa, "ATTN_PROJ_DIM", attn_proj_dim)
    monkeypatch.setattr(attention_csa, "TOPK_SWA", args.window_size)
    monkeypatch.setattr(attention_csa, "TOPK_CSA", args.index_topk)
    monkeypatch.setattr(attention_csa, "TOPK_CSA_TOTAL", args.window_size + args.index_topk)
    monkeypatch.setattr(attention_csa, "TOPK_CSA_COMPRESSED", score_len)
    monkeypatch.setattr(attention_csa, "SOFTMAX_SCALE", args.head_dim**-0.5)
    monkeypatch.setattr(attention_csa, "EPS", args.norm_eps)

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

    monkeypatch.setattr(rope, "ROPE_DIM", args.rope_head_dim)
    monkeypatch.setattr(rope, "ROPE_HALF", args.rope_head_dim // 2)
    monkeypatch.setattr(rope, "N_HEADS", args.n_heads)
    monkeypatch.setattr(rope, "HEAD_DIM", args.head_dim)
    monkeypatch.setattr(rope, "INDEX_HEAD_DIM", args.index_head_dim)
    monkeypatch.setattr(rope, "HEAD_TAIL_OFFSET", args.head_dim - args.rope_head_dim)
    monkeypatch.setattr(rope, "INDEX_TAIL_OFFSET", args.index_head_dim - args.rope_head_dim)

    monkeypatch.setattr(official_model, "rotate_activation", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(official_model, "sparse_attn", torch_sparse_attn)
    monkeypatch.setattr(official_model, "act_quant", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(official_model, "fp4_act_quant", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(official_model, "linear", make_linear_reference())
    monkeypatch.setattr(torch, "einsum", make_einsum_reference(torch.einsum))
    monkeypatch.setattr(torch.Tensor, "square", make_square_reference(torch.Tensor.square))
    return args

def _copy_param(param: torch.nn.Parameter, value: torch.Tensor) -> None:
    param.copy_(value.to(param.dtype))

def _make_official_attention(args) -> torch.nn.Module:
    torch.manual_seed(20260702)
    with official_model.set_dtype(torch.bfloat16):
        attn = official_model.Attention(0, args)

    with torch.no_grad():
        _copy_param(attn.attn_sink, torch.randn(args.n_heads, dtype=torch.float32) * 0.1)
        _copy_param(attn.wq_a.weight, torch.randn(args.q_lora_rank, args.dim, dtype=torch.bfloat16) * 0.1)
        _copy_param(attn.q_norm.weight, torch.rand(args.q_lora_rank, dtype=torch.float32) + 0.5)
        _copy_param(attn.wq_b.weight, torch.randn(args.n_heads * args.head_dim, args.q_lora_rank, dtype=torch.bfloat16) * 0.1)
        _copy_param(attn.wkv.weight, torch.randn(args.head_dim, args.dim, dtype=torch.bfloat16) * 0.1)
        _copy_param(attn.kv_norm.weight, torch.rand(args.head_dim, dtype=torch.float32) + 0.5)
        _copy_param(
            attn.wo_a.weight,
            torch.randn(args.o_groups * args.o_lora_rank, args.n_heads * args.head_dim // args.o_groups, dtype=torch.bfloat16)
            * 0.1,
        )
        _copy_param(attn.wo_b.weight, torch.randn(args.dim, args.o_groups * args.o_lora_rank, dtype=torch.bfloat16) * 0.1)

        _copy_param(attn.compressor.ape, torch.randn(4, 2 * args.head_dim) * 0.02)
        _copy_param(attn.compressor.wkv.weight, torch.randn(2 * args.head_dim, args.dim, dtype=torch.bfloat16) * 0.1)
        _copy_param(attn.compressor.wgate.weight, torch.randn(2 * args.head_dim, args.dim, dtype=torch.bfloat16) * 0.1)
        _copy_param(attn.compressor.norm.weight, torch.rand(args.head_dim, dtype=torch.float32) + 0.5)

        _copy_param(
            attn.indexer.wq_b.weight,
            torch.randn(args.index_n_heads * args.index_head_dim, args.q_lora_rank, dtype=torch.bfloat16) * 0.1,
        )
        _copy_param(attn.indexer.weights_proj.weight, torch.randn(args.index_n_heads, args.dim, dtype=torch.bfloat16) * 0.1)
        _copy_param(attn.indexer.compressor.ape, torch.randn(4, 2 * args.index_head_dim) * 0.02)
        _copy_param(attn.indexer.compressor.wkv.weight, torch.randn(2 * args.index_head_dim, args.dim, dtype=torch.bfloat16) * 0.1)
        _copy_param(
            attn.indexer.compressor.wgate.weight,
            torch.randn(2 * args.index_head_dim, args.dim, dtype=torch.bfloat16) * 0.1,
        )
        _copy_param(attn.indexer.compressor.norm.weight, torch.rand(args.index_head_dim, dtype=torch.float32) + 0.5)
    return attn

def _base_tensors(attn: torch.nn.Module, x: torch.Tensor, args, start_pos: int) -> dict[str, torch.Tensor]:
    seq_len = x.shape[1]
    blocks = seq_len // 4
    cos, sin = rope_cos_sin(attn, start_pos, seq_len)
    comp_cos, comp_sin = compressor_cos_sin(attn, 4, seq_len, start_pos)
    window_topk = official_model.get_window_topk_idxs(args.window_size, 1, seq_len, start_pos)
    offset = seq_len if start_pos == 0 else args.window_size
    return {
        "x": x.clone(),
        "wq_a_t": attn.wq_a.weight.detach().t().contiguous().to(torch.bfloat16),
        "q_norm_w": attn.q_norm.weight.detach().clone(),
        "wq_b_t": attn.wq_b.weight.detach().t().contiguous().to(torch.bfloat16),
        "wkv_t": attn.wkv.weight.detach().t().contiguous().to(torch.bfloat16),
        "kv_norm_w": attn.kv_norm.weight.detach().clone(),
        "attn_sink": attn.attn_sink.detach().clone(),
        "window_topk_idxs": pad_last_dim(window_topk, args.window_size),
        "wo_a_t": attn.wo_a.weight.detach().t().contiguous().to(torch.bfloat16),
        "wo_b_t": attn.wo_b.weight.detach().t().contiguous().to(torch.bfloat16),
        "cos": cos,
        "sin": sin,
        "attn_comp_wkv_t": attn.compressor.wkv.weight.detach().t().contiguous().to(torch.bfloat16),
        "attn_comp_wgate_t": attn.compressor.wgate.weight.detach().t().contiguous().to(torch.bfloat16),
        "attn_comp_ape": attn.compressor.ape.detach().clone(),
        "attn_comp_norm_w": attn.compressor.norm.weight.detach().clone(),
        "attn_comp_cos": comp_cos,
        "attn_comp_sin": comp_sin,
        "attn_comp_block_count": torch.tensor([blocks], dtype=torch.int32),
        "idx_wq_b_t": attn.indexer.wq_b.weight.detach().t().contiguous().to(torch.bfloat16),
        "idx_weights_proj_t": attn.indexer.weights_proj.weight.detach().t().contiguous().to(torch.bfloat16),
        "idx_offset": torch.tensor([offset], dtype=torch.int32),
        "idx_comp_wkv_t": attn.indexer.compressor.wkv.weight.detach().t().contiguous().to(torch.bfloat16),
        "idx_comp_wgate_t": attn.indexer.compressor.wgate.weight.detach().t().contiguous().to(torch.bfloat16),
        "idx_comp_ape": attn.indexer.compressor.ape.detach().clone(),
        "idx_comp_norm_w": attn.indexer.compressor.norm.weight.detach().clone(),
        "idx_comp_cos": comp_cos.clone(),
        "idx_comp_sin": comp_sin.clone(),
        "idx_comp_block_count": torch.tensor([blocks], dtype=torch.int32),
    }

def _add_output_tensors(tensors: dict[str, torch.Tensor], args, seq_len: int, *, decode: bool) -> None:
    score_len = args.max_seq_len // 4
    tensors.update(
        {
            "kv_cache_out": torch.zeros(1, args.window_size, args.head_dim, dtype=torch.bfloat16),
            "attn_comp_kv_state_out": torch.zeros(1, 8, 2 * args.head_dim, dtype=torch.float32),
            "attn_comp_score_state_out": torch.zeros(1, 8, 2 * args.head_dim, dtype=torch.float32),
            "attn_comp_cache_out": torch.zeros(1, score_len, args.head_dim, dtype=torch.bfloat16),
            "idx_kv_cache_out": torch.zeros(1, score_len, args.index_head_dim, dtype=torch.bfloat16),
            "idx_comp_kv_state_out": torch.zeros(1, 8, 2 * args.index_head_dim, dtype=torch.float32),
            "idx_comp_score_state_out": torch.zeros(1, 8, 2 * args.index_head_dim, dtype=torch.float32),
            "out": torch.zeros(1, seq_len, args.dim, dtype=torch.bfloat16),
        }
    )

def _assert_score_state_matches(actual: torch.Tensor, expected: torch.Tensor) -> None:
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], rtol=0, atol=0)

@pytest.mark.parametrize("seq_len", PREFILL_SEQ_LENS)
def test_attention_csa_prefill_golden_matches_official_model(tiny_args, seq_len: int) -> None:
    attn = _make_official_attention(tiny_args)
    x = (torch.randn(1, seq_len, tiny_args.dim, dtype=torch.float32) * 0.1).to(torch.bfloat16)

    with torch.no_grad():
        expected = attn(x.clone(), start_pos=0)

    tensors = _base_tensors(attn, x, tiny_args, start_pos=0)
    _add_output_tensors(tensors, tiny_args, seq_len, decode=False)

    attention_csa.golden_attention_csa_forward(tensors, start_pos=0)

    torch.testing.assert_close(tensors["out"], expected, rtol=0, atol=0)
    torch.testing.assert_close(tensors["kv_cache_out"], attn.kv_cache[:, : tiny_args.window_size], rtol=0, atol=0)
    torch.testing.assert_close(tensors["attn_comp_cache_out"], attn.kv_cache[:, tiny_args.window_size :], rtol=0, atol=0)
    torch.testing.assert_close(tensors["attn_comp_kv_state_out"], attn.compressor.kv_state, rtol=0, atol=0)
    _assert_score_state_matches(tensors["attn_comp_score_state_out"], attn.compressor.score_state)
    torch.testing.assert_close(tensors["idx_kv_cache_out"], attn.indexer.kv_cache, rtol=0, atol=0)
    torch.testing.assert_close(tensors["idx_comp_kv_state_out"], attn.indexer.compressor.kv_state, rtol=0, atol=0)
    _assert_score_state_matches(tensors["idx_comp_score_state_out"], attn.indexer.compressor.score_state)

@pytest.mark.parametrize("start_pos", DECODE_START_POSITIONS)
def test_attention_csa_decode_golden_matches_official_model(tiny_args, start_pos: int) -> None:
    attn = _make_official_attention(tiny_args)
    x_prefill = (torch.randn(1, start_pos, tiny_args.dim, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    x_decode = (torch.randn(1, 1, tiny_args.dim, dtype=torch.float32) * 0.1).to(torch.bfloat16)

    with torch.no_grad():
        attn(x_prefill.clone(), start_pos=0)

    kv_cache_before = attn.kv_cache[:, : tiny_args.window_size].detach().clone()
    attn_comp_cache_before = attn.kv_cache[:, tiny_args.window_size :].detach().clone()
    attn_comp_kv_state_before = attn.compressor.kv_state.detach().clone()
    attn_comp_score_state_before = attn.compressor.score_state.detach().clone()
    idx_kv_cache_before = attn.indexer.kv_cache.detach().clone()
    idx_comp_kv_state_before = attn.indexer.compressor.kv_state.detach().clone()
    idx_comp_score_state_before = attn.indexer.compressor.score_state.detach().clone()

    with torch.no_grad():
        expected = attn(x_decode.clone(), start_pos=start_pos)

    tensors = _base_tensors(attn, x_decode, tiny_args, start_pos=start_pos)
    tensors["kv_cache"] = kv_cache_before
    tensors["attn_comp_kv_state"] = attn_comp_kv_state_before
    tensors["attn_comp_score_state"] = attn_comp_score_state_before
    tensors["attn_comp_cache"] = attn_comp_cache_before
    tensors["idx_kv_cache_in"] = idx_kv_cache_before
    tensors["idx_comp_kv_state"] = idx_comp_kv_state_before
    tensors["idx_comp_score_state"] = idx_comp_score_state_before
    tensors["cache_pos"] = torch.tensor([start_pos % tiny_args.window_size], dtype=torch.int32)
    tensors["comp_slot"] = torch.tensor([start_pos % 4], dtype=torch.int32)
    tensors["comp_cache_slot"] = torch.tensor([start_pos // 4], dtype=torch.int32)
    tensors["comp_should_compress"] = torch.tensor([int((start_pos + 1) % 4 == 0)], dtype=torch.int32)
    _add_output_tensors(tensors, tiny_args, seq_len=1, decode=True)
    attention_csa.golden_attention_csa_forward(tensors, start_pos=start_pos)

    torch.testing.assert_close(tensors["out"], expected, rtol=0, atol=0)
    torch.testing.assert_close(tensors["kv_cache_out"], attn.kv_cache[:, : tiny_args.window_size], rtol=0, atol=0)
    torch.testing.assert_close(tensors["attn_comp_cache_out"], attn.kv_cache[:, tiny_args.window_size :], rtol=0, atol=0)
    torch.testing.assert_close(tensors["attn_comp_kv_state_out"], attn.compressor.kv_state, rtol=0, atol=0)
    _assert_score_state_matches(tensors["attn_comp_score_state_out"], attn.compressor.score_state)
    torch.testing.assert_close(tensors["idx_kv_cache_out"], attn.indexer.kv_cache, rtol=0, atol=0)
    torch.testing.assert_close(tensors["idx_comp_kv_state_out"], attn.indexer.compressor.kv_state, rtol=0, atol=0)
    _assert_score_state_matches(tensors["idx_comp_score_state_out"], attn.indexer.compressor.score_state)
