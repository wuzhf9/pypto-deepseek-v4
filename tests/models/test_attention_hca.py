"""Tests for HCA attention golden logic against official ``model.py``."""

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

import models.attention_hca as attention_hca  # noqa: E402
import models.compressor_ratio128 as compressor_ratio128  # noqa: E402
import models.rope as rope  # noqa: E402

official_model = importlib.import_module("official.model")

COMMON_PREFILL_SEQ_LENS = [3, 4, 7, 8, 13]
COMMON_DECODE_START_POSITIONS = [1, 3, 4, 7, 8, 13]
PREFILL_SEQ_LENS = [*COMMON_PREFILL_SEQ_LENS, 128, 130]
DECODE_START_POSITIONS = [*COMMON_DECODE_START_POSITIONS, 126, 127]

@pytest.fixture()
def tiny_args(monkeypatch):
    args = official_model.ModelArgs(
        max_batch_size=1,
        max_seq_len=512,
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
        compress_ratios=(128,),
        rope_factor=1,
        beta_fast=32,
        beta_slow=1,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=1,
        hc_mult=2,
        hc_sinkhorn_iters=2,
    )

    for module in (attention_hca,):
        monkeypatch.setattr(module, "HIDDEN", args.dim)
        monkeypatch.setattr(module, "Q_LORA_RANK", args.q_lora_rank)
        monkeypatch.setattr(module, "N_HEADS", args.n_heads)
        monkeypatch.setattr(module, "HEAD_DIM", args.head_dim)
        monkeypatch.setattr(module, "ATTN_Q_OUT", args.n_heads * args.head_dim)
        monkeypatch.setattr(module, "O_GROUPS", args.o_groups)
        monkeypatch.setattr(module, "O_LORA_RANK", args.o_lora_rank)
        monkeypatch.setattr(module, "HEADS_PER_GROUP", args.n_heads // args.o_groups)
        monkeypatch.setattr(module, "O_GROUP_IN", args.n_heads * args.head_dim // args.o_groups)
        monkeypatch.setattr(module, "ATTN_OUT_IN", args.o_groups * args.o_lora_rank)
        monkeypatch.setattr(module, "ROPE_HALF", args.rope_head_dim // 2)
        monkeypatch.setattr(module, "WINDOW_SIZE", args.window_size)
        monkeypatch.setattr(module, "TOPK_SWA", args.window_size)
        monkeypatch.setattr(module, "TOPK_HCA", args.max_seq_len // 128)
        monkeypatch.setattr(module, "TOPK_HCA_TOTAL", args.window_size + args.max_seq_len // 128)
        monkeypatch.setattr(module, "SOFTMAX_SCALE", args.head_dim**-0.5)
        monkeypatch.setattr(module, "EPS", args.norm_eps)

    monkeypatch.setattr(compressor_ratio128, "HIDDEN", args.dim)
    monkeypatch.setattr(compressor_ratio128, "HEAD_DIM", args.head_dim)
    monkeypatch.setattr(compressor_ratio128, "ROPE_HALF", args.rope_head_dim // 2)
    monkeypatch.setattr(compressor_ratio128, "TOPK_HCA", args.max_seq_len // 128)

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
    torch.manual_seed(20260701)
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
        attn.compressor.ape.copy_(torch.randn(128, args.head_dim) * 0.02)
        attn.compressor.wkv.weight.copy_(torch.randn(args.head_dim, args.dim, dtype=torch.bfloat16) * 0.1)
        attn.compressor.wgate.weight.copy_(torch.randn(args.head_dim, args.dim, dtype=torch.bfloat16) * 0.1)
        attn.compressor.norm.weight.copy_(torch.rand(args.head_dim, dtype=torch.float32) + 0.5)
    return attn

def _topk_idxs(attn: torch.nn.Module, seq_len: int, start_pos: int) -> torch.Tensor:
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

def _base_tensors(attn: torch.nn.Module, x: torch.Tensor, start_pos: int) -> dict[str, torch.Tensor]:
    seq_len = x.shape[1]
    cos, sin = rope_cos_sin(attn, start_pos, seq_len)
    comp_cos, comp_sin = compressor_cos_sin(attn, 128, seq_len, start_pos)
    return {
        "x": x.clone(),
        "wq_a_t": attn.wq_a.weight.t().contiguous(),
        "q_norm_w": attn.q_norm.weight.detach().clone(),
        "wq_b_t": attn.wq_b.weight.t().contiguous(),
        "wkv_t": attn.wkv.weight.t().contiguous(),
        "kv_norm_w": attn.kv_norm.weight.detach().clone(),
        "attn_sink": attn.attn_sink.detach().clone(),
        "topk_idxs": _topk_idxs(attn, seq_len, start_pos),
        "wo_a_t": attn.wo_a.weight.t().contiguous(),
        "wo_b_t": attn.wo_b.weight.t().contiguous(),
        "cos": cos,
        "sin": sin,
        "comp_wkv_t": attn.compressor.wkv.weight.t().contiguous(),
        "comp_wgate_t": attn.compressor.wgate.weight.t().contiguous(),
        "comp_ape": attn.compressor.ape.detach().clone(),
        "comp_norm_w": attn.compressor.norm.weight.detach().clone(),
        "comp_cos": comp_cos,
        "comp_sin": comp_sin,
    }

def _add_output_tensors(tensors: dict[str, torch.Tensor], args, seq_len: int, *, decode: bool) -> None:
    tensors.update(
        {
            "kv_cache_out": torch.zeros(1, args.window_size, args.head_dim, dtype=torch.bfloat16),
            "comp_kv_state_out": torch.zeros(1, 128, args.head_dim, dtype=torch.float32),
            "comp_score_state_out": torch.zeros(1, 128, args.head_dim, dtype=torch.float32),
            "comp_cache_out": torch.zeros(1, args.max_seq_len // 128, args.head_dim, dtype=torch.bfloat16),
            "out": torch.zeros(1, seq_len, args.dim, dtype=torch.bfloat16),
        }
    )

@pytest.mark.parametrize("seq_len", PREFILL_SEQ_LENS)
def test_attention_hca_prefill_golden_matches_official_model(tiny_args, seq_len: int) -> None:
    attn = _make_official_attention(tiny_args)
    x = torch.randn(1, seq_len, tiny_args.dim, dtype=torch.bfloat16)

    with torch.no_grad():
        expected = attn(x.clone(), start_pos=0)
    tensors = _base_tensors(attn, x, start_pos=0)
    tensors["comp_block_count"] = torch.tensor([seq_len // 128], dtype=torch.int32)
    _add_output_tensors(tensors, tiny_args, seq_len=x.shape[1], decode=False)

    attention_hca.golden_attention_hca_forward(tensors, start_pos=0)

    torch.testing.assert_close(tensors["out"], expected, rtol=0, atol=0)
    torch.testing.assert_close(tensors["kv_cache_out"], attn.kv_cache[:, : tiny_args.window_size], rtol=0, atol=0)
    torch.testing.assert_close(tensors["comp_cache_out"], attn.kv_cache[:, tiny_args.window_size :], rtol=0, atol=0)
    torch.testing.assert_close(tensors["comp_kv_state_out"], attn.compressor.kv_state, rtol=0, atol=0)
    valid = torch.isfinite(attn.compressor.score_state)
    torch.testing.assert_close(tensors["comp_score_state_out"][valid], attn.compressor.score_state[valid], rtol=0, atol=0)

@pytest.mark.parametrize("start_pos", DECODE_START_POSITIONS)
def test_attention_hca_decode_golden_matches_official_model(tiny_args, start_pos: int) -> None:
    attn = _make_official_attention(tiny_args)
    token = torch.randn(1, 1, tiny_args.dim, dtype=torch.bfloat16)
    with torch.no_grad():
        attn.kv_cache.copy_((torch.randn_like(attn.kv_cache.float()) * 0.1).to(torch.bfloat16))
        attn.compressor.kv_state.copy_(torch.randn_like(attn.compressor.kv_state) * 0.1)
        attn.compressor.score_state.copy_(torch.randn_like(attn.compressor.score_state) * 0.1)
        attn.compressor.kv_cache = attn.kv_cache[:, tiny_args.window_size :]
        attn.compressor.freqs_cis = attn.freqs_cis

    kv_cache_before = attn.kv_cache[:, : tiny_args.window_size].clone()
    comp_cache_before = attn.kv_cache[:, tiny_args.window_size :].clone()
    comp_kv_state_before = attn.compressor.kv_state.clone()
    comp_score_state_before = attn.compressor.score_state.clone()

    with torch.no_grad():
        expected = attn(token.clone(), start_pos=start_pos)

    tensors = _base_tensors(attn, token, start_pos=start_pos)
    tensors["kv_cache"] = kv_cache_before
    tensors["comp_kv_state"] = comp_kv_state_before
    tensors["comp_score_state"] = comp_score_state_before
    tensors["comp_cache"] = comp_cache_before
    tensors["cache_pos"] = torch.tensor([start_pos % tiny_args.window_size], dtype=torch.int32)
    tensors["comp_slot"] = torch.tensor([start_pos % 128], dtype=torch.int32)
    tensors["comp_cache_slot"] = torch.tensor([start_pos // 128], dtype=torch.int32)
    tensors["comp_should_compress"] = torch.tensor([int((start_pos + 1) % 128 == 0)], dtype=torch.int32)
    _add_output_tensors(tensors, tiny_args, seq_len=1, decode=True)

    attention_hca.golden_attention_hca_forward(tensors, start_pos=start_pos)

    torch.testing.assert_close(tensors["out"], expected, rtol=0, atol=0)
    torch.testing.assert_close(tensors["kv_cache_out"], attn.kv_cache[:, : tiny_args.window_size], rtol=0, atol=0)
    torch.testing.assert_close(tensors["comp_cache_out"], attn.kv_cache[:, tiny_args.window_size :], rtol=0, atol=0)
    torch.testing.assert_close(tensors["comp_kv_state_out"], attn.compressor.kv_state, rtol=0, atol=0)
    torch.testing.assert_close(tensors["comp_score_state_out"], attn.compressor.score_state, rtol=0, atol=0)

def test_attention_hca_continuous_decode_crosses_ratio128_boundary(tiny_args) -> None:
    attn = _make_official_attention(tiny_args)
    prefill_len = 126
    torch.manual_seed(20260706)
    prefill_x = torch.randn(1, prefill_len, tiny_args.dim, dtype=torch.bfloat16)

    with torch.no_grad():
        expected = attn(prefill_x.clone(), start_pos=0)
    tensors = _base_tensors(attn, prefill_x, start_pos=0)
    tensors["comp_block_count"] = torch.tensor([prefill_len // 128], dtype=torch.int32)
    _add_output_tensors(tensors, tiny_args, seq_len=prefill_len, decode=False)

    attention_hca.golden_attention_hca_forward(tensors, start_pos=0)

    torch.testing.assert_close(tensors["out"], expected, rtol=0, atol=0)
    torch.testing.assert_close(tensors["kv_cache_out"], attn.kv_cache[:, : tiny_args.window_size], rtol=0, atol=0)
    torch.testing.assert_close(tensors["comp_cache_out"], attn.kv_cache[:, tiny_args.window_size :], rtol=0, atol=0)
    torch.testing.assert_close(tensors["comp_kv_state_out"], attn.compressor.kv_state, rtol=0, atol=0)
    valid = torch.isfinite(attn.compressor.score_state)
    torch.testing.assert_close(tensors["comp_score_state_out"][valid], attn.compressor.score_state[valid], rtol=0, atol=0)

    for start_pos in (126, 127, 128, 129):
        token = torch.randn(1, 1, tiny_args.dim, dtype=torch.bfloat16)
        kv_cache_before = attn.kv_cache[:, : tiny_args.window_size].clone()
        comp_cache_before = attn.kv_cache[:, tiny_args.window_size :].clone()
        comp_kv_state_before = attn.compressor.kv_state.clone()
        comp_score_state_before = attn.compressor.score_state.clone()

        with torch.no_grad():
            expected = attn(token.clone(), start_pos=start_pos)

        tensors = _base_tensors(attn, token, start_pos=start_pos)
        tensors["kv_cache"] = kv_cache_before
        tensors["comp_kv_state"] = comp_kv_state_before
        tensors["comp_score_state"] = comp_score_state_before
        tensors["comp_cache"] = comp_cache_before
        tensors["cache_pos"] = torch.tensor([start_pos % tiny_args.window_size], dtype=torch.int32)
        tensors["comp_slot"] = torch.tensor([start_pos % 128], dtype=torch.int32)
        tensors["comp_cache_slot"] = torch.tensor([start_pos // 128], dtype=torch.int32)
        tensors["comp_should_compress"] = torch.tensor([int((start_pos + 1) % 128 == 0)], dtype=torch.int32)
        _add_output_tensors(tensors, tiny_args, seq_len=1, decode=True)

        attention_hca.golden_attention_hca_forward(tensors, start_pos=start_pos)

        torch.testing.assert_close(tensors["out"], expected, rtol=0, atol=0)
        torch.testing.assert_close(tensors["kv_cache_out"], attn.kv_cache[:, : tiny_args.window_size], rtol=0, atol=0)
        torch.testing.assert_close(tensors["comp_cache_out"], attn.kv_cache[:, tiny_args.window_size :], rtol=0, atol=0)
        torch.testing.assert_close(tensors["comp_kv_state_out"], attn.compressor.kv_state, rtol=0, atol=0)
        torch.testing.assert_close(tensors["comp_score_state_out"], attn.compressor.score_state, rtol=0, atol=0)
