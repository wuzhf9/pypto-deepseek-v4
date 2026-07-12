"""Tests for Indexer golden logic against official ``model.py``."""

import importlib

import pytest
import torch
from conftest import make_linear_reference

import models.compressor_ratio4 as compressor_ratio4  # noqa: E402
import models.indexer as indexer  # noqa: E402
import models.rope as rope  # noqa: E402

official_model = importlib.import_module("official.model")

@pytest.fixture()
def tiny_indexer_args(monkeypatch):
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

    index_q_out = args.index_n_heads * args.index_head_dim
    index_proj_dim = 2 * args.index_head_dim
    score_len = args.max_seq_len // 4

    monkeypatch.setattr(indexer, "HIDDEN", args.dim)
    monkeypatch.setattr(indexer, "Q_LORA_RANK", args.q_lora_rank)
    monkeypatch.setattr(indexer, "INDEX_N_HEADS", args.index_n_heads)
    monkeypatch.setattr(indexer, "INDEX_HEAD_DIM", args.index_head_dim)
    monkeypatch.setattr(indexer, "INDEX_Q_OUT", index_q_out)
    monkeypatch.setattr(indexer, "INDEX_PROJ_DIM", index_proj_dim)
    monkeypatch.setattr(indexer, "INDEX_TOPK", args.index_topk)
    monkeypatch.setattr(indexer, "INDEX_SCORE_LEN", score_len)
    monkeypatch.setattr(indexer, "ROPE_HALF", args.rope_head_dim // 2)
    monkeypatch.setattr(
        indexer,
        "INDEX_WEIGHTS_SCALE",
        (args.index_head_dim**-0.5) * (args.index_n_heads**-0.5),
    )

    monkeypatch.setattr(compressor_ratio4, "HIDDEN", args.dim)
    monkeypatch.setattr(compressor_ratio4, "INDEX_HEAD_DIM", args.index_head_dim)
    monkeypatch.setattr(compressor_ratio4, "INDEX_PROJ_DIM", index_proj_dim)
    monkeypatch.setattr(compressor_ratio4, "ROPE_HALF", args.rope_head_dim // 2)
    monkeypatch.setattr(compressor_ratio4, "TOPK_CSA_COMPRESSED", score_len)

    monkeypatch.setattr(rope, "ROPE_DIM", args.rope_head_dim)
    monkeypatch.setattr(rope, "ROPE_HALF", args.rope_head_dim // 2)
    monkeypatch.setattr(rope, "INDEX_HEAD_DIM", args.index_head_dim)
    monkeypatch.setattr(rope, "INDEX_TAIL_OFFSET", args.index_head_dim - args.rope_head_dim)

    monkeypatch.setattr(official_model, "rotate_activation", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(official_model, "fp4_act_quant", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(official_model, "act_quant", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(official_model, "linear", make_linear_reference())
    return args

def _make_official_indexer(args) -> torch.nn.Module:
    torch.manual_seed(20260702)
    with official_model.set_dtype(torch.bfloat16):
        module = official_model.Indexer(args, compress_ratio=4)

    index_q_out = args.index_n_heads * args.index_head_dim
    index_proj_dim = 2 * args.index_head_dim
    with torch.no_grad():
        module.wq_b.weight.copy_(
            (torch.randn(index_q_out, args.q_lora_rank, dtype=torch.bfloat16) * 0.02).float()
        )
        module.weights_proj.weight.copy_(
            (torch.randn(args.index_n_heads, args.dim, dtype=torch.bfloat16) * 0.02).float()
        )
        module.compressor.ape.copy_(torch.randn(4, index_proj_dim) * 0.02)
        module.compressor.wkv.weight.copy_(
            (torch.randn(index_proj_dim, args.dim, dtype=torch.bfloat16) * 0.02).float()
        )
        module.compressor.wgate.weight.copy_(
            (torch.randn(index_proj_dim, args.dim, dtype=torch.bfloat16) * 0.02).float()
        )
        module.compressor.norm.weight.copy_(torch.randn(args.index_head_dim) * 0.1 + 1.0)

    module.kv_cache.zero_()
    module.freqs_cis = official_model.precompute_freqs_cis(
        args.rope_head_dim,
        args.max_seq_len,
        args.original_seq_len,
        args.compress_rope_theta,
        args.rope_factor,
        args.beta_fast,
        args.beta_slow,
    )
    return module

def _pad_topk(expected: torch.Tensor, seq_len: int, topk: int) -> torch.Tensor:
    padded = torch.full((1, seq_len, topk), -1, dtype=torch.int32)
    if expected.shape[-1] > 0:
        padded[:, :, : expected.shape[-1]] = expected.to(torch.int32)
    return padded

def _assert_score_state_matches(actual: torch.Tensor, expected: torch.Tensor) -> None:
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], rtol=0, atol=0)

def _common_tensors(module: torch.nn.Module, x: torch.Tensor, qr: torch.Tensor, args, start_pos: int, offset: int):
    freqs_cis = module.freqs_cis[start_pos : start_pos + x.shape[1]]
    return {
        "x": x.clone(),
        "qr": qr.clone(),
        "wq_b_t": module.wq_b.weight.detach().t().contiguous().to(torch.bfloat16),
        "weights_proj_t": module.weights_proj.weight.detach().t().contiguous().to(torch.bfloat16),
        "cos": freqs_cis.real.contiguous(),
        "sin": freqs_cis.imag.contiguous(),
        "offset": torch.tensor([offset], dtype=torch.int32),
        "comp_wkv_t": module.compressor.wkv.weight.detach().t().contiguous().to(torch.bfloat16),
        "comp_wgate_t": module.compressor.wgate.weight.detach().t().contiguous().to(torch.bfloat16),
        "comp_ape": module.compressor.ape.detach().clone(),
        "comp_norm_w": module.compressor.norm.weight.detach().clone(),
        "topk_idxs": torch.full((1, x.shape[1], args.index_topk), -1, dtype=torch.int32),
        "index_kv_cache": module.kv_cache.detach().clone(),
        "comp_kv_state_out": torch.zeros_like(module.compressor.kv_state),
        "comp_score_state_out": torch.zeros_like(module.compressor.score_state),
    }

def _prefill_tensors(module: torch.nn.Module, x: torch.Tensor, qr: torch.Tensor, args, offset: int):
    seq_len = x.shape[1]
    cutoff = seq_len - seq_len % 4
    freqs_cis = module.freqs_cis[:cutoff:4]
    blocks = freqs_cis.shape[0]
    if blocks == 0:
        freqs_cis = module.freqs_cis[:1]
    compressed_len = max(1, blocks)

    tensors = _common_tensors(module, x, qr, args, start_pos=0, offset=offset)
    tensors.update(
        {
            "comp_cos": freqs_cis.real.contiguous(),
            "comp_sin": freqs_cis.imag.contiguous(),
            "comp_block_count": torch.tensor([blocks], dtype=torch.int32),
        }
    )
    return tensors

def _decode_tensors(module: torch.nn.Module, x: torch.Tensor, qr: torch.Tensor, args, start_pos: int, offset: int):
    should_compress = int((start_pos + 1) % 4 == 0)
    if should_compress:
        rope_pos = start_pos + 1 - 4
        freqs_cis = module.freqs_cis[rope_pos : rope_pos + 1]
        comp_cos = freqs_cis.real.contiguous()
        comp_sin = freqs_cis.imag.contiguous()
    else:
        comp_cos = torch.zeros(1, args.rope_head_dim // 2, dtype=torch.float32)
        comp_sin = torch.zeros(1, args.rope_head_dim // 2, dtype=torch.float32)

    tensors = _common_tensors(module, x, qr, args, start_pos=start_pos, offset=offset)
    tensors.update(
        {
            "comp_kv_state": module.compressor.kv_state.detach().clone(),
            "comp_score_state": module.compressor.score_state.detach().clone(),
            "comp_slot": torch.tensor([start_pos % 4], dtype=torch.int32),
            "comp_cache_slot": torch.tensor([start_pos // 4], dtype=torch.int32),
            "comp_should_compress": torch.tensor([should_compress], dtype=torch.int32),
            "comp_cos": comp_cos,
            "comp_sin": comp_sin,
        }
    )
    tensors["index_kv_cache_in"] = tensors["index_kv_cache"].clone()
    return tensors

@pytest.mark.parametrize("seq_len", [3, 4, 7, 8, 13, 16, 32])
def test_golden_indexer_prefill_matches_official_model(tiny_indexer_args, seq_len: int) -> None:
    module = _make_official_indexer(tiny_indexer_args)
    x = (torch.randn(1, seq_len, tiny_indexer_args.dim, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    qr = (torch.randn(1, seq_len, tiny_indexer_args.q_lora_rank, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    offset = seq_len

    tensors = _prefill_tensors(module, x, qr, tiny_indexer_args, offset)
    expected = module(x.clone(), qr.clone(), start_pos=0, offset=offset)

    indexer.golden_indexer_prefill(tensors)

    torch.testing.assert_close(
        tensors["topk_idxs"],
        _pad_topk(expected, seq_len, tiny_indexer_args.index_topk),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(tensors["index_kv_cache"], module.kv_cache, rtol=0, atol=0)
    torch.testing.assert_close(tensors["comp_kv_state_out"], module.compressor.kv_state, rtol=0, atol=0)
    _assert_score_state_matches(tensors["comp_score_state_out"], module.compressor.score_state)

@pytest.mark.parametrize("start_pos", [1, 2, 3, 7])
def test_golden_indexer_decode_matches_official_model(tiny_indexer_args, start_pos: int) -> None:
    module = _make_official_indexer(tiny_indexer_args)
    x = (torch.randn(1, 1, tiny_indexer_args.dim, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    qr = (torch.randn(1, 1, tiny_indexer_args.q_lora_rank, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    offset = tiny_indexer_args.window_size
    with torch.no_grad():
        module.kv_cache.copy_((torch.randn_like(module.kv_cache.float()) * 0.1).to(torch.bfloat16))
        module.compressor.kv_state.copy_(torch.randn_like(module.compressor.kv_state) * 0.1)
        module.compressor.score_state.copy_(torch.randn_like(module.compressor.score_state) * 0.1)

    tensors = _decode_tensors(module, x, qr, tiny_indexer_args, start_pos, offset)
    expected = module(x.clone(), qr.clone(), start_pos=start_pos, offset=offset)

    indexer.golden_indexer_decode(tensors)

    torch.testing.assert_close(
        tensors["topk_idxs"],
        _pad_topk(expected, 1, tiny_indexer_args.index_topk),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(tensors["index_kv_cache"], module.kv_cache, rtol=0, atol=0)
    torch.testing.assert_close(tensors["comp_kv_state_out"], module.compressor.kv_state, rtol=0, atol=0)
    _assert_score_state_matches(tensors["comp_score_state_out"], module.compressor.score_state)
