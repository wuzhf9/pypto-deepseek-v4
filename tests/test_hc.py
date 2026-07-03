"""Tests for Hyper-Connections golden logic against official ``model.py``."""

import importlib

import pytest
import torch

import models.hc as hc  # noqa: E402

official_model = importlib.import_module("official.model")

DIM = 16
HC_MULT = 4
MIX_HC = (2 + HC_MULT) * HC_MULT
HC_DIM = HC_MULT * DIM
SINKHORN_ITERS = 3
HC_EPS = 1e-6
NORM_EPS = 1e-6
SEQ_LENS = [1, 3, 8]


@pytest.fixture()
def tiny_block_args(monkeypatch):
    args = official_model.ModelArgs(
        max_batch_size=1,
        max_seq_len=32,
        dtype="bf16",
        scale_dtype="fp32",
        expert_dtype=None,
        vocab_size=32,
        dim=DIM,
        moe_inter_dim=8,
        n_layers=2,
        n_hash_layers=1,
        n_heads=2,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        score_func="sqrtsoftplus",
        route_scale=1.5,
        swiglu_limit=1.5,
        q_lora_rank=4,
        head_dim=8,
        rope_head_dim=2,
        norm_eps=NORM_EPS,
        o_groups=1,
        o_lora_rank=4,
        window_size=4,
        compress_ratios=(0, 0),
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        hc_mult=HC_MULT,
        hc_sinkhorn_iters=SINKHORN_ITERS,
        hc_eps=HC_EPS,
    )

    monkeypatch.setattr(official_model, "world_size", 1)
    monkeypatch.setattr(official_model, "rank", 0)
    monkeypatch.setattr(hc, "HIDDEN", args.dim)
    monkeypatch.setattr(hc, "HC_MULT", args.hc_mult)
    monkeypatch.setattr(hc, "HC_DIM", args.hc_mult * args.dim)
    monkeypatch.setattr(hc, "MIX_HC", (2 + args.hc_mult) * args.hc_mult)
    monkeypatch.setattr(hc, "HC_SINKHORN_ITERS", args.hc_sinkhorn_iters)
    monkeypatch.setattr(hc, "RMS_NORM_EPS", args.norm_eps)
    monkeypatch.setattr(hc, "HC_EPS", args.hc_eps)
    return args


def _official_hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = HC_MULT,
    sinkhorn_iters: int = SINKHORN_ITERS,
    eps: float = HC_EPS,
):
    return hc.split_sinkhorn_golden(
        mixes,
        hc_scale,
        hc_base,
        hc_mult=hc_mult,
        sinkhorn_iters=sinkhorn_iters,
        eps=eps,
    )


def _make_official_block(args) -> torch.nn.Module:
    torch.manual_seed(20260703)
    with official_model.set_dtype(torch.bfloat16):
        block = official_model.Block(0, args)

    gen = torch.Generator().manual_seed(20260704)
    with torch.no_grad():
        block.hc_attn_fn.copy_(torch.randn(MIX_HC, HC_DIM, generator=gen, dtype=torch.float32) * 0.1)
        block.hc_attn_scale.copy_(torch.tensor([0.4, 0.7, 0.9], dtype=torch.float32))
        block.hc_attn_base.copy_(torch.randn(MIX_HC, generator=gen, dtype=torch.float32) * 0.05)
    return block


@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_golden_hc_pre_matches_official_block(tiny_block_args, monkeypatch, seq_len: int) -> None:
    monkeypatch.setattr(official_model, "hc_split_sinkhorn", _official_hc_split_sinkhorn)
    block = _make_official_block(tiny_block_args)
    x = (torch.randn(1, seq_len, HC_MULT, DIM, dtype=torch.float32) * 0.5).to(torch.bfloat16)

    with torch.no_grad():
        expected_x, expected_post, expected_comb = block.hc_pre(
            x.clone(),
            block.hc_attn_fn,
            block.hc_attn_scale,
            block.hc_attn_base,
        )

    tensors = {
        "x": x.clone(),
        "hc_fn": block.hc_attn_fn.detach().clone(),
        "hc_scale": block.hc_attn_scale.detach().clone(),
        "hc_base": block.hc_attn_base.detach().clone(),
        "x_mixed": torch.zeros(1, seq_len, DIM, dtype=torch.bfloat16),
        "post": torch.zeros(1, seq_len, HC_MULT, dtype=torch.float32),
        "comb": torch.zeros(1, seq_len, HC_MULT * HC_MULT, dtype=torch.float32),
    }
    hc.golden_hc_pre(tensors)

    torch.testing.assert_close(tensors["x_mixed"], expected_x, rtol=0, atol=0)
    torch.testing.assert_close(tensors["post"], expected_post, rtol=0, atol=0)
    torch.testing.assert_close(tensors["comb"].view(1, seq_len, HC_MULT, HC_MULT), expected_comb, rtol=0, atol=0)


@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_golden_hc_post_matches_official_block(tiny_block_args, seq_len: int) -> None:
    block = _make_official_block(tiny_block_args)
    x = (torch.randn(1, seq_len, DIM, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    residual = (torch.randn(1, seq_len, HC_MULT, DIM, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    post = torch.sigmoid(torch.randn(1, seq_len, HC_MULT, dtype=torch.float32)) * 2.0
    comb = torch.randn(1, seq_len, HC_MULT, HC_MULT, dtype=torch.float32)

    with torch.no_grad():
        expected = block.hc_post(x.clone(), residual.clone(), post.clone(), comb.clone())

    post_padded = torch.zeros(1, seq_len, hc.HC_PAD, dtype=torch.float32)
    post_padded[..., :HC_MULT] = post
    tensors = {
        "x": x.clone(),
        "residual": residual.clone(),
        "post": post_padded,
        "comb": comb.reshape(1, seq_len, HC_MULT * HC_MULT).clone(),
        "out": torch.zeros(1, seq_len, HC_MULT, DIM, dtype=torch.bfloat16),
    }
    hc.golden_hc_post(tensors)

    torch.testing.assert_close(tensors["out"], expected, rtol=0, atol=0)
