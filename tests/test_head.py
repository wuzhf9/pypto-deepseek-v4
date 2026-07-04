"""Tests for head golden logic against official ``model.py``."""

import importlib

import pytest
import torch

import models.head as head_model


official_model = importlib.import_module("official.model")

VOCAB = 32
HIDDEN = 16
HC_MULT = 4
HC_DIM = HC_MULT * HIDDEN
NORM_EPS = 1e-6
HC_EPS = 1e-6
SEQ_LENS = [1, 5, 13]


@pytest.fixture()
def tiny_head(monkeypatch):
    monkeypatch.setattr(official_model, "world_size", 1)
    monkeypatch.setattr(official_model, "rank", 0)
    monkeypatch.setattr(head_model, "VOCAB", VOCAB)
    monkeypatch.setattr(head_model, "HIDDEN", HIDDEN)
    monkeypatch.setattr(head_model, "HC_MULT", HC_MULT)
    monkeypatch.setattr(head_model, "HC_DIM", HC_DIM)
    monkeypatch.setattr(head_model, "NORM_EPS", NORM_EPS)
    monkeypatch.setattr(head_model, "HC_EPS", HC_EPS)

    head = official_model.ParallelHead(VOCAB, HIDDEN, NORM_EPS, HC_EPS)
    norm = official_model.RMSNorm(HIDDEN, NORM_EPS)
    gen = torch.Generator().manual_seed(20260704)

    with torch.no_grad():
        head.weight.copy_(torch.randn(VOCAB, HIDDEN, generator=gen, dtype=torch.float32) * 0.2)
        norm.weight.copy_(torch.randn(HIDDEN, generator=gen, dtype=torch.float32) * 0.1 + 1.0)

    return head, norm, gen


def _make_inputs(gen: torch.Generator, seq_len: int):
    x = (torch.randn(1, seq_len, HC_MULT, HIDDEN, generator=gen, dtype=torch.float32) * 0.3).to(torch.bfloat16)
    hc_fn = torch.randn(HC_MULT, HC_DIM, generator=gen, dtype=torch.float32) * 0.05
    hc_scale = torch.randn(1, generator=gen, dtype=torch.float32) * 0.1 + 1.0
    hc_base = torch.randn(HC_MULT, generator=gen, dtype=torch.float32) * 0.03
    return x, hc_fn, hc_scale, hc_base


def _head_tensors(
    head: torch.nn.Module,
    norm: torch.nn.Module,
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
) -> dict[str, torch.Tensor]:
    seq_len = x.shape[1]
    return {
        "x": x.clone(),
        "hc_fn": hc_fn.clone(),
        "hc_scale": hc_scale.clone(),
        "hc_base": hc_base.clone(),
        "norm_w": norm.weight.detach().clone(),
        "head_w": head.weight.detach().clone(),
        "logits": torch.zeros(1, VOCAB, dtype=torch.float32),
    }


@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_golden_head_matches_official_model(tiny_head, seq_len: int) -> None:
    head, norm, gen = tiny_head
    x, hc_fn, hc_scale, hc_base = _make_inputs(gen, seq_len)

    with torch.no_grad():
        expected_logits = head(x.clone(), hc_fn.clone(), hc_scale.clone(), hc_base.clone(), norm)

    tensors = _head_tensors(head, norm, x, hc_fn, hc_scale, hc_base)
    head_model.golden_head(tensors)

    torch.testing.assert_close(tensors["logits"], expected_logits, rtol=0, atol=0)


def test_build_head_specs_shapes_and_dtypes(monkeypatch) -> None:
    monkeypatch.setattr(head_model, "VOCAB", VOCAB)
    monkeypatch.setattr(head_model, "HIDDEN", HIDDEN)
    monkeypatch.setattr(head_model, "HC_MULT", HC_MULT)
    monkeypatch.setattr(head_model, "HC_DIM", HC_DIM)
    seq_len = 5

    specs = head_model.build_head_specs(seq_len)
    tensors = {spec.name: spec.create_tensor() for spec in specs}
    padded_seq_len = ((seq_len + head_model.T_TILE - 1) // head_model.T_TILE) * head_model.T_TILE

    assert tensors["x"].shape == (1, seq_len, HC_MULT, HIDDEN)
    assert tensors["x"].dtype == torch.bfloat16
    assert tensors["x_pad"].shape == (1, padded_seq_len, HC_MULT, HIDDEN)
    assert tensors["x_pad"].dtype == torch.bfloat16
    assert tensors["hc_fn"].shape == (head_model.HC_PAD, HC_DIM)
    assert tensors["hc_fn"].dtype == torch.float32
    assert tensors["hc_scale"].shape == (1,)
    assert tensors["hc_base"].shape == (head_model.HC_PAD,)
    assert tensors["norm_w"].shape == (HIDDEN,)
    assert tensors["norm_w"].dtype == torch.bfloat16
    assert tensors["head_w"].shape == (VOCAB, HIDDEN)
    assert tensors["head_w"].dtype == torch.float32
    assert tensors["pre"].shape == (1, padded_seq_len, head_model.HC_PAD)
    assert tensors["hc_out_pad"].shape == (1, padded_seq_len, HIDDEN)
    assert tensors["logits"].shape == (1, VOCAB)
