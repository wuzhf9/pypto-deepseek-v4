"""Tests for MoE Expert golden logic against official ``model.py``."""

import importlib

import pytest
import torch
from conftest import make_linear_reference

import models.expert as expert  # noqa: E402

official_model = importlib.import_module("official.model")

DIM = 16
INTER_DIM = 8
SWIGLU_LIMIT = 1.5
SEQ_LENS = [1, 3, 13]

@pytest.fixture()
def tiny_expert(monkeypatch):
    monkeypatch.setattr(expert, "HIDDEN", DIM)
    monkeypatch.setattr(expert, "MOE_INTER_DIM", INTER_DIM)
    monkeypatch.setattr(expert, "SWIGLU_LIMIT", SWIGLU_LIMIT)
    monkeypatch.setattr(official_model, "linear", make_linear_reference())

    torch.manual_seed(20260703)
    with official_model.set_dtype(torch.bfloat16):
        module = official_model.Expert(DIM, INTER_DIM, swiglu_limit=SWIGLU_LIMIT)

    with torch.no_grad():
        module.w1.weight.copy_((torch.randn(INTER_DIM, DIM, dtype=torch.bfloat16) * 0.5).to(module.w1.weight.dtype))
        module.w2.weight.copy_((torch.randn(DIM, INTER_DIM, dtype=torch.bfloat16) * 0.5).to(module.w2.weight.dtype))
        module.w3.weight.copy_((torch.randn(INTER_DIM, DIM, dtype=torch.bfloat16) * 0.5).to(module.w3.weight.dtype))
    return module

def _base_tensors(module: torch.nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    seq_len = x.shape[1]
    return {
        "x": x.clone(),
        "w1_t": module.w1.weight.detach().t().contiguous().to(torch.bfloat16),
        "w2_t": module.w2.weight.detach().t().contiguous().to(torch.bfloat16),
        "w3_t": module.w3.weight.detach().t().contiguous().to(torch.bfloat16),
        "out": torch.zeros(1, seq_len, DIM, dtype=torch.bfloat16),
    }

@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_golden_expert_shared_matches_official_model(tiny_expert, seq_len: int) -> None:
    x = (torch.randn(1, seq_len, DIM, dtype=torch.float32) * 0.8).to(torch.bfloat16)

    with torch.no_grad():
        expected = tiny_expert(x.clone())

    tensors = _base_tensors(tiny_expert, x)
    expert.golden_expert_shared(tensors)

    torch.testing.assert_close(tensors["out"], expected, rtol=0, atol=0)

@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_golden_expert_routed_matches_official_model(tiny_expert, seq_len: int) -> None:
    x = (torch.randn(1, seq_len, DIM, dtype=torch.float32) * 0.8).to(torch.bfloat16)
    weights = torch.rand(1, seq_len, 1, dtype=torch.float32) * 0.9 + 0.1

    with torch.no_grad():
        expected = tiny_expert(x.clone(), weights=weights)

    tensors = _base_tensors(tiny_expert, x)
    tensors["weights"] = weights.clone()
    expert.golden_expert_routed(tensors)

    torch.testing.assert_close(tensors["out"], expected, rtol=0, atol=0)
