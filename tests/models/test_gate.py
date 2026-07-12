"""Tests for MoE Gate golden logic against official ``model.py``."""

import importlib

import pytest
import torch
from conftest import make_linear_reference

import models.gate as gate  # noqa: E402

official_model = importlib.import_module("official.model")

DIM = 16
N_EXPERTS = 8
TOPK = 3
VOCAB = 32
ROUTE_SCALE = 1.5
SEQ_LENS = [1, 3, 13]

@pytest.fixture()
def tiny_gate_args(monkeypatch):
    args = official_model.ModelArgs(
        max_batch_size=1,
        max_seq_len=32,
        dtype="bf16",
        scale_dtype="fp32",
        expert_dtype=None,
        vocab_size=VOCAB,
        dim=DIM,
        moe_inter_dim=8,
        n_layers=2,
        n_hash_layers=1,
        n_heads=2,
        n_routed_experts=N_EXPERTS,
        n_shared_experts=1,
        n_activated_experts=TOPK,
        score_func="sqrtsoftplus",
        route_scale=ROUTE_SCALE,
        q_lora_rank=4,
        head_dim=8,
        rope_head_dim=2,
        norm_eps=1e-6,
        o_groups=1,
        o_lora_rank=4,
        window_size=4,
        compress_ratios=(0, 0),
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        hc_mult=2,
        hc_sinkhorn_iters=2,
    )

    monkeypatch.setattr(gate, "HIDDEN", args.dim)
    monkeypatch.setattr(gate, "N_EXPERTS", args.n_routed_experts)
    monkeypatch.setattr(gate, "TOPK", args.n_activated_experts)
    monkeypatch.setattr(gate, "VOCAB", args.vocab_size)
    monkeypatch.setattr(gate, "ROUTE_SCALE", args.route_scale)
    monkeypatch.setattr(official_model, "linear", make_linear_reference())
    return args

def _make_official_gate(args, *, layer_id: int) -> torch.nn.Module:
    torch.manual_seed(20260703 + layer_id)
    with official_model.set_dtype(torch.bfloat16):
        module = official_model.Gate(layer_id, args)

    with torch.no_grad():
        module.weight.copy_((torch.randn(args.n_routed_experts, args.dim, dtype=torch.bfloat16) * 0.5).to(module.weight.dtype))
        if module.bias is not None:
            module.bias.copy_(torch.randn(args.n_routed_experts, dtype=torch.float32) * 0.2)
        if hasattr(module, "tid2eid") and module.tid2eid is not None:
            base = torch.arange(args.n_activated_experts, dtype=torch.int32).view(1, args.n_activated_experts)
            token_offsets = torch.arange(args.vocab_size, dtype=torch.int32).view(args.vocab_size, 1)
            module.tid2eid.copy_((base + token_offsets) % args.n_routed_experts)
    return module

def _base_tensors(module: torch.nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    bsz, seq_len, _ = x.shape
    return {
        "x": x.clone(),
        "gate_w_t": module.weight.detach().t().contiguous().to(torch.bfloat16),
        "indices": torch.zeros(bsz, seq_len, TOPK, dtype=torch.int32),
        "weights": torch.zeros(bsz, seq_len, TOPK, dtype=torch.float32),
    }

@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_golden_gate_hash_matches_official_model(tiny_gate_args, seq_len: int) -> None:
    module = _make_official_gate(tiny_gate_args, layer_id=0)
    x = (torch.randn(1, seq_len, DIM, dtype=torch.float32) * 0.8).to(torch.bfloat16)
    input_ids = torch.randint(0, VOCAB, (1, seq_len), dtype=torch.int64)

    with torch.no_grad():
        expected_weights, expected_indices = module(x.view(-1, DIM), input_ids.flatten())

    tensors = _base_tensors(module, x)
    tensors["tid2eid"] = module.tid2eid.detach().clone()
    tensors["input_ids"] = input_ids.clone()
    gate.golden_gate_hash(tensors)

    torch.testing.assert_close(tensors["indices"], expected_indices.view(1, seq_len, TOPK).to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(tensors["weights"], expected_weights.view(1, seq_len, TOPK), rtol=0, atol=0)

@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_golden_gate_topk_matches_official_model(tiny_gate_args, seq_len: int) -> None:
    module = _make_official_gate(tiny_gate_args, layer_id=1)
    x = (torch.randn(1, seq_len, DIM, dtype=torch.float32) * 0.8).to(torch.bfloat16)

    with torch.no_grad():
        expected_weights, expected_indices = module(x.view(-1, DIM))

    tensors = _base_tensors(module, x)
    tensors["gate_bias"] = module.bias.detach().clone()
    gate.golden_gate_topk(tensors)

    torch.testing.assert_close(tensors["indices"], expected_indices.view(1, seq_len, TOPK).to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(tensors["weights"], expected_weights.view(1, seq_len, TOPK), rtol=0, atol=0)
