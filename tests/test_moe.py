"""Tests for route-major MoE golden logic against official ``model.py``."""

import importlib

import pytest
import torch
from conftest import make_linear_reference

import models.gate as gate  # noqa: E402
import models.expert as expert  # noqa: E402
import models.moe as moe  # noqa: E402

official_model = importlib.import_module("official.model")

DIM = 16
INTER_DIM = 8
N_EXPERTS = 8
TOPK = 3
VOCAB = 32
ROUTE_SCALE = 1.5
SWIGLU_LIMIT = 1.5
SEQ_LENS = [1, 3, 13]

@pytest.fixture()
def tiny_moe_args(monkeypatch):
    args = official_model.ModelArgs(
        max_batch_size=1,
        max_seq_len=32,
        dtype="bf16",
        scale_dtype="fp32",
        expert_dtype=None,
        vocab_size=VOCAB,
        dim=DIM,
        moe_inter_dim=INTER_DIM,
        n_layers=2,
        n_hash_layers=1,
        n_heads=2,
        n_routed_experts=N_EXPERTS,
        n_shared_experts=1,
        n_activated_experts=TOPK,
        score_func="sqrtsoftplus",
        route_scale=ROUTE_SCALE,
        swiglu_limit=SWIGLU_LIMIT,
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
    monkeypatch.setattr(official_model, "world_size", 1)
    monkeypatch.setattr(official_model, "rank", 0)
    monkeypatch.setattr(official_model, "linear", make_linear_reference())
    return args

def _copy_expert_weights(module: torch.nn.Module, *, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for expert_module in module.experts:
            assert expert_module is not None
            expert_module.w1.weight.copy_((torch.randn(INTER_DIM, DIM, generator=gen, dtype=torch.bfloat16) * 0.5).to(expert_module.w1.weight.dtype))
            expert_module.w2.weight.copy_((torch.randn(DIM, INTER_DIM, generator=gen, dtype=torch.bfloat16) * 0.5).to(expert_module.w2.weight.dtype))
            expert_module.w3.weight.copy_((torch.randn(INTER_DIM, DIM, generator=gen, dtype=torch.bfloat16) * 0.5).to(expert_module.w3.weight.dtype))

        shared = module.shared_experts
        shared.w1.weight.copy_((torch.randn(INTER_DIM, DIM, generator=gen, dtype=torch.bfloat16) * 0.5).to(shared.w1.weight.dtype))
        shared.w2.weight.copy_((torch.randn(DIM, INTER_DIM, generator=gen, dtype=torch.bfloat16) * 0.5).to(shared.w2.weight.dtype))
        shared.w3.weight.copy_((torch.randn(INTER_DIM, DIM, generator=gen, dtype=torch.bfloat16) * 0.5).to(shared.w3.weight.dtype))

def _make_official_moe(args, *, layer_id: int) -> torch.nn.Module:
    torch.manual_seed(20260703 + layer_id)
    with official_model.set_dtype(torch.bfloat16):
        module = official_model.MoE(layer_id, args)

    with torch.no_grad():
        module.gate.weight.copy_((torch.randn(args.n_routed_experts, args.dim, dtype=torch.bfloat16) * 0.5).to(module.gate.weight.dtype))
        if module.gate.bias is not None:
            module.gate.bias.copy_(torch.randn(args.n_routed_experts, dtype=torch.float32) * 0.2)
        if hasattr(module.gate, "tid2eid") and module.gate.tid2eid is not None:
            base = torch.arange(args.n_activated_experts, dtype=torch.int32).view(1, args.n_activated_experts)
            token_offsets = torch.arange(args.vocab_size, dtype=torch.int32).view(args.vocab_size, 1)
            module.gate.tid2eid.copy_((base + token_offsets) % args.n_routed_experts)

    _copy_expert_weights(module, seed=20260713 + layer_id)
    return module

def _moe_tensors(module: torch.nn.Module, x: torch.Tensor, input_ids: torch.Tensor, *, hash_route: bool) -> dict[str, torch.Tensor]:
    bsz, seq_len, _ = x.shape
    routed_w1_t = torch.stack([expert_module.w1.weight.detach().t().contiguous().to(torch.bfloat16) for expert_module in module.experts])
    routed_w2_t = torch.stack([expert_module.w2.weight.detach().t().contiguous().to(torch.bfloat16) for expert_module in module.experts])
    routed_w3_t = torch.stack([expert_module.w3.weight.detach().t().contiguous().to(torch.bfloat16) for expert_module in module.experts])

    tensors = {
        "x": x.clone(),
        "gate_w_t": module.gate.weight.detach().t().contiguous().to(torch.bfloat16),
        "routed_w1_t": routed_w1_t,
        "routed_w2_t": routed_w2_t,
        "routed_w3_t": routed_w3_t,
        "shared_w1_t": module.shared_experts.w1.weight.detach().t().contiguous().to(torch.bfloat16),
        "shared_w2_t": module.shared_experts.w2.weight.detach().t().contiguous().to(torch.bfloat16),
        "shared_w3_t": module.shared_experts.w3.weight.detach().t().contiguous().to(torch.bfloat16),
        "out": torch.zeros(bsz, seq_len, DIM, dtype=torch.bfloat16),
    }
    if hash_route:
        tensors["tid2eid"] = module.gate.tid2eid.detach().clone()
        tensors["input_ids"] = input_ids.clone()
    else:
        tensors["gate_bias"] = module.gate.bias.detach().clone()
    return tensors

@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_golden_moe_hash_matches_official_model(tiny_moe_args, seq_len: int) -> None:
    module = _make_official_moe(tiny_moe_args, layer_id=0)
    x = (torch.randn(1, seq_len, DIM, dtype=torch.float32) * 0.8).to(torch.bfloat16)
    input_ids = torch.randint(0, VOCAB, (1, seq_len), dtype=torch.int64)

    with torch.no_grad():
        expected = module(x.clone(), input_ids)

    tensors = _moe_tensors(module, x, input_ids, hash_route=True)
    moe.golden_moe_hash(tensors)

    torch.testing.assert_close(tensors["out"], expected, rtol=0, atol=0)

@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_golden_moe_topk_matches_official_model(tiny_moe_args, seq_len: int) -> None:
    module = _make_official_moe(tiny_moe_args, layer_id=1)
    x = (torch.randn(1, seq_len, DIM, dtype=torch.float32) * 0.8).to(torch.bfloat16)
    input_ids = torch.randint(0, VOCAB, (1, seq_len), dtype=torch.int64)

    with torch.no_grad():
        expected = module(x.clone(), input_ids)

    tensors = _moe_tensors(module, x, input_ids, hash_route=False)
    moe.golden_moe_topk(tensors)

    torch.testing.assert_close(tensors["out"], expected, rtol=0, atol=0)
