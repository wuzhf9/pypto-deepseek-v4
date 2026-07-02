"""Tests for MoE Gate golden logic against official ``model.py``."""

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_pypto_language_stub() -> None:
    """Allow importing PyPTO kernel modules in host-only unit tests."""

    if "pypto.language" in sys.modules:
        return

    class _Tensor:
        def __class_getitem__(cls, _item):
            return cls

    class _Jit:
        def __call__(self, fn):
            return fn

        def inline(self, fn):
            return fn

    language = types.ModuleType("pypto.language")
    language.Tensor = _Tensor
    language.Out = _Tensor
    language.BF16 = object()
    language.FP32 = object()
    language.INT32 = object()
    language.INT64 = object()
    language.INDEX = object()
    language.jit = _Jit()
    language.dynamic = lambda _name: 1

    pypto = types.ModuleType("pypto")
    pypto.language = language
    sys.modules["pypto"] = pypto
    sys.modules["pypto.language"] = language


def _install_official_kernel_stub() -> None:
    kernel = types.ModuleType("kernel")
    kernel.act_quant = lambda x, *args, **kwargs: x
    kernel.fp4_act_quant = lambda x, *args, **kwargs: x
    kernel.fp8_gemm = None
    kernel.fp4_gemm = None
    kernel.sparse_attn = None
    kernel.hc_split_sinkhorn = None
    sys.modules["kernel"] = kernel


_install_pypto_language_stub()
_install_official_kernel_stub()

import models.gate as gate  # noqa: E402

official_model = importlib.import_module("official.model")


DIM = 16
N_EXPERTS = 8
TOPK = 3
VOCAB = 32
ROUTE_SCALE = 1.5
SEQ_LENS = [1, 3, 13]


def _make_linear_reference():
    def linear(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert bias is None
        return torch.matmul(x.float(), weight.t().contiguous().float()).to(x.dtype)

    return linear


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
    monkeypatch.setattr(official_model, "linear", _make_linear_reference())
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
        "logits": torch.zeros(bsz, seq_len, N_EXPERTS, dtype=torch.float32),
        "scores": torch.zeros(bsz, seq_len, N_EXPERTS, dtype=torch.float32),
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
