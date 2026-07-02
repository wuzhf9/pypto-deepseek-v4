"""Tests for MoE Expert golden logic against official ``model.py``."""

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

import models.expert as expert  # noqa: E402

official_model = importlib.import_module("official.model")


DIM = 16
INTER_DIM = 8
SWIGLU_LIMIT = 1.5
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
def tiny_expert(monkeypatch):
    monkeypatch.setattr(expert, "HIDDEN", DIM)
    monkeypatch.setattr(expert, "MOE_INTER_DIM", INTER_DIM)
    monkeypatch.setattr(expert, "SWIGLU_LIMIT", SWIGLU_LIMIT)
    monkeypatch.setattr(official_model, "linear", _make_linear_reference())

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
        "gate": torch.zeros(1, seq_len, INTER_DIM, dtype=torch.bfloat16),
        "up": torch.zeros(1, seq_len, INTER_DIM, dtype=torch.bfloat16),
        "hidden": torch.zeros(1, seq_len, INTER_DIM, dtype=torch.bfloat16),
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
