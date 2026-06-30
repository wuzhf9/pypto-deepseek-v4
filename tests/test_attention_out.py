"""Tests for attention output projection golden weight layout."""

import sys
import types
from pathlib import Path

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
    language.jit = _Jit()
    language.dynamic = lambda _name: 1

    pypto = types.ModuleType("pypto")
    pypto.language = language
    sys.modules["pypto"] = pypto
    sys.modules["pypto.language"] = language


_install_pypto_language_stub()

from models.attention_out import (  # noqa: E402
    ATTN_OUT_IN,
    B,
    HIDDEN,
    N_HEADS,
    O_GROUP_IN,
    O_GROUPS,
    O_LORA_RANK,
    golden_attention_out,
)
from models.config import FLASH_CONFIG as M  # noqa: E402


def _official_attention_out(
    o: torch.Tensor,
    wo_a_weight: torch.Tensor,
    wo_b_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror ``model.py:537-542`` with the original untransposed weights."""

    seq_len = o.shape[1]
    o_grouped = o.view(B, seq_len, O_GROUPS, -1)
    wo_a = wo_a_weight.view(O_GROUPS, O_LORA_RANK, -1)
    proj = torch.einsum("bsgd,grd->bsgr", o_grouped.float(), wo_a.float()).to(torch.bfloat16)
    out = torch.matmul(proj.flatten(2).float(), wo_b_weight.t().float()).to(torch.bfloat16)
    return proj.flatten(2), out


def test_attention_out_golden_uses_same_transposed_weight_as_model_path() -> None:
    torch.manual_seed(20260630)
    seq_len = 7

    o = (torch.randn(B, seq_len, N_HEADS, M.head_dim) * 0.1).to(torch.bfloat16)
    wo_a_weight = (torch.randn(ATTN_OUT_IN, O_GROUP_IN) * 0.02).to(torch.bfloat16)
    wo_b_weight = (torch.randn(HIDDEN, ATTN_OUT_IN) * 0.02).to(torch.bfloat16)

    expected_proj, expected_out = _official_attention_out(o, wo_a_weight, wo_b_weight)

    tensors = {
        "o": o.clone(),
        "wo_a_t": wo_a_weight.t().contiguous(),
        "wo_b_t": wo_b_weight.t().contiguous(),
        "proj": torch.zeros(B, seq_len, ATTN_OUT_IN, dtype=torch.bfloat16),
        "out": torch.zeros(B, seq_len, HIDDEN, dtype=torch.bfloat16),
    }
    golden_attention_out(tensors)

    torch.testing.assert_close(tensors["proj"], expected_proj, rtol=0, atol=0)
    torch.testing.assert_close(tensors["out"], expected_out, rtol=0, atol=0)
