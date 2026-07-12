"""Tests for attention output projection golden weight layout."""

import torch
from conftest import official_apply_full_head_rope, official_apply_rotary_emb

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
from models.rope import build_deepseek_v4_rope_tables, materialize_rope_range  # noqa: E402

def _official_attention_out(
    o: torch.Tensor,
    wo_a_weight: torch.Tensor,
    wo_b_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Mirror ``model.py:534,537-542`` with original weights."""

    seq_len = o.shape[1]
    o = official_apply_full_head_rope(o, cos, sin, inverse=True)
    o_grouped = o.view(B, seq_len, O_GROUPS, -1)
    wo_a = wo_a_weight.view(O_GROUPS, O_LORA_RANK, -1)
    proj = torch.einsum("bsgd,grd->bsgr", o_grouped.float(), wo_a.float()).to(torch.bfloat16)
    out = torch.matmul(proj.flatten(2).float(), wo_b_weight.t().float()).to(torch.bfloat16)
    return out

def test_attention_out_golden_uses_same_transposed_weight_as_model_path() -> None:
    torch.manual_seed(20260630)
    seq_len = 7
    start_pos = 3

    o = (torch.randn(B, seq_len, N_HEADS, M.head_dim) * 0.1).to(torch.bfloat16)
    wo_a_weight = (torch.randn(ATTN_OUT_IN, O_GROUP_IN) * 0.02).to(torch.bfloat16)
    wo_b_weight = (torch.randn(HIDDEN, ATTN_OUT_IN) * 0.02).to(torch.bfloat16)
    freqs_cos, freqs_sin = build_deepseek_v4_rope_tables(max_seq_len=start_pos + seq_len)
    cos, sin = materialize_rope_range(freqs_cos, freqs_sin, start_pos, seq_len)

    expected_out = _official_attention_out(
        o,
        wo_a_weight,
        wo_b_weight,
        cos,
        sin,
    )

    tensors = {
        "o": o.clone(),
        "wo_a_t": wo_a_weight.t().contiguous(),
        "wo_b_t": wo_b_weight.t().contiguous(),
        "cos": cos,
        "sin": sin,
        "out": torch.zeros(B, seq_len, HIDDEN, dtype=torch.bfloat16),
    }
    golden_attention_out(tensors)

    torch.testing.assert_close(tensors["out"], expected_out, rtol=0, atol=0)
