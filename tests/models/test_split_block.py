"""Tests for split decode Block golden logic."""

import pytest
import torch

import models.gate as gate
import models.split_block as split_block
from test_block import (
    DECODE_START_POS,
    DIM,
    HC_MULT,
    TOPK,
    VOCAB,
    _block_tensors,
    _assert_score_state_close,
    _capture_attention_state,
    _make_block,
    tiny_args,
)


SPLIT_SELECTED_CASES = [
    (
        "swa_hash",
        0,
        "swa",
        True,
        "golden_block_swa_hash_decode",
        "golden_swa_hash_selected_decode_pre_moe",
        "golden_swa_hash_selected_decode_post_moe",
    ),
    (
        "csa_hash",
        2,
        "csa",
        True,
        "golden_block_csa_hash_decode",
        "golden_csa_hash_selected_decode_pre_moe",
        "golden_csa_hash_selected_decode_post_moe",
    ),
    (
        "hca_topk",
        3,
        "hca",
        False,
        "golden_block_hca_topk_decode",
        "golden_hca_topk_selected_decode_pre_moe",
        "golden_hca_topk_selected_decode_post_moe",
    ),
    (
        "csa_topk",
        4,
        "csa",
        False,
        "golden_block_csa_topk_decode",
        "golden_csa_topk_selected_decode_pre_moe",
        "golden_csa_topk_selected_decode_post_moe",
    ),
]


def _attach_selected_expert_weights(tensors: dict[str, torch.Tensor]) -> None:
    indices = tensors["indices"][0, 0].long()
    assert indices.numel() == TOPK
    tensors["selected_w1_t"] = tensors["routed_w1_t"][indices].contiguous()
    tensors["selected_w2_t"] = tensors["routed_w2_t"][indices].contiguous()
    tensors["selected_w3_t"] = tensors["routed_w3_t"][indices].contiguous()


@pytest.mark.parametrize(
    ("case_name", "layer_id", "attention_kind", "hash_route", "packed_fn_name", "pre_fn_name", "post_fn_name"),
    SPLIT_SELECTED_CASES,
)
def test_selected_decode_split_matches_packed_golden(
    tiny_args,
    case_name: str,
    layer_id: int,
    attention_kind: str,
    hash_route: bool,
    packed_fn_name: str,
    pre_fn_name: str,
    post_fn_name: str,
) -> None:
    del case_name
    module = _make_block(tiny_args, layer_id=layer_id)
    prompt = (torch.randn(1, DECODE_START_POS, HC_MULT, DIM, dtype=torch.float32) * 0.2).to(torch.bfloat16)
    prompt_ids = torch.randint(0, VOCAB, (1, DECODE_START_POS), dtype=torch.int64)
    token = (torch.randn(1, 1, HC_MULT, DIM, dtype=torch.float32) * 0.2).to(torch.bfloat16)
    token_ids = torch.randint(0, VOCAB, (1, 1), dtype=torch.int64)

    with torch.no_grad():
        module(prompt.clone(), start_pos=0, input_ids=prompt_ids.clone())

    state = _capture_attention_state(module.attn, attention_kind, tiny_args)
    packed_tensors = _block_tensors(
        module,
        token,
        token_ids,
        tiny_args,
        start_pos=DECODE_START_POS,
        attention_kind=attention_kind,
        hash_route=hash_route,
        decode=True,
        state=state,
    )
    split_tensors = _block_tensors(
        module,
        token,
        token_ids,
        tiny_args,
        start_pos=DECODE_START_POS,
        attention_kind=attention_kind,
        hash_route=hash_route,
        decode=True,
        state=state,
    )

    import models.block as block_golden

    getattr(block_golden, packed_fn_name)(packed_tensors, DECODE_START_POS)
    getattr(split_block, pre_fn_name)(split_tensors, DECODE_START_POS)
    _attach_selected_expert_weights(split_tensors)
    getattr(split_block, post_fn_name)(split_tensors)

    packed_gate_tensors = dict(packed_tensors)
    packed_gate_tensors["x"] = packed_tensors["ffn_normed"]
    packed_gate_tensors["indices"] = torch.zeros_like(split_tensors["indices"])
    packed_gate_tensors["weights"] = torch.zeros_like(split_tensors["weights"])
    gate.golden_gate_forward(packed_gate_tensors, hash_route=hash_route)

    torch.testing.assert_close(split_tensors["indices"], packed_gate_tensors["indices"], rtol=0, atol=0)
    torch.testing.assert_close(split_tensors["weights"], packed_gate_tensors["weights"], rtol=0, atol=0)
    torch.testing.assert_close(split_tensors["ffn_normed"], packed_tensors["ffn_normed"], rtol=0, atol=0)
    torch.testing.assert_close(split_tensors["ffn_hc_post"], packed_tensors["ffn_hc_post"], rtol=0, atol=0)
    torch.testing.assert_close(split_tensors["ffn_hc_comb"], packed_tensors["ffn_hc_comb"], rtol=0, atol=0)
    torch.testing.assert_close(split_tensors["out"], packed_tensors["out"], rtol=0, atol=0)
    torch.testing.assert_close(split_tensors["kv_cache_out"], packed_tensors["kv_cache_out"], rtol=0, atol=0)
    if attention_kind == "hca":
        torch.testing.assert_close(split_tensors["comp_cache_out"], packed_tensors["comp_cache_out"], rtol=0, atol=0)
        torch.testing.assert_close(split_tensors["comp_kv_state_out"], packed_tensors["comp_kv_state_out"], rtol=0, atol=0)
        _assert_score_state_close(split_tensors["comp_score_state_out"], packed_tensors["comp_score_state_out"])
    if attention_kind == "csa":
        torch.testing.assert_close(split_tensors["attn_comp_cache_out"], packed_tensors["attn_comp_cache_out"], rtol=0, atol=0)
        torch.testing.assert_close(split_tensors["attn_comp_kv_state_out"], packed_tensors["attn_comp_kv_state_out"], rtol=0, atol=0)
        _assert_score_state_close(split_tensors["attn_comp_score_state_out"], packed_tensors["attn_comp_score_state_out"])
        torch.testing.assert_close(split_tensors["idx_kv_cache_out"], packed_tensors["idx_kv_cache_out"], rtol=0, atol=0)
        torch.testing.assert_close(split_tensors["idx_comp_kv_state_out"], packed_tensors["idx_comp_kv_state_out"], rtol=0, atol=0)
        _assert_score_state_close(split_tensors["idx_comp_score_state_out"], packed_tensors["idx_comp_score_state_out"])
