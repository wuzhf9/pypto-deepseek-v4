import importlib

import pytest
import torch

from models.config import FLASH_CONFIG
from serving.state import (
    COMPRESS_RATIO4,
    COMPRESS_RATIO128,
    DEFAULT_MAX_SEQ_LEN,
    DeepSeekV4State,
    build_compress_topk_idxs,
    build_deepseek_v4_rope_tables,
    build_window_topk_idxs,
    materialize_compressor_rope,
    materialize_rope_range,
)


official_model = importlib.import_module("official.model")


def _assert_matches_official_with_padding(actual: torch.Tensor, expected: torch.Tensor) -> None:
    expected = expected.to(actual.dtype)
    assert actual.shape[:2] == expected.shape[:2]
    assert actual.shape[2] >= expected.shape[2]

    torch.testing.assert_close(actual[:, :, : expected.shape[2]], expected, rtol=0, atol=0)
    if actual.shape[2] > expected.shape[2]:
        torch.testing.assert_close(
            actual[:, :, expected.shape[2] :],
            torch.full_like(actual[:, :, expected.shape[2] :], -1),
            rtol=0,
            atol=0,
        )


def _official_cos_sin(compress_ratio: int, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    if compress_ratio:
        original_seq_len = FLASH_CONFIG.original_seq_len
        rope_theta = FLASH_CONFIG.compress_rope_theta
    else:
        original_seq_len = 0
        rope_theta = FLASH_CONFIG.rope_theta
    freqs = official_model.precompute_freqs_cis(
        FLASH_CONFIG.rope_head_dim,
        max_seq_len,
        original_seq_len,
        rope_theta,
        FLASH_CONFIG.rope_factor,
        FLASH_CONFIG.beta_fast,
        FLASH_CONFIG.beta_slow,
    )
    return freqs.real.contiguous(), freqs.imag.contiguous()


def _tensor_from_spec(specs, name: str) -> torch.Tensor:
    for spec in specs:
        if spec.name == name:
            return spec.create_tensor()
    raise AssertionError(f"missing TensorSpec: {name}")


def test_layer_specs_follow_official_config():
    state = DeepSeekV4State()
    assert len(state.layers) == FLASH_CONFIG.n_layers
    for layer_id in range(FLASH_CONFIG.n_layers):
        spec = state.layer_spec(layer_id)
        assert spec.layer_id == layer_id
        assert spec.ratio == FLASH_CONFIG.compress_ratios[layer_id]
        assert spec.hash_route == (layer_id < FLASH_CONFIG.n_hash_layers)

    assert state.layer_spec(0).ratio == 0
    assert state.layer_spec(2).ratio == COMPRESS_RATIO4
    assert state.layer_spec(3).ratio == COMPRESS_RATIO128
    assert state.layer_spec(4).ratio == COMPRESS_RATIO4


def test_layer_state_shapes_and_dtypes():
    state = DeepSeekV4State()

    swa = state.layer_state(0)
    assert swa.kv_cache.shape == (1, FLASH_CONFIG.window_size, FLASH_CONFIG.head_dim)
    assert swa.kv_cache.dtype is torch.bfloat16
    assert swa.comp_cache is None
    assert swa.idx_kv_cache is None

    hca = state.layer_state(3)
    assert hca.comp_cache.shape == (1, DEFAULT_MAX_SEQ_LEN // COMPRESS_RATIO128, FLASH_CONFIG.head_dim)
    assert hca.comp_cache.dtype is torch.bfloat16
    assert hca.comp_kv_state.shape == (1, COMPRESS_RATIO128, FLASH_CONFIG.head_dim)
    assert hca.comp_kv_state.dtype is torch.float32
    assert torch.all(hca.comp_score_state == -torch.finfo(torch.float32).max)

    csa = state.layer_state(2)
    assert csa.attn_comp_cache.shape == (1, DEFAULT_MAX_SEQ_LEN // COMPRESS_RATIO4, FLASH_CONFIG.head_dim)
    assert csa.attn_comp_kv_state.shape == (1, 2 * COMPRESS_RATIO4, 2 * FLASH_CONFIG.head_dim)
    assert csa.idx_kv_cache.shape == (1, DEFAULT_MAX_SEQ_LEN // COMPRESS_RATIO4, FLASH_CONFIG.index_head_dim)
    assert csa.idx_comp_kv_state.shape == (1, 2 * COMPRESS_RATIO4, 2 * FLASH_CONFIG.index_head_dim)
    assert torch.all(csa.idx_comp_score_state == -torch.finfo(torch.float32).max)


def test_topk_helpers_match_kernel_helpers():
    from models.sparse_attn import build_compress_topk_idxs as kernel_compress_topk
    from models.sparse_attn import build_window_topk_idxs as kernel_window_topk

    for seq_len in (1, 3, 4, 127, 128, 129):
        assert torch.equal(build_window_topk_idxs(seq_len, 0), kernel_window_topk(seq_len, 0))
        assert torch.equal(
            build_compress_topk_idxs(COMPRESS_RATIO128, seq_len, 0, offset=seq_len, topk_max=32),
            kernel_compress_topk(COMPRESS_RATIO128, seq_len, 0, offset=seq_len, topk_max=32),
        )

    for start_pos in (1, 3, 4, 127, 128, 129):
        assert torch.equal(build_window_topk_idxs(1, start_pos), kernel_window_topk(1, start_pos))
        assert torch.equal(
            build_compress_topk_idxs(COMPRESS_RATIO4, 1, start_pos, offset=FLASH_CONFIG.window_size, topk_max=512),
            kernel_compress_topk(COMPRESS_RATIO4, 1, start_pos, offset=FLASH_CONFIG.window_size, topk_max=512),
        )


def test_state_topk_inputs_match_official_model_helpers():
    state = DeepSeekV4State()

    for seq_len in (1, 13, 127, 128, 129):
        swa = state.build_prefill_inputs(0, seq_len)
        expected_window = official_model.get_window_topk_idxs(
            FLASH_CONFIG.window_size,
            1,
            seq_len,
            0,
        ).int()
        _assert_matches_official_with_padding(swa["topk_idxs"], expected_window)

        hca = state.build_prefill_inputs(3, seq_len)
        expected_compress = official_model.get_compress_topk_idxs(
            COMPRESS_RATIO128,
            1,
            seq_len,
            0,
            seq_len,
        ).int()
        _assert_matches_official_with_padding(hca["topk_idxs"][:, :, : FLASH_CONFIG.window_size], expected_window)
        _assert_matches_official_with_padding(hca["topk_idxs"][:, :, FLASH_CONFIG.window_size :], expected_compress)

        csa = state.build_prefill_inputs(2, seq_len)
        _assert_matches_official_with_padding(csa["window_topk_idxs"], expected_window)

    for start_pos in (1, 13, 126, 127, 128, 129, 255):
        expected_window = official_model.get_window_topk_idxs(
            FLASH_CONFIG.window_size,
            1,
            1,
            start_pos,
        ).int()
        swa = state.build_decode_inputs(0, start_pos)
        _assert_matches_official_with_padding(swa["topk_idxs"], expected_window)

        hca = state.build_decode_inputs(3, start_pos)
        expected_compress = official_model.get_compress_topk_idxs(
            COMPRESS_RATIO128,
            1,
            1,
            start_pos,
            FLASH_CONFIG.window_size,
        ).int()
        _assert_matches_official_with_padding(hca["topk_idxs"][:, :, : FLASH_CONFIG.window_size], expected_window)
        _assert_matches_official_with_padding(hca["topk_idxs"][:, :, FLASH_CONFIG.window_size :], expected_compress)

        csa = state.build_decode_inputs(2, start_pos)
        _assert_matches_official_with_padding(csa["window_topk_idxs"], expected_window)


def test_rope_helpers_match_kernel_helpers():
    from models.rope import build_deepseek_v4_rope_tables as kernel_rope_tables
    from models.rope import materialize_compressor_rope as kernel_compressor_rope
    from models.rope import materialize_rope_range as kernel_rope_range

    cos, sin = build_deepseek_v4_rope_tables(max_seq_len=130)
    kernel_cos, kernel_sin = kernel_rope_tables(max_seq_len=130)
    assert torch.equal(cos, kernel_cos)
    assert torch.equal(sin, kernel_sin)
    assert all(torch.equal(a, b) for a, b in zip(materialize_rope_range(cos, sin, 3, 5), kernel_rope_range(kernel_cos, kernel_sin, 3, 5)))

    comp_cos, comp_sin = build_deepseek_v4_rope_tables(compress_ratio=COMPRESS_RATIO4, max_seq_len=130)
    ours = materialize_compressor_rope(comp_cos, comp_sin, 129, COMPRESS_RATIO4)
    expected = kernel_compressor_rope(comp_cos, comp_sin, 129, COMPRESS_RATIO4)
    assert torch.equal(ours[0], expected[0])
    assert torch.equal(ours[1], expected[1])


def test_state_rope_inputs_match_official_model_profiles():
    state = DeepSeekV4State()
    normal_cos, normal_sin = _official_cos_sin(0, DEFAULT_MAX_SEQ_LEN)
    compress4_cos, compress4_sin = _official_cos_sin(COMPRESS_RATIO4, DEFAULT_MAX_SEQ_LEN)
    compress128_cos, compress128_sin = _official_cos_sin(COMPRESS_RATIO128, DEFAULT_MAX_SEQ_LEN)

    swa = state.build_decode_inputs(0, 13)
    torch.testing.assert_close(swa["cos"], normal_cos[13:14], rtol=0, atol=0)
    torch.testing.assert_close(swa["sin"], normal_sin[13:14], rtol=0, atol=0)

    csa = state.build_prefill_inputs(2, 13)
    torch.testing.assert_close(csa["cos"], compress4_cos[:13], rtol=0, atol=0)
    torch.testing.assert_close(csa["sin"], compress4_sin[:13], rtol=0, atol=0)
    torch.testing.assert_close(csa["attn_comp_cos"], compress4_cos[:12:COMPRESS_RATIO4], rtol=0, atol=0)
    torch.testing.assert_close(csa["attn_comp_sin"], compress4_sin[:12:COMPRESS_RATIO4], rtol=0, atol=0)

    hca = state.build_prefill_inputs(3, 129)
    torch.testing.assert_close(hca["cos"], compress128_cos[:129], rtol=0, atol=0)
    torch.testing.assert_close(hca["sin"], compress128_sin[:129], rtol=0, atol=0)
    torch.testing.assert_close(hca["comp_cos"], compress128_cos[:128:COMPRESS_RATIO128], rtol=0, atol=0)
    torch.testing.assert_close(hca["comp_sin"], compress128_sin[:128:COMPRESS_RATIO128], rtol=0, atol=0)


def test_state_decode_compressor_boundaries_match_official_model_formulas():
    state = DeepSeekV4State()
    compress4_cos, compress4_sin = _official_cos_sin(COMPRESS_RATIO4, DEFAULT_MAX_SEQ_LEN)
    compress128_cos, compress128_sin = _official_cos_sin(COMPRESS_RATIO128, DEFAULT_MAX_SEQ_LEN)

    csa_compress = state.build_decode_inputs(2, 3)
    assert csa_compress["comp_should_compress"].item() == 1
    torch.testing.assert_close(csa_compress["attn_comp_cos"], compress4_cos[0:1], rtol=0, atol=0)
    torch.testing.assert_close(csa_compress["attn_comp_sin"], compress4_sin[0:1], rtol=0, atol=0)

    csa_no_compress = state.build_decode_inputs(2, 4)
    assert csa_no_compress["comp_should_compress"].item() == 0
    assert torch.count_nonzero(csa_no_compress["attn_comp_cos"]) == 0
    assert torch.count_nonzero(csa_no_compress["attn_comp_sin"]) == 0

    hca_compress = state.build_decode_inputs(3, 127)
    assert hca_compress["comp_should_compress"].item() == 1
    torch.testing.assert_close(hca_compress["comp_cos"], compress128_cos[0:1], rtol=0, atol=0)
    torch.testing.assert_close(hca_compress["comp_sin"], compress128_sin[0:1], rtol=0, atol=0)

    hca_no_compress = state.build_decode_inputs(3, 128)
    assert hca_no_compress["comp_should_compress"].item() == 0
    assert torch.count_nonzero(hca_no_compress["comp_cos"]) == 0
    assert torch.count_nonzero(hca_no_compress["comp_sin"]) == 0


@pytest.mark.parametrize("seq_len", [1, 3, 4, 127, 128, 129])
def test_prefill_inputs_for_layer_types(seq_len):
    state = DeepSeekV4State()

    swa = state.build_prefill_inputs(0, seq_len)
    assert swa["topk_idxs"].shape == (1, seq_len, FLASH_CONFIG.window_size)
    assert swa["cos"].shape == (seq_len, FLASH_CONFIG.rope_head_dim // 2)
    assert swa["sin"].shape == (seq_len, FLASH_CONFIG.rope_head_dim // 2)

    hca = state.build_prefill_inputs(3, seq_len)
    assert hca["topk_idxs"].shape == (1, seq_len, FLASH_CONFIG.window_size + DEFAULT_MAX_SEQ_LEN // COMPRESS_RATIO128)
    assert hca["comp_block_count"].item() == seq_len // COMPRESS_RATIO128
    assert hca["comp_cos"].shape[0] == max(1, seq_len // COMPRESS_RATIO128)
    assert hca["comp_sin"].shape == hca["comp_cos"].shape

    csa = state.build_prefill_inputs(2, seq_len)
    assert csa["window_topk_idxs"].shape == (1, seq_len, FLASH_CONFIG.window_size)
    assert csa["attn_comp_block_count"].item() == seq_len // COMPRESS_RATIO4
    assert csa["idx_comp_block_count"].item() == seq_len // COMPRESS_RATIO4
    assert csa["idx_offset"].item() == seq_len
    assert csa["attn_comp_cos"].shape[0] == max(1, seq_len // COMPRESS_RATIO4)
    assert torch.equal(csa["idx_comp_cos"], csa["attn_comp_cos"])


@pytest.mark.parametrize("start_pos", [1, 3, 4, 127, 128, 129])
def test_decode_inputs_for_layer_types(start_pos):
    state = DeepSeekV4State()

    swa = state.build_decode_inputs(0, start_pos)
    assert swa["kv_cache"].shape == (1, FLASH_CONFIG.window_size, FLASH_CONFIG.head_dim)
    assert swa["cache_pos"].item() == start_pos % FLASH_CONFIG.window_size
    assert swa["topk_idxs"].shape == (1, 1, FLASH_CONFIG.window_size)

    hca = state.build_decode_inputs(3, start_pos)
    assert hca["topk_idxs"].shape == (1, 1, FLASH_CONFIG.window_size + DEFAULT_MAX_SEQ_LEN // COMPRESS_RATIO128)
    assert hca["comp_slot"].item() == start_pos % COMPRESS_RATIO128
    assert hca["comp_cache_slot"].item() == start_pos // COMPRESS_RATIO128
    assert hca["comp_should_compress"].item() == int((start_pos + 1) % COMPRESS_RATIO128 == 0)
    assert hca["comp_cos"].shape == (1, FLASH_CONFIG.rope_head_dim // 2)
    if hca["comp_should_compress"].item() == 0:
        assert torch.count_nonzero(hca["comp_cos"]) == 0

    csa = state.build_decode_inputs(2, start_pos)
    assert csa["window_topk_idxs"].shape == (1, 1, FLASH_CONFIG.window_size)
    assert csa["idx_offset"].item() == FLASH_CONFIG.window_size
    assert csa["comp_slot"].item() == start_pos % COMPRESS_RATIO4
    assert csa["comp_cache_slot"].item() == start_pos // COMPRESS_RATIO4
    assert csa["comp_should_compress"].item() == int((start_pos + 1) % COMPRESS_RATIO4 == 0)
    assert torch.equal(csa["idx_comp_cos"], csa["attn_comp_cos"])


def test_ratio4_main_rope_uses_compress_profile():
    state = DeepSeekV4State()
    csa = state.build_prefill_inputs(2, 8)
    normal_cos, _ = build_deepseek_v4_rope_tables(compress_ratio=0, max_seq_len=8)
    compress_cos, _ = build_deepseek_v4_rope_tables(compress_ratio=COMPRESS_RATIO4, max_seq_len=8)
    assert torch.equal(csa["cos"], compress_cos)
    assert not torch.equal(csa["cos"][1:], normal_cos[1:])


def test_ratio128_main_rope_uses_compress_profile():
    state = DeepSeekV4State()
    hca_prefill = state.build_prefill_inputs(3, 8)
    hca_decode = state.build_decode_inputs(3, 7)
    normal_cos, _ = build_deepseek_v4_rope_tables(compress_ratio=0, max_seq_len=8)
    compress_cos, _ = build_deepseek_v4_rope_tables(compress_ratio=COMPRESS_RATIO128, max_seq_len=8)
    assert torch.equal(hca_prefill["cos"], compress_cos)
    assert torch.equal(hca_decode["cos"], compress_cos[7:8])
    assert not torch.equal(hca_prefill["cos"][1:], normal_cos[1:])


def test_ratio128_standalone_specs_use_compress_profile():
    import models.attention_hca as attention_hca
    import models.block as block

    seq_len = 8
    start_pos = 7
    normal_cos, _ = build_deepseek_v4_rope_tables(compress_ratio=0, max_seq_len=seq_len)
    compress_cos, _ = build_deepseek_v4_rope_tables(compress_ratio=COMPRESS_RATIO128, max_seq_len=seq_len)

    attn_prefill_cos = _tensor_from_spec(attention_hca.build_hca_prefill_specs(seq_len), "cos")
    attn_decode_cos = _tensor_from_spec(attention_hca.build_hca_decode_specs(start_pos), "cos")
    block_prefill_cos = _tensor_from_spec(block.build_hca_topk_prefill_specs(seq_len), "cos")
    block_decode_cos = _tensor_from_spec(block.build_hca_topk_decode_specs(start_pos), "cos")

    torch.testing.assert_close(attn_prefill_cos, compress_cos, rtol=0, atol=0)
    torch.testing.assert_close(block_prefill_cos, compress_cos, rtol=0, atol=0)
    torch.testing.assert_close(attn_decode_cos, compress_cos[start_pos : start_pos + 1], rtol=0, atol=0)
    torch.testing.assert_close(block_decode_cos, compress_cos[start_pos : start_pos + 1], rtol=0, atol=0)
    assert not torch.equal(attn_prefill_cos[1:], normal_cos[1:])
    assert not torch.equal(block_prefill_cos[1:], normal_cos[1:])


def test_update_layer_state_replaces_expected_tensors():
    state = DeepSeekV4State()

    new_swa_cache = torch.ones_like(state.layer_state(0).kv_cache)
    state.update_layer_state(0, {"kv_cache_out": new_swa_cache})
    assert state.layer_state(0).kv_cache is new_swa_cache

    hca = state.layer_state(3)
    hca_outputs = {
        "kv_cache_out": torch.ones_like(hca.kv_cache),
        "comp_kv_state_out": torch.ones_like(hca.comp_kv_state),
        "comp_score_state_out": torch.ones_like(hca.comp_score_state),
        "comp_cache_out": torch.ones_like(hca.comp_cache),
    }
    state.update_layer_state(3, hca_outputs)
    assert state.layer_state(3).comp_cache is hca_outputs["comp_cache_out"]

    csa = state.layer_state(2)
    csa_outputs = {
        "kv_cache_out": torch.ones_like(csa.kv_cache),
        "attn_comp_kv_state_out": torch.ones_like(csa.attn_comp_kv_state),
        "attn_comp_score_state_out": torch.ones_like(csa.attn_comp_score_state),
        "attn_comp_cache_out": torch.ones_like(csa.attn_comp_cache),
        "idx_kv_cache_out": torch.ones_like(csa.idx_kv_cache),
        "idx_comp_kv_state_out": torch.ones_like(csa.idx_comp_kv_state),
        "idx_comp_score_state_out": torch.ones_like(csa.idx_comp_score_state),
    }
    state.update_layer_state(2, csa_outputs)
    assert state.layer_state(2).idx_kv_cache is csa_outputs["idx_kv_cache_out"]


def test_state_validates_fixed_runtime_shape_contract():
    with pytest.raises(ValueError, match="batch_size"):
        DeepSeekV4State(batch_size=2)
    with pytest.raises(ValueError, match="max_seq_len"):
        DeepSeekV4State(max_seq_len=8192)
    state = DeepSeekV4State()
    with pytest.raises(ValueError, match="decode start_pos"):
        state.build_decode_inputs(0, 0)
    with pytest.raises(ValueError, match="exceeds"):
        state.build_decode_inputs(0, DEFAULT_MAX_SEQ_LEN)
