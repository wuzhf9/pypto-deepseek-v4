"""Tests for sparse attention topk index builders against official logic."""

import importlib

import pytest
import torch

import models.sparse_attn as sparse_attn  # noqa: E402

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

@pytest.mark.parametrize("seq_len", [1, 13, sparse_attn.WINDOW_SIZE, sparse_attn.WINDOW_SIZE + 5])
def test_build_window_topk_idxs_prefill_matches_official(seq_len: int) -> None:
    actual = sparse_attn.build_window_topk_idxs(seq_len, start_pos=0, topk_max=sparse_attn.TOPK_SWA)
    expected = official_model.get_window_topk_idxs(sparse_attn.WINDOW_SIZE, sparse_attn.B, seq_len, 0).int()

    _assert_matches_official_with_padding(actual, expected)

@pytest.mark.parametrize(
    "start_pos",
    [1, 13, sparse_attn.WINDOW_SIZE - 2, sparse_attn.WINDOW_SIZE - 1, sparse_attn.WINDOW_SIZE, 255],
)
def test_build_window_topk_idxs_decode_matches_official(start_pos: int) -> None:
    actual = sparse_attn.build_window_topk_idxs(1, start_pos=start_pos, topk_max=sparse_attn.TOPK_SWA)
    expected = official_model.get_window_topk_idxs(sparse_attn.WINDOW_SIZE, sparse_attn.B, 1, start_pos).int()

    _assert_matches_official_with_padding(actual, expected)

@pytest.mark.parametrize("seq_len", [1, sparse_attn.HCA_COMPRESS_RATIO - 1, sparse_attn.HCA_COMPRESS_RATIO, 256, 4096])
def test_build_compress_topk_idxs_prefill_matches_official(seq_len: int) -> None:
    offset = seq_len
    actual = sparse_attn.build_compress_topk_idxs(
        sparse_attn.HCA_COMPRESS_RATIO,
        seq_len,
        start_pos=0,
        offset=offset,
        topk_max=sparse_attn.TOPK_HCA,
    )
    expected = official_model.get_compress_topk_idxs(
        sparse_attn.HCA_COMPRESS_RATIO,
        sparse_attn.B,
        seq_len,
        0,
        offset,
    ).int()

    _assert_matches_official_with_padding(actual, expected)

@pytest.mark.parametrize(
    "start_pos",
    [
        1,
        sparse_attn.HCA_COMPRESS_RATIO - 1,
        sparse_attn.HCA_COMPRESS_RATIO,
        sparse_attn.HCA_COMPRESS_RATIO * 2 - 1,
        sparse_attn.HCA_MAX_POSITION_EMBEDDINGS - 1,
    ],
)
def test_build_compress_topk_idxs_decode_matches_official(start_pos: int) -> None:
    offset = sparse_attn.WINDOW_SIZE
    actual = sparse_attn.build_compress_topk_idxs(
        sparse_attn.HCA_COMPRESS_RATIO,
        1,
        start_pos=start_pos,
        offset=offset,
        topk_max=sparse_attn.TOPK_HCA,
    )
    expected = official_model.get_compress_topk_idxs(
        sparse_attn.HCA_COMPRESS_RATIO,
        sparse_attn.B,
        1,
        start_pos,
        offset,
    ).int()

    _assert_matches_official_with_padding(actual, expected)

@pytest.mark.parametrize("seq_len", [1, sparse_attn.CSA_COMPRESS_RATIO - 1, sparse_attn.CSA_COMPRESS_RATIO, 13])
def test_build_csa_synthetic_topk_idxs_prefill_shape_and_visibility(seq_len: int) -> None:
    actual = sparse_attn.build_csa_synthetic_topk_idxs(seq_len, start_pos=0, offset=seq_len)
    window_expected = official_model.get_window_topk_idxs(sparse_attn.WINDOW_SIZE, sparse_attn.B, seq_len, 0).int()

    assert actual.shape == (sparse_attn.B, seq_len, sparse_attn.TOPK_CSA_TOTAL)
    _assert_matches_official_with_padding(actual[:, :, : sparse_attn.TOPK_SWA], window_expected)

    compressed = actual[:, :, sparse_attn.TOPK_SWA :]
    for token_idx in range(seq_len):
        visible_blocks = min(sparse_attn.TOPK_CSA, (token_idx + 1) // sparse_attn.CSA_COMPRESS_RATIO)
        if visible_blocks > 0:
            expected = torch.arange(visible_blocks, dtype=torch.int32) + seq_len
            torch.testing.assert_close(compressed[0, token_idx, :visible_blocks], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            compressed[0, token_idx, visible_blocks:],
            torch.full_like(compressed[0, token_idx, visible_blocks:], -1),
            rtol=0,
            atol=0,
        )

@pytest.mark.parametrize(
    "start_pos",
    [
        1,
        sparse_attn.CSA_COMPRESS_RATIO - 2,
        sparse_attn.CSA_COMPRESS_RATIO - 1,
        sparse_attn.CSA_COMPRESS_RATIO,
        sparse_attn.CSA_COMPRESS_RATIO * 4 - 1,
    ],
)
def test_build_csa_synthetic_topk_idxs_decode_shape_and_visibility(start_pos: int) -> None:
    actual = sparse_attn.build_csa_synthetic_topk_idxs(1, start_pos=start_pos, offset=sparse_attn.WINDOW_SIZE)
    window_expected = official_model.get_window_topk_idxs(sparse_attn.WINDOW_SIZE, sparse_attn.B, 1, start_pos).int()

    assert actual.shape == (sparse_attn.B, 1, sparse_attn.TOPK_CSA_TOTAL)
    _assert_matches_official_with_padding(actual[:, :, : sparse_attn.TOPK_SWA], window_expected)

    compressed = actual[:, :, sparse_attn.TOPK_SWA :]
    visible_blocks = min(sparse_attn.TOPK_CSA, (start_pos + 1) // sparse_attn.CSA_COMPRESS_RATIO)
    if visible_blocks > 0:
        expected = torch.arange(visible_blocks, dtype=torch.int32) + sparse_attn.WINDOW_SIZE
        torch.testing.assert_close(compressed[0, 0, :visible_blocks], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        compressed[0, 0, visible_blocks:],
        torch.full_like(compressed[0, 0, visible_blocks:], -1),
        rtol=0,
        atol=0,
    )
