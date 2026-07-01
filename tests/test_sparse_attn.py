"""Tests for sparse attention topk index builders against official logic."""

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
    if "kernel" in sys.modules:
        return

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
