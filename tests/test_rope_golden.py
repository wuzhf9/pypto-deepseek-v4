"""Tests for the host-side RoPE tables and manual golden rotation."""

import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_pypto_language_stub() -> None:
    """Allow importing ``models.rope`` in host-only unit tests."""

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

from models.config import FLASH_CONFIG as M  # noqa: E402
from models.rope import (  # noqa: E402
    _apply_rope_golden,
    precompute_freqs_cos_sin,
)


def _official_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> torch.Tensor:
    import math

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_value, max_value, dim):
        if min_value == max_value:
            max_value += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_value) / (max_value - min_value)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _official_apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool,
) -> torch.Tensor:
    freqs_cis = torch.complex(cos.float(), sin.float())
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x_complex.ndim == 3:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), x_complex.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    out = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return out.to(x.dtype)


@pytest.mark.parametrize("compress_ratio", [0, 4, 128])
def test_rope_tables_match_official_freqs_cis(compress_ratio: int) -> None:
    seqlen = 19
    base = M.compress_rope_theta if compress_ratio else M.rope_theta
    original_seq_len = M.original_seq_len if compress_ratio else 0

    cos, sin = precompute_freqs_cos_sin(
        M.rope_head_dim,
        seqlen,
        original_seq_len,
        base,
        M.rope_factor,
        M.beta_fast,
        M.beta_slow,
    )
    freqs_cis = _official_freqs_cis(
        M.rope_head_dim,
        seqlen,
        original_seq_len,
        base,
        M.rope_factor,
        M.beta_fast,
        M.beta_slow,
    )

    assert cos.shape == (seqlen, M.rope_head_dim // 2)
    assert sin.shape == (seqlen, M.rope_head_dim // 2)
    assert cos.dtype == torch.float32
    assert sin.dtype == torch.float32
    torch.testing.assert_close(cos, freqs_cis.real, rtol=0, atol=0)
    torch.testing.assert_close(sin, freqs_cis.imag, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("shape", "inverse"),
    [
        ((1, 1, 64), False),
        ((1, 13, 64), False),
        ((1, 13, 64), True),
        ((1, 1, 64, 64), False),
        ((1, 13, 64, 64), False),
        ((1, 13, 64, 64), True),
    ],
)
def test_manual_rope_golden_matches_official_complex_path(
    shape: tuple[int, ...],
    inverse: bool,
) -> None:
    torch.manual_seed(20260630)
    seq_len = shape[1]
    x = (torch.randn(shape, dtype=torch.float32) * 0.2).to(torch.bfloat16)
    cos, sin = precompute_freqs_cos_sin(
        M.rope_head_dim,
        seq_len,
        M.original_seq_len,
        M.compress_rope_theta,
        M.rope_factor,
        M.beta_fast,
        M.beta_slow,
    )

    actual = _apply_rope_golden(x, cos, sin, inverse=inverse)
    expected = _official_apply_rotary_emb(x, cos, sin, inverse=inverse)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
