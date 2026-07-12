"""Shared host-only stubs and helpers for importing kernel modules in tests."""

import os
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--official-checkpoint",
        action="store",
        default=os.environ.get("DSV4_OFFICIAL_CHECKPOINT"),
        help=(
            "Path to the official DeepSeek V4 checkpoint. Defaults to "
            "$DSV4_OFFICIAL_CHECKPOINT or ../deepseek_v4_flash."
        ),
    )


@pytest.fixture(scope="session")
def official_checkpoint_path(request) -> Path:
    configured = request.config.getoption("--official-checkpoint")
    path = Path(configured).expanduser() if configured else REPO_ROOT.parent / "deepseek_v4_flash"
    index = path / "model.safetensors.index.json"
    if not index.is_file():
        pytest.skip(
            f"Official checkpoint is not available at {path}; "
            "pass --official-checkpoint or set DSV4_OFFICIAL_CHECKPOINT"
        )
    return path


def pytest_configure():
    install_pypto_language_stub()
    install_official_kernel_stub()


def install_pypto_language_stub() -> None:
    if "pypto.language" in sys.modules:
        language = sys.modules["pypto.language"]
        for name in ("BF16", "FP32", "INT32", "INT64", "UINT32", "INDEX"):
            if not hasattr(language, name):
                setattr(language, name, object())
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
    language.UINT32 = object()
    language.INDEX = object()
    language.jit = _Jit()
    language.dynamic = lambda _name: 1

    pypto = types.ModuleType("pypto")
    pypto.language = language
    sys.modules["pypto"] = pypto
    sys.modules["pypto.language"] = language


def install_official_kernel_stub() -> None:
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


def make_linear_reference():
    def linear(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert bias is None
        return torch.matmul(x.float(), weight.t().contiguous().float()).to(x.dtype)

    return linear


def torch_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    from models.sparse_attn import golden_sparse_attn

    tensors = {
        "q": q,
        "kv": kv,
        "attn_sink": attn_sink,
        "topk_idxs": topk_idxs,
        "softmax_scale": softmax_scale,
        "out": torch.empty_like(q),
    }
    golden_sparse_attn(tensors)
    return tensors["out"]


def make_einsum_reference(original_einsum):
    def einsum(equation, *operands, **kwargs):
        if equation == "bsgd,grd->bsgr":
            out_dtype = operands[0].dtype
            return original_einsum(equation, *(operand.float() for operand in operands), **kwargs).to(out_dtype)
        return original_einsum(equation, *operands, **kwargs)

    return einsum


def make_square_reference(original_square):
    def square(tensor, *args, **kwargs):
        if tensor.dtype is torch.bfloat16:
            return original_square(tensor.float(), *args, **kwargs)
        return original_square(tensor, *args, **kwargs)

    return square


def pad_last_dim(tensor: torch.Tensor, width: int, value: int = -1) -> torch.Tensor:
    if tensor.shape[-1] == width:
        return tensor.to(torch.int32)
    return F.pad(tensor.to(torch.int32), (0, width - tensor.shape[-1]), value=value)


def rope_cos_sin(attn: torch.nn.Module, start_pos: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = attn.freqs_cis[start_pos : start_pos + seq_len]
    return freqs.real.contiguous(), freqs.imag.contiguous()


def compressor_cos_sin(
    attn: torch.nn.Module,
    ratio: int,
    seq_len: int,
    start_pos: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if start_pos == 0:
        cutoff = seq_len - seq_len % ratio
        freqs = attn.freqs_cis[:cutoff:ratio]
        if freqs.shape[0] == 0:
            freqs = attn.freqs_cis[:1]
    elif (start_pos + 1) % ratio == 0:
        freqs = attn.freqs_cis[start_pos + 1 - ratio : start_pos + 2 - ratio]
    else:
        return (
            torch.zeros(1, attn.rope_head_dim // 2, dtype=torch.float32),
            torch.zeros(1, attn.rope_head_dim // 2, dtype=torch.float32),
        )
    return freqs.real.contiguous(), freqs.imag.contiguous()


def official_apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool,
) -> torch.Tensor:
    freqs_cis = torch.complex(cos.float(), sin.float())
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), x_complex.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    out = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return out.to(x.dtype)


def official_apply_full_head_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool,
    rope_head_dim: int | None = None,
) -> torch.Tensor:
    if rope_head_dim is None:
        rope_head_dim = cos.shape[-1] * 2
    out = x.clone()
    out[..., -rope_head_dim:] = official_apply_rotary_emb(out[..., -rope_head_dim:], cos, sin, inverse)
    return out
