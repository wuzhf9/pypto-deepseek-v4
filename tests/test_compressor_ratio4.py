"""Tests for ratio-4 Indexer compressor golden logic against official ``model.py``."""

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

import models.compressor_ratio4 as compressor_ratio4  # noqa: E402
import models.rope as rope  # noqa: E402

official_model = importlib.import_module("official.model")


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
def tiny_ratio4_args(monkeypatch):
    args = official_model.ModelArgs(
        max_batch_size=1,
        max_seq_len=32,
        dtype="bf16",
        scale_dtype="fp32",
        expert_dtype=None,
        vocab_size=16,
        dim=16,
        moe_inter_dim=16,
        n_layers=1,
        n_hash_layers=0,
        n_heads=2,
        n_routed_experts=2,
        n_shared_experts=1,
        n_activated_experts=1,
        q_lora_rank=4,
        head_dim=8,
        rope_head_dim=2,
        norm_eps=1e-6,
        o_groups=1,
        o_lora_rank=4,
        window_size=4,
        compress_ratios=(4,),
        rope_factor=1,
        beta_fast=32,
        beta_slow=1,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=1,
        hc_mult=2,
        hc_sinkhorn_iters=2,
    )

    monkeypatch.setattr(compressor_ratio4, "HIDDEN", args.dim)
    monkeypatch.setattr(compressor_ratio4, "ATTN_HEAD_DIM", args.head_dim)
    monkeypatch.setattr(compressor_ratio4, "ATTN_PROJ_DIM", 2 * args.head_dim)
    monkeypatch.setattr(compressor_ratio4, "INDEX_HEAD_DIM", args.index_head_dim)
    monkeypatch.setattr(compressor_ratio4, "INDEX_PROJ_DIM", 2 * args.index_head_dim)
    monkeypatch.setattr(compressor_ratio4, "ROPE_HALF", args.rope_head_dim // 2)
    monkeypatch.setattr(compressor_ratio4, "TOPK_CSA_COMPRESSED", args.max_seq_len // 4)
    monkeypatch.setattr(rope, "ROPE_DIM", args.rope_head_dim)
    monkeypatch.setattr(rope, "ROPE_HALF", args.rope_head_dim // 2)
    monkeypatch.setattr(rope, "INDEX_HEAD_DIM", args.index_head_dim)
    monkeypatch.setattr(rope, "INDEX_TAIL_OFFSET", args.index_head_dim - args.rope_head_dim)
    monkeypatch.setattr(official_model, "rotate_activation", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(official_model, "fp4_act_quant", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(official_model, "linear", _make_linear_reference())
    return args


def _make_official_indexer_compressor(args) -> torch.nn.Module:
    torch.manual_seed(20260702)
    with official_model.set_dtype(torch.bfloat16):
        compressor = official_model.Compressor(args, compress_ratio=4, head_dim=args.index_head_dim, rotate=True)

    proj_dim = 2 * args.index_head_dim
    with torch.no_grad():
        compressor.ape.copy_(torch.randn(4, proj_dim) * 0.02)
        compressor.wkv.weight.copy_((torch.randn(proj_dim, args.dim, dtype=torch.bfloat16) * 0.02).float())
        compressor.wgate.weight.copy_((torch.randn(proj_dim, args.dim, dtype=torch.bfloat16) * 0.02).float())
        compressor.norm.weight.copy_(torch.randn(args.index_head_dim) * 0.1 + 1.0)

    compressor.kv_cache = torch.zeros(1, args.max_seq_len // 4, args.index_head_dim, dtype=torch.bfloat16)
    compressor.freqs_cis = official_model.precompute_freqs_cis(
        args.rope_head_dim,
        args.max_seq_len,
        args.original_seq_len,
        args.compress_rope_theta,
        args.rope_factor,
        args.beta_fast,
        args.beta_slow,
    )
    return compressor


def _make_official_attention_compressor(args) -> torch.nn.Module:
    torch.manual_seed(20260702)
    with official_model.set_dtype(torch.bfloat16):
        compressor = official_model.Compressor(args, compress_ratio=4, head_dim=args.head_dim)

    proj_dim = 2 * args.head_dim
    with torch.no_grad():
        compressor.ape.copy_(torch.randn(4, proj_dim) * 0.02)
        compressor.wkv.weight.copy_((torch.randn(proj_dim, args.dim, dtype=torch.bfloat16) * 0.02).float())
        compressor.wgate.weight.copy_((torch.randn(proj_dim, args.dim, dtype=torch.bfloat16) * 0.02).float())
        compressor.norm.weight.copy_(torch.randn(args.head_dim) * 0.1 + 1.0)

    compressor.kv_cache = torch.zeros(1, args.max_seq_len // 4, args.head_dim, dtype=torch.bfloat16)
    compressor.freqs_cis = official_model.precompute_freqs_cis(
        args.rope_head_dim,
        args.max_seq_len,
        args.original_seq_len,
        args.compress_rope_theta,
        args.rope_factor,
        args.beta_fast,
        args.beta_slow,
    )
    return compressor


def _compressor_tensors(compressor: torch.nn.Module, x: torch.Tensor, args) -> dict[str, torch.Tensor]:
    seq_len = x.shape[1]
    cutoff = seq_len - seq_len % 4
    freqs_cis = compressor.freqs_cis[:cutoff:4]
    actual_compressed_len = freqs_cis.shape[0]
    compressed_len = max(1, actual_compressed_len)
    if actual_compressed_len == 0:
        freqs_cis = compressor.freqs_cis[:1]

    proj_dim = 2 * args.index_head_dim
    return {
        "x": x.clone(),
        "wkv_t": compressor.wkv.weight.detach().t().contiguous().to(torch.bfloat16),
        "wgate_t": compressor.wgate.weight.detach().t().contiguous().to(torch.bfloat16),
        "ape": compressor.ape.detach().clone(),
        "norm_w": compressor.norm.weight.detach().clone(),
        "cos": freqs_cis.real.contiguous(),
        "sin": freqs_cis.imag.contiguous(),
        "block_count": torch.tensor([actual_compressed_len], dtype=torch.int32),
        "kv_state_out": torch.zeros(1, 8, proj_dim, dtype=torch.float32),
        "score_state_out": torch.zeros(1, 8, proj_dim, dtype=torch.float32),
        "compressed_cache_out": torch.zeros(1, args.max_seq_len // 4, args.index_head_dim, dtype=torch.bfloat16),
        "kv_proj": torch.zeros(1, seq_len, proj_dim, dtype=torch.float32),
        "score_proj": torch.zeros(1, seq_len, proj_dim, dtype=torch.float32),
        "pooled": torch.zeros(1, compressed_len, args.index_head_dim, dtype=torch.bfloat16),
        "normed": torch.zeros(1, compressed_len, args.index_head_dim, dtype=torch.bfloat16),
        "compressed": torch.zeros(1, compressed_len, args.index_head_dim, dtype=torch.bfloat16),
    }


def _attention_compressor_tensors(compressor: torch.nn.Module, x: torch.Tensor, args) -> dict[str, torch.Tensor]:
    seq_len = x.shape[1]
    cutoff = seq_len - seq_len % 4
    freqs_cis = compressor.freqs_cis[:cutoff:4]
    actual_compressed_len = freqs_cis.shape[0]
    compressed_len = max(1, actual_compressed_len)
    if actual_compressed_len == 0:
        freqs_cis = compressor.freqs_cis[:1]

    proj_dim = 2 * args.head_dim
    return {
        "x": x.clone(),
        "wkv_t": compressor.wkv.weight.detach().t().contiguous().to(torch.bfloat16),
        "wgate_t": compressor.wgate.weight.detach().t().contiguous().to(torch.bfloat16),
        "ape": compressor.ape.detach().clone(),
        "norm_w": compressor.norm.weight.detach().clone(),
        "cos": freqs_cis.real.contiguous(),
        "sin": freqs_cis.imag.contiguous(),
        "block_count": torch.tensor([actual_compressed_len], dtype=torch.int32),
        "kv_state_out": torch.zeros(1, 8, proj_dim, dtype=torch.float32),
        "score_state_out": torch.zeros(1, 8, proj_dim, dtype=torch.float32),
        "compressed_cache_out": torch.zeros(1, args.max_seq_len // 4, args.head_dim, dtype=torch.bfloat16),
        "kv_proj": torch.zeros(1, seq_len, proj_dim, dtype=torch.float32),
        "score_proj": torch.zeros(1, seq_len, proj_dim, dtype=torch.float32),
        "pooled": torch.zeros(1, compressed_len, args.head_dim, dtype=torch.bfloat16),
        "normed": torch.zeros(1, compressed_len, args.head_dim, dtype=torch.bfloat16),
        "compressed": torch.zeros(1, compressed_len, args.head_dim, dtype=torch.bfloat16),
    }


def _compressor_decode_tensors(
    compressor: torch.nn.Module,
    x: torch.Tensor,
    args,
    start_pos: int,
) -> dict[str, torch.Tensor]:
    if start_pos <= 0:
        raise ValueError(f"decode start_pos must be greater than 0, got {start_pos}")

    slot = start_pos % 4
    cache_slot = start_pos // 4
    should_compress = int((start_pos + 1) % 4 == 0)
    if should_compress:
        rope_pos = start_pos + 1 - 4
        freqs_cis = compressor.freqs_cis[rope_pos : rope_pos + 1]
        cos = freqs_cis.real.contiguous()
        sin = freqs_cis.imag.contiguous()
    else:
        cos = torch.zeros(1, args.rope_head_dim // 2, dtype=torch.float32)
        sin = torch.zeros(1, args.rope_head_dim // 2, dtype=torch.float32)

    proj_dim = 2 * args.index_head_dim
    return {
        "x": x.clone(),
        "kv_state": compressor.kv_state.detach().clone(),
        "score_state": compressor.score_state.detach().clone(),
        "compressed_cache": compressor.kv_cache.detach().clone(),
        "slot": torch.tensor([slot], dtype=torch.int32),
        "cache_slot": torch.tensor([cache_slot], dtype=torch.int32),
        "should_compress": torch.tensor([should_compress], dtype=torch.int32),
        "wkv_t": compressor.wkv.weight.detach().t().contiguous().to(torch.bfloat16),
        "wgate_t": compressor.wgate.weight.detach().t().contiguous().to(torch.bfloat16),
        "ape": compressor.ape.detach().clone(),
        "norm_w": compressor.norm.weight.detach().clone(),
        "cos": cos,
        "sin": sin,
        "kv_proj": torch.zeros(1, 1, proj_dim, dtype=torch.float32),
        "score_proj": torch.zeros(1, 1, proj_dim, dtype=torch.float32),
        "pooled": torch.zeros(1, 1, args.index_head_dim, dtype=torch.bfloat16),
        "normed": torch.zeros(1, 1, args.index_head_dim, dtype=torch.bfloat16),
        "compressed": torch.zeros(1, 1, args.index_head_dim, dtype=torch.bfloat16),
        "kv_state_out": torch.zeros_like(compressor.kv_state),
        "score_state_out": torch.zeros_like(compressor.score_state),
        "compressed_cache_out": torch.zeros_like(compressor.kv_cache),
    }


def _attention_compressor_decode_tensors(
    compressor: torch.nn.Module,
    x: torch.Tensor,
    args,
    start_pos: int,
) -> dict[str, torch.Tensor]:
    if start_pos <= 0:
        raise ValueError(f"decode start_pos must be greater than 0, got {start_pos}")

    slot = start_pos % 4
    cache_slot = start_pos // 4
    should_compress = int((start_pos + 1) % 4 == 0)
    if should_compress:
        rope_pos = start_pos + 1 - 4
        freqs_cis = compressor.freqs_cis[rope_pos : rope_pos + 1]
        cos = freqs_cis.real.contiguous()
        sin = freqs_cis.imag.contiguous()
    else:
        cos = torch.zeros(1, args.rope_head_dim // 2, dtype=torch.float32)
        sin = torch.zeros(1, args.rope_head_dim // 2, dtype=torch.float32)

    proj_dim = 2 * args.head_dim
    return {
        "x": x.clone(),
        "kv_state": compressor.kv_state.detach().clone(),
        "score_state": compressor.score_state.detach().clone(),
        "compressed_cache": compressor.kv_cache.detach().clone(),
        "slot": torch.tensor([slot], dtype=torch.int32),
        "cache_slot": torch.tensor([cache_slot], dtype=torch.int32),
        "should_compress": torch.tensor([should_compress], dtype=torch.int32),
        "wkv_t": compressor.wkv.weight.detach().t().contiguous().to(torch.bfloat16),
        "wgate_t": compressor.wgate.weight.detach().t().contiguous().to(torch.bfloat16),
        "ape": compressor.ape.detach().clone(),
        "norm_w": compressor.norm.weight.detach().clone(),
        "cos": cos,
        "sin": sin,
        "kv_proj": torch.zeros(1, 1, proj_dim, dtype=torch.float32),
        "score_proj": torch.zeros(1, 1, proj_dim, dtype=torch.float32),
        "pooled": torch.zeros(1, 1, args.head_dim, dtype=torch.bfloat16),
        "normed": torch.zeros(1, 1, args.head_dim, dtype=torch.bfloat16),
        "compressed": torch.zeros(1, 1, args.head_dim, dtype=torch.bfloat16),
        "kv_state_out": torch.zeros_like(compressor.kv_state),
        "score_state_out": torch.zeros_like(compressor.score_state),
        "compressed_cache_out": torch.zeros_like(compressor.kv_cache),
    }


def _assert_score_state_matches(actual: torch.Tensor, expected: torch.Tensor) -> None:
    valid = torch.isfinite(expected)
    torch.testing.assert_close(actual[valid], expected[valid], rtol=0, atol=0)
    assert torch.isneginf(expected[~valid]).all()
    assert (actual[~valid] <= compressor_ratio4.NEG_INF / 2).all()


@pytest.mark.parametrize("seq_len", [3, 4, 6, 7, 8, 13, 16, 32])
def test_golden_compressor_ratio4_indexer_prefill_matches_official_model(
    tiny_ratio4_args,
    seq_len: int,
) -> None:
    compressor = _make_official_indexer_compressor(tiny_ratio4_args)
    x = (torch.randn(1, seq_len, tiny_ratio4_args.dim, dtype=torch.float32) * 0.1).to(torch.bfloat16)

    expected_return = compressor(x.clone(), start_pos=0)
    tensors = _compressor_tensors(compressor, x, tiny_ratio4_args)

    compressor_ratio4.golden_compressor_ratio4_indexer_forward(tensors, start_pos=0)

    torch.testing.assert_close(tensors["kv_state_out"], compressor.kv_state, rtol=0, atol=0)
    _assert_score_state_matches(tensors["score_state_out"], compressor.score_state)
    torch.testing.assert_close(tensors["compressed_cache_out"], compressor.kv_cache, rtol=0, atol=0)
    if expected_return is None:
        assert int(tensors["block_count"][0].item()) == 0
        torch.testing.assert_close(tensors["compressed"], torch.zeros_like(tensors["compressed"]), rtol=0, atol=0)
    else:
        torch.testing.assert_close(tensors["compressed"][:, : expected_return.shape[1]], expected_return, rtol=0, atol=0)
        torch.testing.assert_close(
            tensors["compressed_cache_out"][:, : expected_return.shape[1]],
            expected_return,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("start_pos", [1, 2, 3, 7])
def test_golden_compressor_ratio4_indexer_decode_matches_official_model(
    tiny_ratio4_args,
    start_pos: int,
) -> None:
    compressor = _make_official_indexer_compressor(tiny_ratio4_args)
    x = (torch.randn(1, 1, tiny_ratio4_args.dim, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    with torch.no_grad():
        compressor.kv_state.copy_(torch.randn_like(compressor.kv_state) * 0.1)
        compressor.score_state.copy_(torch.randn_like(compressor.score_state) * 0.1)
        compressor.kv_cache.copy_((torch.randn_like(compressor.kv_cache.float()) * 0.1).to(torch.bfloat16))

    tensors = _compressor_decode_tensors(compressor, x, tiny_ratio4_args, start_pos)
    expected_return = compressor(x.clone(), start_pos=start_pos)

    compressor_ratio4.golden_compressor_ratio4_indexer_forward(tensors, start_pos=start_pos)

    torch.testing.assert_close(tensors["kv_state_out"], compressor.kv_state, rtol=0, atol=0)
    _assert_score_state_matches(tensors["score_state_out"], compressor.score_state)
    torch.testing.assert_close(tensors["compressed_cache_out"], compressor.kv_cache, rtol=0, atol=0)
    if expected_return is None:
        torch.testing.assert_close(tensors["compressed"], torch.zeros_like(tensors["compressed"]), rtol=0, atol=0)
    else:
        torch.testing.assert_close(tensors["compressed"], expected_return, rtol=0, atol=0)
        cache_slot = start_pos // 4
        torch.testing.assert_close(
            tensors["compressed_cache_out"][:, cache_slot : cache_slot + 1],
            expected_return,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("seq_len", [3, 4, 6, 7, 8, 13, 16, 32])
def test_golden_compressor_ratio4_attention_prefill_matches_official_model(
    tiny_ratio4_args,
    seq_len: int,
) -> None:
    compressor = _make_official_attention_compressor(tiny_ratio4_args)
    x = (torch.randn(1, seq_len, tiny_ratio4_args.dim, dtype=torch.float32) * 0.1).to(torch.bfloat16)

    expected_return = compressor(x.clone(), start_pos=0)
    tensors = _attention_compressor_tensors(compressor, x, tiny_ratio4_args)

    compressor_ratio4.golden_compressor_ratio4_attention_forward(tensors, start_pos=0)

    torch.testing.assert_close(tensors["kv_state_out"], compressor.kv_state, rtol=0, atol=0)
    _assert_score_state_matches(tensors["score_state_out"], compressor.score_state)
    torch.testing.assert_close(tensors["compressed_cache_out"], compressor.kv_cache, rtol=0, atol=0)
    if expected_return is None:
        assert int(tensors["block_count"][0].item()) == 0
        torch.testing.assert_close(tensors["compressed"], torch.zeros_like(tensors["compressed"]), rtol=0, atol=0)
    else:
        torch.testing.assert_close(tensors["compressed"][:, : expected_return.shape[1]], expected_return, rtol=0, atol=0)
        torch.testing.assert_close(
            tensors["compressed_cache_out"][:, : expected_return.shape[1]],
            expected_return,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("start_pos", [1, 2, 3, 7])
def test_golden_compressor_ratio4_attention_decode_matches_official_model(
    tiny_ratio4_args,
    start_pos: int,
) -> None:
    compressor = _make_official_attention_compressor(tiny_ratio4_args)
    x = (torch.randn(1, 1, tiny_ratio4_args.dim, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    with torch.no_grad():
        compressor.kv_state.copy_(torch.randn_like(compressor.kv_state) * 0.1)
        compressor.score_state.copy_(torch.randn_like(compressor.score_state) * 0.1)
        compressor.kv_cache.copy_((torch.randn_like(compressor.kv_cache.float()) * 0.1).to(torch.bfloat16))

    tensors = _attention_compressor_decode_tensors(compressor, x, tiny_ratio4_args, start_pos)
    expected_return = compressor(x.clone(), start_pos=start_pos)

    compressor_ratio4.golden_compressor_ratio4_attention_forward(tensors, start_pos=start_pos)

    torch.testing.assert_close(tensors["kv_state_out"], compressor.kv_state, rtol=0, atol=0)
    _assert_score_state_matches(tensors["score_state_out"], compressor.score_state)
    torch.testing.assert_close(tensors["compressed_cache_out"], compressor.kv_cache, rtol=0, atol=0)
    if expected_return is None:
        torch.testing.assert_close(tensors["compressed"], torch.zeros_like(tensors["compressed"]), rtol=0, atol=0)
    else:
        torch.testing.assert_close(tensors["compressed"], expected_return, rtol=0, atol=0)
        cache_slot = start_pos // 4
        torch.testing.assert_close(
            tensors["compressed_cache_out"][:, cache_slot : cache_slot + 1],
            expected_return,
            rtol=0,
            atol=0,
        )
