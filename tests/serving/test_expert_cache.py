"""Tests for the packed routed-expert cache reader."""

from dataclasses import replace
import json

import pytest
import torch
from safetensors.torch import save_file

from models.config import FLASH_CONFIG
from serving.expert_cache import (
    EXPERT_CACHE_FORMAT,
    EXPERT_CACHE_VERSION,
    ExpertCacheReader,
    layer_expert_cache_filename,
)


def _config():
    return replace(
        FLASH_CONFIG,
        dim=4,
        n_layers=2,
        n_routed_experts=3,
        n_activated_experts=2,
        moe_inter_dim=3,
    )


def _packed_tensors():
    return {
        "routed_w1_t": torch.stack(
            [torch.full((4, 3), float(i + 1), dtype=torch.bfloat16) for i in range(3)]
        ),
        "routed_w2_t": torch.stack(
            [torch.full((3, 4), float(i + 11), dtype=torch.bfloat16) for i in range(3)]
        ),
        "routed_w3_t": torch.stack(
            [torch.full((4, 3), float(i + 21), dtype=torch.bfloat16) for i in range(3)]
        ),
    }


def _write_manifest(directory, *, layers=(0,), **overrides):
    cfg = _config()
    data = {
        "format": EXPERT_CACHE_FORMAT,
        "version": EXPERT_CACHE_VERSION,
        "source_checkpoint": "checkpoint",
        "n_layers": cfg.n_layers,
        "n_routed_experts": cfg.n_routed_experts,
        "dim": cfg.dim,
        "moe_inter_dim": cfg.moe_inter_dim,
        "dtype": "bfloat16",
        "layers": {
            str(layer_id): {"file": layer_expert_cache_filename(layer_id)}
            for layer_id in layers
        },
    }
    data.update(overrides)
    (directory / "manifest.json").write_text(json.dumps(data), encoding="utf-8")


def _write_cache(directory) -> None:
    _write_manifest(directory)
    save_file(_packed_tensors(), directory / layer_expert_cache_filename(0))


def test_disabled_reader_returns_uncached_layer() -> None:
    reader = ExpertCacheReader(None, config=_config())

    assert reader.inspect_layer(0) is None
    assert reader.load_routed_pack(0, device=torch.device("cpu")) is None
    assert not reader.copy_selected_into(
        0,
        [0, 1],
        out_w1=torch.empty(2, 4, 3, dtype=torch.bfloat16),
        out_w2=torch.empty(2, 3, 4, dtype=torch.bfloat16),
        out_w3=torch.empty(2, 4, 3, dtype=torch.bfloat16),
    )


def test_reader_requires_manifest_when_cache_directory_is_configured(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="requires manifest.json"):
        ExpertCacheReader(tmp_path, config=_config())


def test_manifest_omitted_layer_is_uncached_but_declared_missing_file_is_error(tmp_path) -> None:
    _write_manifest(tmp_path, layers=())
    reader = ExpertCacheReader(tmp_path, config=_config())
    assert reader.inspect_layer(0) is None

    _write_manifest(tmp_path, layers=(0,))
    reader = ExpertCacheReader(tmp_path, config=_config())
    with pytest.raises(FileNotFoundError, match="missing layer file"):
        reader.inspect_layer(0)


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"format": "unknown"}, "Unsupported expert cache format"),
        ({"version": EXPERT_CACHE_VERSION - 1}, "Unsupported expert cache version"),
        ({"dim": 99}, "dim mismatch"),
    ],
)
def test_manifest_format_version_and_config_are_strict(tmp_path, override, error) -> None:
    _write_manifest(tmp_path, **override)

    with pytest.raises(ValueError, match=error):
        ExpertCacheReader(tmp_path, config=_config())


def test_reader_inspects_complete_pack_reuses_handle_and_closes_idempotently(tmp_path) -> None:
    _write_cache(tmp_path)
    reader = ExpertCacheReader(tmp_path, config=_config())

    first = reader.inspect_layer(0)
    second = reader.inspect_layer(0)

    assert first is second
    assert first is not None
    assert first.keys == {"routed_w1_t", "routed_w2_t", "routed_w3_t"}
    assert reader.open_handle_count == 1
    reader.close()
    reader.close()
    assert reader.open_handle_count == 0


def test_selected_copy_preserves_order_duplicates_and_profiles(tmp_path) -> None:
    tensors = _packed_tensors()
    _write_cache(tmp_path)
    events = []
    reader = ExpertCacheReader(
        tmp_path,
        config=_config(),
        profile_callback=lambda name, start: events.append((name, start)),
    )
    out_w1 = torch.empty(3, 4, 3, dtype=torch.bfloat16)
    out_w2 = torch.empty(3, 3, 4, dtype=torch.bfloat16)
    out_w3 = torch.empty(3, 4, 3, dtype=torch.bfloat16)

    assert reader.copy_selected_into(
        0,
        [2, 0, 2],
        out_w1=out_w1,
        out_w2=out_w2,
        out_w3=out_w3,
    )

    assert torch.equal(out_w1, tensors["routed_w1_t"][[2, 0, 2]])
    assert torch.equal(out_w2, tensors["routed_w2_t"][[2, 0, 2]])
    assert torch.equal(out_w3, tensors["routed_w3_t"][[2, 0, 2]])
    assert [name for name, _ in events] == ["expert_cache.selected_slice_copy"]


def test_reader_loads_independent_routed_pack_and_profiles(tmp_path) -> None:
    tensors = _packed_tensors()
    _write_cache(tmp_path)
    events = []
    reader = ExpertCacheReader(
        tmp_path,
        config=_config(),
        profile_callback=lambda name, start: events.append((name, start)),
    )

    packed = reader.load_routed_pack(0, device=torch.device("cpu"))

    assert packed is not None
    assert torch.equal(packed[0], tensors["routed_w1_t"])
    assert torch.equal(packed[1], tensors["routed_w2_t"])
    assert torch.equal(packed[2], tensors["routed_w3_t"])
    assert all(tensor.is_contiguous() for tensor in packed)
    assert packed[0].untyped_storage().data_ptr() != tensors["routed_w1_t"].untyped_storage().data_ptr()
    assert [name for name, _ in events] == ["expert_cache.routed_pack"]


def test_selected_copy_validates_all_outputs_before_writing(tmp_path) -> None:
    _write_cache(tmp_path)
    reader = ExpertCacheReader(tmp_path, config=_config())
    out_w1 = torch.full((2, 4, 3), 7.0, dtype=torch.bfloat16)
    out_w2 = torch.full((2, 3, 4), 7.0, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="out_w3 selected expert shape mismatch"):
        reader.copy_selected_into(
            0,
            [0, 1],
            out_w1=out_w1,
            out_w2=out_w2,
            out_w3=torch.empty(2, 4, 4, dtype=torch.bfloat16),
        )

    assert torch.equal(out_w1, torch.full_like(out_w1, 7.0))
    assert torch.equal(out_w2, torch.full_like(out_w2, 7.0))


@pytest.mark.parametrize(
    ("name", "tensor", "error"),
    [
        ("routed_w1_t", torch.zeros(3, 5, 3, dtype=torch.bfloat16), "shape mismatch"),
        ("routed_w2_t", torch.zeros(3, 3, 4, dtype=torch.float32), "dtype mismatch"),
    ],
)
def test_reader_rejects_invalid_metadata(tmp_path, name, tensor, error) -> None:
    tensors = _packed_tensors()
    tensors[name] = tensor
    _write_manifest(tmp_path)
    save_file(tensors, tmp_path / layer_expert_cache_filename(0))
    reader = ExpertCacheReader(tmp_path, config=_config())

    with pytest.raises((ValueError, TypeError), match=error):
        reader.inspect_layer(0)


@pytest.mark.parametrize(
    "tensors",
    [
        {"routed_w1_t": torch.zeros(3, 4, 3, dtype=torch.bfloat16)},
        {
            **_packed_tensors(),
            "expert_000.w1_t": torch.zeros(4, 3, dtype=torch.bfloat16),
        },
    ],
)
def test_reader_rejects_missing_or_unexpected_keys(tmp_path, tensors) -> None:
    _write_manifest(tmp_path)
    save_file(tensors, tmp_path / layer_expert_cache_filename(0))
    reader = ExpertCacheReader(tmp_path, config=_config())

    with pytest.raises(ValueError, match="keys mismatch"):
        reader.inspect_layer(0)
