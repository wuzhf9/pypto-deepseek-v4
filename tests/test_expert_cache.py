"""Tests for versioned routed-expert cache metadata and v1 loading."""

from dataclasses import replace
import json

import pytest
import torch
from safetensors.torch import save_file

from models.config import FLASH_CONFIG
from serving.expert_cache import (
    EXPERT_CACHE_FORMAT,
    EXPERT_CACHE_V1,
    EXPERT_CACHE_V2,
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


def _expert_tensors(expert_ids=(0, 1, 2)):
    tensors = {}
    for expert_id in expert_ids:
        value = float(expert_id + 1)
        prefix = f"expert_{expert_id:03d}"
        tensors[f"{prefix}.w1_t"] = torch.full((4, 3), value, dtype=torch.bfloat16)
        tensors[f"{prefix}.w2_t"] = torch.full((3, 4), value + 10, dtype=torch.bfloat16)
        tensors[f"{prefix}.w3_t"] = torch.full((4, 3), value + 20, dtype=torch.bfloat16)
    return tensors


def _write_manifest(directory, *, version=EXPERT_CACHE_V1, experts=(0, 1, 2), **overrides):
    cfg = _config()
    data = {
        "format": EXPERT_CACHE_FORMAT,
        "version": version,
        "source_checkpoint": "checkpoint",
        "n_layers": cfg.n_layers,
        "n_routed_experts": cfg.n_routed_experts,
        "dim": cfg.dim,
        "moe_inter_dim": cfg.moe_inter_dim,
        "dtype": "bfloat16",
        "layers": {
            "0": {
                "file": layer_expert_cache_filename(0),
                "experts": list(experts),
            }
        },
    }
    data.update(overrides)
    (directory / "manifest.json").write_text(json.dumps(data), encoding="utf-8")


def test_v1_reader_loads_clones_reuses_handle_and_profiles(tmp_path) -> None:
    path = tmp_path / layer_expert_cache_filename(0)
    save_file(_expert_tensors(), path)
    events = []
    reader = ExpertCacheReader(
        tmp_path,
        config=_config(),
        profile_callback=lambda name, start: events.append((name, start)),
    )

    first = reader.load_expert(0, 1, device=torch.device("cpu"))
    second = reader.load_expert(0, 2, device=torch.device("cpu"))

    assert first is not None and second is not None
    assert torch.equal(first[0], torch.full((4, 3), 2.0, dtype=torch.bfloat16))
    assert torch.equal(first[1], torch.full((3, 4), 12.0, dtype=torch.bfloat16))
    assert torch.equal(first[2], torch.full((4, 3), 22.0, dtype=torch.bfloat16))
    assert first[0].is_contiguous()
    assert first[0].untyped_storage().data_ptr() != second[0].untyped_storage().data_ptr()
    assert reader.open_handle_count == 1
    assert [name for name, _ in events] == ["expert_cache.load", "expert_cache.load"]

    reader.close()
    reader.close()
    assert reader.open_handle_count == 0


def test_reader_returns_none_for_missing_layer_or_uncached_legacy_expert(tmp_path) -> None:
    reader = ExpertCacheReader(tmp_path, config=_config())
    assert reader.load_expert(0, 0, device=torch.device("cpu")) is None

    save_file(_expert_tensors((0,)), tmp_path / layer_expert_cache_filename(0))
    assert reader.load_expert(0, 1, device=torch.device("cpu")) is None


def test_manifest_declared_expert_missing_key_is_corruption(tmp_path) -> None:
    save_file(_expert_tensors((0,)), tmp_path / layer_expert_cache_filename(0))
    _write_manifest(tmp_path, experts=(0, 1))
    reader = ExpertCacheReader(tmp_path, config=_config())

    with pytest.raises(KeyError, match="declares layer 0 expert 1"):
        reader.load_expert(0, 1, device=torch.device("cpu"))


def test_manifest_layer_file_missing_is_error(tmp_path) -> None:
    _write_manifest(tmp_path)
    reader = ExpertCacheReader(tmp_path, config=_config())

    with pytest.raises(FileNotFoundError, match="missing layer file"):
        reader.inspect_layer(0)


def test_manifest_config_mismatch_is_rejected(tmp_path) -> None:
    _write_manifest(tmp_path, dim=99)

    with pytest.raises(ValueError, match="dim mismatch"):
        ExpertCacheReader(tmp_path, config=_config())


def test_reader_detects_complete_v2_file_without_materializing_it(tmp_path) -> None:
    save_file(
        {
            "routed_w1_t": torch.zeros(3, 4, 3, dtype=torch.bfloat16),
            "routed_w2_t": torch.zeros(3, 3, 4, dtype=torch.bfloat16),
            "routed_w3_t": torch.zeros(3, 4, 3, dtype=torch.bfloat16),
        },
        tmp_path / layer_expert_cache_filename(0),
    )
    reader = ExpertCacheReader(tmp_path, config=_config())

    info = reader.inspect_layer(0)
    assert info is not None
    assert info.version == EXPERT_CACHE_V2
    expert = reader.load_expert(0, 0, device=torch.device("cpu"))
    assert expert is not None
    assert expert[0].shape == (4, 3)
    assert expert[1].shape == (3, 4)
    assert expert[2].shape == (4, 3)


def test_v2_selected_copy_preserves_order_duplicates_and_uses_slice_profile(tmp_path) -> None:
    packed_w1 = torch.stack([torch.full((4, 3), float(i + 1), dtype=torch.bfloat16) for i in range(3)])
    packed_w2 = torch.stack([torch.full((3, 4), float(i + 11), dtype=torch.bfloat16) for i in range(3)])
    packed_w3 = torch.stack([torch.full((4, 3), float(i + 21), dtype=torch.bfloat16) for i in range(3)])
    save_file(
        {
            "routed_w1_t": packed_w1,
            "routed_w2_t": packed_w2,
            "routed_w3_t": packed_w3,
        },
        tmp_path / layer_expert_cache_filename(0),
    )
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

    assert torch.equal(out_w1, packed_w1[[2, 0, 2]])
    assert torch.equal(out_w2, packed_w2[[2, 0, 2]])
    assert torch.equal(out_w3, packed_w3[[2, 0, 2]])
    assert [name for name, _ in events] == ["expert_cache.v2.selected_slice_copy"]


def test_v2_reader_loads_full_packed_clones_and_profiles(tmp_path) -> None:
    packed_w1 = torch.arange(3 * 4 * 3, dtype=torch.bfloat16).reshape(3, 4, 3)
    packed_w2 = torch.arange(3 * 3 * 4, dtype=torch.bfloat16).reshape(3, 3, 4) + 100
    packed_w3 = torch.arange(3 * 4 * 3, dtype=torch.bfloat16).reshape(3, 4, 3) + 200
    save_file(
        {
            "routed_w1_t": packed_w1,
            "routed_w2_t": packed_w2,
            "routed_w3_t": packed_w3,
        },
        tmp_path / layer_expert_cache_filename(0),
    )
    events = []
    reader = ExpertCacheReader(
        tmp_path,
        config=_config(),
        profile_callback=lambda name, start: events.append((name, start)),
    )

    packed = reader.load_packed_clone(0, device=torch.device("cpu"))

    assert packed is not None
    assert torch.equal(packed[0], packed_w1)
    assert torch.equal(packed[1], packed_w2)
    assert torch.equal(packed[2], packed_w3)
    assert all(tensor.is_contiguous() for tensor in packed)
    assert packed[0].untyped_storage().data_ptr() != packed_w1.untyped_storage().data_ptr()
    assert [name for name, _ in events] == ["expert_cache.v2.packed_clone"]


def test_v1_packed_clone_returns_none_for_weight_loader_fallback(tmp_path) -> None:
    save_file(_expert_tensors(), tmp_path / layer_expert_cache_filename(0))
    reader = ExpertCacheReader(tmp_path, config=_config())

    assert reader.load_packed_clone(0, device=torch.device("cpu")) is None


def test_v1_experts_and_v2_full_pack_are_elementwise_equal(tmp_path) -> None:
    v1_dir = tmp_path / "v1"
    v2_dir = tmp_path / "v2"
    v1_dir.mkdir()
    v2_dir.mkdir()
    v1_tensors = _expert_tensors()
    save_file(v1_tensors, v1_dir / layer_expert_cache_filename(0))
    save_file(
        {
            "routed_w1_t": torch.stack(
                [v1_tensors[f"expert_{expert_id:03d}.w1_t"] for expert_id in range(3)]
            ),
            "routed_w2_t": torch.stack(
                [v1_tensors[f"expert_{expert_id:03d}.w2_t"] for expert_id in range(3)]
            ),
            "routed_w3_t": torch.stack(
                [v1_tensors[f"expert_{expert_id:03d}.w3_t"] for expert_id in range(3)]
            ),
        },
        v2_dir / layer_expert_cache_filename(0),
    )
    v1_reader = ExpertCacheReader(v1_dir, config=_config())
    v2_reader = ExpertCacheReader(v2_dir, config=_config())

    v2_pack = v2_reader.load_packed_clone(0, device=torch.device("cpu"))

    assert v2_pack is not None
    for expert_id in range(3):
        v1_expert = v1_reader.load_expert(0, expert_id, device=torch.device("cpu"))
        assert v1_expert is not None
        assert all(
            torch.equal(v1_tensor, v2_tensor[expert_id])
            for v1_tensor, v2_tensor in zip(v1_expert, v2_pack)
        )


def test_v2_selected_copy_validates_all_outputs_before_writing(tmp_path) -> None:
    save_file(
        {
            "routed_w1_t": torch.zeros(3, 4, 3, dtype=torch.bfloat16),
            "routed_w2_t": torch.zeros(3, 3, 4, dtype=torch.bfloat16),
            "routed_w3_t": torch.zeros(3, 4, 3, dtype=torch.bfloat16),
        },
        tmp_path / layer_expert_cache_filename(0),
    )
    reader = ExpertCacheReader(tmp_path, config=_config())
    out_w1 = torch.full((2, 4, 3), 7.0, dtype=torch.bfloat16)
    out_w2 = torch.full((2, 3, 4), 7.0, dtype=torch.bfloat16)
    invalid_w3 = torch.empty(2, 4, 4, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="out_w3 selected expert shape mismatch"):
        reader.copy_selected_into(
            0,
            [0, 1],
            out_w1=out_w1,
            out_w2=out_w2,
            out_w3=invalid_w3,
        )

    assert torch.equal(out_w1, torch.full_like(out_w1, 7.0))
    assert torch.equal(out_w2, torch.full_like(out_w2, 7.0))


def test_v1_selected_copy_returns_false_for_weight_loader_fallback(tmp_path) -> None:
    save_file(_expert_tensors(), tmp_path / layer_expert_cache_filename(0))
    reader = ExpertCacheReader(tmp_path, config=_config())

    assert not reader.copy_selected_into(
        0,
        [0, 1],
        out_w1=torch.empty(2, 4, 3, dtype=torch.bfloat16),
        out_w2=torch.empty(2, 3, 4, dtype=torch.bfloat16),
        out_w3=torch.empty(2, 4, 3, dtype=torch.bfloat16),
    )


@pytest.mark.parametrize(
    ("name", "tensor", "error"),
    [
        ("routed_w1_t", torch.zeros(3, 5, 3, dtype=torch.bfloat16), "shape mismatch"),
        ("routed_w2_t", torch.zeros(3, 3, 4, dtype=torch.float32), "dtype mismatch"),
    ],
)
def test_reader_rejects_invalid_v2_metadata(tmp_path, name, tensor, error) -> None:
    tensors = {
        "routed_w1_t": torch.zeros(3, 4, 3, dtype=torch.bfloat16),
        "routed_w2_t": torch.zeros(3, 3, 4, dtype=torch.bfloat16),
        "routed_w3_t": torch.zeros(3, 4, 3, dtype=torch.bfloat16),
    }
    tensors[name] = tensor
    save_file(tensors, tmp_path / layer_expert_cache_filename(0))
    reader = ExpertCacheReader(tmp_path, config=_config())

    with pytest.raises((ValueError, TypeError), match=error):
        reader.inspect_layer(0)


def test_reader_rejects_partial_or_mixed_v2_keys(tmp_path) -> None:
    path = tmp_path / layer_expert_cache_filename(0)
    save_file({"routed_w1_t": torch.zeros(3, 4, 3, dtype=torch.bfloat16)}, path)
    reader = ExpertCacheReader(tmp_path, config=_config())
    with pytest.raises(ValueError, match="Incomplete packed expert cache"):
        reader.inspect_layer(0)
    reader.close()

    path.unlink()
    tensors = _expert_tensors((0,))
    tensors.update(
        {
            "routed_w1_t": torch.zeros(3, 4, 3, dtype=torch.bfloat16),
            "routed_w2_t": torch.zeros(3, 3, 4, dtype=torch.bfloat16),
            "routed_w3_t": torch.zeros(3, 4, 3, dtype=torch.bfloat16),
        }
    )
    save_file(tensors, path)
    reader = ExpertCacheReader(tmp_path, config=_config())
    with pytest.raises(ValueError, match="Mixed v1/v2"):
        reader.inspect_layer(0)


def test_manifest_version_must_match_detected_file(tmp_path) -> None:
    save_file(_expert_tensors(), tmp_path / layer_expert_cache_filename(0))
    _write_manifest(tmp_path, version=EXPERT_CACHE_V2, experts=())
    reader = ExpertCacheReader(tmp_path, config=_config())

    with pytest.raises(ValueError, match="does not match"):
        reader.inspect_layer(0)
