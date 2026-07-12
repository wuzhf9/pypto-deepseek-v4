"""Tests for the packed BF16 expert-cache converter."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import safe_open, save_file

from models.config import FLASH_CONFIG
import serving.convert_expert_cache as converter
from serving.expert_cache import (
    EXPERT_CACHE_FORMAT,
    EXPERT_CACHE_VERSION,
    PACKED_KEYS,
    PACKED_W1,
    PACKED_W2,
    PACKED_W3,
    layer_expert_cache_filename,
)
from serving.weight_loader import DeepSeekV4WeightLoader


def _config():
    return replace(
        FLASH_CONFIG,
        dim=4,
        n_layers=2,
        n_routed_experts=2,
        n_activated_experts=2,
        moe_inter_dim=3,
    )


def _checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    tensors = {}
    for layer_id in range(2):
        for expert_id in range(2):
            base = float(layer_id * 10 + expert_id + 1)
            prefix = f"model.layers.{layer_id}.mlp.experts.{expert_id}"
            tensors[f"{prefix}.gate_proj.weight"] = torch.full((3, 4), base, dtype=torch.bfloat16)
            tensors[f"{prefix}.down_proj.weight"] = torch.full((4, 3), base + 2, dtype=torch.bfloat16)
            tensors[f"{prefix}.up_proj.weight"] = torch.full((3, 4), base + 4, dtype=torch.bfloat16)
    save_file(tensors, checkpoint / "model.safetensors")
    index = {"weight_map": {name: "model.safetensors" for name in tensors}}
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
    return checkpoint, index


def _args(checkpoint, output, **overrides):
    values = {
        "checkpoint": str(checkpoint),
        "output": str(output),
        "layers": "0",
        "overwrite": False,
        "profile": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _packed_tensors(value: float = 1.0):
    cfg = _config()
    return {
        PACKED_W1: torch.full((cfg.n_routed_experts, cfg.dim, cfg.moe_inter_dim), value, dtype=torch.bfloat16),
        PACKED_W2: torch.full((cfg.n_routed_experts, cfg.moe_inter_dim, cfg.dim), value + 1, dtype=torch.bfloat16),
        PACKED_W3: torch.full((cfg.n_routed_experts, cfg.dim, cfg.moe_inter_dim), value + 2, dtype=torch.bfloat16),
    }


def test_parse_layer_ids_supports_ranges_and_rejects_invalid_values() -> None:
    assert converter.parse_layer_ids(None, count=4) == [0, 1, 2, 3]
    assert converter.parse_layer_ids("3,1-2,2", count=4) == [1, 2, 3]
    with pytest.raises(ValueError, match="invalid layer range"):
        converter.parse_layer_ids("2-1", count=4)
    with pytest.raises(ValueError, match="layer ids"):
        converter.parse_layer_ids("4", count=4)


def test_build_packed_layer_places_experts_on_first_dimension(tmp_path) -> None:
    checkpoint, index = _checkpoint(tmp_path)
    cfg = _config()
    loader = DeepSeekV4WeightLoader(checkpoint, index, config=cfg)

    packed = converter.build_packed_layer(loader, 1, config=cfg)

    assert set(packed) == set(PACKED_KEYS)
    assert packed[PACKED_W1].shape == (2, 4, 3)
    assert packed[PACKED_W2].shape == (2, 3, 4)
    assert packed[PACKED_W3].shape == (2, 4, 3)
    assert torch.equal(packed[PACKED_W1][0], torch.full((4, 3), 11.0, dtype=torch.bfloat16))
    assert torch.equal(packed[PACKED_W2][1], torch.full((3, 4), 14.0, dtype=torch.bfloat16))
    assert torch.equal(packed[PACKED_W3][1], torch.full((4, 3), 16.0, dtype=torch.bfloat16))
    assert all(tensor.is_contiguous() for tensor in packed.values())
    loader.close()


def test_converter_writes_manifest_and_round_trips_one_layer(tmp_path) -> None:
    checkpoint, _ = _checkpoint(tmp_path)
    output = tmp_path / "packed"
    cfg = _config()

    converter.convert_experts(_args(checkpoint, output), config=cfg)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == EXPERT_CACHE_FORMAT
    assert manifest["version"] == EXPERT_CACHE_VERSION
    assert set(manifest["layers"]) == {"0"}
    assert manifest["layers"]["0"]["packed"] is True
    path = output / layer_expert_cache_filename(0)
    with safe_open(path, framework="pt", device="cpu") as handle:
        assert set(handle.keys()) == set(PACKED_KEYS)
        assert torch.equal(handle.get_tensor(PACKED_W1)[1], torch.full((4, 3), 2.0, dtype=torch.bfloat16))
        assert torch.equal(handle.get_tensor(PACKED_W2)[0], torch.full((3, 4), 3.0, dtype=torch.bfloat16))
        assert torch.equal(handle.get_tensor(PACKED_W3)[0], torch.full((4, 3), 5.0, dtype=torch.bfloat16))
        assert handle.metadata() == {"format": EXPERT_CACHE_FORMAT, "version": str(EXPERT_CACHE_VERSION)}


def test_converter_skips_valid_existing_layer_without_overwrite(tmp_path, capsys) -> None:
    checkpoint, _ = _checkpoint(tmp_path)
    output = tmp_path / "packed"
    cfg = _config()
    args = _args(checkpoint, output)
    converter.convert_experts(args, config=cfg)
    path = output / layer_expert_cache_filename(0)
    before = path.stat().st_mtime_ns

    converter.convert_experts(args, config=cfg)

    assert path.stat().st_mtime_ns == before
    assert "skip existing" in capsys.readouterr().out


def test_converter_rejects_nonempty_unmanifested_or_wrong_version_output(tmp_path) -> None:
    checkpoint, _ = _checkpoint(tmp_path)
    cfg = _config()
    output = tmp_path / "nonempty"
    output.mkdir()
    (output / "unknown").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty but has no manifest"):
        converter.convert_experts(_args(checkpoint, output), config=cfg)

    output = tmp_path / "wrong_version"
    output.mkdir()
    manifest = {
        "format": EXPERT_CACHE_FORMAT,
        "version": EXPERT_CACHE_VERSION - 1,
        "source_checkpoint": str(checkpoint.resolve()),
        "n_layers": cfg.n_layers,
        "n_routed_experts": cfg.n_routed_experts,
        "dim": cfg.dim,
        "moe_inter_dim": cfg.moe_inter_dim,
        "dtype": "bfloat16",
        "layers": {},
    }
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="version mismatch"):
        converter.convert_experts(_args(checkpoint, output), config=cfg)


def test_atomic_layer_write_preserves_existing_file_when_validation_fails(tmp_path, monkeypatch) -> None:
    path = tmp_path / layer_expert_cache_filename(0)
    old_tensors = _packed_tensors(1.0)
    save_file(old_tensors, path)
    original = path.read_bytes()

    def fail_validation(_path, *, config):
        del config
        raise RuntimeError("validation failed")

    monkeypatch.setattr(converter, "_validate_packed_file", fail_validation)
    with pytest.raises(RuntimeError, match="validation failed"):
        converter._write_packed_file_atomic(path, _packed_tensors(9.0), config=_config())

    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_cli_does_not_accept_partial_experts_argument() -> None:
    with pytest.raises(SystemExit):
        converter.parse_args(
            [
                "--checkpoint",
                "checkpoint",
                "--output",
                "output",
                "--experts",
                "0",
            ]
        )


def test_cli_does_not_accept_weight_index_argument() -> None:
    with pytest.raises(SystemExit):
        converter.parse_args(
            [
                "--checkpoint",
                "checkpoint",
                "--output",
                "output",
                "--weight-index",
                "index.json",
            ]
        )
