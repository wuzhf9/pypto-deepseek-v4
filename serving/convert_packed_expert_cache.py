"""Convert routed expert weights to packed per-layer BF16 cache files."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import time
from typing import Any

import torch
from safetensors.torch import safe_open, save_file

from models.config import DeepSeekV4FlashConfig, FLASH_CONFIG
from serving.expert_cache import (
    EXPERT_CACHE_FORMAT,
    EXPERT_CACHE_V2,
    PACKED_KEYS,
    PACKED_W1,
    PACKED_W2,
    PACKED_W3,
    layer_expert_cache_filename,
)
from serving.weight_loader import DeepSeekV4WeightLoader, tensor_nbytes


def parse_layer_ids(spec: str | None, *, count: int) -> list[int]:
    """Parse a comma-separated layer list with inclusive ranges."""
    if spec is None or spec.strip() == "":
        return list(range(count))

    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"invalid layer range: {part}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))

    layer_ids = sorted(set(values))
    invalid = [layer_id for layer_id in layer_ids if not 0 <= layer_id < count]
    if invalid:
        raise ValueError(f"layer ids must be in [0, {count}), got {invalid}")
    return layer_ids


def build_packed_layer(
    loader: DeepSeekV4WeightLoader,
    layer_id: int,
    *,
    config: DeepSeekV4FlashConfig,
) -> dict[str, torch.Tensor]:
    """Build one complete three-tensor packed BF16 layer on CPU."""
    routed_w1_t = torch.empty(
        config.n_routed_experts,
        config.dim,
        config.moe_inter_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )
    routed_w2_t = torch.empty(
        config.n_routed_experts,
        config.moe_inter_dim,
        config.dim,
        dtype=torch.bfloat16,
        device="cpu",
    )
    routed_w3_t = torch.empty(
        config.n_routed_experts,
        config.dim,
        config.moe_inter_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )

    for expert_id in range(config.n_routed_experts):
        expert = loader.get_moe_routed_expert(layer_id, expert_id, device="cpu")
        routed_w1_t[expert_id].copy_(expert.w1_t)
        routed_w2_t[expert_id].copy_(expert.w2_t)
        routed_w3_t[expert_id].copy_(expert.w3_t)
        loader.release_prefix(f"layers.{layer_id}.ffn.experts.{expert_id}.")

    tensors = {
        PACKED_W1: routed_w1_t,
        PACKED_W2: routed_w2_t,
        PACKED_W3: routed_w3_t,
    }
    _validate_packed_tensors(tensors, config=config)
    return tensors


def convert_packed_experts(
    args: argparse.Namespace,
    *,
    config: DeepSeekV4FlashConfig = FLASH_CONFIG,
) -> None:
    """Convert selected complete layers into an independent format v2 directory."""
    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    manifest = _load_or_create_manifest(output, args.checkpoint, config=config)
    _write_manifest_atomic(output, manifest)
    layers = parse_layer_ids(args.layers, count=config.n_layers)

    loader = DeepSeekV4WeightLoader(
        args.checkpoint,
        weight_index=args.weight_index,
        config=config,
        default_device="cpu",
        profile=args.profile,
    )
    try:
        for layer_id in layers:
            filename = layer_expert_cache_filename(layer_id)
            path = output / filename
            if path.exists() and not args.overwrite:
                _validate_packed_file(path, config=config)
                manifest["layers"][str(layer_id)] = {
                    "file": filename,
                    "bytes": path.stat().st_size,
                    "packed": True,
                }
                _write_manifest_atomic(output, manifest)
                print(f"[PACKED_CACHE] layer {layer_id}: skip existing {path}", flush=True)
                continue

            if args.profile:
                loader.reset_profile_stats()
            start = time.perf_counter()
            tensors = build_packed_layer(loader, layer_id, config=config)
            build_s = time.perf_counter() - start
            logical_bytes = sum(tensor_nbytes(tensor) for tensor in tensors.values())

            start = time.perf_counter()
            _write_packed_file_atomic(path, tensors, config=config)
            save_s = time.perf_counter() - start
            manifest["layers"][str(layer_id)] = {
                "file": filename,
                "bytes": path.stat().st_size,
                "logical_bytes": logical_bytes,
                "packed": True,
            }
            _write_manifest_atomic(output, manifest)

            print(
                f"[PACKED_CACHE] layer {layer_id}: wrote {path} "
                f"experts={config.n_routed_experts} logical_bytes={logical_bytes} "
                f"build={build_s:.3f}s save={save_s:.3f}s",
                flush=True,
            )
            if args.profile:
                _print_profile(loader)
    finally:
        loader.close()


def _load_or_create_manifest(
    output: Path,
    checkpoint: str | os.PathLike[str],
    *,
    config: DeepSeekV4FlashConfig,
) -> dict[str, Any]:
    path = output / "manifest.json"
    source_checkpoint = str(Path(checkpoint).expanduser().resolve())
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        _validate_manifest(manifest, source_checkpoint=source_checkpoint, config=config, path=path)
        return manifest

    existing = list(output.iterdir())
    if existing:
        raise ValueError(
            f"Packed expert cache output directory is non-empty but has no manifest: {output}"
        )
    return {
        "format": EXPERT_CACHE_FORMAT,
        "version": EXPERT_CACHE_V2,
        "source_checkpoint": source_checkpoint,
        "n_layers": config.n_layers,
        "n_routed_experts": config.n_routed_experts,
        "dim": config.dim,
        "moe_inter_dim": config.moe_inter_dim,
        "dtype": "bfloat16",
        "layout": {
            PACKED_W1: ["n_routed_experts", "dim", "moe_inter_dim"],
            PACKED_W2: ["n_routed_experts", "moe_inter_dim", "dim"],
            PACKED_W3: ["n_routed_experts", "dim", "moe_inter_dim"],
        },
        "layers": {},
    }


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    source_checkpoint: str,
    config: DeepSeekV4FlashConfig,
    path: Path,
) -> None:
    expected = {
        "format": EXPERT_CACHE_FORMAT,
        "version": EXPERT_CACHE_V2,
        "source_checkpoint": source_checkpoint,
        "n_layers": config.n_layers,
        "n_routed_experts": config.n_routed_experts,
        "dim": config.dim,
        "moe_inter_dim": config.moe_inter_dim,
        "dtype": "bfloat16",
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(
                f"Packed expert cache manifest {name} mismatch in {path}: "
                f"expected {value!r}, got {manifest.get(name)!r}"
            )
    if not isinstance(manifest.get("layers"), dict):
        raise ValueError(f"Packed expert cache manifest layers must be an object: {path}")


def _write_packed_file_atomic(
    path: Path,
    tensors: dict[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
) -> None:
    _validate_packed_tensors(tensors, config=config)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        if temp_path.exists():
            temp_path.unlink()
        save_file(
            tensors,
            temp_path,
            metadata={
                "format": EXPERT_CACHE_FORMAT,
                "version": str(EXPERT_CACHE_V2),
            },
        )
        _validate_packed_file(temp_path, config=config)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _write_manifest_atomic(output: Path, manifest: dict[str, Any]) -> None:
    path = output / "manifest.json"
    temp_path = output / f".manifest.json.tmp-{os.getpid()}"
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _validate_packed_file(path: Path, *, config: DeepSeekV4FlashConfig) -> None:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = frozenset(str(key) for key in handle.keys())
        if keys != frozenset(PACKED_KEYS):
            raise ValueError(
                f"Packed expert cache {path} keys mismatch: "
                f"expected {sorted(PACKED_KEYS)}, got {sorted(keys)}"
            )
        tensors = {name: handle.get_tensor(name) for name in PACKED_KEYS}
        _validate_packed_tensors(tensors, config=config)


def _validate_packed_tensors(
    tensors: Mapping[str, torch.Tensor] | dict[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
) -> None:
    expected_shapes = {
        PACKED_W1: (config.n_routed_experts, config.dim, config.moe_inter_dim),
        PACKED_W2: (config.n_routed_experts, config.moe_inter_dim, config.dim),
        PACKED_W3: (config.n_routed_experts, config.dim, config.moe_inter_dim),
    }
    if frozenset(tensors) != frozenset(PACKED_KEYS):
        raise ValueError(f"Packed expert tensors must contain exactly {sorted(PACKED_KEYS)}")
    for name, shape in expected_shapes.items():
        tensor = tensors[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(f"Packed expert {name} shape mismatch: expected {shape}, got {tuple(tensor.shape)}")
        if tensor.dtype is not torch.bfloat16:
            raise TypeError(f"Packed expert {name} dtype mismatch: expected torch.bfloat16, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"Packed expert {name} must be contiguous")


def _print_profile(loader: DeepSeekV4WeightLoader) -> None:
    parts = [f"{name}={elapsed_ms:.3f}ms/{count}" for name, count, elapsed_ms in loader.profile_summary()]
    print(f"[PACKED_CACHE_PROFILE] {' '.join(parts)}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DeepSeek V4 packed BF16 per-layer routed expert cache.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--weight-index", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--profile", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    convert_packed_experts(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
