"""Convert routed expert weights to BF16 packed runtime cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from safetensors.torch import save_file

from models.config import FLASH_CONFIG
from serving.weight_loader import DeepSeekV4WeightLoader, tensor_nbytes


FORMAT = "dsv4_bf16_routed_pack"
VERSION = 1


def parse_layers(spec: str | None, *, n_layers: int) -> list[int]:
    if spec is None or spec.strip() == "":
        return list(range(n_layers))

    layers: list[int] = []
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
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))

    deduped = sorted(set(layers))
    invalid = [layer for layer in deduped if not 0 <= layer < n_layers]
    if invalid:
        raise ValueError(f"layer ids must be in [0, {n_layers}), got {invalid}")
    return deduped


def convert_layers(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    layers = parse_layers(args.layers, n_layers=FLASH_CONFIG.n_layers)
    loader = DeepSeekV4WeightLoader(
        args.checkpoint,
        weight_index=args.weight_index,
        config=FLASH_CONFIG,
        default_device="cpu",
        profile=args.profile,
    )
    manifest = _load_or_create_manifest(output, args.checkpoint)

    try:
        for layer_id in layers:
            path = output / _cache_filename(layer_id)
            if path.exists() and not args.overwrite:
                print(f"[CACHE] layer {layer_id}: skip existing {path}", flush=True)
                manifest["layers"][str(layer_id)] = path.name
                continue

            loader.reset_profile_stats()
            start = time.perf_counter()
            pack = loader.get_layer_moe_routed_pack(layer_id)
            build_s = time.perf_counter() - start
            tensors = {
                "routed_w1_t": pack.routed_w1_t,
                "routed_w2_t": pack.routed_w2_t,
                "routed_w3_t": pack.routed_w3_t,
            }
            start = time.perf_counter()
            save_file(tensors, path)
            save_s = time.perf_counter() - start
            nbytes = sum(tensor_nbytes(tensor) for tensor in tensors.values())
            manifest["layers"][str(layer_id)] = path.name
            _write_manifest(output, manifest)
            loader.release_prefix(f"layers.{layer_id}.")

            print(
                f"[CACHE] layer {layer_id}: wrote {path} "
                f"bytes={nbytes} build={build_s:.3f}s save={save_s:.3f}s",
                flush=True,
            )
            if args.profile:
                _print_profile(loader)
    finally:
        loader.close()


def _load_or_create_manifest(output: Path, checkpoint: str) -> dict:
    path = output / "manifest.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    return {
        "format": FORMAT,
        "version": VERSION,
        "source_checkpoint": str(Path(checkpoint).expanduser()),
        "n_layers": FLASH_CONFIG.n_layers,
        "n_routed_experts": FLASH_CONFIG.n_routed_experts,
        "dim": FLASH_CONFIG.dim,
        "moe_inter_dim": FLASH_CONFIG.moe_inter_dim,
        "dtype": "bfloat16",
        "layout": {
            "routed_w1_t": ["n_routed_experts", "dim", "moe_inter_dim"],
            "routed_w2_t": ["n_routed_experts", "moe_inter_dim", "dim"],
            "routed_w3_t": ["n_routed_experts", "dim", "moe_inter_dim"],
        },
        "layers": {},
    }


def _write_manifest(output: Path, manifest: dict) -> None:
    with open(output / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _cache_filename(layer_id: int) -> str:
    return f"layer_{layer_id:03d}_routed_pack.safetensors"


def _print_profile(loader: DeepSeekV4WeightLoader) -> None:
    parts = [f"{name}={elapsed_ms:.3f}ms/{count}" for name, count, elapsed_ms in loader.profile_summary()]
    print(f"[CACHE_PROFILE] {' '.join(parts)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build DeepSeek V4 BF16 routed expert packed cache.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--weight-index", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--profile", action="store_true", default=False)
    args = parser.parse_args()
    convert_layers(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
