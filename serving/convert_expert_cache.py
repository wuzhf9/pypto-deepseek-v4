"""Convert routed expert weights to per-layer BF16 expert cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from safetensors.torch import save_file

from models.config import FLASH_CONFIG
from serving.weight_loader import DeepSeekV4WeightLoader, tensor_nbytes


FORMAT = "dsv4_bf16_layer_experts"
VERSION = 1


def parse_ids(spec: str | None, *, count: int, label: str) -> list[int]:
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
                raise ValueError(f"invalid {label} range: {part}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))

    deduped = sorted(set(values))
    invalid = [value for value in deduped if not 0 <= value < count]
    if invalid:
        raise ValueError(f"{label} ids must be in [0, {count}), got {invalid}")
    return deduped


def convert_experts(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    layers = parse_ids(args.layers, count=FLASH_CONFIG.n_layers, label="layer")
    experts = parse_ids(args.experts, count=FLASH_CONFIG.n_routed_experts, label="expert")
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
                manifest["layers"][str(layer_id)] = {
                    "file": _cache_filename(layer_id),
                    "experts": experts,
                }
                _write_manifest(output, manifest)
                continue

            tensors = {}
            build_s = 0.0
            nbytes = 0
            for expert_id in experts:
                loader.reset_profile_stats()
                start = time.perf_counter()
                expert = loader.get_moe_routed_expert(layer_id, expert_id)
                build_s += time.perf_counter() - start
                prefix = _expert_prefix(expert_id)
                tensors[f"{prefix}.w1_t"] = expert.w1_t
                tensors[f"{prefix}.w2_t"] = expert.w2_t
                tensors[f"{prefix}.w3_t"] = expert.w3_t
                nbytes += tensor_nbytes(expert.w1_t) + tensor_nbytes(expert.w2_t) + tensor_nbytes(expert.w3_t)
                loader.release_prefix(f"layers.{layer_id}.ffn.experts.{expert_id}.")

                if args.profile:
                    print(f"[CACHE] layer {layer_id} expert {expert_id}: build done", flush=True)
                    _print_profile(loader)

            start = time.perf_counter()
            save_file(tensors, path)
            save_s = time.perf_counter() - start
            manifest["layers"][str(layer_id)] = {
                "file": _cache_filename(layer_id),
                "experts": experts,
            }
            _write_manifest(output, manifest)

            print(
                f"[CACHE] layer {layer_id}: wrote {path} experts={len(experts)} "
                f"bytes={nbytes} build={build_s:.3f}s save={save_s:.3f}s",
                flush=True,
            )
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
            "expert_{expert_id:03d}.w1_t": ["dim", "moe_inter_dim"],
            "expert_{expert_id:03d}.w2_t": ["moe_inter_dim", "dim"],
            "expert_{expert_id:03d}.w3_t": ["dim", "moe_inter_dim"],
        },
        "layers": {},
    }


def _write_manifest(output: Path, manifest: dict) -> None:
    with open(output / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _cache_filename(layer_id: int) -> str:
    return f"layer_{layer_id:03d}_experts.safetensors"


def _expert_prefix(expert_id: int) -> str:
    return f"expert_{expert_id:03d}"


def _print_profile(loader: DeepSeekV4WeightLoader) -> None:
    parts = [f"{name}={elapsed_ms:.3f}ms/{count}" for name, count, elapsed_ms in loader.profile_summary()]
    print(f"[CACHE_PROFILE] {' '.join(parts)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build DeepSeek V4 BF16 per-layer routed expert cache.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--weight-index", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--experts", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--profile", action="store_true", default=False)
    args = parser.parse_args()
    convert_experts(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
