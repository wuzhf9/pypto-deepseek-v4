"""Packed BF16 routed-expert cache metadata and tensor reader."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import torch
from safetensors.torch import safe_open

from models.config import DeepSeekV4FlashConfig


EXPERT_CACHE_FORMAT = "dsv4_bf16_layer_experts"
EXPERT_CACHE_VERSION = 2

PACKED_W1 = "routed_w1_t"
PACKED_W2 = "routed_w2_t"
PACKED_W3 = "routed_w3_t"
PACKED_KEYS = (PACKED_W1, PACKED_W2, PACKED_W3)


def layer_expert_cache_filename(layer_id: int) -> str:
    """Return the stable per-layer expert-cache filename."""
    return f"layer_{int(layer_id):03d}_experts.safetensors"


@dataclass(frozen=True)
class ExpertCacheLayerManifest:
    """One layer entry from an expert-cache manifest."""

    file: str


@dataclass(frozen=True)
class ExpertCacheManifest:
    """Validated expert-cache directory metadata."""

    version: int
    n_layers: int
    n_routed_experts: int
    dim: int
    moe_inter_dim: int
    dtype: str
    layers: Mapping[int, ExpertCacheLayerManifest]


@dataclass(frozen=True)
class LayerExpertCacheInfo:
    """Validated keys for one packed layer cache file."""

    layer_id: int
    path: Path
    keys: frozenset[str]


class ExpertCacheReader:
    """Read the final packed expert-cache format."""

    def __init__(
        self,
        directory: str | Path | None,
        *,
        config: DeepSeekV4FlashConfig,
        profile_callback: Callable[[str, float], None] | None = None,
    ) -> None:
        self.directory = Path(directory) if directory is not None else None
        self.config = config
        self._profile_callback = profile_callback
        self._manifest = self._load_manifest()
        self._handles: dict[Path, Any] = {}
        self._layer_info: dict[int, LayerExpertCacheInfo] = {}

    @property
    def manifest(self) -> ExpertCacheManifest | None:
        return self._manifest

    @property
    def open_handle_count(self) -> int:
        return len(self._handles)

    def layer_path(self, layer_id: int) -> Path | None:
        self._validate_layer_id(layer_id)
        if self.directory is None:
            return None
        assert self._manifest is not None
        entry = self._manifest.layers.get(layer_id)
        if entry is None:
            return None
        return self.directory / entry.file

    def inspect_layer(self, layer_id: int) -> LayerExpertCacheInfo | None:
        """Validate and cache one packed layer without cloning tensors."""
        self._validate_layer_id(layer_id)
        cached = self._layer_info.get(layer_id)
        if cached is not None:
            return cached

        path = self.layer_path(layer_id)
        if path is None:
            return None
        if not path.exists():
            raise FileNotFoundError(f"Expert cache manifest references missing layer file: {path}")

        handle = self._get_handle(path)
        keys = frozenset(str(key) for key in handle.keys())
        expected_keys = frozenset(PACKED_KEYS)
        if keys != expected_keys:
            missing = expected_keys.difference(keys)
            unexpected = keys.difference(expected_keys)
            raise ValueError(
                f"Packed expert cache {path} keys mismatch: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        self._validate_packed_metadata(handle, path)

        info = LayerExpertCacheInfo(layer_id=layer_id, path=path, keys=keys)
        self._layer_info[layer_id] = info
        return info

    def copy_selected_into(
        self,
        layer_id: int,
        expert_ids: Sequence[int],
        *,
        out_w1: torch.Tensor,
        out_w2: torch.Tensor,
        out_w3: torch.Tensor,
    ) -> bool:
        """Copy selected expert slices, returning False for an uncached layer."""
        ids = tuple(int(expert_id) for expert_id in expert_ids)
        for expert_id in ids:
            self._validate_expert_id(expert_id)
        info = self.inspect_layer(layer_id)
        if info is None:
            return False

        count = len(ids)
        self._validate_selected_output("out_w1", out_w1, (count, self.config.dim, self.config.moe_inter_dim))
        self._validate_selected_output("out_w2", out_w2, (count, self.config.moe_inter_dim, self.config.dim))
        self._validate_selected_output("out_w3", out_w3, (count, self.config.dim, self.config.moe_inter_dim))

        start = time.perf_counter()
        handle = self._get_handle(info.path)
        w1_slice = handle.get_slice(PACKED_W1)
        w2_slice = handle.get_slice(PACKED_W2)
        w3_slice = handle.get_slice(PACKED_W3)
        selected = [
            (
                w1_slice[expert_id],
                w2_slice[expert_id],
                w3_slice[expert_id],
            )
            for expert_id in ids
        ]
        for w1_t, w2_t, w3_t in selected:
            self._validate_expert_tensor("w1_t", w1_t, (self.config.dim, self.config.moe_inter_dim))
            self._validate_expert_tensor("w2_t", w2_t, (self.config.moe_inter_dim, self.config.dim))
            self._validate_expert_tensor("w3_t", w3_t, (self.config.dim, self.config.moe_inter_dim))
        for slot, (w1_t, w2_t, w3_t) in enumerate(selected):
            out_w1[slot].copy_(w1_t)
            out_w2[slot].copy_(w2_t)
            out_w3[slot].copy_(w3_t)
        self._record_profile("expert_cache.selected_slice_copy", start)
        return True

    def load_routed_pack(
        self,
        layer_id: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Clone one complete routed-expert pack for prefill."""
        info = self.inspect_layer(layer_id)
        if info is None:
            return None

        start = time.perf_counter()
        handle = self._get_handle(info.path)
        routed_w1_t = self._materialize_tensor(handle.get_tensor(PACKED_W1), device)
        routed_w2_t = self._materialize_tensor(handle.get_tensor(PACKED_W2), device)
        routed_w3_t = self._materialize_tensor(handle.get_tensor(PACKED_W3), device)
        self._record_profile("expert_cache.routed_pack", start)
        return routed_w1_t, routed_w2_t, routed_w3_t

    def close(self) -> None:
        """Close cached safe_open handles; repeated calls are harmless."""
        for handle in self._handles.values():
            handle.__exit__(None, None, None)
        self._handles.clear()
        self._layer_info.clear()

    def _load_manifest(self) -> ExpertCacheManifest | None:
        if self.directory is None:
            return None
        path = self.directory / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Expert cache directory requires manifest.json: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("format") != EXPERT_CACHE_FORMAT:
            raise ValueError(f"Unsupported expert cache format in {path}: {data.get('format')!r}")
        version = int(data.get("version", -1))
        if version != EXPERT_CACHE_VERSION:
            raise ValueError(
                f"Unsupported expert cache version in {path}: "
                f"expected {EXPERT_CACHE_VERSION}, got {version}"
            )

        expected = {
            "n_layers": self.config.n_layers,
            "n_routed_experts": self.config.n_routed_experts,
            "dim": self.config.dim,
            "moe_inter_dim": self.config.moe_inter_dim,
            "dtype": "bfloat16",
        }
        for name, value in expected.items():
            if data.get(name) != value:
                raise ValueError(
                    f"Expert cache manifest {name} mismatch in {path}: "
                    f"expected {value!r}, got {data.get(name)!r}"
                )

        raw_layers = data.get("layers")
        if not isinstance(raw_layers, dict):
            raise ValueError(f"Expert cache manifest layers must be an object: {path}")
        layers: dict[int, ExpertCacheLayerManifest] = {}
        for raw_layer_id, raw_entry in raw_layers.items():
            layer_id = int(raw_layer_id)
            self._validate_layer_id(layer_id)
            if not isinstance(raw_entry, dict) or not isinstance(raw_entry.get("file"), str):
                raise ValueError(f"Invalid expert cache layer entry {raw_layer_id!r} in {path}")
            layers[layer_id] = ExpertCacheLayerManifest(file=raw_entry["file"])

        return ExpertCacheManifest(
            version=version,
            n_layers=int(data["n_layers"]),
            n_routed_experts=int(data["n_routed_experts"]),
            dim=int(data["dim"]),
            moe_inter_dim=int(data["moe_inter_dim"]),
            dtype=str(data["dtype"]),
            layers=layers,
        )

    def _get_handle(self, path: Path) -> Any:
        path = path.resolve()
        handle = self._handles.get(path)
        if handle is None:
            handle = safe_open(path, framework="pt", device="cpu")
            self._handles[path] = handle
        return handle

    def _validate_packed_metadata(self, handle: Any, path: Path) -> None:
        expected = {
            PACKED_W1: (self.config.n_routed_experts, self.config.dim, self.config.moe_inter_dim),
            PACKED_W2: (self.config.n_routed_experts, self.config.moe_inter_dim, self.config.dim),
            PACKED_W3: (self.config.n_routed_experts, self.config.dim, self.config.moe_inter_dim),
        }
        for name, shape in expected.items():
            tensor_slice = handle.get_slice(name)
            actual_shape = tuple(int(dim) for dim in tensor_slice.get_shape())
            if actual_shape != shape:
                raise ValueError(
                    f"Packed expert {name} shape mismatch in {path}: expected {shape}, got {actual_shape}"
                )
            dtype = tensor_slice.get_dtype()
            if dtype != "BF16":
                raise TypeError(
                    f"Packed expert {name} dtype mismatch in {path}: expected BF16, got {dtype}"
                )

    def _record_profile(self, name: str, start: float) -> None:
        if self._profile_callback is not None:
            self._profile_callback(name, start)

    def _validate_layer_id(self, layer_id: int) -> None:
        if not 0 <= layer_id < self.config.n_layers:
            raise ValueError(f"layer_id must be in [0, {self.config.n_layers}), got {layer_id}")

    def _validate_expert_id(self, expert_id: int) -> None:
        if not 0 <= expert_id < self.config.n_routed_experts:
            raise ValueError(
                f"expert_id must be in [0, {self.config.n_routed_experts}), got {expert_id}"
            )

    @staticmethod
    def _validate_expert_tensor(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} expert cache shape mismatch: expected {shape}, got {tuple(tensor.shape)}")
        if tensor.dtype is not torch.bfloat16:
            raise TypeError(f"{name} expert cache dtype mismatch: expected torch.bfloat16, got {tensor.dtype}")

    @staticmethod
    def _validate_selected_output(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} selected expert shape mismatch: expected {shape}, got {tuple(tensor.shape)}")
        if tensor.dtype is not torch.bfloat16:
            raise TypeError(f"{name} selected expert dtype mismatch: expected torch.bfloat16, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} selected expert output must be contiguous")

    @staticmethod
    def _materialize_tensor(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        if device.type == "cpu":
            return tensor.clone().contiguous()
        return tensor.to(device).contiguous()


__all__ = [
    "EXPERT_CACHE_FORMAT",
    "EXPERT_CACHE_VERSION",
    "PACKED_KEYS",
    "PACKED_W1",
    "PACKED_W2",
    "PACKED_W3",
    "ExpertCacheLayerManifest",
    "ExpertCacheManifest",
    "ExpertCacheReader",
    "LayerExpertCacheInfo",
    "layer_expert_cache_filename",
]
