"""Versioned BF16 routed-expert cache metadata and tensor reader."""

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
EXPERT_CACHE_V1 = 1
EXPERT_CACHE_V2 = 2

PACKED_W1 = "routed_w1_t"
PACKED_W2 = "routed_w2_t"
PACKED_W3 = "routed_w3_t"
PACKED_KEYS = (PACKED_W1, PACKED_W2, PACKED_W3)


def layer_expert_cache_filename(layer_id: int) -> str:
    """Return the stable per-layer expert-cache filename."""
    return f"layer_{int(layer_id):03d}_experts.safetensors"


@dataclass(frozen=True)
class ExpertCacheLayerManifest:
    """One optional layer entry from an expert-cache manifest."""

    file: str
    experts: tuple[int, ...] | None


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
    """Detected format and keys for one layer cache file."""

    layer_id: int
    path: Path
    version: int
    keys: frozenset[str]


class ExpertCacheReader:
    """Read v1 per-expert and v2 packed caches behind one format boundary."""

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
        if self._manifest is not None:
            entry = self._manifest.layers.get(layer_id)
            if entry is None:
                return None
            return self.directory / entry.file
        return self.directory / layer_expert_cache_filename(layer_id)

    def inspect_layer(self, layer_id: int) -> LayerExpertCacheInfo | None:
        """Detect and cache one layer file's format without cloning tensors."""
        self._validate_layer_id(layer_id)
        cached = self._layer_info.get(layer_id)
        if cached is not None:
            return cached

        path = self.layer_path(layer_id)
        if path is None:
            return None
        if not path.exists():
            if self._manifest is not None and layer_id in self._manifest.layers:
                raise FileNotFoundError(f"Expert cache manifest references missing layer file: {path}")
            return None

        handle = self._get_handle(path)
        keys = frozenset(str(key) for key in handle.keys())
        packed_present = frozenset(PACKED_KEYS).intersection(keys)
        v1_present = frozenset(key for key in keys if key.startswith("expert_"))

        if packed_present:
            missing = frozenset(PACKED_KEYS).difference(keys)
            if missing:
                raise ValueError(f"Incomplete packed expert cache {path}: missing keys {sorted(missing)}")
            if v1_present:
                raise ValueError(f"Mixed v1/v2 expert cache keys are not supported: {path}")
            unexpected = keys.difference(PACKED_KEYS)
            if unexpected:
                raise ValueError(f"Packed expert cache {path} has unexpected keys {sorted(unexpected)}")
            version = EXPERT_CACHE_V2
            self._validate_packed_metadata(handle, path)
        else:
            if not v1_present:
                raise ValueError(f"Expert cache file has no recognized expert tensors: {path}")
            version = EXPERT_CACHE_V1

        if self._manifest is not None and self._manifest.version != version:
            raise ValueError(
                f"Expert cache manifest version {self._manifest.version} does not match "
                f"layer {layer_id} file version {version}: {path}"
            )

        info = LayerExpertCacheInfo(layer_id=layer_id, path=path, version=version, keys=keys)
        self._layer_info[layer_id] = info
        return info

    def load_expert(
        self,
        layer_id: int,
        expert_id: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Load one v1 expert, or return None when it is not cached."""
        self._validate_expert_id(expert_id)
        info = self.inspect_layer(layer_id)
        if info is None:
            return None
        if info.version == EXPERT_CACHE_V2:
            start = time.perf_counter()
            handle = self._get_handle(info.path)
            w1_t = self._materialize_tensor(handle.get_slice(PACKED_W1)[expert_id], device)
            w2_t = self._materialize_tensor(handle.get_slice(PACKED_W2)[expert_id], device)
            w3_t = self._materialize_tensor(handle.get_slice(PACKED_W3)[expert_id], device)
            self._record_profile("expert_cache.v2.expert_slice", start)
            self._validate_expert_tensor("w1_t", w1_t, (self.config.dim, self.config.moe_inter_dim))
            self._validate_expert_tensor("w2_t", w2_t, (self.config.moe_inter_dim, self.config.dim))
            self._validate_expert_tensor("w3_t", w3_t, (self.config.dim, self.config.moe_inter_dim))
            return w1_t.contiguous(), w2_t.contiguous(), w3_t.contiguous()

        prefix = f"expert_{expert_id:03d}"
        names = (
            f"{prefix}.w1_t",
            f"{prefix}.w2_t",
            f"{prefix}.w3_t",
        )
        missing = [name for name in names if name not in info.keys]
        if missing:
            if self._manifest_declares_expert(layer_id, expert_id):
                raise KeyError(
                    f"Expert cache manifest declares layer {layer_id} expert {expert_id}, "
                    f"but {info.path} is missing keys {missing}"
                )
            return None

        start = time.perf_counter()
        handle = self._get_handle(info.path)
        w1_t = self._materialize_tensor(handle.get_tensor(names[0]), device)
        w2_t = self._materialize_tensor(handle.get_tensor(names[1]), device)
        w3_t = self._materialize_tensor(handle.get_tensor(names[2]), device)
        self._record_profile("expert_cache.load", start)

        self._validate_expert_tensor("w1_t", w1_t, (self.config.dim, self.config.moe_inter_dim))
        self._validate_expert_tensor("w2_t", w2_t, (self.config.moe_inter_dim, self.config.dim))
        self._validate_expert_tensor("w3_t", w3_t, (self.config.dim, self.config.moe_inter_dim))
        return w1_t.contiguous(), w2_t.contiguous(), w3_t.contiguous()

    def copy_selected_into(
        self,
        layer_id: int,
        expert_ids: Sequence[int],
        *,
        out_w1: torch.Tensor,
        out_w2: torch.Tensor,
        out_w3: torch.Tensor,
    ) -> bool:
        """Copy selected v2 expert slices into preallocated Host packs.

        Returns False for a missing layer or a v1 file so WeightLoader can
        preserve its existing per-expert fallback path.
        """
        ids = tuple(int(expert_id) for expert_id in expert_ids)
        for expert_id in ids:
            self._validate_expert_id(expert_id)
        info = self.inspect_layer(layer_id)
        if info is None or info.version == EXPERT_CACHE_V1:
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
        self._record_profile("expert_cache.v2.selected_slice_copy", start)
        return True

    def load_packed_clone(
        self,
        layer_id: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Clone all three v2 packed tensors, or return None for v1/missing layers."""
        info = self.inspect_layer(layer_id)
        if info is None or info.version == EXPERT_CACHE_V1:
            return None

        start = time.perf_counter()
        handle = self._get_handle(info.path)
        routed_w1_t = self._materialize_tensor(handle.get_tensor(PACKED_W1), device)
        routed_w2_t = self._materialize_tensor(handle.get_tensor(PACKED_W2), device)
        routed_w3_t = self._materialize_tensor(handle.get_tensor(PACKED_W3), device)
        self._record_profile("expert_cache.v2.packed_clone", start)
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
            return None
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("format") != EXPERT_CACHE_FORMAT:
            raise ValueError(f"Unsupported expert cache format in {path}: {data.get('format')!r}")
        version = int(data.get("version", -1))
        if version not in {EXPERT_CACHE_V1, EXPERT_CACHE_V2}:
            raise ValueError(f"Unsupported expert cache version in {path}: {version}")

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
            raw_experts = raw_entry.get("experts")
            experts = None if raw_experts is None else tuple(int(expert_id) for expert_id in raw_experts)
            if experts is not None:
                for expert_id in experts:
                    self._validate_expert_id(expert_id)
            layers[layer_id] = ExpertCacheLayerManifest(file=raw_entry["file"], experts=experts)

        return ExpertCacheManifest(
            version=version,
            n_layers=int(data["n_layers"]),
            n_routed_experts=int(data["n_routed_experts"]),
            dim=int(data["dim"]),
            moe_inter_dim=int(data["moe_inter_dim"]),
            dtype=str(data["dtype"]),
            layers=layers,
        )

    def _manifest_declares_expert(self, layer_id: int, expert_id: int) -> bool:
        if self._manifest is None:
            return False
        entry = self._manifest.layers.get(layer_id)
        if entry is None or entry.experts is None:
            return self._manifest.version == EXPERT_CACHE_V2
        return expert_id in entry.experts

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
    "EXPERT_CACHE_V1",
    "EXPERT_CACHE_V2",
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
