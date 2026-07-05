"""Checkpoint weight loading and layout conversion for the PyPTO runtime.

This module is intentionally host-only.  It converts DeepSeek V4 Flash
checkpoint tensors into the layouts expected by the kernels under ``models/``:
quantized weights are materialized as BF16, ordinary linear weights are
transposed to ``[in, out]`` except the LM head, and packed routed experts are
stacked into the route-major tensors used by ``models/moe.py``.
"""

from collections import OrderedDict
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open

from models.config import DeepSeekV4FlashConfig, FLASH_CONFIG


FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)

KEY_MAPPING = {
    "embed_tokens": "embed",
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "ffn_norm",
    "q_proj": "wq",
    "q_a_proj": "wq_a",
    "q_a_layernorm": "q_norm",
    "q_b_proj": "wq_b",
    "kv_a_proj_with_mqa": "wkv",
    "kv_a_layernorm": "kv_norm",
    "kv_b_proj": "wkv_b",
    "o_proj": "wo",
    "gate_proj": "w1",
    "down_proj": "w2",
    "up_proj": "w3",
    "lm_head": "head",
    "embed": "embed",
    "wq_b": "wq_b",
    "wo_a": "wo_a",
    "wo_b": "wo_b",
    "head": "head",
    "attn_sink": "attn_sink",
    "weights_proj": "weights_proj",
}

HC_HEAD_PAD = 16


@dataclass(frozen=True)
class HeadWeights:
    hc_fn_t: torch.Tensor
    hc_scale: torch.Tensor
    hc_base: torch.Tensor
    norm_w: torch.Tensor
    head_w: torch.Tensor


@dataclass(frozen=True)
class LayerHCWeights:
    attn_hc_fn_t: torch.Tensor
    attn_hc_scale: torch.Tensor
    attn_hc_base: torch.Tensor
    ffn_hc_fn_t: torch.Tensor
    ffn_hc_scale: torch.Tensor
    ffn_hc_base: torch.Tensor


@dataclass(frozen=True)
class LayerAttentionWeights:
    attn_norm_w: torch.Tensor
    wq_a_t: torch.Tensor
    q_norm_w: torch.Tensor
    wq_b_t: torch.Tensor
    wkv_t: torch.Tensor
    kv_norm_w: torch.Tensor
    attn_sink: torch.Tensor
    wo_a_t: torch.Tensor
    wo_b_t: torch.Tensor


@dataclass(frozen=True)
class CompressorWeights:
    wkv_t: torch.Tensor
    wgate_t: torch.Tensor
    ape: torch.Tensor
    norm_w: torch.Tensor


@dataclass(frozen=True)
class IndexerWeights:
    idx_wq_b_t: torch.Tensor
    idx_weights_proj_t: torch.Tensor
    idx_comp_wkv_t: torch.Tensor
    idx_comp_wgate_t: torch.Tensor
    idx_comp_ape: torch.Tensor
    idx_comp_norm_w: torch.Tensor


@dataclass(frozen=True)
class MoEGateWeights:
    gate_w_t: torch.Tensor
    tid2eid: torch.Tensor | None = None
    gate_bias: torch.Tensor | None = None


@dataclass(frozen=True)
class MoESharedWeights:
    shared_w1_t: torch.Tensor
    shared_w2_t: torch.Tensor
    shared_w3_t: torch.Tensor


@dataclass(frozen=True)
class MoERoutedExpertWeights:
    w1_t: torch.Tensor
    w2_t: torch.Tensor
    w3_t: torch.Tensor


@dataclass(frozen=True)
class MoERoutedPackWeights:
    routed_w1_t: torch.Tensor
    routed_w2_t: torch.Tensor
    routed_w3_t: torch.Tensor


def normalize_param_name(name: str) -> str:
    """Map HF checkpoint parameter names to inference-side names."""
    if name.startswith("model."):
        name = name[len("model.") :]
    name = name.replace("self_attn", "attn")
    name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")

    parts = name.split(".")
    if len(parts) < 2:
        return name
    if any(x in name for x in ("hc", "attn_sink", "tid2eid", "tie2eid", "ape")):
        key_idx = len(parts) - 1
    else:
        key_idx = len(parts) - 2
    parts[key_idx] = KEY_MAPPING.get(parts[key_idx], parts[key_idx])
    return ".".join(parts)


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def dequant_fp8_weight_to_bf16(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize per-128-block FP8 weights to BF16."""
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"Expected float8_e4m3fn weight, got {weight.dtype}")
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("FP8 weight and scale must both be 2D tensors")
    out_dim, in_dim = weight.shape
    if out_dim % 128 != 0 or in_dim % 128 != 0:
        raise ValueError(f"FP8 weight shape must be divisible by 128, got {tuple(weight.shape)}")
    expected_scale = (out_dim // 128, in_dim // 128)
    if tuple(scale.shape) != expected_scale:
        raise ValueError(f"Expected FP8 scale shape {expected_scale}, got {tuple(scale.shape)}")
    x = weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float()
    x = x * scale.float()[:, None, :, None]
    return x.flatten(2, 3).flatten(0, 1).bfloat16()


def dequant_fp4_weight_to_bf16(packed_i8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize packed E2M1 FP4 expert weights to BF16."""
    if packed_i8.dtype != torch.int8:
        raise TypeError(f"Expected int8 packed FP4 weight, got {packed_i8.dtype}")
    if packed_i8.ndim != 2 or scale.ndim != 2:
        raise ValueError("FP4 weight and scale must both be 2D tensors")
    out_dim, packed_in_dim = packed_i8.shape
    in_dim = packed_in_dim * 2
    if in_dim % 32 != 0:
        raise ValueError(f"Unpacked FP4 input dim must be divisible by 32, got {in_dim}")
    expected_scale = (out_dim, in_dim // 32)
    if tuple(scale.shape) != expected_scale:
        raise ValueError(f"Expected FP4 scale shape {expected_scale}, got {tuple(scale.shape)}")

    table = FP4_TABLE.to(device=packed_i8.device)
    x = packed_i8.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    unpacked = torch.stack([table[low.long()], table[high.long()]], dim=-1).flatten(1)
    unpacked = unpacked.float() * scale.float().repeat_interleave(32, dim=1)
    return unpacked.bfloat16()


class DeepSeekV4WeightLoader:
    """Lazy checkpoint loader for the host-side layer-by-layer runtime."""

    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str],
        weight_index: str | os.PathLike[str] | dict[str, Any] | None = None,
        *,
        config: DeepSeekV4FlashConfig = FLASH_CONFIG,
        default_device: str | torch.device = "cpu",
        max_cache_bytes: int = 0,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.index = self._load_index(weight_index)
        self.config = config
        self.default_device = torch.device(default_device)
        self.max_cache_bytes = max_cache_bytes
        self._cache: OrderedDict[tuple[str, str, bool, str | None], torch.Tensor] = OrderedDict()
        self._cache_bytes = 0

    @property
    def cache_bytes(self) -> int:
        return self._cache_bytes

    def has_tensor(self, name: str) -> bool:
        return name in self.index

    def entry(self, name: str) -> dict[str, Any]:
        try:
            return self.index[name]
        except KeyError as exc:
            raise KeyError(f"Unknown checkpoint tensor: {name}") from exc

    def get_tensor(
        self,
        name: str,
        *,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
        dequantize: bool = True,
        cache: bool = True,
    ) -> torch.Tensor:
        target = torch.device(device) if device is not None else self.default_device
        key = (name, str(target), dequantize, str(dtype) if dtype is not None else None)
        if key in self._cache:
            tensor = self._cache.pop(key)
            self._cache[key] = tensor
            return tensor

        tensor = self._load_indexed_tensor(name, device=target, dequantize=dequantize)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        tensor = tensor.contiguous()
        if cache:
            self._insert_cache(key, tensor)
        return tensor

    def get_linear_weight(
        self,
        name: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
        cache: bool = True,
    ) -> torch.Tensor:
        tensor = self.get_tensor(name, dtype=dtype, device=device, dequantize=True, cache=cache)
        if not tensor.is_floating_point():
            raise TypeError(f"Expected floating point linear weight for {name}, got {tensor.dtype}")
        return tensor

    def get_linear_t(
        self,
        name: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
        cache: bool = True,
    ) -> torch.Tensor:
        return self.get_linear_weight(name, dtype=dtype, device=device, cache=cache).t().contiguous()

    def get_embedding_weight(self, *, device: str | torch.device | None = None) -> torch.Tensor:
        return self.get_tensor("embed.weight", dtype=torch.bfloat16, device=device)

    def get_head_weights(self, *, device: str | torch.device | None = None) -> HeadWeights:
        hc_fn = self.get_tensor("hc_head_fn", dtype=torch.float32, device=device)
        if hc_fn.ndim != 2 or hc_fn.shape[0] != self.config.hc_mult:
            raise ValueError(f"Expected hc_head_fn shape [{self.config.hc_mult}, HC_DIM], got {tuple(hc_fn.shape)}")
        if hc_fn.shape[0] > HC_HEAD_PAD:
            raise ValueError(f"hc_mult={hc_fn.shape[0]} exceeds HC head pad width {HC_HEAD_PAD}")

        hc_fn_t = torch.zeros(hc_fn.shape[1], HC_HEAD_PAD, dtype=torch.float32, device=hc_fn.device)
        hc_fn_t[:, : hc_fn.shape[0]] = hc_fn.t().contiguous()

        hc_base_raw = self.get_tensor("hc_head_base", dtype=torch.float32, device=device)
        hc_base = torch.zeros(HC_HEAD_PAD, dtype=torch.float32, device=hc_base_raw.device)
        hc_base[: hc_base_raw.numel()] = hc_base_raw.reshape(-1)

        return HeadWeights(
            hc_fn_t=hc_fn_t.contiguous(),
            hc_scale=self.get_tensor("hc_head_scale", dtype=torch.float32, device=device),
            hc_base=hc_base.contiguous(),
            norm_w=self.get_tensor("norm.weight", dtype=torch.bfloat16, device=device),
            head_w=self.get_tensor("head.weight", dtype=torch.float32, device=device),
        )

    def get_layer_hc(self, layer_id: int, *, device: str | torch.device | None = None) -> LayerHCWeights:
        prefix = self._layer_prefix(layer_id)
        return LayerHCWeights(
            attn_hc_fn_t=self.get_tensor(f"{prefix}.hc_attn_fn", dtype=torch.float32, device=device).t().contiguous(),
            attn_hc_scale=self.get_tensor(f"{prefix}.hc_attn_scale", dtype=torch.float32, device=device),
            attn_hc_base=self.get_tensor(f"{prefix}.hc_attn_base", dtype=torch.float32, device=device),
            ffn_hc_fn_t=self.get_tensor(f"{prefix}.hc_ffn_fn", dtype=torch.float32, device=device).t().contiguous(),
            ffn_hc_scale=self.get_tensor(f"{prefix}.hc_ffn_scale", dtype=torch.float32, device=device),
            ffn_hc_base=self.get_tensor(f"{prefix}.hc_ffn_base", dtype=torch.float32, device=device),
        )

    def get_layer_attention_common(
        self,
        layer_id: int,
        *,
        device: str | torch.device | None = None,
    ) -> LayerAttentionWeights:
        prefix = f"{self._layer_prefix(layer_id)}.attn"
        return LayerAttentionWeights(
            attn_norm_w=self.get_tensor(f"{self._layer_prefix(layer_id)}.attn_norm.weight", dtype=torch.bfloat16, device=device),
            wq_a_t=self.get_linear_t(f"{prefix}.wq_a.weight", device=device),
            q_norm_w=self.get_tensor(f"{prefix}.q_norm.weight", dtype=torch.bfloat16, device=device),
            wq_b_t=self.get_linear_t(f"{prefix}.wq_b.weight", device=device),
            wkv_t=self.get_linear_t(f"{prefix}.wkv.weight", device=device),
            kv_norm_w=self.get_tensor(f"{prefix}.kv_norm.weight", dtype=torch.bfloat16, device=device),
            attn_sink=self.get_tensor(f"{prefix}.attn_sink", dtype=torch.float32, device=device),
            wo_a_t=self.get_linear_t(f"{prefix}.wo_a.weight", device=device),
            wo_b_t=self.get_linear_t(f"{prefix}.wo_b.weight", device=device),
        )

    def get_layer_compressor_ratio128(
        self,
        layer_id: int,
        *,
        device: str | torch.device | None = None,
    ) -> CompressorWeights:
        return self._get_compressor(f"{self._layer_prefix(layer_id)}.attn.compressor", device=device)

    def get_layer_compressor_ratio4_attention(
        self,
        layer_id: int,
        *,
        device: str | torch.device | None = None,
    ) -> CompressorWeights:
        return self._get_compressor(f"{self._layer_prefix(layer_id)}.attn.compressor", device=device)

    def get_layer_indexer(self, layer_id: int, *, device: str | torch.device | None = None) -> IndexerWeights:
        prefix = f"{self._layer_prefix(layer_id)}.attn.indexer"
        comp_prefix = f"{prefix}.compressor"
        return IndexerWeights(
            idx_wq_b_t=self.get_linear_t(f"{prefix}.wq_b.weight", device=device),
            idx_weights_proj_t=self.get_linear_t(f"{prefix}.weights_proj.weight", device=device),
            idx_comp_wkv_t=self.get_linear_t(f"{comp_prefix}.wkv.weight", device=device),
            idx_comp_wgate_t=self.get_linear_t(f"{comp_prefix}.wgate.weight", device=device),
            idx_comp_ape=self.get_tensor(f"{comp_prefix}.ape", dtype=torch.float32, device=device),
            idx_comp_norm_w=self.get_tensor(f"{comp_prefix}.norm.weight", dtype=torch.bfloat16, device=device),
        )

    def get_layer_moe_gate(
        self,
        layer_id: int,
        *,
        hash_route: bool,
        device: str | torch.device | None = None,
    ) -> MoEGateWeights:
        prefix = f"{self._layer_prefix(layer_id)}.ffn.gate"
        gate_w_t = self.get_linear_t(f"{prefix}.weight", device=device)
        if hash_route:
            tid_name = f"{prefix}.tid2eid"
            if not self.has_tensor(tid_name):
                tid_name = f"{prefix}.tie2eid"
            return MoEGateWeights(
                gate_w_t=gate_w_t,
                tid2eid=self.get_tensor(tid_name, dtype=torch.int32, device=device),
            )
        return MoEGateWeights(
            gate_w_t=gate_w_t,
            gate_bias=self.get_tensor(f"{prefix}.bias", dtype=torch.float32, device=device),
        )

    def get_layer_moe_shared(self, layer_id: int, *, device: str | torch.device | None = None) -> MoESharedWeights:
        prefix = f"{self._layer_prefix(layer_id)}.ffn.shared_experts"
        return MoESharedWeights(
            shared_w1_t=self.get_linear_t(f"{prefix}.w1.weight", device=device),
            shared_w2_t=self.get_linear_t(f"{prefix}.w2.weight", device=device),
            shared_w3_t=self.get_linear_t(f"{prefix}.w3.weight", device=device),
        )

    def get_moe_routed_expert(
        self,
        layer_id: int,
        expert_id: int,
        *,
        device: str | torch.device | None = None,
    ) -> MoERoutedExpertWeights:
        if not 0 <= expert_id < self.config.n_routed_experts:
            raise ValueError(f"expert_id must be in [0, {self.config.n_routed_experts}), got {expert_id}")
        prefix = f"{self._layer_prefix(layer_id)}.ffn.experts.{expert_id}"
        return MoERoutedExpertWeights(
            w1_t=self.get_linear_t(f"{prefix}.w1.weight", device=device),
            w2_t=self.get_linear_t(f"{prefix}.w2.weight", device=device),
            w3_t=self.get_linear_t(f"{prefix}.w3.weight", device=device),
        )

    def get_layer_moe_routed_pack(
        self,
        layer_id: int,
        *,
        device: str | torch.device | None = None,
        release_each_expert: bool = False,
    ) -> MoERoutedPackWeights:
        w1_tensors = []
        w2_tensors = []
        w3_tensors = []
        for expert_id in range(self.config.n_routed_experts):
            expert = self.get_moe_routed_expert(layer_id, expert_id, device=device)
            w1_tensors.append(expert.w1_t)
            w2_tensors.append(expert.w2_t)
            w3_tensors.append(expert.w3_t)
            if release_each_expert:
                self.release_prefix(f"{self._layer_prefix(layer_id)}.ffn.experts.{expert_id}.")
        return MoERoutedPackWeights(
            routed_w1_t=torch.stack(w1_tensors, dim=0).contiguous(),
            routed_w2_t=torch.stack(w2_tensors, dim=0).contiguous(),
            routed_w3_t=torch.stack(w3_tensors, dim=0).contiguous(),
        )

    def release(self, name: str | None = None) -> None:
        if name is None:
            self._cache.clear()
            self._cache_bytes = 0
            return

        removed = [key for key in self._cache if key[0] == name]
        for key in removed:
            tensor = self._cache.pop(key)
            self._cache_bytes -= tensor_nbytes(tensor)

    def release_prefix(self, prefix: str) -> None:
        removed = [key for key in self._cache if key[0].startswith(prefix)]
        for key in removed:
            tensor = self._cache.pop(key)
            self._cache_bytes -= tensor_nbytes(tensor)

    def _get_compressor(self, prefix: str, *, device: str | torch.device | None = None) -> CompressorWeights:
        return CompressorWeights(
            wkv_t=self.get_linear_t(f"{prefix}.wkv.weight", device=device),
            wgate_t=self.get_linear_t(f"{prefix}.wgate.weight", device=device),
            ape=self.get_tensor(f"{prefix}.ape", dtype=torch.float32, device=device),
            norm_w=self.get_tensor(f"{prefix}.norm.weight", dtype=torch.bfloat16, device=device),
        )

    def _layer_prefix(self, layer_id: int) -> str:
        if not 0 <= layer_id < self.config.n_layers:
            raise ValueError(f"layer_id must be in [0, {self.config.n_layers}), got {layer_id}")
        return f"layers.{layer_id}"

    def _load_index(self, weight_index: str | os.PathLike[str] | dict[str, Any] | None) -> dict[str, dict[str, Any]]:
        if weight_index is None:
            candidates = [self.checkpoint_path / "weight_index.json", self.checkpoint_path / "model.safetensors.index.json"]
            for candidate in candidates:
                if candidate.exists():
                    weight_index = candidate
                    break
            if weight_index is None:
                raise FileNotFoundError(f"No weight index found under {self.checkpoint_path}")

        if isinstance(weight_index, (str, os.PathLike)):
            with open(weight_index, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        else:
            data = weight_index

        if "weight_map" in data:
            return self._normalize_weight_map(data["weight_map"])
        return self._normalize_full_index(data)

    @staticmethod
    def _normalize_weight_map(weight_map: dict[str, str]) -> dict[str, dict[str, Any]]:
        entries: dict[str, dict[str, Any]] = {}
        for raw_name, rel_file in weight_map.items():
            if raw_name.startswith("mtp.0.") or raw_name.startswith("model.mtp.0."):
                continue
            name = normalize_param_name(raw_name)
            entries[name] = {
                "file": rel_file,
                "raw_name": raw_name,
                "kind": "unknown",
            }
        return entries

    @staticmethod
    def _normalize_full_index(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
        entries: dict[str, dict[str, Any]] = {}
        for key, raw_entry in data.items():
            if not isinstance(raw_entry, dict):
                continue
            raw_name = raw_entry.get("raw_name", key)
            if raw_name.startswith("mtp.0.") or raw_name.startswith("model.mtp.0."):
                continue
            name = normalize_param_name(key)
            entry = dict(raw_entry)
            entry.setdefault("raw_name", raw_name)
            if "file" not in entry:
                raise KeyError(f"Missing file for weight index entry {key}")
            entry.setdefault("kind", "unknown")
            entries[name] = entry
        return entries

    def _load_indexed_tensor(
        self,
        name: str,
        *,
        device: torch.device,
        dequantize: bool,
    ) -> torch.Tensor:
        entry = self.entry(name)
        tensor = self._load_raw_tensor(entry)
        kind = self._resolve_kind(name, entry, tensor)
        if not dequantize or kind not in {"fp8_weight", "fp4_packed_weight"}:
            return tensor.to(device)

        scale_name = entry.get("scale") or name.replace(".weight", ".scale")
        if scale_name in self.index:
            scale_entry = self.entry(scale_name)
        elif "scale_file" in entry and "scale_raw_name" in entry:
            scale_entry = {"file": entry["scale_file"], "raw_name": entry["scale_raw_name"]}
        else:
            raise KeyError(f"Missing scale tensor for quantized weight: {name}")
        scale = self._load_raw_tensor(scale_entry)
        if kind == "fp8_weight":
            return dequant_fp8_weight_to_bf16(tensor, scale).to(device)
        return dequant_fp4_weight_to_bf16(tensor, scale).to(device)

    def _load_raw_tensor(self, entry: dict[str, Any]) -> torch.Tensor:
        path = self.checkpoint_path / entry["file"]
        with safe_open(path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(entry["raw_name"])

    @staticmethod
    def _resolve_kind(name: str, entry: dict[str, Any], tensor: torch.Tensor) -> str:
        kind = entry.get("kind", "unknown")
        if kind != "unknown":
            return kind
        if name.endswith(".weight") and tensor.dtype == torch.float8_e4m3fn:
            return "fp8_weight"
        if name.endswith(".weight") and tensor.dtype == torch.int8:
            return "fp4_packed_weight"
        if name.endswith(".scale"):
            return "scale"
        if tensor.is_floating_point():
            return "plain_tensor"
        if tensor.dtype in {torch.int32, torch.int64}:
            return "integer_tensor"
        return "unknown"

    def _insert_cache(self, key: tuple[str, str, bool, str | None], tensor: torch.Tensor) -> None:
        if key in self._cache:
            old = self._cache.pop(key)
            self._cache_bytes -= tensor_nbytes(old)
        self._cache[key] = tensor
        self._cache_bytes += tensor_nbytes(tensor)
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        if self.max_cache_bytes <= 0:
            return
        while self._cache and self._cache_bytes > self.max_cache_bytes:
            _, tensor = self._cache.popitem(last=False)
            self._cache_bytes -= tensor_nbytes(tensor)


def load_weight_loader_from_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    default_device: str | torch.device = "cpu",
    max_cache_bytes: int = 0,
) -> DeepSeekV4WeightLoader:
    return DeepSeekV4WeightLoader(
        checkpoint_path,
        default_device=default_device,
        max_cache_bytes=max_cache_bytes,
    )


__all__ = [
    "CompressorWeights",
    "DeepSeekV4WeightLoader",
    "HeadWeights",
    "IndexerWeights",
    "LayerAttentionWeights",
    "LayerHCWeights",
    "MoEGateWeights",
    "MoERoutedExpertWeights",
    "MoERoutedPackWeights",
    "MoESharedWeights",
    "dequant_fp4_weight_to_bf16",
    "dequant_fp8_weight_to_bf16",
    "load_weight_loader_from_checkpoint",
    "normalize_param_name",
]
