"""Checkpoint weight loading and layout conversion for the PyPTO runtime.

This module is intentionally host-only.  It converts DeepSeek V4 Flash
checkpoint tensors into the layouts expected by the kernels under ``models/``:
quantized weights are materialized as BF16, ordinary linear weights are
transposed to ``[in, out]`` except the LM head, and routed expert weights are
materialized as the layouts expected by ``models/moe.py`` and
``models/split_block.py``.
"""

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

import torch
from safetensors.torch import safe_open

from models.config import DeepSeekV4FlashConfig, FLASH_CONFIG
from serving.expert_cache import ExpertCacheReader
from serving.runtime_types import HostStagingTensor, RuntimeWeight, RuntimeWeightKey, StagingKind


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
    hc_fn_t: RuntimeWeight
    hc_scale: RuntimeWeight
    hc_base: RuntimeWeight
    norm_w: RuntimeWeight
    head_w: RuntimeWeight


@dataclass(frozen=True)
class LayerHCWeights:
    attn_hc_fn_t: RuntimeWeight
    attn_hc_scale: RuntimeWeight
    attn_hc_base: RuntimeWeight
    ffn_hc_fn_t: RuntimeWeight
    ffn_hc_scale: RuntimeWeight
    ffn_hc_base: RuntimeWeight


@dataclass(frozen=True)
class LayerAttentionWeights:
    attn_norm_w: RuntimeWeight
    wq_a_t: RuntimeWeight
    q_norm_w: RuntimeWeight
    wq_b_t: RuntimeWeight
    wkv_t: RuntimeWeight
    kv_norm_w: RuntimeWeight
    attn_sink: RuntimeWeight
    wo_a_t: RuntimeWeight
    wo_b_t: RuntimeWeight


@dataclass(frozen=True)
class CompressorWeights:
    wkv_t: RuntimeWeight
    wgate_t: RuntimeWeight
    ape: RuntimeWeight
    norm_w: RuntimeWeight


@dataclass(frozen=True)
class IndexerWeights:
    idx_wq_b_t: RuntimeWeight
    idx_weights_proj_t: RuntimeWeight
    idx_comp_wkv_t: RuntimeWeight
    idx_comp_wgate_t: RuntimeWeight
    idx_comp_ape: RuntimeWeight
    idx_comp_norm_w: RuntimeWeight


@dataclass(frozen=True)
class MoEGateWeights:
    gate_w_t: RuntimeWeight
    tid2eid: RuntimeWeight | None = None
    gate_bias: RuntimeWeight | None = None


@dataclass(frozen=True)
class MoESharedWeights:
    shared_w1_t: RuntimeWeight
    shared_w2_t: RuntimeWeight
    shared_w3_t: RuntimeWeight


@dataclass(frozen=True)
class MoERoutedExpertWeights:
    w1_t: torch.Tensor
    w2_t: torch.Tensor
    w3_t: torch.Tensor


@dataclass(frozen=True)
class MoERoutedPackWeights:
    routed_w1_t: HostStagingTensor
    routed_w2_t: HostStagingTensor
    routed_w3_t: HostStagingTensor


@dataclass(frozen=True)
class MoESelectedExpertWeights:
    selected_w1_t: HostStagingTensor
    selected_w2_t: HostStagingTensor
    selected_w3_t: HostStagingTensor


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
        profile: bool = False,
        expert_cache_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.index = self._load_index(weight_index)
        self.config = config
        self.default_device = torch.device(default_device)
        self.profile = bool(profile)
        self.expert_cache_dir = Path(expert_cache_dir) if expert_cache_dir is not None else None
        self._layout_cache: dict[tuple[RuntimeWeightKey, str], RuntimeWeight] = {}
        self._layout_cache_bytes = 0
        self._file_handles: dict[Path, Any] = {}
        self._profile_stats: OrderedDict[str, dict[str, float | int]] = OrderedDict()
        self._expert_cache = ExpertCacheReader(
            self.expert_cache_dir,
            config=self.config,
            profile_callback=self._record_profile,
        )

    @property
    def layout_cache_bytes(self) -> int:
        return self._layout_cache_bytes

    def has_tensor(self, name: str) -> bool:
        return name in self.index

    def entry(self, name: str) -> dict[str, Any]:
        try:
            return self.index[name]
        except KeyError as exc:
            raise KeyError(f"Unknown checkpoint tensor: {name}") from exc

    def _load_tensor(
        self,
        name: str,
        *,
        dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
        dequantize: bool = True,
    ) -> torch.Tensor:
        target = torch.device(device) if device is not None else self.default_device
        tensor = self._load_indexed_tensor(name, device=target, dequantize=dequantize)
        if dtype is not None:
            start = time.perf_counter()
            tensor = tensor.to(dtype=dtype)
            self._record_profile(f"dtype_cast.{dtype}", start)
        return tensor.contiguous()

    def _get_runtime_weight(
        self,
        name: str,
        *,
        dtype: torch.dtype,
        layout: str = "identity",
        device: str | torch.device | None = None,
        cache: bool = True,
        layout_version: int = 1,
        padding_profile: str | None = None,
        build: Callable[[], torch.Tensor] | None = None,
    ) -> RuntimeWeight:
        """Return a checkpoint tensor descriptor in its final kernel-facing layout.

        Identity layouts are loaded directly from the checkpoint.  Non-identity
        layouts must provide a builder so a raw tensor cannot be cached under a
        transformed-layout key by mistake.
        """
        target = torch.device(device) if device is not None else self.default_device
        if build is None and layout != "identity":
            raise ValueError(f"Runtime layout {layout!r} for {name} requires an explicit builder")
        key = RuntimeWeightKey(
            name,
            dtype,
            layout,
            layout_version=layout_version,
            padding_profile=padding_profile,
        )
        cache_key = (key, str(target))
        lookup_start = time.perf_counter()
        if cache and cache_key in self._layout_cache:
            self._record_profile("cache.layout.hit", lookup_start)
            return self._layout_cache[cache_key]
        self._record_profile("cache.layout.miss", lookup_start)

        if build is None:
            build = lambda: self._load_tensor(
                name,
                dtype=dtype,
                device=target,
                dequantize=True,
            )
        tensor = build()
        if tensor.dtype != key.dtype:
            raise TypeError(
                f"Runtime layout {key.layout!r} for {key.name} expected dtype {key.dtype}, got {tensor.dtype}"
            )
        if tensor.device != target:
            raise ValueError(
                f"Runtime layout {key.layout!r} for {key.name} expected device {target}, got {tensor.device}"
            )
        weight = RuntimeWeight(key=key, host_tensor=tensor.contiguous())
        if cache:
            self._insert_layout_cache(cache_key, weight)
        return weight

    def _load_linear_weight(
        self,
        name: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        tensor = self._load_tensor(name, dtype=dtype, device=device, dequantize=True)
        if not tensor.is_floating_point():
            raise TypeError(f"Expected floating point linear weight for {name}, got {tensor.dtype}")
        return tensor

    def get_embedding_weight(self, *, device: str | torch.device | None = None) -> RuntimeWeight:
        return self._get_runtime_weight("embed.weight", dtype=torch.bfloat16, device=device)

    def get_head_weights(self, *, device: str | torch.device | None = None) -> HeadWeights:
        target = torch.device(device) if device is not None else self.default_device
        padding_profile = f"width={HC_HEAD_PAD}"
        hc_fn_t = self._get_runtime_weight(
            "hc_head_fn",
            dtype=torch.float32,
            layout="hc_head_padded_t",
            padding_profile=padding_profile,
            build=lambda: self._build_head_hc_fn_t(device=target),
            device=target,
        )
        hc_base = self._get_runtime_weight(
            "hc_head_base",
            dtype=torch.float32,
            layout="hc_head_base_padded",
            padding_profile=padding_profile,
            build=lambda: self._build_head_hc_base(device=target),
            device=target,
        )

        return HeadWeights(
            hc_fn_t=hc_fn_t,
            hc_scale=self._get_runtime_weight("hc_head_scale", dtype=torch.float32, device=device),
            hc_base=hc_base,
            norm_w=self._get_runtime_weight("norm.weight", dtype=torch.bfloat16, device=device),
            head_w=self._get_runtime_weight("head.weight", dtype=torch.float32, device=device),
        )

    def get_layer_hc(self, layer_id: int, *, device: str | torch.device | None = None) -> LayerHCWeights:
        prefix = self._layer_prefix(layer_id)
        return LayerHCWeights(
            attn_hc_fn_t=self._get_transposed_weight(
                f"{prefix}.hc_attn_fn",
                dtype=torch.float32,
                device=device,
                layout="hc_t",
            ),
            attn_hc_scale=self._get_runtime_weight(f"{prefix}.hc_attn_scale", dtype=torch.float32, device=device),
            attn_hc_base=self._get_runtime_weight(f"{prefix}.hc_attn_base", dtype=torch.float32, device=device),
            ffn_hc_fn_t=self._get_transposed_weight(
                f"{prefix}.hc_ffn_fn",
                dtype=torch.float32,
                device=device,
                layout="hc_t",
            ),
            ffn_hc_scale=self._get_runtime_weight(f"{prefix}.hc_ffn_scale", dtype=torch.float32, device=device),
            ffn_hc_base=self._get_runtime_weight(f"{prefix}.hc_ffn_base", dtype=torch.float32, device=device),
        )

    def get_layer_ffn_norm(self, layer_id: int, *, device: str | torch.device | None = None) -> RuntimeWeight:
        return self._get_runtime_weight(
            f"{self._layer_prefix(layer_id)}.ffn_norm.weight",
            dtype=torch.bfloat16,
            device=device,
        )

    def get_layer_attention_common(
        self,
        layer_id: int,
        *,
        device: str | torch.device | None = None,
    ) -> LayerAttentionWeights:
        prefix = f"{self._layer_prefix(layer_id)}.attn"
        return LayerAttentionWeights(
            attn_norm_w=self._get_runtime_weight(
                f"{self._layer_prefix(layer_id)}.attn_norm.weight",
                dtype=torch.bfloat16,
                device=device,
            ),
            wq_a_t=self._get_transposed_weight(f"{prefix}.wq_a.weight", device=device),
            q_norm_w=self._get_runtime_weight(f"{prefix}.q_norm.weight", dtype=torch.bfloat16, device=device),
            wq_b_t=self._get_transposed_weight(f"{prefix}.wq_b.weight", device=device),
            wkv_t=self._get_transposed_weight(f"{prefix}.wkv.weight", device=device),
            kv_norm_w=self._get_runtime_weight(f"{prefix}.kv_norm.weight", dtype=torch.bfloat16, device=device),
            attn_sink=self._get_runtime_weight(f"{prefix}.attn_sink", dtype=torch.float32, device=device),
            wo_a_t=self._get_transposed_weight(f"{prefix}.wo_a.weight", device=device),
            wo_b_t=self._get_transposed_weight(f"{prefix}.wo_b.weight", device=device),
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
            idx_wq_b_t=self._get_transposed_weight(f"{prefix}.wq_b.weight", device=device),
            idx_weights_proj_t=self._get_transposed_weight(f"{prefix}.weights_proj.weight", device=device),
            idx_comp_wkv_t=self._get_transposed_weight(f"{comp_prefix}.wkv.weight", device=device),
            idx_comp_wgate_t=self._get_transposed_weight(f"{comp_prefix}.wgate.weight", device=device),
            idx_comp_ape=self._get_runtime_weight(f"{comp_prefix}.ape", dtype=torch.float32, device=device),
            idx_comp_norm_w=self._get_runtime_weight(
                f"{comp_prefix}.norm.weight",
                dtype=torch.bfloat16,
                device=device,
            ),
        )

    def get_layer_moe_gate(
        self,
        layer_id: int,
        *,
        hash_route: bool,
        device: str | torch.device | None = None,
    ) -> MoEGateWeights:
        prefix = f"{self._layer_prefix(layer_id)}.ffn.gate"
        gate_w_t = self._get_transposed_weight(f"{prefix}.weight", device=device)
        if hash_route:
            tid_name = f"{prefix}.tid2eid"
            if not self.has_tensor(tid_name):
                tid_name = f"{prefix}.tie2eid"
            return MoEGateWeights(
                gate_w_t=gate_w_t,
                tid2eid=self._get_runtime_weight(tid_name, dtype=torch.int32, device=device),
            )
        return MoEGateWeights(
            gate_w_t=gate_w_t,
            gate_bias=self._get_runtime_weight(f"{prefix}.bias", dtype=torch.float32, device=device),
        )

    def get_layer_moe_shared(self, layer_id: int, *, device: str | torch.device | None = None) -> MoESharedWeights:
        prefix = f"{self._layer_prefix(layer_id)}.ffn.shared_experts"
        return MoESharedWeights(
            shared_w1_t=self._get_transposed_weight(f"{prefix}.w1.weight", device=device),
            shared_w2_t=self._get_transposed_weight(f"{prefix}.w2.weight", device=device),
            shared_w3_t=self._get_transposed_weight(f"{prefix}.w3.weight", device=device),
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
        target = torch.device(device) if device is not None else self.default_device
        prefix = f"{self._layer_prefix(layer_id)}.ffn.experts.{expert_id}"
        return MoERoutedExpertWeights(
            w1_t=self._get_transposed_weight(f"{prefix}.w1.weight", device=target, cache=False).host_tensor,
            w2_t=self._get_transposed_weight(f"{prefix}.w2.weight", device=target, cache=False).host_tensor,
            w3_t=self._get_transposed_weight(f"{prefix}.w3.weight", device=target, cache=False).host_tensor,
        )

    def get_layer_moe_selected_experts(
        self,
        layer_id: int,
        expert_ids: torch.Tensor | list[int] | tuple[int, ...],
        *,
        device: str | torch.device | None = None,
    ) -> MoESelectedExpertWeights:
        target = torch.device(device) if device is not None else self.default_device
        ids = self._normalize_selected_expert_ids(expert_ids)
        selected_w1_t = torch.empty(
            self.config.n_activated_experts,
            self.config.dim,
            self.config.moe_inter_dim,
            dtype=torch.bfloat16,
            device=target,
        )
        selected_w2_t = torch.empty(
            self.config.n_activated_experts,
            self.config.moe_inter_dim,
            self.config.dim,
            dtype=torch.bfloat16,
            device=target,
        )
        selected_w3_t = torch.empty(
            self.config.n_activated_experts,
            self.config.dim,
            self.config.moe_inter_dim,
            dtype=torch.bfloat16,
            device=target,
        )

        start = time.perf_counter()
        cache_hit = self._expert_cache.copy_selected_into(
            layer_id,
            ids,
            out_w1=selected_w1_t,
            out_w2=selected_w2_t,
            out_w3=selected_w3_t,
        )
        if not cache_hit:
            for slot, expert_id in enumerate(ids):
                expert = self.get_moe_routed_expert(layer_id, expert_id, device=target)
                selected_w1_t[slot].copy_(expert.w1_t)
                selected_w2_t[slot].copy_(expert.w2_t)
                selected_w3_t[slot].copy_(expert.w3_t)
        self._record_profile("selected_experts.build", start)
        return MoESelectedExpertWeights(
            selected_w1_t=HostStagingTensor(
                selected_w1_t.contiguous(), StagingKind.DECODE_SELECTED, "w1_t"
            ),
            selected_w2_t=HostStagingTensor(
                selected_w2_t.contiguous(), StagingKind.DECODE_SELECTED, "w2_t"
            ),
            selected_w3_t=HostStagingTensor(
                selected_w3_t.contiguous(), StagingKind.DECODE_SELECTED, "w3_t"
            ),
        )

    def get_layer_moe_routed_pack(
        self,
        layer_id: int,
        *,
        device: str | torch.device | None = None,
        release_each_expert: bool = False,
    ) -> MoERoutedPackWeights:
        del release_each_expert
        target = torch.device(device) if device is not None else self.default_device

        cached_pack = self._expert_cache.load_routed_pack(layer_id, device=target)
        if cached_pack is not None:
            routed_w1_t, routed_w2_t, routed_w3_t = cached_pack
            return MoERoutedPackWeights(
                routed_w1_t=HostStagingTensor(routed_w1_t, StagingKind.PREFILL_ROUTED, "w1_t"),
                routed_w2_t=HostStagingTensor(routed_w2_t, StagingKind.PREFILL_ROUTED, "w2_t"),
                routed_w3_t=HostStagingTensor(routed_w3_t, StagingKind.PREFILL_ROUTED, "w3_t"),
            )

        routed_w1_t = torch.empty(
            self.config.n_routed_experts,
            self.config.dim,
            self.config.moe_inter_dim,
            dtype=torch.bfloat16,
            device=target,
        )
        routed_w2_t = torch.empty(
            self.config.n_routed_experts,
            self.config.moe_inter_dim,
            self.config.dim,
            dtype=torch.bfloat16,
            device=target,
        )
        routed_w3_t = torch.empty(
            self.config.n_routed_experts,
            self.config.dim,
            self.config.moe_inter_dim,
            dtype=torch.bfloat16,
            device=target,
        )

        for expert_id in range(self.config.n_routed_experts):
            expert = self.get_moe_routed_expert(layer_id, expert_id, device=target)
            routed_w1_t[expert_id].copy_(expert.w1_t)
            routed_w2_t[expert_id].copy_(expert.w2_t)
            routed_w3_t[expert_id].copy_(expert.w3_t)
        return MoERoutedPackWeights(
            routed_w1_t=HostStagingTensor(routed_w1_t.contiguous(), StagingKind.PREFILL_ROUTED, "w1_t"),
            routed_w2_t=HostStagingTensor(routed_w2_t.contiguous(), StagingKind.PREFILL_ROUTED, "w2_t"),
            routed_w3_t=HostStagingTensor(routed_w3_t.contiguous(), StagingKind.PREFILL_ROUTED, "w3_t"),
        )

    def release(self, name: str | None = None) -> None:
        """Release fixed runtime layouts for one parameter or for the whole loader."""
        if name is None:
            self._layout_cache.clear()
            self._layout_cache_bytes = 0
            self._expert_cache.close()
            self._close_file_handles()
            return
        self._release_layout_keys(key for key in self._layout_cache if key[0].name == name)

    def close(self) -> None:
        self.release()

    def release_prefix(self, prefix: str) -> None:
        """Release fixed runtime layouts whose parameter names match ``prefix``."""
        self._release_layout_keys(key for key in self._layout_cache if key[0].name.startswith(prefix))

    def _get_transposed_weight(
        self,
        name: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
        cache: bool = True,
        layout: str = "linear_t",
    ) -> RuntimeWeight:
        target = torch.device(device) if device is not None else self.default_device

        def build() -> torch.Tensor:
            tensor = self._load_linear_weight(name, dtype=dtype, device=target)
            start = time.perf_counter()
            out = tensor.t().contiguous()
            self._record_profile(f"transpose.{layout}", start)
            return out

        return self._get_runtime_weight(
            name,
            dtype=dtype,
            layout=layout,
            build=build,
            device=target,
            cache=cache,
        )

    def _build_head_hc_fn_t(self, *, device: torch.device) -> torch.Tensor:
        hc_fn = self._load_tensor("hc_head_fn", dtype=torch.float32, device=device)
        if hc_fn.ndim != 2 or hc_fn.shape[0] != self.config.hc_mult:
            raise ValueError(f"Expected hc_head_fn shape [{self.config.hc_mult}, HC_DIM], got {tuple(hc_fn.shape)}")
        if hc_fn.shape[0] > HC_HEAD_PAD:
            raise ValueError(f"hc_mult={hc_fn.shape[0]} exceeds HC head pad width {HC_HEAD_PAD}")
        out = torch.zeros(hc_fn.shape[1], HC_HEAD_PAD, dtype=torch.float32, device=hc_fn.device)
        out[:, : hc_fn.shape[0]] = hc_fn.t()
        return out.contiguous()

    def _build_head_hc_base(self, *, device: torch.device) -> torch.Tensor:
        hc_base_raw = self._load_tensor("hc_head_base", dtype=torch.float32, device=device)
        if hc_base_raw.numel() > HC_HEAD_PAD:
            raise ValueError(f"hc_head_base has {hc_base_raw.numel()} entries, exceeds pad width {HC_HEAD_PAD}")
        out = torch.zeros(HC_HEAD_PAD, dtype=torch.float32, device=hc_base_raw.device)
        out[: hc_base_raw.numel()] = hc_base_raw.reshape(-1)
        return out.contiguous()

    def _release_layout_keys(self, keys: Iterable[tuple[RuntimeWeightKey, str]]) -> None:
        for key in list(keys):
            weight = self._layout_cache.pop(key)
            self._layout_cache_bytes -= tensor_nbytes(weight.host_tensor)

    def _get_compressor(self, prefix: str, *, device: str | torch.device | None = None) -> CompressorWeights:
        return CompressorWeights(
            wkv_t=self._get_transposed_weight(f"{prefix}.wkv.weight", device=device),
            wgate_t=self._get_transposed_weight(f"{prefix}.wgate.weight", device=device),
            ape=self._get_runtime_weight(f"{prefix}.ape", dtype=torch.float32, device=device),
            norm_w=self._get_runtime_weight(f"{prefix}.norm.weight", dtype=torch.bfloat16, device=device),
        )

    def _layer_prefix(self, layer_id: int) -> str:
        self._validate_layer_id(layer_id)
        return f"layers.{layer_id}"

    def _normalize_selected_expert_ids(self, expert_ids: torch.Tensor | list[int] | tuple[int, ...]) -> list[int]:
        if isinstance(expert_ids, torch.Tensor):
            ids = [int(x) for x in expert_ids.detach().cpu().reshape(-1).tolist()]
        else:
            ids = [int(x) for x in expert_ids]
        expected = self.config.n_activated_experts
        if len(ids) != expected:
            raise ValueError(f"selected expert ids must contain {expected} entries, got {len(ids)}")
        for expert_id in ids:
            if not 0 <= expert_id < self.config.n_routed_experts:
                raise ValueError(f"expert_id must be in [0, {self.config.n_routed_experts}), got {expert_id}")
        return ids

    def _copy_linear_t_into(self, out: torch.Tensor, name: str, *, device: torch.device) -> None:
        tensor = self._get_transposed_weight(name, device=device, cache=False).host_tensor
        if tuple(tensor.shape) != tuple(out.shape):
            raise ValueError(f"{name} transposed shape mismatch: expected {tuple(out.shape)}, got {tuple(tensor.shape)}")
        start = time.perf_counter()
        out.copy_(tensor)
        self._record_profile("copy_linear_t", start)

    def _validate_layer_id(self, layer_id: int) -> None:
        if not 0 <= layer_id < self.config.n_layers:
            raise ValueError(f"layer_id must be in [0, {self.config.n_layers}), got {layer_id}")

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
        start = time.perf_counter()
        tensor = self._load_raw_tensor(entry)
        kind = self._resolve_kind(name, entry, tensor)
        self._record_profile(f"raw_load.{kind}", start)
        if not dequantize or kind not in {"fp8_weight", "fp4_packed_weight"}:
            start = time.perf_counter()
            out = tensor.to(device)
            self._record_profile(f"to_device.{kind}", start)
            return out

        scale_name = entry.get("scale") or name.replace(".weight", ".scale")
        if scale_name in self.index:
            scale_entry = self.entry(scale_name)
        elif "scale_file" in entry and "scale_raw_name" in entry:
            scale_entry = {"file": entry["scale_file"], "raw_name": entry["scale_raw_name"]}
        else:
            raise KeyError(f"Missing scale tensor for quantized weight: {name}")
        start = time.perf_counter()
        scale = self._load_raw_tensor(scale_entry)
        self._record_profile(f"scale_load.{kind}", start)
        start = time.perf_counter()
        if kind == "fp8_weight":
            converted = dequant_fp8_weight_to_bf16(tensor, scale)
        else:
            converted = dequant_fp4_weight_to_bf16(tensor, scale)
        self._record_profile(f"dequant.{kind}", start)
        start = time.perf_counter()
        out = converted.to(device)
        self._record_profile(f"to_device.{kind}", start)
        return out

    def _load_raw_tensor(self, entry: dict[str, Any]) -> torch.Tensor:
        path = self.checkpoint_path / entry["file"]
        handle = self._get_file_handle(path)
        return handle.get_tensor(entry["raw_name"])

    def _get_file_handle(self, path: Path) -> Any:
        path = path.resolve()
        handle = self._file_handles.get(path)
        if handle is None:
            handle = safe_open(path, framework="pt", device="cpu")
            self._file_handles[path] = handle
        return handle

    def _close_file_handles(self) -> None:
        for handle in self._file_handles.values():
            handle.__exit__(None, None, None)
        self._file_handles.clear()

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

    def _insert_layout_cache(self, key: tuple[RuntimeWeightKey, str], weight: RuntimeWeight) -> None:
        if key in self._layout_cache:
            old = self._layout_cache[key]
            self._layout_cache_bytes -= tensor_nbytes(old.host_tensor)
        self._layout_cache[key] = weight
        self._layout_cache_bytes += tensor_nbytes(weight.host_tensor)

    def reset_profile_stats(self) -> None:
        self._profile_stats.clear()

    def profile_summary(self) -> list[tuple[str, int, float]]:
        return [
            (name, int(stats["count"]), float(stats["seconds"]) * 1000.0)
            for name, stats in self._profile_stats.items()
        ]

    def _record_profile(self, name: str, start: float) -> None:
        if not self.profile:
            return
        stats = self._profile_stats.setdefault(name, {"count": 0, "seconds": 0.0})
        stats["count"] = int(stats["count"]) + 1
        stats["seconds"] = float(stats["seconds"]) + (time.perf_counter() - start)


def load_weight_loader_from_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    default_device: str | torch.device = "cpu",
    expert_cache_dir: str | os.PathLike[str] | None = None,
) -> DeepSeekV4WeightLoader:
    return DeepSeekV4WeightLoader(
        checkpoint_path,
        default_device=default_device,
        expert_cache_dir=expert_cache_dir,
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
    "MoESelectedExpertWeights",
    "MoESharedWeights",
    "dequant_fp4_weight_to_bf16",
    "dequant_fp8_weight_to_bf16",
    "load_weight_loader_from_checkpoint",
    "normalize_param_name",
]
