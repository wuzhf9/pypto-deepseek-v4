"""Host-side state and auxiliary input management for DeepSeek V4 Flash."""

from dataclasses import dataclass
import math
from typing import Any

import torch

from models.config import FLASH_CONFIG, DeepSeekV4FlashConfig


DEFAULT_MAX_SEQ_LEN = 4096
DEFAULT_BATCH_SIZE = 1
COMPRESS_RATIO4 = 4
COMPRESS_RATIO128 = 128
RATIO4_STATE_ROWS = 2 * COMPRESS_RATIO4


@dataclass(frozen=True)
class LayerSpec:
    layer_id: int
    ratio: int
    hash_route: bool


@dataclass
class LayerState:
    spec: LayerSpec
    kv_cache: torch.Tensor
    comp_cache: torch.Tensor | None = None
    comp_kv_state: torch.Tensor | None = None
    comp_score_state: torch.Tensor | None = None
    attn_comp_cache: torch.Tensor | None = None
    attn_comp_kv_state: torch.Tensor | None = None
    attn_comp_score_state: torch.Tensor | None = None
    idx_kv_cache: torch.Tensor | None = None
    idx_comp_kv_state: torch.Tensor | None = None
    idx_comp_score_state: torch.Tensor | None = None


def build_window_topk_idxs(
    seq_len: int,
    start_pos: int = 0,
    topk_max: int = FLASH_CONFIG.window_size,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> torch.Tensor:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if start_pos < 0:
        raise ValueError(f"start_pos must be non-negative, got {start_pos}")
    if start_pos > 0 and seq_len != 1:
        raise ValueError(f"decode-style window topk expects seq_len=1, got {seq_len}")
    if topk_max < FLASH_CONFIG.window_size:
        raise ValueError(f"topk_max must be at least {FLASH_CONFIG.window_size}, got {topk_max}")
    if batch_size != DEFAULT_BATCH_SIZE:
        raise ValueError(f"current kernels require batch_size={DEFAULT_BATCH_SIZE}, got {batch_size}")

    window_size = FLASH_CONFIG.window_size
    topk = torch.full((batch_size, seq_len, topk_max), -1, dtype=torch.int32)
    if start_pos >= window_size - 1:
        pos = start_pos % window_size
        idxs = torch.cat(
            [
                torch.arange(pos + 1, window_size, dtype=torch.int32),
                torch.arange(0, pos + 1, dtype=torch.int32),
            ]
        )
        topk[0, 0, : idxs.numel()] = idxs
    elif start_pos > 0:
        idxs = torch.arange(0, start_pos + 1, dtype=torch.int32)
        topk[0, 0, : idxs.numel()] = idxs
    else:
        for token_id in range(seq_len):
            start = max(0, token_id - window_size + 1)
            idxs = torch.arange(start, token_id + 1, dtype=torch.int32)
            topk[0, token_id, : idxs.numel()] = idxs
    return topk


def build_compress_topk_idxs(
    ratio: int,
    seq_len: int,
    start_pos: int,
    offset: int,
    topk_max: int,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> torch.Tensor:
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if start_pos < 0:
        raise ValueError(f"start_pos must be non-negative, got {start_pos}")
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")
    if topk_max < 0:
        raise ValueError(f"topk_max must be non-negative, got {topk_max}")
    if batch_size != DEFAULT_BATCH_SIZE:
        raise ValueError(f"current kernels require batch_size={DEFAULT_BATCH_SIZE}, got {batch_size}")

    if start_pos > 0:
        if seq_len != 1:
            raise ValueError(f"decode-style compress topk expects seq_len=1, got {seq_len}")
        matrix = torch.arange(0, (start_pos + 1) // ratio, dtype=torch.int32) + offset
        matrix = matrix.view(1, -1)
    else:
        matrix = torch.arange(seq_len // ratio, dtype=torch.int32).repeat(seq_len, 1)
        mask = matrix >= torch.arange(1, seq_len + 1, dtype=torch.int32).unsqueeze(1) // ratio
        matrix = torch.where(mask, torch.full_like(matrix, -1), matrix + offset)

    actual_topk = matrix.shape[1]
    if actual_topk > topk_max:
        raise ValueError(f"topk_max={topk_max} is smaller than required compressed topk={actual_topk}")

    topk = torch.full((batch_size, matrix.shape[0], topk_max), -1, dtype=torch.int32)
    if actual_topk > 0:
        topk[:, :, :actual_topk] = matrix.unsqueeze(0).expand(batch_size, -1, -1)
    return topk


def rope_profile_for_compress_ratio(config: Any, compress_ratio: int) -> tuple[float, int]:
    if compress_ratio:
        return float(config.compress_rope_theta), int(config.original_seq_len)
    return float(config.rope_theta), 0


def _linear_ramp_factor(low: int, high: int, dim: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    if low == high:
        high = high + 0.001
    ramp = (torch.arange(dim, dtype=torch.float32, device=device) - low) / (high - low)
    return torch.clamp(ramp, 0, 1)


def _find_correction_dim(num_rotations: int, dim: int, base: float, max_seq_len: int) -> float:
    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def _find_correction_range(low_rot: int, high_rot: int, dim: int, base: float, max_seq_len: int) -> tuple[int, int]:
    low = math.floor(_find_correction_dim(low_rot, dim, base, max_seq_len))
    high = math.ceil(_find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)


def precompute_freqs_cos_sin(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"RoPE dim must be a positive even integer, got {dim}")
    if seqlen <= 0:
        raise ValueError(f"RoPE sequence length must be positive, got {seqlen}")

    half_dim = dim // 2
    inv_freq = 1.0 / (float(base) ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    if original_seq_len > 0:
        low, high = _find_correction_range(beta_fast, beta_slow, dim, float(base), int(original_seq_len))
        smooth = 1 - _linear_ramp_factor(low, high, half_dim, device=device)
        inv_freq = inv_freq / float(factor) * (1 - smooth) + inv_freq * smooth

    positions = torch.arange(seqlen, dtype=torch.float32, device=device)
    angles = torch.outer(positions, inv_freq)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis.real.contiguous(), freqs_cis.imag.contiguous()


def build_deepseek_v4_rope_tables(
    config: Any = FLASH_CONFIG,
    compress_ratio: int = 0,
    *,
    max_seq_len: int,
    rope_dim: int | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    base, original_seq_len = rope_profile_for_compress_ratio(config, compress_ratio)
    dim = int(rope_dim if rope_dim is not None else config.rope_head_dim)
    return precompute_freqs_cos_sin(
        dim,
        int(max_seq_len),
        original_seq_len,
        base,
        float(config.rope_factor),
        int(config.beta_fast),
        int(config.beta_slow),
        device=device,
    )


def materialize_rope_range(
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    start_pos: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if start_pos < 0:
        raise ValueError(f"start_pos must be non-negative, got {start_pos}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    return freqs_cos[start_pos : start_pos + seq_len].contiguous(), freqs_sin[start_pos : start_pos + seq_len].contiguous()


def materialize_compressor_rope(
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    seq_len: int,
    ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    cutoff = seq_len - seq_len % ratio
    if cutoff == 0:
        return freqs_cos[:1].contiguous(), freqs_sin[:1].contiguous()
    return freqs_cos[:cutoff:ratio].contiguous(), freqs_sin[:cutoff:ratio].contiguous()


def materialize_decode_compressor_rope(
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    start_pos: int,
    ratio: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if start_pos < 0:
        raise ValueError(f"start_pos must be non-negative, got {start_pos}")
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    should_compress = (start_pos + 1) % ratio == 0
    if not should_compress:
        return (
            torch.zeros(1, freqs_cos.shape[-1], dtype=torch.float32, device=freqs_cos.device),
            torch.zeros(1, freqs_sin.shape[-1], dtype=torch.float32, device=freqs_sin.device),
        )
    rope_pos = start_pos + 1 - ratio
    return freqs_cos[rope_pos : rope_pos + 1].contiguous(), freqs_sin[rope_pos : rope_pos + 1].contiguous()


class DeepSeekV4State:
    """Own layer persistent states and build per-step host inputs."""

    def __init__(
        self,
        config: DeepSeekV4FlashConfig = FLASH_CONFIG,
        *,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | torch.device = "cpu",
    ) -> None:
        if batch_size != DEFAULT_BATCH_SIZE:
            raise ValueError(f"current kernels require batch_size={DEFAULT_BATCH_SIZE}, got {batch_size}")
        if max_seq_len != DEFAULT_MAX_SEQ_LEN:
            raise ValueError(f"current kernels require max_seq_len={DEFAULT_MAX_SEQ_LEN}, got {max_seq_len}")
        if config.window_size != 128:
            raise ValueError(f"current kernels require window_size=128, got {config.window_size}")

        self.config = config
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.topk_hca = max_seq_len // COMPRESS_RATIO128
        self.index_score_len = max_seq_len // COMPRESS_RATIO4
        self.layers = [self._init_layer_state(layer_id) for layer_id in range(config.n_layers)]
        self._normal_rope = build_deepseek_v4_rope_tables(
            config,
            compress_ratio=0,
            max_seq_len=max_seq_len,
            device=self.device,
        )
        self._compress4_rope = build_deepseek_v4_rope_tables(
            config,
            compress_ratio=COMPRESS_RATIO4,
            max_seq_len=max_seq_len,
            device=self.device,
        )
        self._compress128_rope = build_deepseek_v4_rope_tables(
            config,
            compress_ratio=COMPRESS_RATIO128,
            max_seq_len=max_seq_len,
            device=self.device,
        )

    def layer_spec(self, layer_id: int) -> LayerSpec:
        return self.layer_state(layer_id).spec

    def layer_state(self, layer_id: int) -> LayerState:
        self._validate_layer_id(layer_id)
        return self.layers[layer_id]

    def build_prefill_inputs(self, layer_id: int, seq_len: int) -> dict[str, torch.Tensor]:
        self._validate_seq_len(seq_len)
        spec = self.layer_spec(layer_id)
        cos, sin = self._main_rope_for_layer(spec, 0, seq_len)

        if spec.ratio == 0:
            return {
                "topk_idxs": build_window_topk_idxs(seq_len, start_pos=0, topk_max=self.config.window_size),
                "cos": cos,
                "sin": sin,
            }
        if spec.ratio == COMPRESS_RATIO128:
            window_topk = build_window_topk_idxs(seq_len, start_pos=0, topk_max=self.config.window_size)
            compress_topk = build_compress_topk_idxs(
                COMPRESS_RATIO128,
                seq_len,
                start_pos=0,
                offset=seq_len,
                topk_max=self.topk_hca,
            )
            comp_cos, comp_sin = materialize_compressor_rope(
                self._compress128_rope[0],
                self._compress128_rope[1],
                seq_len,
                COMPRESS_RATIO128,
            )
            return {
                "topk_idxs": torch.cat([window_topk, compress_topk], dim=-1),
                "cos": cos,
                "sin": sin,
                "comp_cos": comp_cos,
                "comp_sin": comp_sin,
                "comp_block_count": self._scalar_int(seq_len // COMPRESS_RATIO128),
            }
        if spec.ratio == COMPRESS_RATIO4:
            comp_cos, comp_sin = materialize_compressor_rope(
                self._compress4_rope[0],
                self._compress4_rope[1],
                seq_len,
                COMPRESS_RATIO4,
            )
            block_count = seq_len // COMPRESS_RATIO4
            return {
                "window_topk_idxs": build_window_topk_idxs(seq_len, start_pos=0, topk_max=self.config.window_size),
                "cos": cos,
                "sin": sin,
                "attn_comp_cos": comp_cos,
                "attn_comp_sin": comp_sin,
                "attn_comp_block_count": self._scalar_int(block_count),
                "idx_offset": self._scalar_int(seq_len),
                "idx_comp_cos": comp_cos.clone(),
                "idx_comp_sin": comp_sin.clone(),
                "idx_comp_block_count": self._scalar_int(block_count),
            }
        raise ValueError(f"Unsupported compress ratio: {spec.ratio}")

    def build_decode_inputs(self, layer_id: int, start_pos: int) -> dict[str, torch.Tensor]:
        if start_pos <= 0:
            raise ValueError(f"decode start_pos must be positive, got {start_pos}")
        self._validate_start_pos(start_pos)
        state = self.layer_state(layer_id)
        spec = state.spec
        cos, sin = self._main_rope_for_layer(spec, start_pos, 1)
        inputs = {
            "kv_cache": state.kv_cache,
            "cache_pos": self._scalar_int(start_pos % self.config.window_size),
            "cos": cos,
            "sin": sin,
        }

        if spec.ratio == 0:
            inputs["topk_idxs"] = build_window_topk_idxs(1, start_pos=start_pos, topk_max=self.config.window_size)
            return inputs

        if spec.ratio == COMPRESS_RATIO128:
            window_topk = build_window_topk_idxs(1, start_pos=start_pos, topk_max=self.config.window_size)
            compress_topk = build_compress_topk_idxs(
                COMPRESS_RATIO128,
                1,
                start_pos=start_pos,
                offset=self.config.window_size,
                topk_max=self.topk_hca,
            )
            comp_cos, comp_sin = materialize_decode_compressor_rope(
                self._compress128_rope[0],
                self._compress128_rope[1],
                start_pos,
                COMPRESS_RATIO128,
            )
            inputs.update(
                {
                    "comp_kv_state": self._required(state.comp_kv_state, "comp_kv_state"),
                    "comp_score_state": self._required(state.comp_score_state, "comp_score_state"),
                    "comp_cache": self._required(state.comp_cache, "comp_cache"),
                    "topk_idxs": torch.cat([window_topk, compress_topk], dim=-1),
                    "comp_slot": self._scalar_int(start_pos % COMPRESS_RATIO128),
                    "comp_cache_slot": self._scalar_int(start_pos // COMPRESS_RATIO128),
                    "comp_should_compress": self._scalar_int(int((start_pos + 1) % COMPRESS_RATIO128 == 0)),
                    "comp_cos": comp_cos,
                    "comp_sin": comp_sin,
                }
            )
            return inputs

        if spec.ratio == COMPRESS_RATIO4:
            comp_cos, comp_sin = materialize_decode_compressor_rope(
                self._compress4_rope[0],
                self._compress4_rope[1],
                start_pos,
                COMPRESS_RATIO4,
            )
            inputs.update(
                {
                    "attn_comp_kv_state": self._required(state.attn_comp_kv_state, "attn_comp_kv_state"),
                    "attn_comp_score_state": self._required(state.attn_comp_score_state, "attn_comp_score_state"),
                    "attn_comp_cache": self._required(state.attn_comp_cache, "attn_comp_cache"),
                    "idx_kv_cache_in": self._required(state.idx_kv_cache, "idx_kv_cache"),
                    "idx_comp_kv_state": self._required(state.idx_comp_kv_state, "idx_comp_kv_state"),
                    "idx_comp_score_state": self._required(state.idx_comp_score_state, "idx_comp_score_state"),
                    "window_topk_idxs": build_window_topk_idxs(1, start_pos=start_pos, topk_max=self.config.window_size),
                    "comp_slot": self._scalar_int(start_pos % COMPRESS_RATIO4),
                    "comp_cache_slot": self._scalar_int(start_pos // COMPRESS_RATIO4),
                    "comp_should_compress": self._scalar_int(int((start_pos + 1) % COMPRESS_RATIO4 == 0)),
                    "idx_offset": self._scalar_int(self.config.window_size),
                    "attn_comp_cos": comp_cos,
                    "attn_comp_sin": comp_sin,
                    "idx_comp_cos": comp_cos.clone(),
                    "idx_comp_sin": comp_sin.clone(),
                }
            )
            return inputs

        raise ValueError(f"Unsupported compress ratio: {spec.ratio}")

    def update_layer_state(self, layer_id: int, outputs: dict[str, torch.Tensor]) -> None:
        state = self.layer_state(layer_id)
        state.kv_cache = self._take_output(outputs, "kv_cache_out", state.kv_cache)
        if state.spec.ratio == COMPRESS_RATIO128:
            state.comp_kv_state = self._take_output(outputs, "comp_kv_state_out", self._required(state.comp_kv_state, "comp_kv_state"))
            state.comp_score_state = self._take_output(
                outputs,
                "comp_score_state_out",
                self._required(state.comp_score_state, "comp_score_state"),
            )
            state.comp_cache = self._take_output(outputs, "comp_cache_out", self._required(state.comp_cache, "comp_cache"))
        elif state.spec.ratio == COMPRESS_RATIO4:
            state.attn_comp_kv_state = self._take_output(
                outputs,
                "attn_comp_kv_state_out",
                self._required(state.attn_comp_kv_state, "attn_comp_kv_state"),
            )
            state.attn_comp_score_state = self._take_output(
                outputs,
                "attn_comp_score_state_out",
                self._required(state.attn_comp_score_state, "attn_comp_score_state"),
            )
            state.attn_comp_cache = self._take_output(
                outputs,
                "attn_comp_cache_out",
                self._required(state.attn_comp_cache, "attn_comp_cache"),
            )
            state.idx_kv_cache = self._take_output(outputs, "idx_kv_cache_out", self._required(state.idx_kv_cache, "idx_kv_cache"))
            state.idx_comp_kv_state = self._take_output(
                outputs,
                "idx_comp_kv_state_out",
                self._required(state.idx_comp_kv_state, "idx_comp_kv_state"),
            )
            state.idx_comp_score_state = self._take_output(
                outputs,
                "idx_comp_score_state_out",
                self._required(state.idx_comp_score_state, "idx_comp_score_state"),
            )

    def _init_layer_state(self, layer_id: int) -> LayerState:
        ratio = int(self.config.compress_ratios[layer_id])
        spec = LayerSpec(layer_id=layer_id, ratio=ratio, hash_route=layer_id < self.config.n_hash_layers)
        state = LayerState(
            spec=spec,
            kv_cache=torch.zeros(
                self.batch_size,
                self.config.window_size,
                self.config.head_dim,
                dtype=torch.bfloat16,
                device=self.device,
            ),
        )
        if ratio == COMPRESS_RATIO128:
            state.comp_cache = torch.zeros(self.batch_size, self.topk_hca, self.config.head_dim, dtype=torch.bfloat16, device=self.device)
            state.comp_kv_state = torch.zeros(
                self.batch_size,
                COMPRESS_RATIO128,
                self.config.head_dim,
                dtype=torch.float32,
                device=self.device,
            )
            state.comp_score_state = torch.full(
                (self.batch_size, COMPRESS_RATIO128, self.config.head_dim),
                -torch.finfo(torch.float32).max,
                dtype=torch.float32,
                device=self.device,
            )
        elif ratio == COMPRESS_RATIO4:
            attn_proj_dim = 2 * self.config.head_dim
            index_proj_dim = 2 * self.config.index_head_dim
            state.attn_comp_cache = torch.zeros(
                self.batch_size,
                self.index_score_len,
                self.config.head_dim,
                dtype=torch.bfloat16,
                device=self.device,
            )
            state.attn_comp_kv_state = torch.zeros(
                self.batch_size,
                RATIO4_STATE_ROWS,
                attn_proj_dim,
                dtype=torch.float32,
                device=self.device,
            )
            state.attn_comp_score_state = torch.full(
                (self.batch_size, RATIO4_STATE_ROWS, attn_proj_dim),
                -torch.finfo(torch.float32).max,
                dtype=torch.float32,
                device=self.device,
            )
            state.idx_kv_cache = torch.zeros(
                self.batch_size,
                self.index_score_len,
                self.config.index_head_dim,
                dtype=torch.bfloat16,
                device=self.device,
            )
            state.idx_comp_kv_state = torch.zeros(
                self.batch_size,
                RATIO4_STATE_ROWS,
                index_proj_dim,
                dtype=torch.float32,
                device=self.device,
            )
            state.idx_comp_score_state = torch.full(
                (self.batch_size, RATIO4_STATE_ROWS, index_proj_dim),
                -torch.finfo(torch.float32).max,
                dtype=torch.float32,
                device=self.device,
            )
        elif ratio != 0:
            raise ValueError(f"Unsupported compress ratio at layer {layer_id}: {ratio}")
        return state

    def _main_rope_for_layer(self, spec: LayerSpec, start_pos: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        rope = self._compress4_rope if spec.ratio == COMPRESS_RATIO4 else self._normal_rope
        return materialize_rope_range(rope[0], rope[1], start_pos, seq_len)

    def _validate_layer_id(self, layer_id: int) -> None:
        if not 0 <= layer_id < self.config.n_layers:
            raise ValueError(f"layer_id must be in [0, {self.config.n_layers}), got {layer_id}")

    def _validate_seq_len(self, seq_len: int) -> None:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

    def _validate_start_pos(self, start_pos: int) -> None:
        if start_pos >= self.max_seq_len:
            raise ValueError(f"start_pos={start_pos} exceeds max decode position {self.max_seq_len - 1}")

    def _scalar_int(self, value: int) -> torch.Tensor:
        return torch.tensor([int(value)], dtype=torch.int32, device=self.device)

    @staticmethod
    def _required(tensor: torch.Tensor | None, name: str) -> torch.Tensor:
        if tensor is None:
            raise ValueError(f"Layer state is missing required tensor: {name}")
        return tensor

    @staticmethod
    def _take_output(outputs: dict[str, torch.Tensor], name: str, current: torch.Tensor) -> torch.Tensor:
        if name not in outputs:
            raise KeyError(f"Missing state output tensor: {name}")
        tensor = outputs[name]
        if tuple(tensor.shape) != tuple(current.shape):
            raise ValueError(f"{name} shape mismatch: expected {tuple(current.shape)}, got {tuple(tensor.shape)}")
        if tensor.dtype != current.dtype:
            raise TypeError(f"{name} dtype mismatch: expected {current.dtype}, got {tensor.dtype}")
        return tensor.contiguous()


__all__ = [
    "COMPRESS_RATIO4",
    "COMPRESS_RATIO128",
    "DEFAULT_MAX_SEQ_LEN",
    "DeepSeekV4State",
    "LayerSpec",
    "LayerState",
    "RATIO4_STATE_ROWS",
    "build_compress_topk_idxs",
    "build_deepseek_v4_rope_tables",
    "build_window_topk_idxs",
    "materialize_compressor_rope",
    "materialize_decode_compressor_rope",
    "materialize_rope_range",
    "precompute_freqs_cos_sin",
    "rope_profile_for_compress_ratio",
]

