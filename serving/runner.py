"""Host-side DeepSeek V4 Flash PyPTO runner.

This module owns whole-model orchestration only.  The model math stays in
``models/`` kernels; this runner loads converted weights, builds per-layer
state/auxiliary inputs, dispatches kernels, and carries the hidden tensor
between layers on the host.
"""

from dataclasses import dataclass
import time
from typing import Any, Literal

import torch

from models import block as block_kernels
from models import embedding as embedding_kernel
from models import head as head_kernel
from models.config import FLASH_CONFIG, DeepSeekV4FlashConfig
from models.golden import TensorSpec
from serving.profile import ProfileRecorder
from serving.state import COMPRESS_RATIO4, COMPRESS_RATIO128, DEFAULT_MAX_SEQ_LEN, DeepSeekV4State, LayerSpec
from serving.weight_loader import DeepSeekV4WeightLoader


BackendName = Literal["direct", "worker"]


@dataclass(frozen=True)
class _KernelCase:
    name: str
    fn: Any
    spec_builder: Any


class _DirectBackend:
    """Compile and run kernels directly with host tensors.

    This mirrors ``models.golden.run_jit`` and is the first validation backend.
    It keeps hidden/state tensors on the host between dispatches.  The worker
    backend can reuse the same runner argument assembly once the shared weight
    buffer pool is added.
    """

    def __init__(self, *, platform: str, device_id: int, runtime_cfg: dict[str, Any] | None = None) -> None:
        self._platform = platform
        self._device_id = device_id
        self._runtime_cfg = dict(runtime_cfg or {})
        self._compiled: dict[tuple[str, tuple[tuple[int, ...], ...], tuple[torch.dtype, ...]], Any] = {}
        self.last_compile_seconds = 0.0
        self.last_run_seconds = 0.0
        self.last_compile_cache_hit = False

    def run(self, case: _KernelCase, specs: list[TensorSpec], tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        start = time.perf_counter()
        compiled = self._compile(case, specs)
        self.last_compile_seconds = time.perf_counter() - start
        ordered_args = [tensors[spec.name] for spec in specs]
        start = time.perf_counter()
        compiled(*ordered_args, config=self._run_config())
        self.last_run_seconds = time.perf_counter() - start
        return {spec.name: tensors[spec.name] for spec in specs if spec.is_output}

    def close(self) -> None:
        self._compiled.clear()

    def _compile(self, case: _KernelCase, specs: list[TensorSpec]) -> Any:
        key = (
            case.name,
            tuple(tuple(spec.shape) for spec in specs),
            tuple(spec.dtype for spec in specs),
        )
        compiled = self._compiled.get(key)
        if compiled is not None:
            self.last_compile_cache_hit = True
            return compiled

        self.last_compile_cache_hit = False
        dummy_args = [torch.empty(spec.shape, dtype=spec.dtype) for spec in specs]
        compiled = case.fn.compile(*dummy_args, config=self._run_config())
        self._compiled[key] = compiled
        return compiled

    def _run_config(self) -> Any:
        from pypto.runtime import RunConfig

        return RunConfig(platform=self._platform, device_id=self._device_id, **self._runtime_cfg)


class _WorkerBackend:
    """Placeholder for the experimental worker-resident backend."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise NotImplementedError(
            "worker backend was removed after profiling showed kernel runtime dominates; "
            "use backend='direct'"
        )


class DeepSeekV4Runner:
    """Correctness-first host runner for DeepSeek V4 Flash PyPTO kernels."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        weight_index: str | dict[str, Any] | None = None,
        config: DeepSeekV4FlashConfig = FLASH_CONFIG,
        device_id: int = 0,
        platform: str = "a2a3",
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        backend: BackendName = "direct",
        max_layers: int | None = 1,
        run_head: bool = True,
        profile: bool = False,
        verbose_layer_log: bool = False,
        routed_pack_cache_dir: str | None = None,
        runtime_cfg: dict[str, Any] | None = None,
    ) -> None:
        if max_layers is not None and not 0 <= max_layers <= config.n_layers:
            raise ValueError(f"max_layers must be in [0, {config.n_layers}], got {max_layers}")

        self.config = config
        self.max_layers = config.n_layers if max_layers is None else int(max_layers)
        self.run_head = bool(run_head)
        self.profiler = ProfileRecorder(enabled=profile)
        self.verbose_layer_log = bool(verbose_layer_log)
        self.weight_loader = DeepSeekV4WeightLoader(
            checkpoint_path,
            weight_index=weight_index,
            config=config,
            default_device="cpu",
            profile=self.profiler.enabled,
            routed_pack_cache_dir=routed_pack_cache_dir,
        )
        self.state = DeepSeekV4State(config=config, max_seq_len=max_seq_len, device="cpu")

        if backend == "direct":
            self.backend: Any = _DirectBackend(platform=platform, device_id=device_id, runtime_cfg=runtime_cfg)
        elif backend == "worker":
            self.backend = _WorkerBackend(platform=platform, device_id=device_id, runtime_cfg=runtime_cfg)
        else:
            raise ValueError(f"unsupported backend: {backend!r}")

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = self._validate_prefill_input_ids(input_ids)
        seq_len = int(input_ids.shape[1])
        with self.profiler.timer("prefill.total", seq_len=seq_len, max_layers=self.max_layers, run_head=self.run_head):
            hidden_3d = self._run_embedding(input_ids)
            hidden = hidden_3d.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()

            for layer_id in range(self.max_layers):
                hidden = self._run_block(layer_id, hidden, input_ids=input_ids, start_pos=0, decode=False)
                with self.profiler.timer("layer.release", layer=layer_id, mode="prefill"):
                    self._release_layer_weights(layer_id)

            if not self.run_head:
                return hidden
            return self._run_head(hidden)

    def decode(self, input_ids: torch.Tensor, *, start_pos: int) -> torch.Tensor:
        input_ids = self._validate_decode_input_ids(input_ids, start_pos=start_pos)
        with self.profiler.timer("decode.total", start_pos=start_pos, max_layers=self.max_layers, run_head=self.run_head):
            hidden_3d = self._run_embedding(input_ids)
            hidden = hidden_3d.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()

            for layer_id in range(self.max_layers):
                hidden = self._run_block(layer_id, hidden, input_ids=input_ids, start_pos=start_pos, decode=True)
                with self.profiler.timer("layer.release", layer=layer_id, mode="decode"):
                    self._release_layer_weights(layer_id)

            if not self.run_head:
                return hidden
            return self._run_head(hidden)

    def close(self) -> None:
        self.backend.close()
        self.weight_loader.close()

    def _run_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = int(input_ids.shape[1])
        with self.profiler.timer("embedding.total", seq_len=seq_len):
            specs = embedding_kernel.build_embedding_specs(seq_len=seq_len)
            with self.profiler.timer("embedding.weight"):
                weight = self.weight_loader.get_embedding_weight()
            with self.profiler.timer("embedding.materialize"):
                tensors = self._materialize_specs(
                    specs,
                    {
                        "input_ids": input_ids,
                        "weight": weight,
                    },
                )
            with self.profiler.backend_timer("embedding.kernel", self.backend):
                outputs = self.backend.run(
                    _KernelCase("embedding_test", embedding_kernel.embedding_test, embedding_kernel.build_embedding_specs),
                    specs,
                    tensors,
                )
            return outputs["out"].contiguous()

    def _run_head(self, hidden: torch.Tensor) -> torch.Tensor:
        seq_len = int(hidden.shape[1])
        with self.profiler.timer("head.total", seq_len=seq_len):
            specs = head_kernel.build_head_specs(seq_len=seq_len)
            with self.profiler.timer("head.weight"):
                weights = self.weight_loader.get_head_weights()
            with self.profiler.timer("head.materialize"):
                tensors = self._materialize_specs(
                    specs,
                    {
                        "x": hidden,
                        "hc_fn_t": weights.hc_fn_t,
                        "hc_scale": weights.hc_scale,
                        "hc_base": weights.hc_base,
                        "norm_w": weights.norm_w,
                        "head_w": weights.head_w,
                    },
                )
            with self.profiler.backend_timer("head.kernel", self.backend):
                outputs = self.backend.run(
                    _KernelCase("head_test", head_kernel.head_test, head_kernel.build_head_specs),
                    specs,
                    tensors,
                )
            return outputs["logits"].contiguous()

    def _run_block(
        self,
        layer_id: int,
        hidden: torch.Tensor,
        *,
        input_ids: torch.Tensor,
        start_pos: int,
        decode: bool,
    ) -> torch.Tensor:
        spec = self.state.layer_spec(layer_id)
        seq_len = int(hidden.shape[1])
        case = self._block_case(spec, decode=decode, start_pos=start_pos, seq_len=seq_len)
        specs = case.spec_builder(start_pos) if decode else case.spec_builder(seq_len)
        mode = "decode" if decode else "prefill"
        if self.verbose_layer_log:
            print(
                f"[RUNNER] layer {layer_id} start: mode={mode} ratio={spec.ratio} "
                f"hash_route={spec.hash_route} kernel={case.name} input={tuple(hidden.shape)}",
                flush=True,
            )

        with self.profiler.timer("layer.total", layer=layer_id, mode=mode, ratio=spec.ratio, kernel=case.name):
            with self.profiler.timer("layer.values", layer=layer_id, mode=mode, ratio=spec.ratio):
                if self.profiler.enabled:
                    self.weight_loader.reset_profile_stats()
                values = self._layer_values(layer_id, hidden, input_ids=input_ids, start_pos=start_pos, decode=decode)
            self.profiler.record_weight_loader("layer.weight_loader", self.weight_loader, layer=layer_id, mode=mode, ratio=spec.ratio)
            with self.profiler.timer("layer.materialize", layer=layer_id, mode=mode, ratio=spec.ratio):
                tensors = self._materialize_specs(specs, values)
            with self.profiler.backend_timer("layer.kernel", self.backend, layer=layer_id, mode=mode, ratio=spec.ratio, kernel=case.name):
                outputs = self.backend.run(case, specs, tensors)
            with self.profiler.timer("layer.state_update", layer=layer_id, mode=mode, ratio=spec.ratio):
                self.state.update_layer_state(layer_id, outputs)
            out = outputs["out"].contiguous()
            if self.verbose_layer_log:
                finite = bool(torch.isfinite(out.float()).all().item())
                print(
                    f"[RUNNER] layer {layer_id} done: output={tuple(out.shape)} "
                    f"dtype={out.dtype} finite={finite}",
                    flush=True,
                )
            return out

    def _layer_values(
        self,
        layer_id: int,
        hidden: torch.Tensor,
        *,
        input_ids: torch.Tensor,
        start_pos: int,
        decode: bool,
    ) -> dict[str, torch.Tensor]:
        spec = self.state.layer_spec(layer_id)
        mode = "decode" if decode else "prefill"
        with self.profiler.timer("layer.values.aux", layer=layer_id, mode=mode, ratio=spec.ratio):
            aux = self.state.build_decode_inputs(layer_id, start_pos) if decode else self.state.build_prefill_inputs(layer_id, int(hidden.shape[1]))
        with self.profiler.timer("layer.values.hc", layer=layer_id, mode=mode, ratio=spec.ratio):
            hc = self.weight_loader.get_layer_hc(layer_id)
        with self.profiler.timer("layer.values.attn", layer=layer_id, mode=mode, ratio=spec.ratio):
            attn = self.weight_loader.get_layer_attention_common(layer_id)
        with self.profiler.timer("layer.values.gate", layer=layer_id, mode=mode, ratio=spec.ratio):
            gate = self.weight_loader.get_layer_moe_gate(layer_id, hash_route=spec.hash_route)
        with self.profiler.timer("layer.values.shared", layer=layer_id, mode=mode, ratio=spec.ratio):
            shared = self.weight_loader.get_layer_moe_shared(layer_id)
        with self.profiler.timer("layer.values.routed_pack", layer=layer_id, mode=mode, ratio=spec.ratio):
            routed = self.weight_loader.get_layer_moe_routed_pack(layer_id, release_each_expert=True)
        with self.profiler.timer("layer.values.ffn_norm", layer=layer_id, mode=mode, ratio=spec.ratio):
            ffn_norm_w = self.weight_loader.get_tensor(
                f"layers.{layer_id}.ffn_norm.weight",
                dtype=torch.bfloat16,
            )

        values: dict[str, torch.Tensor] = {
            "x": hidden,
            "attn_hc_fn_t": hc.attn_hc_fn_t,
            "attn_hc_scale": hc.attn_hc_scale,
            "attn_hc_base": hc.attn_hc_base,
            "attn_norm_w": attn.attn_norm_w,
            "wq_a_t": attn.wq_a_t,
            "q_norm_w": attn.q_norm_w,
            "wq_b_t": attn.wq_b_t,
            "wkv_t": attn.wkv_t,
            "kv_norm_w": attn.kv_norm_w,
            "attn_sink": attn.attn_sink,
            "wo_a_t": attn.wo_a_t,
            "wo_b_t": attn.wo_b_t,
            "ffn_hc_fn_t": hc.ffn_hc_fn_t,
            "ffn_hc_scale": hc.ffn_hc_scale,
            "ffn_hc_base": hc.ffn_hc_base,
            "ffn_norm_w": ffn_norm_w,
            "gate_w_t": gate.gate_w_t,
            "routed_w1_t": routed.routed_w1_t,
            "routed_w2_t": routed.routed_w2_t,
            "routed_w3_t": routed.routed_w3_t,
            "shared_w1_t": shared.shared_w1_t,
            "shared_w2_t": shared.shared_w2_t,
            "shared_w3_t": shared.shared_w3_t,
            **aux,
        }

        if spec.hash_route:
            if gate.tid2eid is None:
                raise ValueError(f"layer {layer_id} hash route requires tid2eid")
            values["tid2eid"] = gate.tid2eid
            values["input_ids"] = input_ids
        else:
            if gate.gate_bias is None:
                raise ValueError(f"layer {layer_id} topk route requires gate_bias")
            values["gate_bias"] = gate.gate_bias

        if spec.ratio == COMPRESS_RATIO128:
            with self.profiler.timer("layer.values.compressor128", layer=layer_id, mode=mode, ratio=spec.ratio):
                comp = self.weight_loader.get_layer_compressor_ratio128(layer_id)
            values.update(
                {
                    "comp_wkv_t": comp.wkv_t,
                    "comp_wgate_t": comp.wgate_t,
                    "comp_ape": comp.ape,
                    "comp_norm_w": comp.norm_w,
                }
            )
        elif spec.ratio == COMPRESS_RATIO4:
            with self.profiler.timer("layer.values.attn_compressor4", layer=layer_id, mode=mode, ratio=spec.ratio):
                attn_comp = self.weight_loader.get_layer_compressor_ratio4_attention(layer_id)
            with self.profiler.timer("layer.values.indexer", layer=layer_id, mode=mode, ratio=spec.ratio):
                indexer = self.weight_loader.get_layer_indexer(layer_id)
            values.update(
                {
                    "attn_comp_wkv_t": attn_comp.wkv_t,
                    "attn_comp_wgate_t": attn_comp.wgate_t,
                    "attn_comp_ape": attn_comp.ape,
                    "attn_comp_norm_w": attn_comp.norm_w,
                    "idx_wq_b_t": indexer.idx_wq_b_t,
                    "idx_weights_proj_t": indexer.idx_weights_proj_t,
                    "idx_comp_wkv_t": indexer.idx_comp_wkv_t,
                    "idx_comp_wgate_t": indexer.idx_comp_wgate_t,
                    "idx_comp_ape": indexer.idx_comp_ape,
                    "idx_comp_norm_w": indexer.idx_comp_norm_w,
                }
            )

        return values

    @staticmethod
    def _materialize_specs(specs: list[TensorSpec], values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tensors: dict[str, torch.Tensor] = {}
        missing_required: list[str] = []
        for spec in specs:
            value = values.get(spec.name)
            if value is not None:
                tensors[spec.name] = DeepSeekV4Runner._coerce_tensor(spec, value)
            elif spec.is_output or spec.init_value is None:
                tensors[spec.name] = spec.create_tensor()
            else:
                missing_required.append(spec.name)
        if missing_required:
            raise KeyError(f"Missing runner tensors for required inputs: {missing_required}")
        return tensors

    @staticmethod
    def _coerce_tensor(spec: TensorSpec, tensor: torch.Tensor) -> torch.Tensor:
        if tuple(tensor.shape) != tuple(spec.shape):
            raise ValueError(f"{spec.name} shape mismatch: expected {tuple(spec.shape)}, got {tuple(tensor.shape)}")
        if tensor.dtype != spec.dtype:
            tensor = tensor.to(dtype=spec.dtype)
        return tensor.contiguous()

    def _block_case(self, spec: LayerSpec, *, decode: bool, start_pos: int, seq_len: int) -> _KernelCase:
        del start_pos, seq_len
        suffix = "decode" if decode else "prefill"
        if spec.ratio == 0 and spec.hash_route:
            return _KernelCase(
                f"block_swa_hash_{suffix}_fwd",
                block_kernels.block_swa_hash_decode_fwd if decode else block_kernels.block_swa_hash_prefill_fwd,
                block_kernels.build_swa_hash_decode_specs if decode else block_kernels.build_swa_hash_prefill_specs,
            )
        if spec.ratio == COMPRESS_RATIO4 and spec.hash_route:
            return _KernelCase(
                f"block_csa_hash_{suffix}_fwd",
                block_kernels.block_csa_hash_decode_fwd if decode else block_kernels.block_csa_hash_prefill_fwd,
                block_kernels.build_csa_hash_decode_specs if decode else block_kernels.build_csa_hash_prefill_specs,
            )
        if spec.ratio == COMPRESS_RATIO128 and not spec.hash_route:
            return _KernelCase(
                f"block_hca_topk_{suffix}_fwd",
                block_kernels.block_hca_topk_decode_fwd if decode else block_kernels.block_hca_topk_prefill_fwd,
                block_kernels.build_hca_topk_decode_specs if decode else block_kernels.build_hca_topk_prefill_specs,
            )
        if spec.ratio == COMPRESS_RATIO4 and not spec.hash_route:
            return _KernelCase(
                f"block_csa_topk_{suffix}_fwd",
                block_kernels.block_csa_topk_decode_fwd if decode else block_kernels.block_csa_topk_prefill_fwd,
                block_kernels.build_csa_topk_decode_specs if decode else block_kernels.build_csa_topk_prefill_specs,
            )
        raise ValueError(f"unsupported block shape: layer={spec.layer_id} ratio={spec.ratio} hash_route={spec.hash_route}")

    def _release_layer_weights(self, layer_id: int) -> None:
        self.weight_loader.release_prefix(f"layers.{layer_id}.")

    def _validate_prefill_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"prefill input_ids must have shape [1, S], got {tuple(input_ids.shape)}")
        if input_ids.shape[1] <= 0:
            raise ValueError("prefill input_ids sequence length must be positive")
        if input_ids.shape[1] > self.state.max_seq_len:
            raise ValueError(f"prefill sequence length {input_ids.shape[1]} exceeds max_seq_len={self.state.max_seq_len}")
        return input_ids.to(dtype=torch.int64, device="cpu").contiguous()

    def _validate_decode_input_ids(self, input_ids: torch.Tensor, *, start_pos: int) -> torch.Tensor:
        if input_ids.ndim != 2 or tuple(input_ids.shape) != (1, 1):
            raise ValueError(f"decode input_ids must have shape [1, 1], got {tuple(input_ids.shape)}")
        if start_pos <= 0:
            raise ValueError(f"decode start_pos must be positive, got {start_pos}")
        if start_pos >= self.state.max_seq_len:
            raise ValueError(f"decode start_pos={start_pos} exceeds max_seq_len={self.state.max_seq_len}")
        return input_ids.to(dtype=torch.int64, device="cpu").contiguous()


__all__ = [
    "BackendName",
    "DeepSeekV4Runner",
]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="DeepSeek V4 Flash PyPTO runner smoke entrypoint.")
    parser.add_argument("--checkpoint", type=str, default="../deepseek_v4_flash")
    parser.add_argument("--weight-index", type=str, default=None)
    parser.add_argument("-p", "--platform", type=str, default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-s", "--seq-len", type=int, default=1)
    parser.add_argument("--max-layers", type=int, default=1)
    parser.add_argument("--backend", choices=["direct", "worker"], default="direct")
    parser.add_argument("--no-head", action="store_true", default=False)
    parser.add_argument("--decode-steps", type=int, default=0)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--verbose-layer-log", action="store_true", default=False)
    parser.add_argument("--routed-pack-cache-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.decode_steps < 0:
        raise ValueError(f"decode steps must be non-negative, got {args.decode_steps}")
    if args.seq_len + args.decode_steps > DEFAULT_MAX_SEQ_LEN:
        raise ValueError(
            f"seq_len + decode_steps must be <= {DEFAULT_MAX_SEQ_LEN}, "
            f"got {args.seq_len} + {args.decode_steps}"
        )

    torch.manual_seed(args.seed)
    input_ids = torch.randint(0, FLASH_CONFIG.vocab_size, (1, args.seq_len), dtype=torch.int64)
    runner = DeepSeekV4Runner(
        args.checkpoint,
        weight_index=args.weight_index,
        platform=args.platform,
        device_id=args.device,
        backend=args.backend,
        max_layers=args.max_layers,
        run_head=not args.no_head,
        profile=args.profile,
        verbose_layer_log=args.verbose_layer_log,
        routed_pack_cache_dir=args.routed_pack_cache_dir,
    )
    try:
        out = runner.prefill(input_ids)
        finite = bool(torch.isfinite(out.float()).all().item())
        print(
            f"[RUNNER] prefill ok: input_ids={tuple(input_ids.shape)} "
            f"output={tuple(out.shape)} dtype={out.dtype} finite={finite}",
            flush=True,
        )
        if not finite:
            return 1

        next_ids = _next_decode_input(out, run_head=not args.no_head)
        for step in range(args.decode_steps):
            start_pos = args.seq_len + step
            out = runner.decode(next_ids, start_pos=start_pos)
            finite = bool(torch.isfinite(out.float()).all().item())
            print(
                f"[RUNNER] decode ok: step={step + 1}/{args.decode_steps} "
                f"start_pos={start_pos} input_ids={tuple(next_ids.shape)} "
                f"output={tuple(out.shape)} dtype={out.dtype} finite={finite}",
                flush=True,
            )
            if not finite:
                return 1
            next_ids = _next_decode_input(out, run_head=not args.no_head)
    finally:
        runner.close()

    return 0


def _next_decode_input(out: torch.Tensor, *, run_head: bool) -> torch.Tensor:
    if run_head:
        return torch.argmax(out, dim=-1).view(1, 1).to(dtype=torch.int64, device="cpu").contiguous()
    return torch.randint(0, FLASH_CONFIG.vocab_size, (1, 1), dtype=torch.int64)


if __name__ == "__main__":
    raise SystemExit(main())
