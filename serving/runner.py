"""Host-side DeepSeek V4 Flash PyPTO runner.

This module owns whole-model orchestration only.  The model math stays in
``models/`` kernels; this runner loads converted weights, builds per-layer
state/auxiliary inputs, dispatches kernels, and carries runtime-owned hidden
tensors between layers.
"""

from typing import Any

import torch

from models import block as block_kernels
from models import embedding as embedding_kernel
from models import head as head_kernel
from models import split_block as split_block_kernels
from models.config import FLASH_CONFIG, DeepSeekV4FlashConfig
from serving.device_runtime import DeviceRuntime
from serving.profiler import ProfileRecorder, block_profile_fields
from serving.runtime_types import KernelCase, StepContext, StepKind
from serving.state import COMPRESS_RATIO4, COMPRESS_RATIO128, DEFAULT_MAX_SEQ_LEN, DeepSeekV4StatePlan, LayerSpec
from serving.weight_loader import DeepSeekV4WeightLoader


class DeepSeekV4Runner:
    """Model runner for DeepSeek V4 Flash PyPTO kernels."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        runtime: DeviceRuntime,
        config: DeepSeekV4FlashConfig = FLASH_CONFIG,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        max_layers: int | None = 1,
        run_head: bool = True,
        profile: bool = False,
        verbose_layer_log: bool = False,
        expert_cache_dir: str | None = None,
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
            config=config,
            default_device="cpu",
            profile=self.profiler.enabled,
            expert_cache_dir=expert_cache_dir,
        )
        self.state_plan = DeepSeekV4StatePlan(config=config, max_seq_len=max_seq_len, device="cpu")

        self.runtime = runtime
        self.runtime.prepare_state(self.state_plan.layer_state_schemas()[: self.max_layers])

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = self._validate_prefill_input_ids(input_ids)
        seq_len = int(input_ids.shape[1])
        with self.profiler.timer("prefill.total", seq_len=seq_len, max_layers=self.max_layers, run_head=self.run_head):
            self.runtime.begin_step(StepContext(kind=StepKind.PREFILL, seq_len=seq_len, start_pos=0))
            try:
                hidden = self._run_embedding(input_ids)

                for layer_id in range(self.max_layers):
                    hidden = self._run_prefill_block(layer_id, hidden, input_ids=input_ids)

                output = self._run_head(hidden, seq_len=seq_len) if self.run_head else hidden
                return self.runtime.export_output(output)
            finally:
                self.runtime.end_step()

    def decode(self, input_ids: torch.Tensor, *, start_pos: int) -> torch.Tensor:
        input_ids = self._validate_decode_input_ids(input_ids, start_pos=start_pos)
        with self.profiler.timer("decode.total", start_pos=start_pos, max_layers=self.max_layers, run_head=self.run_head):
            self.runtime.begin_step(StepContext(kind=StepKind.DECODE, seq_len=1, start_pos=start_pos))
            try:
                hidden = self._run_embedding(input_ids)

                for layer_id in range(self.max_layers):
                    hidden = self._run_decode_block(layer_id, hidden, input_ids=input_ids, start_pos=start_pos)

                output = self._run_head(hidden, seq_len=1) if self.run_head else hidden
                return self.runtime.export_output(output)
            finally:
                self.runtime.end_step()

    def close(self) -> None:
        self.runtime.close()
        self.weight_loader.close()

    def _run_embedding(self, input_ids: torch.Tensor) -> Any:
        seq_len = int(input_ids.shape[1])
        with self.profiler.timer("embedding.total", seq_len=seq_len):
            specs = embedding_kernel.build_embedding_specs(seq_len=seq_len)
            with self.profiler.timer("embedding.weight"):
                weight = self.weight_loader.get_embedding_weight()
            with self.profiler.timer("embedding.materialize"):
                bindings = self.runtime.materialize(
                    specs,
                    {
                        "input_ids": input_ids,
                        "weight": weight,
                    },
                )
            with self.profiler.runtime_timer("embedding.kernel", self.runtime):
                outputs = self.runtime.run(
                    KernelCase("embedding_test", embedding_kernel.embedding_test, embedding_kernel.build_embedding_specs),
                    specs,
                    bindings,
                )
            return outputs["out"]

    def _run_head(self, hidden: Any, *, seq_len: int) -> Any:
        with self.profiler.timer("head.total", seq_len=seq_len):
            specs = head_kernel.build_head_specs(seq_len=seq_len)
            with self.profiler.timer("head.weight"):
                weights = self.weight_loader.get_head_weights()
            with self.profiler.timer("head.materialize"):
                bindings = self.runtime.materialize(
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
            with self.profiler.runtime_timer("head.kernel", self.runtime):
                outputs = self.runtime.run(
                    KernelCase("head_test", head_kernel.head_test, head_kernel.build_head_specs),
                    specs,
                    bindings,
                )
            return outputs["logits"]

    def _run_prefill_block(
        self,
        layer_id: int,
        hidden: Any,
        *,
        input_ids: torch.Tensor,
    ) -> Any:
        spec = self.state_plan.layer_spec(layer_id)
        seq_len = int(input_ids.shape[1])
        case = self._block_case(spec, decode=False, start_pos=0, seq_len=seq_len)
        specs = case.spec_builder(seq_len)
        mode = "prefill"
        if self.verbose_layer_log:
            input_shape = (1, seq_len, self.config.hc_mult, self.config.dim)
            print(
                f"[RUNNER] layer {layer_id} start: mode={mode} ratio={spec.ratio} "
                f"hash_route={spec.hash_route} kernel={case.name} input={input_shape}",
                flush=True,
            )

        profile_fields = block_profile_fields(
            layer=layer_id,
            mode=mode,
            ratio=spec.ratio,
            hash_route=spec.hash_route,
            kernel=case.name,
        )
        with self.profiler.timer("layer.total", **profile_fields):
            with self.profiler.timer("layer.values", **profile_fields):
                if self.profiler.enabled:
                    self.weight_loader.reset_profile_stats()
                values = self._prefill_block_values(layer_id, hidden, input_ids=input_ids)
            self.profiler.record_weight_loader("layer.weight_loader", self.weight_loader, **profile_fields)
            with self.profiler.timer("layer.materialize", **profile_fields):
                bindings = self.runtime.materialize(specs, values)
            with self.profiler.runtime_timer("layer.kernel", self.runtime, **profile_fields):
                outputs = self.runtime.run(case, specs, bindings)
            with self.profiler.timer("layer.state_update", **profile_fields):
                self.runtime.commit_state(layer_id, outputs)
            out = outputs["out"]
            if self.verbose_layer_log:
                debug_out = self.runtime.export_debug_tensor(out)
                finite = bool(torch.isfinite(debug_out.float()).all().item())
                print(
                    f"[RUNNER] layer {layer_id} done: output={tuple(debug_out.shape)} "
                    f"dtype={debug_out.dtype} finite={finite}",
                    flush=True,
                )
            return out

    def _run_decode_block(
        self,
        layer_id: int,
        hidden: Any,
        *,
        input_ids: torch.Tensor,
        start_pos: int,
    ) -> Any:
        spec = self.state_plan.layer_spec(layer_id)
        pre_case = self._selected_decode_pre_case(spec)
        post_case = self._selected_decode_post_case()
        pre_specs = pre_case.spec_builder(start_pos)
        post_specs = post_case.spec_builder(start_pos)
        if self.verbose_layer_log:
            input_shape = (1, 1, self.config.hc_mult, self.config.dim)
            print(
                f"[RUNNER] layer {layer_id} start: mode=decode ratio={spec.ratio} "
                f"hash_route={spec.hash_route} kernel={pre_case.name}+{post_case.name} "
                f"input={input_shape}",
                flush=True,
            )

        profile_fields = block_profile_fields(
            layer=layer_id,
            mode="decode",
            ratio=spec.ratio,
            hash_route=spec.hash_route,
            kernel=pre_case.name,
        )
        post_profile_fields = {
            **profile_fields,
            "kernel": post_case.name,
            "block_shape": "selected_decode_post_moe",
        }
        with self.profiler.timer("layer.total", **profile_fields):
            with self.profiler.timer("layer.selected_decode.pre_values", **profile_fields):
                if self.profiler.enabled:
                    self.weight_loader.reset_profile_stats()
                pre_values = self._decode_pre_moe_values(
                    layer_id,
                    hidden,
                    input_ids=input_ids,
                    start_pos=start_pos,
                )
            self.profiler.record_weight_loader("layer.selected_decode.pre_weight_loader", self.weight_loader, **profile_fields)
            with self.profiler.timer("layer.selected_decode.pre_materialize", **profile_fields):
                pre_bindings = self.runtime.materialize(pre_specs, pre_values)
            with self.profiler.runtime_timer("layer.selected_decode.pre_kernel", self.runtime, **profile_fields):
                pre_outputs = self.runtime.run(pre_case, pre_specs, pre_bindings)
            with self.profiler.timer("layer.selected_decode.state_update", **profile_fields):
                self.runtime.commit_state(layer_id, pre_outputs)

            with self.profiler.timer("layer.selected_decode.post_values", **post_profile_fields):
                if self.profiler.enabled:
                    self.weight_loader.reset_profile_stats()
                post_values = self._decode_post_moe_values(layer_id, pre_outputs)
            self.profiler.record_weight_loader("layer.selected_decode.post_weight_loader", self.weight_loader, **post_profile_fields)
            with self.profiler.timer("layer.selected_decode.post_materialize", **post_profile_fields):
                post_bindings = self.runtime.materialize(post_specs, post_values)
            with self.profiler.runtime_timer("layer.selected_decode.post_kernel", self.runtime, **post_profile_fields):
                post_outputs = self.runtime.run(post_case, post_specs, post_bindings)

            out = post_outputs["out"]
            if self.verbose_layer_log:
                debug_out = self.runtime.export_debug_tensor(out)
                finite = bool(torch.isfinite(debug_out.float()).all().item())
                print(
                    f"[RUNNER] layer {layer_id} done: output={tuple(debug_out.shape)} "
                    f"dtype={debug_out.dtype} finite={finite}",
                    flush=True,
                )
            return out

    def _prefill_block_values(
        self,
        layer_id: int,
        hidden: Any,
        *,
        input_ids: torch.Tensor,
    ) -> dict[str, Any]:
        spec = self.state_plan.layer_spec(layer_id)
        mode = "prefill"
        with self.profiler.timer("layer.values.aux", layer=layer_id, mode=mode, ratio=spec.ratio):
            aux = self.state_plan.build_prefill_aux(layer_id, int(input_ids.shape[1]))
        values = self._block_pre_moe_values(
            layer_id,
            hidden,
            input_ids=input_ids,
            mode=mode,
            aux=aux,
        )
        values.update(self.runtime.state_outputs(layer_id))
        with self.profiler.timer("layer.values.shared", layer=layer_id, mode=mode, ratio=spec.ratio):
            shared = self.weight_loader.get_layer_moe_shared(layer_id)
        values.update(
            {
                "shared_w1_t": shared.shared_w1_t,
                "shared_w2_t": shared.shared_w2_t,
                "shared_w3_t": shared.shared_w3_t,
            }
        )

        with self.profiler.timer("layer.values.routed_pack", layer=layer_id, mode=mode, ratio=spec.ratio):
            routed = self.weight_loader.get_layer_moe_routed_pack(layer_id, release_each_expert=True)
        values.update(
            {
                "routed_w1_t": routed.routed_w1_t,
                "routed_w2_t": routed.routed_w2_t,
                "routed_w3_t": routed.routed_w3_t,
            }
        )
        return values

    def _decode_pre_moe_values(
        self,
        layer_id: int,
        hidden: Any,
        *,
        input_ids: torch.Tensor,
        start_pos: int,
    ) -> dict[str, Any]:
        spec = self.state_plan.layer_spec(layer_id)
        mode = "decode"
        with self.profiler.timer("layer.values.aux", layer=layer_id, mode=mode, ratio=spec.ratio):
            aux = self.state_plan.build_decode_aux(layer_id, start_pos)
        values = self._block_pre_moe_values(
            layer_id,
            hidden,
            input_ids=input_ids,
            mode=mode,
            aux=aux,
        )
        values.update(self.runtime.state_inputs(layer_id))
        values.update(self.runtime.state_outputs(layer_id))
        return values

    def _decode_post_moe_values(self, layer_id: int, pre_outputs: dict[str, Any]) -> dict[str, Any]:
        shared = self.weight_loader.get_layer_moe_shared(layer_id)
        indices = self.runtime.read_control(pre_outputs["indices"])
        selected = self.weight_loader.get_layer_moe_selected_experts(layer_id, indices)
        return {
            "ffn_normed": pre_outputs["ffn_normed"],
            "weights": pre_outputs["weights"],
            "selected_w1_t": selected.selected_w1_t,
            "selected_w2_t": selected.selected_w2_t,
            "selected_w3_t": selected.selected_w3_t,
            "shared_w1_t": shared.shared_w1_t,
            "shared_w2_t": shared.shared_w2_t,
            "shared_w3_t": shared.shared_w3_t,
            "attn_hc_out": pre_outputs["attn_hc_out"],
            "ffn_hc_post": pre_outputs["ffn_hc_post"],
            "ffn_hc_comb": pre_outputs["ffn_hc_comb"],
        }

    def _block_pre_moe_values(
        self,
        layer_id: int,
        hidden: Any,
        *,
        input_ids: torch.Tensor,
        mode: str,
        aux: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        spec = self.state_plan.layer_spec(layer_id)
        with self.profiler.timer("layer.values.hc", layer=layer_id, mode=mode, ratio=spec.ratio):
            hc = self.weight_loader.get_layer_hc(layer_id)
        with self.profiler.timer("layer.values.attn", layer=layer_id, mode=mode, ratio=spec.ratio):
            attn = self.weight_loader.get_layer_attention_common(layer_id)
        with self.profiler.timer("layer.values.gate", layer=layer_id, mode=mode, ratio=spec.ratio):
            gate = self.weight_loader.get_layer_moe_gate(layer_id, hash_route=spec.hash_route)
        with self.profiler.timer("layer.values.ffn_norm", layer=layer_id, mode=mode, ratio=spec.ratio):
            ffn_norm_w = self.weight_loader.get_layer_ffn_norm(layer_id)

        values: dict[str, Any] = {
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

    def _block_case(self, spec: LayerSpec, *, decode: bool, start_pos: int, seq_len: int) -> KernelCase:
        del start_pos, seq_len
        suffix = "decode" if decode else "prefill"
        if spec.ratio == 0 and spec.hash_route:
            return KernelCase(
                f"block_swa_hash_{suffix}_fwd",
                block_kernels.block_swa_hash_decode_fwd if decode else block_kernels.block_swa_hash_prefill_fwd,
                block_kernels.build_swa_hash_decode_specs if decode else block_kernels.build_swa_hash_prefill_specs,
            )
        if spec.ratio == COMPRESS_RATIO4 and spec.hash_route:
            return KernelCase(
                f"block_csa_hash_{suffix}_fwd",
                block_kernels.block_csa_hash_decode_fwd if decode else block_kernels.block_csa_hash_prefill_fwd,
                block_kernels.build_csa_hash_decode_specs if decode else block_kernels.build_csa_hash_prefill_specs,
            )
        if spec.ratio == COMPRESS_RATIO128 and not spec.hash_route:
            return KernelCase(
                f"block_hca_topk_{suffix}_fwd",
                block_kernels.block_hca_topk_decode_fwd if decode else block_kernels.block_hca_topk_prefill_fwd,
                block_kernels.build_hca_topk_decode_specs if decode else block_kernels.build_hca_topk_prefill_specs,
            )
        if spec.ratio == COMPRESS_RATIO4 and not spec.hash_route:
            return KernelCase(
                f"block_csa_topk_{suffix}_fwd",
                block_kernels.block_csa_topk_decode_fwd if decode else block_kernels.block_csa_topk_prefill_fwd,
                block_kernels.build_csa_topk_decode_specs if decode else block_kernels.build_csa_topk_prefill_specs,
            )
        raise ValueError(f"unsupported block shape: layer={spec.layer_id} ratio={spec.ratio} hash_route={spec.hash_route}")

    def _selected_decode_pre_case(self, spec: LayerSpec) -> KernelCase:
        if spec.ratio == 0 and spec.hash_route:
            return KernelCase(
                "block_swa_hash_selected_decode_pre_moe_fwd",
                split_block_kernels.swa_hash_selected_decode_pre_moe_fwd,
                split_block_kernels.build_swa_hash_selected_decode_pre_moe_specs,
            )
        if spec.ratio == COMPRESS_RATIO4 and spec.hash_route:
            return KernelCase(
                "block_csa_hash_selected_decode_pre_moe_fwd",
                split_block_kernels.csa_hash_selected_decode_pre_moe_fwd,
                split_block_kernels.build_csa_hash_selected_decode_pre_moe_specs,
            )
        if spec.ratio == COMPRESS_RATIO128 and not spec.hash_route:
            return KernelCase(
                "block_hca_topk_selected_decode_pre_moe_fwd",
                split_block_kernels.hca_topk_selected_decode_pre_moe_fwd,
                split_block_kernels.build_hca_topk_selected_decode_pre_moe_specs,
            )
        if spec.ratio == COMPRESS_RATIO4 and not spec.hash_route:
            return KernelCase(
                "block_csa_topk_selected_decode_pre_moe_fwd",
                split_block_kernels.csa_topk_selected_decode_pre_moe_fwd,
                split_block_kernels.build_csa_topk_selected_decode_pre_moe_specs,
            )
        raise ValueError(f"unsupported selected decode block shape: layer={spec.layer_id} ratio={spec.ratio} hash_route={spec.hash_route}")

    @staticmethod
    def _selected_decode_post_case() -> KernelCase:
        return KernelCase(
            "block_selected_decode_post_moe_fwd",
            split_block_kernels.selected_decode_post_moe_fwd,
            split_block_kernels.build_selected_decode_post_moe_specs,
        )

    def _validate_prefill_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"prefill input_ids must have shape [1, S], got {tuple(input_ids.shape)}")
        if input_ids.shape[1] <= 0:
            raise ValueError("prefill input_ids sequence length must be positive")
        if input_ids.shape[1] > self.state_plan.max_seq_len:
            raise ValueError(f"prefill sequence length {input_ids.shape[1]} exceeds max_seq_len={self.state_plan.max_seq_len}")
        return input_ids.to(dtype=torch.int64, device="cpu").contiguous()

    def _validate_decode_input_ids(self, input_ids: torch.Tensor, *, start_pos: int) -> torch.Tensor:
        if input_ids.ndim != 2 or tuple(input_ids.shape) != (1, 1):
            raise ValueError(f"decode input_ids must have shape [1, 1], got {tuple(input_ids.shape)}")
        if start_pos <= 0:
            raise ValueError(f"decode start_pos must be positive, got {start_pos}")
        if start_pos >= self.state_plan.max_seq_len:
            raise ValueError(f"decode start_pos={start_pos} exceeds max_seq_len={self.state_plan.max_seq_len}")
        return input_ids.to(dtype=torch.int64, device="cpu").contiguous()


__all__ = [
    "DeepSeekV4Runner",
]
