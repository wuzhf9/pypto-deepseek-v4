"""Direct serving backend using host tensors for runtime dispatch."""

from collections.abc import Mapping, Sequence
import time
from typing import Any

import torch

from models.golden import TensorSpec
from serving.backends.base import KernelCase
from serving.backends.direct_state_store import DirectStateStore
from serving.state import LayerStateSchema


class DirectBackend:
    """Compile and run kernels directly with host tensors.

    This mirrors ``models.golden.run_jit`` and is the first validation backend.
    It keeps hidden and state tensors on the host between dispatches.
    """

    def __init__(self, *, platform: str, device_id: int, runtime_cfg: dict[str, Any] | None = None) -> None:
        self._platform = platform
        self._device_id = device_id
        self._runtime_cfg = dict(runtime_cfg or {})
        self._compiled: dict[tuple[str, tuple[tuple[int, ...], ...], tuple[torch.dtype, ...]], Any] = {}
        self.last_compile_seconds = 0.0
        self.last_run_seconds = 0.0
        self.last_compile_cache_hit = False
        self._state_store = DirectStateStore()

    def materialize(
        self,
        specs: list[TensorSpec],
        values: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Bind provided host values and allocate missing scratch/output tensors."""
        tensors: dict[str, torch.Tensor] = {}
        missing_required: list[str] = []
        for spec in specs:
            value = values.get(spec.name)
            if value is not None:
                tensors[spec.name] = self._coerce_tensor(spec, value)
            elif spec.is_output or spec.init_value is None:
                tensors[spec.name] = spec.create_tensor()
            else:
                missing_required.append(spec.name)
        if missing_required:
            raise KeyError(f"Missing backend tensors for required inputs: {missing_required}")
        return tensors

    def run(
        self,
        case: KernelCase,
        specs: list[TensorSpec],
        tensors: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        start = time.perf_counter()
        compiled = self._compile(case, specs)
        self.last_compile_seconds = time.perf_counter() - start
        ordered_args = [tensors[spec.name] for spec in specs]
        start = time.perf_counter()
        compiled(*ordered_args, config=self._run_config())
        self.last_run_seconds = time.perf_counter() - start
        return {spec.name: tensors[spec.name] for spec in specs if spec.is_output}

    def read_control(self, tensor: Any) -> torch.Tensor:
        """Return a host control tensor without copying on the direct path."""
        return self._require_host_tensor(tensor)

    def export_output(self, tensor: Any) -> torch.Tensor:
        """Return a contiguous host tensor at the public Runner boundary."""
        return self._require_host_tensor(tensor).contiguous()

    def export_debug_tensor(self, tensor: Any) -> torch.Tensor:
        """Expose a host tensor for explicitly requested debug inspection."""
        return self._require_host_tensor(tensor)

    def prepare_state(self, schemas: Sequence[LayerStateSchema]) -> None:
        self._state_store.prepare(schemas)

    def state_inputs(self, layer_id: int) -> dict[str, torch.Tensor]:
        return self._state_store.inputs(layer_id)

    def state_outputs(self, layer_id: int) -> dict[str, torch.Tensor]:
        return self._state_store.outputs(layer_id)

    def commit_state(self, layer_id: int, outputs: Mapping[str, Any]) -> None:
        self._state_store.commit(layer_id, outputs)

    def close(self) -> None:
        self._state_store.close()
        self._compiled.clear()

    @staticmethod
    def _coerce_tensor(spec: TensorSpec, tensor: Any) -> torch.Tensor:
        tensor = DirectBackend._require_host_tensor(tensor)
        if tuple(tensor.shape) != tuple(spec.shape):
            raise ValueError(f"{spec.name} shape mismatch: expected {tuple(spec.shape)}, got {tuple(tensor.shape)}")
        if tensor.dtype != spec.dtype:
            tensor = tensor.to(dtype=spec.dtype)
        return tensor.contiguous()

    @staticmethod
    def _require_host_tensor(tensor: Any) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"DirectBackend requires torch.Tensor values, got {type(tensor)!r}")
        return tensor

    def _compile(self, case: KernelCase, specs: list[TensorSpec]) -> Any:
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


__all__ = ["DirectBackend"]
