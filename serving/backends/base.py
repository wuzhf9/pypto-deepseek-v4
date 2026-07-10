"""Runtime-neutral serving backend definitions."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch

from models.golden import TensorSpec
from serving.state import LayerStateSchema


BackendName = Literal["direct", "worker"]


@dataclass(frozen=True)
class KernelCase:
    """A compiled kernel entrypoint and its TensorSpec builder."""

    name: str
    fn: Any
    spec_builder: Any


class Backend(Protocol):
    """Tensor materialization and execution contract for serving backends."""

    last_compile_seconds: float
    last_run_seconds: float
    last_compile_cache_hit: bool

    def materialize(
        self,
        specs: list[TensorSpec],
        values: Mapping[str, Any],
    ) -> dict[str, Any]: ...

    def run(
        self,
        case: KernelCase,
        specs: list[TensorSpec],
        tensors: Mapping[str, Any],
    ) -> dict[str, Any]: ...

    def read_control(self, tensor: Any) -> torch.Tensor: ...

    def export_output(self, tensor: Any) -> torch.Tensor: ...

    def export_debug_tensor(self, tensor: Any) -> torch.Tensor: ...

    def prepare_state(self, schemas: Sequence[LayerStateSchema]) -> None: ...

    def state_inputs(self, layer_id: int) -> dict[str, Any]: ...

    def state_outputs(self, layer_id: int) -> dict[str, Any]: ...

    def commit_state(self, layer_id: int, outputs: Mapping[str, Any]) -> None: ...

    def close(self) -> None: ...


__all__ = [
    "Backend",
    "BackendName",
    "KernelCase",
]
