"""Device-resident mutable state storage for the worker backend."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from serving.backends.device_pool import AllocationCategory, DeviceBufferPool, DeviceLease
from serving.state import LayerStateSchema, StateTensorSpec


@dataclass
class WorkerStatePair:
    """One mutable state tensor backed by swappable device leases."""

    spec: StateTensorSpec
    current: DeviceLease
    next: DeviceLease


class WorkerStateStore:
    """Own per-layer device state as current/next allocation pairs."""

    def __init__(self, pool: DeviceBufferPool) -> None:
        self._pool = pool
        self._layers: dict[int, dict[str, WorkerStatePair]] = {}
        self._prepared = False

    def prepare(self, schemas: Sequence[LayerStateSchema]) -> None:
        if self._prepared:
            raise RuntimeError("WorkerStateStore is already prepared")
        self._validate_schemas(schemas)

        layers: dict[int, dict[str, WorkerStatePair]] = {}
        allocated: list[DeviceLease] = []
        try:
            for schema in schemas:
                pairs: dict[str, WorkerStatePair] = {}
                for tensor_spec in schema.tensors:
                    current = self._pool.allocate_persistent(
                        tensor_spec.shape,
                        tensor_spec.dtype,
                        category=AllocationCategory.STATE,
                        init=tensor_spec.create_tensor(),
                    )
                    allocated.append(current)
                    next_buffer = self._pool.allocate_persistent(
                        tensor_spec.shape,
                        tensor_spec.dtype,
                        category=AllocationCategory.STATE,
                    )
                    allocated.append(next_buffer)
                    pairs[tensor_spec.name] = WorkerStatePair(
                        spec=tensor_spec,
                        current=current,
                        next=next_buffer,
                    )
                layers[schema.spec.layer_id] = pairs
        except BaseException:
            for lease in reversed(allocated):
                self._pool.free(lease)
            raise

        self._layers = layers
        self._prepared = True

    def inputs(self, layer_id: int) -> dict[str, Any]:
        pairs = self._layer(layer_id)
        return {pair.spec.input_name: pair.current.tensor for pair in pairs.values()}

    def outputs(self, layer_id: int) -> dict[str, Any]:
        pairs = self._layer(layer_id)
        return {pair.spec.output_name: pair.next.tensor for pair in pairs.values()}

    def commit(self, layer_id: int, outputs: Mapping[str, Any]) -> None:
        pairs = self._layer(layer_id)
        for pair in pairs.values():
            output_name = pair.spec.output_name
            if output_name not in outputs:
                raise KeyError(f"Missing state output tensor: {output_name}")
            tensor = outputs[output_name]
            if tensor is not pair.next.tensor:
                raise ValueError(f"State output {output_name!r} is not the bound next buffer")

        for pair in pairs.values():
            pair.current, pair.next = pair.next, pair.current

    def close(self) -> None:
        for pairs in self._layers.values():
            for pair in pairs.values():
                self._pool.free(pair.current)
                self._pool.free(pair.next)
        self._layers.clear()
        self._prepared = False

    @staticmethod
    def _validate_schemas(schemas: Sequence[LayerStateSchema]) -> None:
        layer_ids: set[int] = set()
        for schema in schemas:
            layer_id = schema.spec.layer_id
            if layer_id in layer_ids:
                raise ValueError(f"Duplicate state schema for layer {layer_id}")
            layer_ids.add(layer_id)
            tensor_names: set[str] = set()
            input_names: set[str] = set()
            output_names: set[str] = set()
            for tensor_spec in schema.tensors:
                if tensor_spec.name in tensor_names:
                    raise ValueError(f"Duplicate state tensor {tensor_spec.name!r} for layer {layer_id}")
                if tensor_spec.input_name in input_names:
                    raise ValueError(f"Duplicate state input {tensor_spec.input_name!r} for layer {layer_id}")
                if tensor_spec.output_name in output_names:
                    raise ValueError(f"Duplicate state output {tensor_spec.output_name!r} for layer {layer_id}")
                tensor_names.add(tensor_spec.name)
                input_names.add(tensor_spec.input_name)
                output_names.add(tensor_spec.output_name)

    def _layer(self, layer_id: int) -> dict[str, WorkerStatePair]:
        if not self._prepared:
            raise RuntimeError("WorkerStateStore is not prepared")
        try:
            return self._layers[layer_id]
        except KeyError as exc:
            raise ValueError(f"No state schema for layer {layer_id}") from exc


__all__ = ["WorkerStatePair", "WorkerStateStore"]
