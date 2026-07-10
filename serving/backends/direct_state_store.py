"""Host state storage for the direct serving backend."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from serving.state import LayerStateSchema, StateTensorSpec


@dataclass
class DirectStatePair:
    spec: StateTensorSpec
    current: torch.Tensor
    next: torch.Tensor


class DirectStateStore:
    """Own per-layer host state as swappable current/next tensor pairs."""

    def __init__(self) -> None:
        self._layers: dict[int, dict[str, DirectStatePair]] = {}
        self._prepared = False

    def prepare(self, schemas: Sequence[LayerStateSchema]) -> None:
        if self._prepared:
            raise RuntimeError("DirectStateStore is already prepared")

        layers: dict[int, dict[str, DirectStatePair]] = {}
        for schema in schemas:
            layer_id = schema.spec.layer_id
            if layer_id in layers:
                raise ValueError(f"Duplicate state schema for layer {layer_id}")
            pairs: dict[str, DirectStatePair] = {}
            input_names: set[str] = set()
            output_names: set[str] = set()
            for tensor_spec in schema.tensors:
                if tensor_spec.name in pairs:
                    raise ValueError(f"Duplicate state tensor {tensor_spec.name!r} for layer {layer_id}")
                if tensor_spec.input_name in input_names:
                    raise ValueError(f"Duplicate state input {tensor_spec.input_name!r} for layer {layer_id}")
                if tensor_spec.output_name in output_names:
                    raise ValueError(f"Duplicate state output {tensor_spec.output_name!r} for layer {layer_id}")
                input_names.add(tensor_spec.input_name)
                output_names.add(tensor_spec.output_name)
                current = tensor_spec.create_tensor()
                pairs[tensor_spec.name] = DirectStatePair(
                    spec=tensor_spec,
                    current=current,
                    next=torch.empty_like(current),
                )
            layers[layer_id] = pairs

        self._layers = layers
        self._prepared = True

    def inputs(self, layer_id: int) -> dict[str, torch.Tensor]:
        pairs = self._layer(layer_id)
        return {pair.spec.input_name: pair.current for pair in pairs.values()}

    def outputs(self, layer_id: int) -> dict[str, torch.Tensor]:
        pairs = self._layer(layer_id)
        return {pair.spec.output_name: pair.next for pair in pairs.values()}

    def commit(self, layer_id: int, outputs: Mapping[str, Any]) -> None:
        pairs = self._layer(layer_id)
        for pair in pairs.values():
            output_name = pair.spec.output_name
            if output_name not in outputs:
                raise KeyError(f"Missing state output tensor: {output_name}")
            tensor = outputs[output_name]
            if tensor is not pair.next:
                raise ValueError(f"State output {output_name!r} is not the bound next buffer")

        for pair in pairs.values():
            pair.current, pair.next = pair.next, pair.current

    def close(self) -> None:
        self._layers.clear()
        self._prepared = False

    def _layer(self, layer_id: int) -> dict[str, DirectStatePair]:
        if not self._prepared:
            raise RuntimeError("DirectStateStore is not prepared")
        try:
            return self._layers[layer_id]
        except KeyError as exc:
            raise ValueError(f"No state schema for layer {layer_id}") from exc


__all__ = ["DirectStatePair", "DirectStateStore"]
