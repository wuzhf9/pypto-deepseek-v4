"""Tests for the device runtime contract and runner orchestration."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from models.golden import TensorSpec
from serving.device_runtime import KernelBindings
from serving.profiler import ProfileRecorder
from serving.runtime_types import KernelCase, StepContext, StepKind
from serving.runner import DeepSeekV4Runner
from serving.state import LayerSpec


class _OpaqueTensor:
    pass


class _FakeRuntime:
    def __init__(self) -> None:
        self.output = _OpaqueTensor()
        self.state_input = _OpaqueTensor()
        self.state_output = _OpaqueTensor()
        self.materialize_calls: list[tuple[list[TensorSpec], dict[str, Any]]] = []
        self.export_calls: list[Any] = []
        self.state_input_calls: list[int] = []
        self.state_output_calls: list[int] = []
        self.begin_step_calls: list[StepContext] = []
        self.end_step_calls = 0
        self.run_bindings: list[KernelBindings] = []
        self.fail_begin = False

    def materialize(self, specs: list[TensorSpec], values: dict[str, Any]) -> KernelBindings:
        self.materialize_calls.append((specs, values))
        return KernelBindings(dict(values))

    def run(self, _case: KernelCase, _specs: list[TensorSpec], bindings: KernelBindings) -> dict[str, Any]:
        self.run_bindings.append(bindings)
        return {"out": self.output}

    def begin_step(self, context: StepContext) -> None:
        if self.fail_begin:
            raise RuntimeError("begin failed")
        self.begin_step_calls.append(context)

    def end_step(self) -> None:
        self.end_step_calls += 1

    def export_output(self, tensor: Any) -> torch.Tensor:
        self.export_calls.append(tensor)
        return torch.tensor([7.0])

    def state_inputs(self, layer_id: int) -> dict[str, Any]:
        self.state_input_calls.append(layer_id)
        return {"cache": self.state_input}

    def state_outputs(self, layer_id: int) -> dict[str, Any]:
        self.state_output_calls.append(layer_id)
        return {"cache_out": self.state_output}


def test_runner_delegates_embedding_materialize_and_keeps_opaque_output() -> None:
    runner = DeepSeekV4Runner.__new__(DeepSeekV4Runner)
    runtime = _FakeRuntime()
    weight = object()
    runner.runtime = runtime
    runner.profiler = ProfileRecorder(enabled=False)
    runner.weight_loader = SimpleNamespace(get_embedding_weight=lambda: weight)
    input_ids = torch.tensor([[1]], dtype=torch.int64)

    output = runner._run_embedding(input_ids)

    assert output is runtime.output
    assert len(runtime.materialize_calls) == 1
    _, values = runtime.materialize_calls[0]
    assert values == {"input_ids": input_ids, "weight": weight}
    assert len(runtime.run_bindings) == 1
    assert isinstance(runtime.run_bindings[0], KernelBindings)


def test_runner_exports_only_at_public_output_boundary(monkeypatch) -> None:
    runner = DeepSeekV4Runner.__new__(DeepSeekV4Runner)
    runtime = _FakeRuntime()
    runner.runtime = runtime
    runner.state_plan = SimpleNamespace(max_seq_len=8)
    runner.profiler = ProfileRecorder(enabled=False)
    runner.max_layers = 0
    runner.run_head = False
    monkeypatch.setattr(runner, "_run_embedding", lambda _input_ids: runtime.output)

    output = runner.prefill(torch.tensor([[1]], dtype=torch.int64))

    torch.testing.assert_close(output, torch.tensor([7.0]))
    assert runtime.export_calls == [runtime.output]
    assert runtime.begin_step_calls == [StepContext(kind=StepKind.PREFILL, seq_len=1, start_pos=0)]
    assert runtime.end_step_calls == 1


def test_runner_wraps_decode_in_runtime_step(monkeypatch) -> None:
    runner = DeepSeekV4Runner.__new__(DeepSeekV4Runner)
    runtime = _FakeRuntime()
    runner.runtime = runtime
    runner.state_plan = SimpleNamespace(max_seq_len=8)
    runner.profiler = ProfileRecorder(enabled=False)
    runner.max_layers = 0
    runner.run_head = False
    monkeypatch.setattr(runner, "_run_embedding", lambda _input_ids: runtime.output)

    output = runner.decode(torch.tensor([[1]], dtype=torch.int64), start_pos=3)

    torch.testing.assert_close(output, torch.tensor([7.0]))
    assert runtime.begin_step_calls == [StepContext(kind=StepKind.DECODE, seq_len=1, start_pos=3)]
    assert runtime.end_step_calls == 1


def test_runner_ends_step_when_execution_fails(monkeypatch) -> None:
    runner = DeepSeekV4Runner.__new__(DeepSeekV4Runner)
    runtime = _FakeRuntime()
    runner.runtime = runtime
    runner.state_plan = SimpleNamespace(max_seq_len=8)
    runner.profiler = ProfileRecorder(enabled=False)
    runner.max_layers = 0
    runner.run_head = False

    def fail_embedding(_input_ids: torch.Tensor) -> Any:
        raise RuntimeError("embedding failed")

    monkeypatch.setattr(runner, "_run_embedding", fail_embedding)

    with pytest.raises(RuntimeError, match="embedding failed"):
        runner.prefill(torch.tensor([[1]], dtype=torch.int64))

    assert len(runtime.begin_step_calls) == 1
    assert runtime.end_step_calls == 1


def test_runner_does_not_end_step_when_begin_fails(monkeypatch) -> None:
    runner = DeepSeekV4Runner.__new__(DeepSeekV4Runner)
    runtime = _FakeRuntime()
    runtime.fail_begin = True
    runner.runtime = runtime
    runner.state_plan = SimpleNamespace(max_seq_len=8)
    runner.profiler = ProfileRecorder(enabled=False)
    runner.max_layers = 0
    runner.run_head = False
    monkeypatch.setattr(runner, "_run_embedding", lambda _input_ids: runtime.output)

    with pytest.raises(RuntimeError, match="begin failed"):
        runner.prefill(torch.tensor([[1]], dtype=torch.int64))

    assert runtime.begin_step_calls == []
    assert runtime.end_step_calls == 0


def test_runner_reads_selected_expert_indices_through_runtime() -> None:
    runner = DeepSeekV4Runner.__new__(DeepSeekV4Runner)
    device_indices = _OpaqueTensor()
    host_indices = torch.tensor([[[1, 3, 5, 7, 9, 11]]], dtype=torch.int32)
    selected_calls: list[tuple[int, torch.Tensor]] = []
    shared = SimpleNamespace(shared_w1_t=object(), shared_w2_t=object(), shared_w3_t=object())
    selected = SimpleNamespace(selected_w1_t=object(), selected_w2_t=object(), selected_w3_t=object())

    def get_selected(layer_id: int, indices: torch.Tensor) -> Any:
        selected_calls.append((layer_id, indices))
        return selected

    runner.runtime = SimpleNamespace(read_control=lambda tensor: host_indices if tensor is device_indices else None)
    runner.weight_loader = SimpleNamespace(
        get_layer_moe_shared=lambda _layer_id: shared,
        get_layer_moe_selected_experts=get_selected,
    )
    pre_outputs = {
        "indices": device_indices,
        "ffn_normed": _OpaqueTensor(),
        "weights": _OpaqueTensor(),
        "attn_hc_out": _OpaqueTensor(),
        "ffn_hc_post": _OpaqueTensor(),
        "ffn_hc_comb": _OpaqueTensor(),
    }

    values = runner._decode_post_moe_values(4, pre_outputs)

    assert selected_calls == [(4, host_indices)]
    assert values["ffn_normed"] is pre_outputs["ffn_normed"]
    assert values["selected_w1_t"] is selected.selected_w1_t


def test_runner_adds_runtime_state_bindings_to_prefill_and_decode_values(monkeypatch) -> None:
    runner = DeepSeekV4Runner.__new__(DeepSeekV4Runner)
    runtime = _FakeRuntime()
    layer_spec = LayerSpec(layer_id=0, ratio=0, hash_route=True)
    shared = SimpleNamespace(shared_w1_t=object(), shared_w2_t=object(), shared_w3_t=object())
    routed = SimpleNamespace(routed_w1_t=object(), routed_w2_t=object(), routed_w3_t=object())
    runner.runtime = runtime
    runner.profiler = ProfileRecorder(enabled=False)
    runner.state_plan = SimpleNamespace(
        layer_spec=lambda _layer_id: layer_spec,
        build_prefill_aux=lambda _layer_id, _seq_len: {"prefill_aux": object()},
        build_decode_aux=lambda _layer_id, _start_pos: {"decode_aux": object()},
    )
    runner.weight_loader = SimpleNamespace(
        get_layer_moe_shared=lambda _layer_id: shared,
        get_layer_moe_routed_pack=lambda _layer_id, release_each_expert: routed,
    )
    monkeypatch.setattr(
        runner,
        "_block_pre_moe_values",
        lambda _layer_id, hidden, *, input_ids, mode, aux: {"x": hidden, **aux},
    )
    hidden = _OpaqueTensor()
    input_ids = torch.tensor([[1]], dtype=torch.int64)

    prefill_values = runner._prefill_block_values(0, hidden, input_ids=input_ids)
    decode_values = runner._decode_pre_moe_values(0, hidden, input_ids=input_ids, start_pos=1)

    assert prefill_values["cache_out"] is runtime.state_output
    assert "cache" not in prefill_values
    assert decode_values["cache"] is runtime.state_input
    assert decode_values["cache_out"] is runtime.state_output
    assert runtime.state_input_calls == [0]
    assert runtime.state_output_calls == [0, 0]
