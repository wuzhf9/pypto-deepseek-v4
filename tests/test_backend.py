"""Tests for serving backend structure and direct execution behavior."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from models.golden import TensorSpec
from serving.backends.base import KernelBindings, KernelCase
from serving.backends.direct_backend import DirectBackend
from serving.backends.factory import create_backend
from serving.profiler import ProfileRecorder
from serving.runtime_types import (
    HostStagingTensor,
    RuntimeWeight,
    RuntimeWeightKey,
    StagingKind,
    StepContext,
    StepKind,
)
from serving.runner import DeepSeekV4Runner
from serving.state import LayerSpec, LayerStateSchema, StateTensorSpec


class _FakeJitFn:
    def __init__(self) -> None:
        self.compile_calls: list[tuple[tuple[torch.Tensor, ...], Any]] = []
        self.run_calls: list[tuple[tuple[torch.Tensor, ...], Any]] = []

    def compile(self, *args: torch.Tensor, config: Any) -> Any:
        self.compile_calls.append((args, config))

        def compiled(*run_args: torch.Tensor, config: Any) -> None:
            self.run_calls.append((run_args, config))
            run_args[1].copy_(run_args[0] + 1)

        return compiled


def _specs() -> list[TensorSpec]:
    return [
        TensorSpec("x", [2], torch.float32),
        TensorSpec("out", [2], torch.float32, is_output=True),
    ]


def _state_schemas() -> tuple[LayerStateSchema, ...]:
    return (
        LayerStateSchema(
            spec=LayerSpec(layer_id=0, ratio=0, hash_route=True),
            tensors=(
                StateTensorSpec(
                    name="cache",
                    input_name="cache",
                    output_name="cache_out",
                    shape=(2,),
                    dtype=torch.float32,
                ),
            ),
        ),
    )


class _OpaqueTensor:
    pass


class _DelegatingBackend:
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


def test_direct_backend_runs_in_spec_order_and_reuses_compile_cache(monkeypatch) -> None:
    backend = DirectBackend(platform="test", device_id=3, runtime_cfg={"option": True})
    run_config = object()
    monkeypatch.setattr(backend, "_run_config", lambda: run_config)
    fn = _FakeJitFn()
    case = KernelCase("fake", fn, lambda: _specs())
    specs = _specs()
    tensors = {
        "out": torch.zeros(2, dtype=torch.float32),
        "x": torch.tensor([2.0, 4.0], dtype=torch.float32),
    }

    bindings = KernelBindings(tensors)
    outputs = backend.run(case, specs, bindings)

    assert backend.last_compile_cache_hit is False
    assert len(fn.compile_calls) == 1
    assert len(fn.run_calls) == 1
    assert fn.compile_calls[0][1] is run_config
    assert fn.run_calls[0][0] == (tensors["x"], tensors["out"])
    assert fn.run_calls[0][1] is run_config
    assert outputs == {"out": tensors["out"]}
    torch.testing.assert_close(outputs["out"], torch.tensor([3.0, 5.0]))

    backend.run(case, specs, bindings)

    assert backend.last_compile_cache_hit is True
    assert len(fn.compile_calls) == 1
    assert len(fn.run_calls) == 2


def test_direct_backend_close_clears_compile_cache(monkeypatch) -> None:
    backend = DirectBackend(platform="test", device_id=0)
    monkeypatch.setattr(backend, "_run_config", lambda: object())
    fn = _FakeJitFn()
    case = KernelCase("fake", fn, lambda: _specs())
    specs = _specs()
    tensors = {
        "x": torch.zeros(2, dtype=torch.float32),
        "out": torch.zeros(2, dtype=torch.float32),
    }

    bindings = KernelBindings(tensors)
    backend.run(case, specs, bindings)
    backend.close()
    backend.run(case, specs, bindings)

    assert len(fn.compile_calls) == 2
    assert backend.last_compile_cache_hit is False


def test_direct_backend_materializes_host_values_and_allocates_buffers() -> None:
    backend = DirectBackend(platform="test", device_id=0)
    x = torch.arange(6, dtype=torch.float32).reshape(3, 2).t()
    specs = [
        TensorSpec("x", [2, 3], torch.float32, init_value=torch.ones(2, 3)),
        TensorSpec("cast", [2], torch.float32, init_value=torch.ones(2)),
        TensorSpec("scratch", [2], torch.float32),
        TensorSpec("out", [2], torch.float32, is_output=True),
    ]

    bindings = backend.materialize(
        specs,
        {
            "x": x,
            "cast": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        },
    )

    tensors = bindings.tensors
    assert tensors["x"].is_contiguous()
    torch.testing.assert_close(tensors["x"], x)
    assert tensors["cast"].dtype == torch.float32
    torch.testing.assert_close(tensors["cast"], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(tensors["scratch"], torch.zeros(2))
    torch.testing.assert_close(tensors["out"], torch.zeros(2))


def test_direct_backend_unwraps_fixed_and_staging_runtime_values() -> None:
    backend = DirectBackend(platform="test", device_id=0)
    fixed_tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
    staging_tensor = torch.tensor([3.0, 4.0], dtype=torch.float32)
    fixed = RuntimeWeight(
        RuntimeWeightKey(name="weight", dtype=torch.float32, layout="identity"),
        fixed_tensor,
    )
    staging = HostStagingTensor(staging_tensor, StagingKind.DECODE_SELECTED, "w1_t")
    specs = [
        TensorSpec("fixed", [2], torch.float32, init_value=torch.ones(2)),
        TensorSpec("staging", [2], torch.float32, init_value=torch.ones(2)),
    ]

    bindings = backend.materialize(specs, {"fixed": fixed, "staging": staging})
    tensors = bindings.tensors

    assert tensors["fixed"] is fixed_tensor
    assert tensors["staging"] is staging_tensor


def test_direct_backend_validates_step_lifecycle_and_close_cleans_active_step() -> None:
    backend = DirectBackend(platform="test", device_id=0)
    prefill = StepContext(kind=StepKind.PREFILL, seq_len=4, start_pos=0)

    with pytest.raises(RuntimeError, match="not active"):
        backend.end_step()

    backend.begin_step(prefill)
    with pytest.raises(RuntimeError, match="already active"):
        backend.begin_step(StepContext(kind=StepKind.DECODE, seq_len=1, start_pos=4))
    backend.end_step()

    backend.begin_step(prefill)
    backend.close()
    backend.begin_step(prefill)
    backend.end_step()


def test_direct_backend_materialize_validates_required_values() -> None:
    backend = DirectBackend(platform="test", device_id=0)
    required = TensorSpec("required", [2], torch.float32, init_value=torch.ones(2))

    with pytest.raises(KeyError, match="Missing backend tensors.*required"):
        backend.materialize([required], {})
    with pytest.raises(ValueError, match="required shape mismatch"):
        backend.materialize([required], {"required": torch.zeros(3)})
    with pytest.raises(TypeError, match="requires torch.Tensor"):
        backend.materialize([required], {"required": object()})


def test_direct_backend_control_output_and_debug_boundaries() -> None:
    backend = DirectBackend(platform="test", device_id=0)
    control = torch.tensor([3], dtype=torch.int32)
    output = torch.arange(6, dtype=torch.float32).reshape(3, 2).t()

    assert backend.read_control(control) is control
    assert backend.export_debug_tensor(output) is output
    exported = backend.export_output(output)
    assert exported.is_contiguous()
    torch.testing.assert_close(exported, output)

    with pytest.raises(TypeError, match="requires torch.Tensor"):
        backend.read_control(_OpaqueTensor())


def test_direct_backend_state_uses_swappable_current_and_next_buffers() -> None:
    backend = DirectBackend(platform="test", device_id=0)
    backend.prepare_state(_state_schemas())
    initial_inputs = backend.state_inputs(0)
    initial_outputs = backend.state_outputs(0)
    current = initial_inputs["cache"]
    next_buffer = initial_outputs["cache_out"]

    assert current is not next_buffer
    torch.testing.assert_close(current, torch.zeros(2))
    next_buffer.fill_(5.0)
    backend.commit_state(0, initial_outputs)

    assert backend.state_inputs(0)["cache"] is next_buffer
    assert backend.state_outputs(0)["cache_out"] is current
    torch.testing.assert_close(backend.state_inputs(0)["cache"], torch.full((2,), 5.0))


def test_direct_backend_state_commit_validates_all_outputs_before_swap() -> None:
    backend = DirectBackend(platform="test", device_id=0)
    backend.prepare_state(_state_schemas())
    current = backend.state_inputs(0)["cache"]
    next_buffer = backend.state_outputs(0)["cache_out"]

    with pytest.raises(KeyError, match="cache_out"):
        backend.commit_state(0, {})
    assert backend.state_inputs(0)["cache"] is current

    with pytest.raises(ValueError, match="not the bound next buffer"):
        backend.commit_state(0, {"cache_out": torch.empty_like(next_buffer)})
    assert backend.state_inputs(0)["cache"] is current


def test_direct_backend_state_close_is_idempotent_and_allows_reprepare() -> None:
    backend = DirectBackend(platform="test", device_id=0)
    backend.prepare_state(_state_schemas())

    backend.close()
    backend.close()

    with pytest.raises(RuntimeError, match="not prepared"):
        backend.state_inputs(0)
    backend.prepare_state(_state_schemas())
    torch.testing.assert_close(backend.state_inputs(0)["cache"], torch.zeros(2))


def test_runner_delegates_embedding_materialize_and_keeps_opaque_output() -> None:
    runner = DeepSeekV4Runner.__new__(DeepSeekV4Runner)
    backend = _DelegatingBackend()
    weight = object()
    runner.backend = backend
    runner.profiler = ProfileRecorder(enabled=False)
    runner.weight_loader = SimpleNamespace(get_embedding_weight=lambda: weight)
    input_ids = torch.tensor([[1]], dtype=torch.int64)

    output = runner._run_embedding(input_ids)

    assert output is backend.output
    assert len(backend.materialize_calls) == 1
    _, values = backend.materialize_calls[0]
    assert values == {"input_ids": input_ids, "weight": weight}
    assert len(backend.run_bindings) == 1
    assert isinstance(backend.run_bindings[0], KernelBindings)


def test_runner_exports_only_at_public_output_boundary(monkeypatch) -> None:
    runner = DeepSeekV4Runner.__new__(DeepSeekV4Runner)
    backend = _DelegatingBackend()
    runner.backend = backend
    runner.state_plan = SimpleNamespace(max_seq_len=8)
    runner.profiler = ProfileRecorder(enabled=False)
    runner.max_layers = 0
    runner.run_head = False
    monkeypatch.setattr(runner, "_run_embedding", lambda _input_ids: backend.output)

    output = runner.prefill(torch.tensor([[1]], dtype=torch.int64))

    torch.testing.assert_close(output, torch.tensor([7.0]))
    assert backend.export_calls == [backend.output]
    assert backend.begin_step_calls == [StepContext(kind=StepKind.PREFILL, seq_len=1, start_pos=0)]
    assert backend.end_step_calls == 1


def test_runner_wraps_decode_in_backend_step(monkeypatch) -> None:
    runner = DeepSeekV4Runner.__new__(DeepSeekV4Runner)
    backend = _DelegatingBackend()
    runner.backend = backend
    runner.state_plan = SimpleNamespace(max_seq_len=8)
    runner.profiler = ProfileRecorder(enabled=False)
    runner.max_layers = 0
    runner.run_head = False
    monkeypatch.setattr(runner, "_run_embedding", lambda _input_ids: backend.output)

    output = runner.decode(torch.tensor([[1]], dtype=torch.int64), start_pos=3)

    torch.testing.assert_close(output, torch.tensor([7.0]))
    assert backend.begin_step_calls == [StepContext(kind=StepKind.DECODE, seq_len=1, start_pos=3)]
    assert backend.end_step_calls == 1


def test_runner_ends_step_when_execution_fails(monkeypatch) -> None:
    runner = DeepSeekV4Runner.__new__(DeepSeekV4Runner)
    backend = _DelegatingBackend()
    runner.backend = backend
    runner.state_plan = SimpleNamespace(max_seq_len=8)
    runner.profiler = ProfileRecorder(enabled=False)
    runner.max_layers = 0
    runner.run_head = False

    def fail_embedding(_input_ids: torch.Tensor) -> Any:
        raise RuntimeError("embedding failed")

    monkeypatch.setattr(runner, "_run_embedding", fail_embedding)

    with pytest.raises(RuntimeError, match="embedding failed"):
        runner.prefill(torch.tensor([[1]], dtype=torch.int64))

    assert len(backend.begin_step_calls) == 1
    assert backend.end_step_calls == 1


def test_runner_does_not_end_step_when_begin_fails(monkeypatch) -> None:
    runner = DeepSeekV4Runner.__new__(DeepSeekV4Runner)
    backend = _DelegatingBackend()
    backend.fail_begin = True
    runner.backend = backend
    runner.state_plan = SimpleNamespace(max_seq_len=8)
    runner.profiler = ProfileRecorder(enabled=False)
    runner.max_layers = 0
    runner.run_head = False
    monkeypatch.setattr(runner, "_run_embedding", lambda _input_ids: backend.output)

    with pytest.raises(RuntimeError, match="begin failed"):
        runner.prefill(torch.tensor([[1]], dtype=torch.int64))

    assert backend.begin_step_calls == []
    assert backend.end_step_calls == 0


def test_runner_reads_selected_expert_indices_through_backend() -> None:
    runner = DeepSeekV4Runner.__new__(DeepSeekV4Runner)
    device_indices = _OpaqueTensor()
    host_indices = torch.tensor([[[1, 3, 5, 7, 9, 11]]], dtype=torch.int32)
    selected_calls: list[tuple[int, torch.Tensor]] = []
    shared = SimpleNamespace(shared_w1_t=object(), shared_w2_t=object(), shared_w3_t=object())
    selected = SimpleNamespace(selected_w1_t=object(), selected_w2_t=object(), selected_w3_t=object())

    def get_selected(layer_id: int, indices: torch.Tensor) -> Any:
        selected_calls.append((layer_id, indices))
        return selected

    runner.backend = SimpleNamespace(read_control=lambda tensor: host_indices if tensor is device_indices else None)
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


def test_runner_adds_backend_state_bindings_to_prefill_and_decode_values(monkeypatch) -> None:
    runner = DeepSeekV4Runner.__new__(DeepSeekV4Runner)
    backend = _DelegatingBackend()
    layer_spec = LayerSpec(layer_id=0, ratio=0, hash_route=True)
    shared = SimpleNamespace(shared_w1_t=object(), shared_w2_t=object(), shared_w3_t=object())
    routed = SimpleNamespace(routed_w1_t=object(), routed_w2_t=object(), routed_w3_t=object())
    runner.backend = backend
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

    assert prefill_values["cache_out"] is backend.state_output
    assert "cache" not in prefill_values
    assert decode_values["cache"] is backend.state_input
    assert decode_values["cache_out"] is backend.state_output
    assert backend.state_input_calls == [0]
    assert backend.state_output_calls == [0, 0]


def test_factory_creates_direct_backend() -> None:
    backend = create_backend(
        "direct",
        platform="a2a3",
        device_id=2,
        runtime_cfg={"enable_l2_swimlane": True},
    )

    assert isinstance(backend, DirectBackend)
    assert backend._platform == "a2a3"
    assert backend._device_id == 2
    assert backend._runtime_cfg == {"enable_l2_swimlane": True}


def test_factory_creates_worker_backend(monkeypatch) -> None:
    import serving.backends.worker_backend as worker_backend

    captured: dict[str, Any] = {}

    class FakeWorkerBackend:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(worker_backend, "WorkerBackend", FakeWorkerBackend)

    backend = create_backend(
        "worker",
        platform="a2a3",
        device_id=1,
        runtime_cfg={"enable_l2_swimlane": True},
    )

    assert isinstance(backend, FakeWorkerBackend)
    assert captured == {
        "platform": "a2a3",
        "device_id": 1,
        "runtime_cfg": {"enable_l2_swimlane": True},
        "keep_prefill_routed_staging": False,
    }


def test_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unsupported backend: 'unknown'"):
        create_backend("unknown", platform="a2a3", device_id=0)  # type: ignore[arg-type]
