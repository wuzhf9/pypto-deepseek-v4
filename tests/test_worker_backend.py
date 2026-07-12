import ctypes
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from models.golden import TensorSpec
from serving.backends.base import KernelCase
from serving.backends.device_pool import AllocationCategory
from serving.backends.worker_backend import WorkerBackend, WorkerKernelBindings
from serving.runtime_types import (
    HostStagingTensor,
    RuntimeWeight,
    RuntimeWeightKey,
    StagingKind,
    StepContext,
    StepKind,
)


@dataclass(eq=False)
class _FakeDeviceTensor:
    backing: torch.Tensor

    @property
    def data_ptr(self) -> int:
        return self.backing.data_ptr()

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.backing.shape)

    @property
    def dtype(self) -> torch.dtype:
        return self.backing.dtype

    @property
    def nbytes(self) -> int:
        return self.backing.numel() * self.backing.element_size()


class _FakeChipWorker:
    def __init__(self, config: Any) -> None:
        self.config = config
        self.allocated: dict[int, _FakeDeviceTensor] = {}
        self.alloc_calls: list[tuple[tuple[int, ...], torch.dtype, bool]] = []
        self.run_calls: list[tuple[Any, tuple[Any, ...], Any]] = []
        self.close_calls = 0

    def alloc_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        init: torch.Tensor | None = None,
        worker_id: int = 0,
    ) -> _FakeDeviceTensor:
        del worker_id
        backing = torch.empty(shape, dtype=dtype)
        if init is not None:
            backing.copy_(init)
        tensor = _FakeDeviceTensor(backing)
        self.allocated[tensor.data_ptr] = tensor
        self.alloc_calls.append((shape, dtype, init is not None))
        return tensor

    def free_tensor(self, tensor: _FakeDeviceTensor, *, worker_id: int = 0) -> None:
        del worker_id
        if self.allocated.pop(tensor.data_ptr, None) is None:
            raise RuntimeError("fake device tensor already freed")

    def copy_to(self, dst: int, src: int, nbytes: int, *, worker_id: int = 0) -> None:
        del worker_id
        ctypes.memmove(dst, src, nbytes)

    def copy_from(self, dst: int, src: int, nbytes: int, *, worker_id: int = 0) -> None:
        del worker_id
        ctypes.memmove(dst, src, nbytes)

    def run(self, compiled: Any, *args: Any, config: Any) -> None:
        self.run_calls.append((compiled, args, config))
        compiled(*args, config=config)

    def close(self) -> None:
        self.close_calls += 1


class _EmbeddingJitFn:
    def __init__(self) -> None:
        self.compile_calls: list[tuple[tuple[torch.Tensor, ...], Any]] = []

    def compile(self, *args: torch.Tensor, config: Any) -> Any:
        self.compile_calls.append((args, config))

        def compiled(
            input_ids: _FakeDeviceTensor,
            weight: _FakeDeviceTensor,
            out: _FakeDeviceTensor,
            *,
            config: Any,
        ) -> None:
            del config
            embedded = weight.backing[input_ids.backing.long()]
            out.backing.copy_(embedded.unsqueeze(2))

        return compiled


def _backend(*, keep_prefill_routed_staging: bool = False) -> tuple[WorkerBackend, _FakeChipWorker, dict[str, Any]]:
    runtime_config: dict[str, Any] = {}
    worker_holder: dict[str, _FakeChipWorker] = {}

    def run_config_factory(**kwargs: Any) -> dict[str, Any]:
        runtime_config.update(kwargs)
        return runtime_config

    def worker_factory(config: Any) -> _FakeChipWorker:
        worker = _FakeChipWorker(config)
        worker_holder["worker"] = worker
        return worker

    backend = WorkerBackend(
        platform="test",
        device_id=3,
        runtime_cfg={"enable_l2_swimlane": True},
        keep_prefill_routed_staging=keep_prefill_routed_staging,
        worker_factory=worker_factory,
        run_config_factory=run_config_factory,
    )
    return backend, worker_holder["worker"], runtime_config


def _embedding_specs(seq_len: int = 2) -> list[TensorSpec]:
    return [
        TensorSpec("input_ids", [1, seq_len], torch.int64, init_value=torch.zeros(1, seq_len)),
        TensorSpec("weight", [4, 3], torch.float32, init_value=torch.zeros(4, 3)),
        TensorSpec("out", [1, seq_len, 1, 3], torch.float32, is_output=True),
    ]


def _weight() -> RuntimeWeight:
    tensor = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    return RuntimeWeight(RuntimeWeightKey("embed.weight", tensor.dtype, "identity"), tensor)


def _run_embedding_step(
    backend: WorkerBackend,
    case: KernelCase,
    input_ids: torch.Tensor,
    weight: RuntimeWeight,
    *,
    start_pos: int,
) -> torch.Tensor:
    backend.begin_step(StepContext(StepKind.PREFILL, input_ids.shape[1], start_pos))
    try:
        specs = _embedding_specs(input_ids.shape[1])
        bindings = backend.materialize(specs, {"input_ids": input_ids, "weight": weight})
        outputs = backend.run(case, specs, bindings)
        return backend.export_output(outputs["out"])
    finally:
        backend.end_step()


def test_embedding_slice_reuses_fixed_weight_compile_and_step_buffers() -> None:
    backend, worker, runtime_config = _backend()
    jit_fn = _EmbeddingJitFn()
    case = KernelCase("embedding", jit_fn, _embedding_specs)
    weight = _weight()

    first = _run_embedding_step(backend, case, torch.tensor([[1, 3]]), weight, start_pos=0)
    first_stats = backend.pool_stats
    second = _run_embedding_step(backend, case, torch.tensor([[2, 0]]), weight, start_pos=2)

    torch.testing.assert_close(first, weight.host_tensor[torch.tensor([[1, 3]])].unsqueeze(2))
    torch.testing.assert_close(second, weight.host_tensor[torch.tensor([[2, 0]])].unsqueeze(2))
    assert runtime_config == {"platform": "test", "device_id": 3, "enable_l2_swimlane": True}
    assert worker.config is runtime_config
    assert len(jit_fn.compile_calls) == 1
    assert len(worker.run_calls) == 2
    assert backend.last_compile_cache_hit is True
    assert first_stats.category_bytes[AllocationCategory.FIXED_WEIGHT] == weight.host_tensor.numel() * 4
    assert backend.pool_stats.category_bytes[AllocationCategory.FIXED_WEIGHT] == weight.host_tensor.numel() * 4
    assert backend.pool_stats.reuse_count == 2
    assert backend.pool_stats.in_use_count == 1


def test_materialize_reuses_same_raw_host_tensor_within_step() -> None:
    backend, _, _ = _backend()
    host = torch.tensor([1.0, 2.0])
    specs = [
        TensorSpec("left", [2], torch.float32, init_value=torch.zeros(2)),
        TensorSpec("right", [2], torch.float32, init_value=torch.zeros(2)),
    ]
    backend.begin_step(StepContext(StepKind.PREFILL, 1, 0))

    bindings = backend.materialize(specs, {"left": host, "right": host})

    assert bindings.tensors["left"] is bindings.tensors["right"]
    backend.end_step()


def test_materialize_validates_required_host_values() -> None:
    backend, _, _ = _backend()
    required = TensorSpec("required", [2], torch.float32, init_value=torch.ones(2))
    backend.begin_step(StepContext(StepKind.PREFILL, 1, 0))
    try:
        with pytest.raises(KeyError, match="Missing backend tensors.*required"):
            backend.materialize([required], {})
        with pytest.raises(ValueError, match="required shape mismatch"):
            backend.materialize([required], {"required": torch.zeros(3)})
        with pytest.raises(TypeError, match="Host tensor or DeviceTensor-compatible"):
            backend.materialize([required], {"required": object()})
    finally:
        backend.end_step()


def test_prefill_staging_reuses_within_step_and_frees_at_step_end() -> None:
    backend, worker, _ = _backend()
    spec = TensorSpec("weight", [2], torch.float32, init_value=torch.zeros(2))
    staging = HostStagingTensor(torch.ones(2), StagingKind.PREFILL_ROUTED, "w1_t")
    backend.begin_step(StepContext(StepKind.PREFILL, 1, 0))

    class JitFn:
        def compile(self, *args: torch.Tensor, config: Any) -> Any:
            del args, config

            def compiled(*args: _FakeDeviceTensor, config: Any) -> None:
                del args, config

            return compiled

    case = KernelCase("staging", JitFn(), lambda: [spec])
    first = backend.materialize([spec], {"weight": staging})
    first_tensor = first.tensors["weight"]
    backend.run(case, [spec], first)
    second = backend.materialize([spec], {"weight": staging})
    backend.run(case, [spec], second)

    assert second.tensors["weight"] is first_tensor
    assert backend.pool_stats.reuse_count == 1
    assert backend.pool_stats.category_bytes[AllocationCategory.STAGING_ROUTED] == staging.host_tensor.numel() * 4
    backend.end_step()
    assert AllocationCategory.STAGING_ROUTED not in backend.pool_stats.category_bytes
    assert first_tensor.data_ptr not in worker.allocated


def test_prefill_staging_can_be_kept_idle_across_steps() -> None:
    backend, worker, _ = _backend(keep_prefill_routed_staging=True)
    spec = TensorSpec("weight", [2], torch.float32, init_value=torch.zeros(2))
    staging = HostStagingTensor(torch.ones(2), StagingKind.PREFILL_ROUTED, "w1_t")
    backend.begin_step(StepContext(StepKind.PREFILL, 1, 0))
    bindings = backend.materialize([spec], {"weight": staging})
    device_tensor = bindings.tensors["weight"]
    backend.end_step()

    assert backend.pool_stats.category_bytes[AllocationCategory.STAGING_ROUTED] == staging.host_tensor.numel() * 4
    assert device_tensor.data_ptr in worker.allocated
    backend.close()
    assert worker.allocated == {}


def test_block_dispatch_returns_prebound_outputs_and_releases_consumed_intermediate() -> None:
    backend, _, _ = _backend()
    backend.begin_step(StepContext(StepKind.PREFILL, 1, 0))

    class ProducerJit:
        def compile(self, *args: torch.Tensor, config: Any) -> Any:
            del args, config

            def compiled(out: _FakeDeviceTensor, *, config: Any) -> None:
                del config
                out.backing.fill_(2)

            return compiled

    producer_specs = [TensorSpec("out", [2], torch.float32, is_output=True)]
    producer_bindings = backend.materialize(producer_specs, {})
    hidden = backend.run(KernelCase("producer", ProducerJit(), lambda: producer_specs), producer_specs, producer_bindings)[
        "out"
    ]
    state_out = _FakeDeviceTensor(torch.zeros(2, dtype=torch.float32))
    routed = HostStagingTensor(torch.ones(2), StagingKind.PREFILL_ROUTED, "w1_t")

    class BlockJit:
        def compile(self, *args: torch.Tensor, config: Any) -> Any:
            del args, config

            def compiled(
                x: _FakeDeviceTensor,
                routed_w1_t: _FakeDeviceTensor,
                scratch: _FakeDeviceTensor,
                cache_out: _FakeDeviceTensor,
                out: _FakeDeviceTensor,
                *,
                config: Any,
            ) -> None:
                del scratch, config
                cache_out.backing.copy_(x.backing)
                out.backing.copy_(x.backing + routed_w1_t.backing)

            return compiled

    block_specs = [
        TensorSpec("x", [2], torch.float32, init_value=torch.zeros(2)),
        TensorSpec("routed_w1_t", [2], torch.float32, init_value=torch.zeros(2)),
        TensorSpec("scratch", [2], torch.float32),
        TensorSpec("cache_out", [2], torch.float32, is_output=True),
        TensorSpec("out", [2], torch.float32, is_output=True),
    ]
    block_bindings = backend.materialize(
        block_specs,
        {"x": hidden, "routed_w1_t": routed, "cache_out": state_out},
    )
    outputs = backend.run(KernelCase("block", BlockJit(), lambda: block_specs), block_specs, block_bindings)

    assert outputs["cache_out"] is state_out
    assert torch.equal(state_out.backing, torch.full((2,), 2.0))
    torch.testing.assert_close(backend.export_output(outputs["out"]), torch.full((2,), 3.0))
    assert backend.pool_stats.in_use_count == 0
    backend.end_step()
    assert AllocationCategory.STAGING_ROUTED not in backend.pool_stats.category_bytes


def test_selected_decode_reads_control_and_keeps_pre_post_intermediates_on_device() -> None:
    backend, _, _ = _backend()

    class PreJit:
        def compile(self, *args: torch.Tensor, config: Any) -> Any:
            del args, config

            def compiled(
                x: _FakeDeviceTensor,
                ffn_normed: _FakeDeviceTensor,
                weights: _FakeDeviceTensor,
                indices: _FakeDeviceTensor,
                *,
                config: Any,
            ) -> None:
                del config
                ffn_normed.backing.copy_(x.backing)
                weights.backing.fill_(0.25)
                indices.backing.copy_(torch.tensor([1, 0], dtype=torch.int32))

            return compiled

    class PostJit:
        def compile(self, *args: torch.Tensor, config: Any) -> Any:
            del args, config

            def compiled(
                ffn_normed: _FakeDeviceTensor,
                weights: _FakeDeviceTensor,
                selected_w1_t: _FakeDeviceTensor,
                out: _FakeDeviceTensor,
                *,
                config: Any,
            ) -> None:
                del config
                out.backing.copy_(ffn_normed.backing + weights.backing + selected_w1_t.backing)

            return compiled

    pre_specs = [
        TensorSpec("x", [2], torch.float32, init_value=torch.zeros(2)),
        TensorSpec("ffn_normed", [2], torch.float32, is_output=True),
        TensorSpec("weights", [2], torch.float32, is_output=True),
        TensorSpec("indices", [2], torch.int32, is_output=True),
    ]
    post_specs = [
        TensorSpec("ffn_normed", [2], torch.float32, init_value=torch.zeros(2)),
        TensorSpec("weights", [2], torch.float32, init_value=torch.zeros(2)),
        TensorSpec("selected_w1_t", [2], torch.float32, init_value=torch.zeros(2)),
        TensorSpec("out", [2], torch.float32, is_output=True),
    ]
    pre_case = KernelCase("selected_pre", PreJit(), lambda: pre_specs)
    post_case = KernelCase("selected_post", PostJit(), lambda: post_specs)

    def run_step(value: float, start_pos: int) -> torch.Tensor:
        backend.begin_step(StepContext(StepKind.DECODE, 1, start_pos))
        try:
            pre_bindings = backend.materialize(pre_specs, {"x": torch.full((2,), value)})
            pre_outputs = backend.run(pre_case, pre_specs, pre_bindings)
            indices = backend.read_control(pre_outputs["indices"])
            assert torch.equal(indices, torch.tensor([1, 0], dtype=torch.int32))
            selected = HostStagingTensor(
                torch.full((2,), value + 1),
                StagingKind.DECODE_SELECTED,
                "w1_t",
            )
            post_bindings = backend.materialize(
                post_specs,
                {
                    "ffn_normed": pre_outputs["ffn_normed"],
                    "weights": pre_outputs["weights"],
                    "selected_w1_t": selected,
                },
            )
            post_outputs = backend.run(post_case, post_specs, post_bindings)
            # The final output and the step-scoped raw input remain active.
            assert backend.pool_stats.in_use_count == 2
            return backend.export_output(post_outputs["out"])
        finally:
            backend.end_step()

    first = run_step(2.0, 1)
    reuse_after_first = backend.pool_stats.reuse_count
    second = run_step(4.0, 2)

    torch.testing.assert_close(first, torch.full((2,), 5.25))
    torch.testing.assert_close(second, torch.full((2,), 9.25))
    assert backend.pool_stats.reuse_count > reuse_after_first
    assert backend.pool_stats.category_bytes[AllocationCategory.STAGING_SELECTED] == 2 * 4
    assert backend.pool_stats.in_use_count == 0
    expected_d2h = 2 * ((2 * 4) + (2 * 4))
    assert backend.pool_stats.d2h_bytes == expected_d2h


def test_worker_bindings_are_single_use_and_scratch_is_released() -> None:
    backend, _, _ = _backend()

    class JitFn:
        def compile(self, *args: torch.Tensor, config: Any) -> Any:
            del args, config

            def compiled(*args: _FakeDeviceTensor, config: Any) -> None:
                del args, config

            return compiled

    specs = [
        TensorSpec("scratch", [2], torch.float32),
        TensorSpec("out", [2], torch.float32, is_output=True),
    ]
    backend.begin_step(StepContext(StepKind.PREFILL, 1, 0))
    bindings = backend.materialize(specs, {})
    assert isinstance(bindings, WorkerKernelBindings)

    backend.run(KernelCase("scratch", JitFn(), lambda: specs), specs, bindings)
    with pytest.raises(RuntimeError, match="already been consumed"):
        backend.run(KernelCase("scratch", JitFn(), lambda: specs), specs, bindings)
    assert backend.pool_stats.in_use_count == 1
    backend.end_step()
    assert backend.pool_stats.in_use_count == 0


def test_close_cleans_active_step_state_and_worker_once() -> None:
    backend, worker, _ = _backend()
    backend.prepare_state([])
    backend.begin_step(StepContext(StepKind.PREFILL, 1, 0))
    backend.materialize(
        [TensorSpec("input", [1], torch.float32, init_value=torch.zeros(1))],
        {"input": torch.ones(1)},
    )

    backend.close()
    backend.close()

    assert worker.allocated == {}
    assert worker.close_calls == 1
    assert backend.pool_stats.current_bytes == 0
    with pytest.raises(RuntimeError, match="closed"):
        backend.begin_step(StepContext(StepKind.PREFILL, 1, 0))
