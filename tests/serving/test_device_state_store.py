import ctypes
from dataclasses import dataclass

import pytest
import torch

from serving.device_pool import AllocationCategory, DeviceBufferPool
from serving.device_state_store import DeviceStateStore
from serving.state import DeepSeekV4StatePlan, LayerSpec, LayerStateSchema, StateTensorSpec


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
    def __init__(self) -> None:
        self.allocated: dict[int, _FakeDeviceTensor] = {}
        self.free_count = 0

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
        return tensor

    def free_tensor(self, tensor: _FakeDeviceTensor, *, worker_id: int = 0) -> None:
        del worker_id
        if self.allocated.pop(tensor.data_ptr, None) is None:
            raise RuntimeError("fake device tensor already freed")
        self.free_count += 1

    def copy_to(self, dst: int, src: int, nbytes: int, *, worker_id: int = 0) -> None:
        del worker_id
        ctypes.memmove(dst, src, nbytes)

    def copy_from(self, dst: int, src: int, nbytes: int, *, worker_id: int = 0) -> None:
        del worker_id
        ctypes.memmove(dst, src, nbytes)


def _schema(*tensor_specs: StateTensorSpec, layer_id: int = 0) -> LayerStateSchema:
    return LayerStateSchema(
        spec=LayerSpec(layer_id=layer_id, ratio=0, hash_route=True),
        tensors=tuple(tensor_specs),
    )


def _tensor_spec(
    name: str = "cache",
    *,
    input_name: str = "cache",
    output_name: str = "cache_out",
    init_value: float = 0,
) -> StateTensorSpec:
    return StateTensorSpec(
        name=name,
        input_name=input_name,
        output_name=output_name,
        shape=(2,),
        dtype=torch.float32,
        init_value=init_value,
    )


def test_prepare_allocates_initialized_current_and_distinct_next_buffers() -> None:
    worker = _FakeChipWorker()
    pool = DeviceBufferPool(worker)
    store = DeviceStateStore(pool)
    spec = _tensor_spec(init_value=-7)

    store.prepare([_schema(spec)])
    inputs = store.inputs(0)
    outputs = store.outputs(0)

    assert inputs["cache"] is not outputs["cache_out"]
    torch.testing.assert_close(inputs["cache"].backing, torch.full((2,), -7.0))
    assert pool.stats.allocation_count == 2
    assert pool.stats.category_bytes == {AllocationCategory.STATE: 2 * 2 * 4}
    assert pool.stats.h2d_bytes == 2 * 4


def test_commit_validates_all_outputs_before_atomic_swap() -> None:
    pool = DeviceBufferPool(_FakeChipWorker())
    store = DeviceStateStore(pool)
    first = _tensor_spec()
    second = _tensor_spec("scores", input_name="scores", output_name="scores_out")
    store.prepare([_schema(first, second)])
    initial_inputs = store.inputs(0)
    initial_outputs = store.outputs(0)

    with pytest.raises(KeyError, match="scores_out"):
        store.commit(0, {"cache_out": initial_outputs["cache_out"]})
    assert store.inputs(0) == initial_inputs

    with pytest.raises(ValueError, match="not the bound next buffer"):
        store.commit(
            0,
            {
                "cache_out": initial_outputs["cache_out"],
                "scores_out": initial_inputs["scores"],
            },
        )
    assert store.inputs(0) == initial_inputs

    store.commit(0, initial_outputs)

    assert store.inputs(0)["cache"] is initial_outputs["cache_out"]
    assert store.inputs(0)["scores"] is initial_outputs["scores_out"]
    assert store.outputs(0)["cache_out"] is initial_inputs["cache"]
    assert store.outputs(0)["scores_out"] is initial_inputs["scores"]


def test_multiple_commits_swap_the_same_two_device_buffers() -> None:
    pool = DeviceBufferPool(_FakeChipWorker())
    store = DeviceStateStore(pool)
    store.prepare([_schema(_tensor_spec())])
    first_current = store.inputs(0)["cache"]
    first_next = store.outputs(0)["cache_out"]

    store.commit(0, store.outputs(0))
    store.commit(0, store.outputs(0))

    assert store.inputs(0)["cache"] is first_current
    assert store.outputs(0)["cache_out"] is first_next
    assert pool.stats.allocation_count == 2


def test_real_state_plan_ratio_schemas_allocate_expected_pairs_and_initial_values() -> None:
    pool = DeviceBufferPool(_FakeChipWorker())
    store = DeviceStateStore(pool)
    plan = DeepSeekV4StatePlan()
    schemas = [
        plan.layer_state_schema(0),
        plan.layer_state_schema(2),
        plan.layer_state_schema(3),
    ]

    store.prepare(schemas)

    assert len(store.inputs(0)) == 1
    assert len(store.inputs(2)) == 7
    assert len(store.inputs(3)) == 4
    assert pool.stats.allocation_count == 2 * (1 + 7 + 4)
    assert pool.stats.category_bytes[AllocationCategory.STATE] == 2 * sum(
        tensor.numel() * tensor.element_size()
        for schema in schemas
        for tensor in (spec.create_tensor() for spec in schema.tensors)
    )
    ratio4_score = store.inputs(2)["attn_comp_score_state"]
    assert torch.all(ratio4_score.backing == -torch.finfo(torch.float32).max)
    ratio128_score = store.inputs(3)["comp_score_state"]
    assert torch.all(ratio128_score.backing == -torch.finfo(torch.float32).max)


@pytest.mark.parametrize(
    ("schemas", "message"),
    [
        ([_schema(_tensor_spec()), _schema(_tensor_spec())], "Duplicate state schema"),
        ([_schema(_tensor_spec(), _tensor_spec())], "Duplicate state tensor"),
        (
            [_schema(_tensor_spec(), _tensor_spec("other", output_name="other_out"))],
            "Duplicate state input",
        ),
        (
            [_schema(_tensor_spec(), _tensor_spec("other", input_name="other"))],
            "Duplicate state output",
        ),
    ],
)
def test_prepare_rejects_duplicate_schema_names_before_allocation(
    schemas: list[LayerStateSchema],
    message: str,
) -> None:
    pool = DeviceBufferPool(_FakeChipWorker())
    store = DeviceStateStore(pool)

    with pytest.raises(ValueError, match=message):
        store.prepare(schemas)

    assert pool.stats.allocation_count == 0


def test_prepare_rejects_second_prepare_and_unknown_layer() -> None:
    pool = DeviceBufferPool(_FakeChipWorker())
    store = DeviceStateStore(pool)
    schema = _schema(_tensor_spec())
    store.prepare([schema])

    with pytest.raises(RuntimeError, match="already prepared"):
        store.prepare([schema])
    with pytest.raises(ValueError, match="No state schema"):
        store.inputs(1)


def test_close_frees_state_only_is_idempotent_and_allows_reprepare() -> None:
    worker = _FakeChipWorker()
    pool = DeviceBufferPool(worker)
    store = DeviceStateStore(pool)
    scratch = pool.acquire((3,), torch.float32, category=AllocationCategory.SCRATCH)
    store.prepare([_schema(_tensor_spec())])

    store.close()
    store.close()

    assert pool.stats.allocation_count == 1
    assert pool.stats.category_bytes == {AllocationCategory.SCRATCH: scratch.nbytes}
    assert worker.free_count == 2
    with pytest.raises(RuntimeError, match="not prepared"):
        store.inputs(0)

    store.prepare([_schema(_tensor_spec())])
    assert pool.stats.allocation_count == 3
