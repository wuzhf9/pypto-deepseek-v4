import ctypes
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from serving.device_pool import AllocationCategory, DeviceBufferPool


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
        self.alloc_calls: list[tuple[tuple[int, ...], torch.dtype, torch.Tensor | None, int]] = []
        self.free_calls: list[tuple[_FakeDeviceTensor, int]] = []
        self.copy_to_calls: list[tuple[int, int, int, int]] = []
        self.copy_from_calls: list[tuple[int, int, int, int]] = []

    def alloc_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        init: torch.Tensor | None = None,
        worker_id: int = 0,
    ) -> _FakeDeviceTensor:
        backing = torch.empty(shape, dtype=dtype)
        if init is not None:
            backing.copy_(init)
        tensor = _FakeDeviceTensor(backing)
        self.allocated[tensor.data_ptr] = tensor
        self.alloc_calls.append((shape, dtype, init, worker_id))
        return tensor

    def free_tensor(self, tensor: _FakeDeviceTensor, *, worker_id: int = 0) -> None:
        if self.allocated.pop(tensor.data_ptr, None) is None:
            raise RuntimeError("fake device tensor already freed")
        self.free_calls.append((tensor, worker_id))

    def copy_to(self, dst: int, src: int, nbytes: int, *, worker_id: int = 0) -> None:
        ctypes.memmove(dst, src, nbytes)
        self.copy_to_calls.append((dst, src, nbytes, worker_id))

    def copy_from(self, dst: int, src: int, nbytes: int, *, worker_id: int = 0) -> None:
        ctypes.memmove(dst, src, nbytes)
        self.copy_from_calls.append((dst, src, nbytes, worker_id))


def test_persistent_allocation_tracks_init_bytes_and_stats() -> None:
    worker = _FakeChipWorker()
    pool = DeviceBufferPool(worker, worker_id=3)
    init = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    lease = pool.allocate_persistent(
        init.shape,
        init.dtype,
        category=AllocationCategory.FIXED_WEIGHT,
        init=init,
    )

    assert torch.equal(lease.tensor.backing, init)
    assert worker.alloc_calls[0][3] == 3
    assert pool.stats.alloc_count == 1
    assert pool.stats.h2d_bytes == init.numel() * init.element_size()
    assert pool.stats.current_bytes == lease.nbytes
    assert pool.stats.peak_bytes == lease.nbytes
    assert pool.stats.category_bytes == {AllocationCategory.FIXED_WEIGHT: lease.nbytes}


def test_reusable_allocation_reuses_exact_key_and_copies_new_init() -> None:
    worker = _FakeChipWorker()
    pool = DeviceBufferPool(worker)
    first_init = torch.tensor([1.0, 2.0])
    second_init = torch.tensor([3.0, 4.0])

    first = pool.acquire(
        (2,),
        torch.float32,
        category=AllocationCategory.STAGING_SELECTED,
        reuse_key="w1_t",
        init=first_init,
    )
    pool.release(first)
    second = pool.acquire(
        (2,),
        torch.float32,
        category=AllocationCategory.STAGING_SELECTED,
        reuse_key="w1_t",
        init=second_init,
    )

    assert second is first
    assert len(worker.alloc_calls) == 1
    assert len(worker.copy_to_calls) == 1
    assert torch.equal(second.tensor.backing, second_init)
    assert pool.stats.reuse_count == 1
    assert pool.stats.h2d_bytes == 2 * second.nbytes


@pytest.mark.parametrize(
    ("category", "reuse_key", "shape", "dtype"),
    [
        (AllocationCategory.SCRATCH, "w1_t", (2,), torch.float32),
        (AllocationCategory.STAGING_SELECTED, "w2_t", (2,), torch.float32),
        (AllocationCategory.STAGING_SELECTED, "w1_t", (3,), torch.float32),
        (AllocationCategory.STAGING_SELECTED, "w1_t", (2,), torch.bfloat16),
    ],
)
def test_reusable_allocation_does_not_alias_different_pool_key(
    category: AllocationCategory,
    reuse_key: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    worker = _FakeChipWorker()
    pool = DeviceBufferPool(worker)
    first = pool.acquire(
        (2,),
        torch.float32,
        category=AllocationCategory.STAGING_SELECTED,
        reuse_key="w1_t",
    )
    pool.release(first)

    second = pool.acquire(shape, dtype, category=category, reuse_key=reuse_key)

    assert second is not first
    assert len(worker.alloc_calls) == 2


def test_release_rejects_duplicate_persistent_and_cross_pool_leases() -> None:
    first_pool = DeviceBufferPool(_FakeChipWorker())
    second_pool = DeviceBufferPool(_FakeChipWorker())
    reusable = first_pool.acquire((1,), torch.float32, category=AllocationCategory.SCRATCH)
    persistent = first_pool.allocate_persistent(
        (1,), torch.float32, category=AllocationCategory.STATE
    )

    first_pool.release(reusable)
    with pytest.raises(RuntimeError, match="already been released"):
        first_pool.release(reusable)
    with pytest.raises(ValueError, match="persistent allocations"):
        first_pool.release(persistent)
    with pytest.raises(ValueError, match="does not belong"):
        second_pool.release(reusable)


def test_copy_round_trip_and_shape_dtype_validation() -> None:
    worker = _FakeChipWorker()
    pool = DeviceBufferPool(worker, worker_id=2)
    lease = pool.acquire((2, 3), torch.float32, category=AllocationCategory.ACTIVE_UPLOAD)
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    pool.copy_to(lease, source)
    out = pool.copy_from(lease)

    assert torch.equal(out, source)
    assert pool.stats.h2d_bytes == lease.nbytes
    assert pool.stats.d2h_bytes == lease.nbytes
    assert worker.copy_to_calls[0][3] == 2
    assert worker.copy_from_calls[0][3] == 2
    with pytest.raises(ValueError, match="shape mismatch"):
        pool.copy_to(lease, torch.zeros(6, dtype=torch.float32))
    with pytest.raises(TypeError, match="dtype mismatch"):
        pool.copy_to(lease, torch.zeros(2, 3, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="must be contiguous"):
        pool.copy_from(lease, out=torch.empty(3, 2).t())


def test_free_requires_reusable_release_and_detects_freed_lease() -> None:
    worker = _FakeChipWorker()
    pool = DeviceBufferPool(worker)
    lease = pool.acquire((4,), torch.bfloat16, category=AllocationCategory.INTERMEDIATE)

    with pytest.raises(RuntimeError, match="must be released"):
        pool.free(lease)
    pool.release(lease)
    pool.free(lease)

    assert pool.stats.current_bytes == 0
    assert pool.stats.free_count == 1
    with pytest.raises(RuntimeError, match="already been freed"):
        pool.free(lease)


def test_category_mode_and_reuse_key_validation() -> None:
    pool = DeviceBufferPool(_FakeChipWorker())

    with pytest.raises(ValueError, match="persistent allocation"):
        pool.allocate_persistent((1,), torch.float32, category=AllocationCategory.SCRATCH)
    with pytest.raises(ValueError, match="reusable allocation"):
        pool.acquire((1,), torch.float32, category=AllocationCategory.STATE)
    with pytest.raises(TypeError, match="reuse_key must be hashable"):
        pool.acquire((1,), torch.float32, category=AllocationCategory.SCRATCH, reuse_key=[])
    with pytest.raises(ValueError, match="non-negative"):
        pool.acquire((-1,), torch.float32, category=AllocationCategory.SCRATCH)


def test_close_frees_active_and_idle_allocations_and_is_idempotent() -> None:
    worker = _FakeChipWorker()
    pool = DeviceBufferPool(worker)
    persistent = pool.allocate_persistent(
        (2,), torch.float32, category=AllocationCategory.FIXED_WEIGHT
    )
    active = pool.acquire((3,), torch.float32, category=AllocationCategory.INTERMEDIATE)
    idle = pool.acquire((4,), torch.float32, category=AllocationCategory.SCRATCH)
    pool.release(idle)
    peak_bytes = persistent.nbytes + active.nbytes + idle.nbytes

    pool.close()
    pool.close()

    assert len(worker.free_calls) == 3
    assert worker.allocated == {}
    assert pool.stats.current_bytes == 0
    assert pool.stats.peak_bytes == peak_bytes
    assert pool.stats.free_count == 3
    with pytest.raises(RuntimeError, match="closed"):
        pool.acquire((1,), torch.float32, category=AllocationCategory.SCRATCH)
