"""Owned DeviceTensor allocation, reuse, copy, and accounting."""

from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

import torch


class AllocationCategory(Enum):
    """Lifetime category for a WorkerBackend-owned device allocation."""

    FIXED_WEIGHT = "fixed_weight"
    STATE = "state"
    ACTIVE_UPLOAD = "active_upload"
    INTERMEDIATE = "intermediate"
    SCRATCH = "scratch"
    STAGING_ROUTED = "staging_routed"
    STAGING_SELECTED = "staging_selected"


_PERSISTENT_CATEGORIES = frozenset(
    {
        AllocationCategory.FIXED_WEIGHT,
        AllocationCategory.STATE,
    }
)


@dataclass(frozen=True, eq=False)
class DeviceLease:
    """An opaque lease for one pool-owned DeviceTensor."""

    tensor: Any
    category: AllocationCategory
    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int
    reusable: bool
    reuse_key: Hashable | None
    _pool_token: object
    _allocation_id: int


@dataclass(frozen=True)
class DevicePoolStats:
    """A point-in-time snapshot of pool allocation and copy counters."""

    allocation_count: int
    in_use_count: int
    alloc_count: int
    free_count: int
    reuse_count: int
    h2d_bytes: int
    d2h_bytes: int
    current_bytes: int
    active_bytes: int
    peak_bytes: int
    category_bytes: Mapping[AllocationCategory, int]


@dataclass
class _AllocationRecord:
    lease: DeviceLease
    in_use: bool


class DeviceBufferPool:
    """Own and reuse DeviceTensors allocated by one ChipWorker.

    The pool never closes the worker itself.  The owning WorkerBackend must
    close this pool before calling ``ChipWorker.close()``.
    """

    def __init__(self, worker: Any, *, worker_id: int = 0) -> None:
        self._worker = worker
        self._worker_id = int(worker_id)
        self._pool_token = object()
        self._next_allocation_id = 0
        self._records: dict[int, _AllocationRecord] = {}
        self._free_lists: dict[tuple[Any, ...], list[int]] = defaultdict(list)
        self._alloc_count = 0
        self._free_count = 0
        self._reuse_count = 0
        self._h2d_bytes = 0
        self._d2h_bytes = 0
        self._current_bytes = 0
        self._peak_bytes = 0
        self._closed = False

    @property
    def stats(self) -> DevicePoolStats:
        category_bytes: dict[AllocationCategory, int] = defaultdict(int)
        active_bytes = 0
        in_use_count = 0
        for record in self._records.values():
            category_bytes[record.lease.category] += record.lease.nbytes
            if record.in_use:
                in_use_count += 1
                active_bytes += record.lease.nbytes
        return DevicePoolStats(
            allocation_count=len(self._records),
            in_use_count=in_use_count,
            alloc_count=self._alloc_count,
            free_count=self._free_count,
            reuse_count=self._reuse_count,
            h2d_bytes=self._h2d_bytes,
            d2h_bytes=self._d2h_bytes,
            current_bytes=self._current_bytes,
            active_bytes=active_bytes,
            peak_bytes=self._peak_bytes,
            category_bytes=dict(category_bytes),
        )

    def allocate_persistent(
        self,
        shape: Sequence[int],
        dtype: torch.dtype,
        *,
        category: AllocationCategory,
        init: torch.Tensor | None = None,
    ) -> DeviceLease:
        """Allocate a non-reusable fixed-weight or state buffer."""
        self._require_open()
        if category not in _PERSISTENT_CATEGORIES:
            raise ValueError(f"persistent allocation does not support category {category.value!r}")
        return self._allocate(shape, dtype, category=category, init=init, reusable=False, reuse_key=None)

    def acquire(
        self,
        shape: Sequence[int],
        dtype: torch.dtype,
        *,
        category: AllocationCategory,
        reuse_key: Hashable | None = None,
        init: torch.Tensor | None = None,
    ) -> DeviceLease:
        """Acquire a reusable buffer, copying ``init`` on both miss and hit."""
        self._require_open()
        if category in _PERSISTENT_CATEGORIES:
            raise ValueError(f"reusable allocation does not support category {category.value!r}")
        normalized_shape = self._normalize_shape(shape)
        self._validate_reuse_key(reuse_key)
        key = self._free_list_key(category, reuse_key, normalized_shape, dtype)
        free_list = self._free_lists.get(key)
        if free_list:
            allocation_id = free_list.pop()
            if not free_list:
                self._free_lists.pop(key, None)
            record = self._records[allocation_id]
            if record.in_use:
                raise RuntimeError(f"free-list allocation {allocation_id} is already in use")
            record.in_use = True
            self._reuse_count += 1
            if init is not None:
                self.copy_to(record.lease, init)
            return record.lease
        return self._allocate(
            normalized_shape,
            dtype,
            category=category,
            init=init,
            reusable=True,
            reuse_key=reuse_key,
        )

    def release(self, lease: DeviceLease) -> None:
        """Return an acquired reusable buffer to its exact-match free list."""
        self._require_open()
        record = self._record_for(lease)
        if not lease.reusable:
            raise ValueError("persistent allocations cannot be released to a reuse free list")
        if not record.in_use:
            raise RuntimeError(f"allocation {lease._allocation_id} has already been released")
        record.in_use = False
        key = self._free_list_key(lease.category, lease.reuse_key, lease.shape, lease.dtype)
        self._free_lists[key].append(lease._allocation_id)

    def copy_to(self, lease: DeviceLease, host_tensor: torch.Tensor) -> None:
        """Copy one exact-shape Host tensor into a pool-owned DeviceTensor."""
        self._require_open()
        self._record_for(lease, require_in_use=True)
        host = self._validate_host_tensor(lease, host_tensor)
        self._worker.copy_to(
            self._device_data_ptr(lease.tensor),
            host.data_ptr(),
            lease.nbytes,
            worker_id=self._worker_id,
        )
        self._h2d_bytes += lease.nbytes

    def copy_from(self, lease: DeviceLease, *, out: torch.Tensor | None = None) -> torch.Tensor:
        """Copy one exact-shape DeviceTensor into a contiguous Host tensor."""
        self._require_open()
        self._record_for(lease, require_in_use=True)
        host = torch.empty(lease.shape, dtype=lease.dtype) if out is None else out
        host = self._validate_host_tensor(lease, host, require_contiguous=True)
        self._worker.copy_from(
            host.data_ptr(),
            self._device_data_ptr(lease.tensor),
            lease.nbytes,
            worker_id=self._worker_id,
        )
        self._d2h_bytes += lease.nbytes
        return host

    def free(self, lease: DeviceLease) -> None:
        """Free one persistent allocation or one already-released reusable allocation."""
        self._require_open()
        record = self._record_for(lease)
        if lease.reusable and record.in_use:
            raise RuntimeError(f"reusable allocation {lease._allocation_id} must be released before free")
        self._free_record(record)

    def close(self) -> None:
        """Free every owned allocation.  Repeated close calls are harmless."""
        if self._closed:
            return
        for allocation_id in list(self._records):
            self._free_record(self._records[allocation_id], force=True)
        self._free_lists.clear()
        self._closed = True

    def _allocate(
        self,
        shape: Sequence[int],
        dtype: torch.dtype,
        *,
        category: AllocationCategory,
        init: torch.Tensor | None,
        reusable: bool,
        reuse_key: Hashable | None,
    ) -> DeviceLease:
        normalized_shape = self._normalize_shape(shape)
        nbytes = self._tensor_nbytes(normalized_shape, dtype)
        host_init = None
        if init is not None:
            host_init = self._validate_host_shape_dtype(normalized_shape, dtype, init).contiguous()
        tensor = self._worker.alloc_tensor(
            normalized_shape,
            dtype,
            init=host_init,
            worker_id=self._worker_id,
        )
        try:
            self._validate_device_tensor(tensor, normalized_shape, dtype, nbytes)
        except BaseException:
            self._worker.free_tensor(tensor, worker_id=self._worker_id)
            raise
        allocation_id = self._next_allocation_id
        self._next_allocation_id += 1
        lease = DeviceLease(
            tensor=tensor,
            category=category,
            shape=normalized_shape,
            dtype=dtype,
            nbytes=nbytes,
            reusable=reusable,
            reuse_key=reuse_key,
            _pool_token=self._pool_token,
            _allocation_id=allocation_id,
        )
        self._records[allocation_id] = _AllocationRecord(lease=lease, in_use=True)
        self._alloc_count += 1
        if host_init is not None:
            self._h2d_bytes += nbytes
        self._current_bytes += nbytes
        self._peak_bytes = max(self._peak_bytes, self._current_bytes)
        return lease

    def _free_record(self, record: _AllocationRecord, *, force: bool = False) -> None:
        lease = record.lease
        if not force and lease.reusable and record.in_use:
            raise RuntimeError(f"reusable allocation {lease._allocation_id} is still in use")
        self._worker.free_tensor(lease.tensor, worker_id=self._worker_id)
        self._records.pop(lease._allocation_id)
        if lease.reusable:
            key = self._free_list_key(lease.category, lease.reuse_key, lease.shape, lease.dtype)
            free_list = self._free_lists.get(key)
            if free_list and lease._allocation_id in free_list:
                free_list.remove(lease._allocation_id)
                if not free_list:
                    self._free_lists.pop(key, None)
        self._free_count += 1
        self._current_bytes -= lease.nbytes

    def _record_for(self, lease: DeviceLease, *, require_in_use: bool = False) -> _AllocationRecord:
        if not isinstance(lease, DeviceLease) or lease._pool_token is not self._pool_token:
            raise ValueError("device lease does not belong to this pool")
        record = self._records.get(lease._allocation_id)
        if record is None or record.lease is not lease:
            raise RuntimeError(f"allocation {lease._allocation_id} has already been freed")
        if require_in_use and not record.in_use:
            raise RuntimeError(f"allocation {lease._allocation_id} is not in use")
        return record

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("device buffer pool is closed")

    @staticmethod
    def _normalize_shape(shape: Sequence[int]) -> tuple[int, ...]:
        normalized = tuple(int(dim) for dim in shape)
        if any(dim < 0 for dim in normalized):
            raise ValueError(f"device tensor shape must be non-negative, got {normalized}")
        return normalized

    @staticmethod
    def _tensor_nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
        return math.prod(shape) * torch.empty((), dtype=dtype).element_size()

    @staticmethod
    def _validate_reuse_key(reuse_key: Hashable | None) -> None:
        try:
            hash(reuse_key)
        except TypeError as exc:
            raise TypeError("reuse_key must be hashable") from exc

    @staticmethod
    def _free_list_key(
        category: AllocationCategory,
        reuse_key: Hashable | None,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> tuple[Any, ...]:
        return category, reuse_key, shape, dtype

    @staticmethod
    def _validate_host_shape_dtype(
        shape: tuple[int, ...],
        dtype: torch.dtype,
        host_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(host_tensor, torch.Tensor):
            raise TypeError(f"expected a Host torch.Tensor, got {type(host_tensor)!r}")
        if host_tensor.device.type != "cpu":
            raise ValueError(f"expected a CPU Host tensor, got device {host_tensor.device}")
        if tuple(host_tensor.shape) != shape:
            raise ValueError(f"Host tensor shape mismatch: expected {shape}, got {tuple(host_tensor.shape)}")
        if host_tensor.dtype != dtype:
            raise TypeError(f"Host tensor dtype mismatch: expected {dtype}, got {host_tensor.dtype}")
        return host_tensor

    @classmethod
    def _validate_host_tensor(
        cls,
        lease: DeviceLease,
        host_tensor: torch.Tensor,
        *,
        require_contiguous: bool = False,
    ) -> torch.Tensor:
        host = cls._validate_host_shape_dtype(lease.shape, lease.dtype, host_tensor)
        if require_contiguous and not host.is_contiguous():
            raise ValueError("Host output tensor must be contiguous")
        return host.contiguous()

    @staticmethod
    def _device_data_ptr(tensor: Any) -> int:
        data_ptr = getattr(tensor, "data_ptr", None)
        data_ptr = data_ptr() if callable(data_ptr) else data_ptr
        if not isinstance(data_ptr, int):
            raise TypeError("DeviceTensor.data_ptr must be an integer")
        return data_ptr

    @staticmethod
    def _validate_device_tensor(
        tensor: Any,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        nbytes: int,
    ) -> None:
        if hasattr(tensor, "shape") and tuple(tensor.shape) != shape:
            raise ValueError(f"allocated DeviceTensor shape mismatch: expected {shape}, got {tuple(tensor.shape)}")
        if hasattr(tensor, "dtype") and tensor.dtype != dtype:
            raise TypeError(f"allocated DeviceTensor dtype mismatch: expected {dtype}, got {tensor.dtype}")
        device_nbytes = getattr(tensor, "nbytes", nbytes)
        device_nbytes = device_nbytes() if callable(device_nbytes) else device_nbytes
        if int(device_nbytes) != nbytes:
            raise ValueError(f"allocated DeviceTensor nbytes mismatch: expected {nbytes}, got {device_nbytes}")
        DeviceBufferPool._device_data_ptr(tensor)


__all__ = [
    "AllocationCategory",
    "DeviceBufferPool",
    "DeviceLease",
    "DevicePoolStats",
]
