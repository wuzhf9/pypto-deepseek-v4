"""Device-resident serving runtime backed by one ChipWorker."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import time
from typing import Any

import torch

from models.golden import TensorSpec
from serving.device_pool import AllocationCategory, DeviceBufferPool, DeviceLease
from serving.device_state_store import DeviceStateStore
from serving.runtime_types import HostStagingTensor, KernelCase, RuntimeWeight, RuntimeWeightKey, StagingKind, StepContext
from serving.state import LayerStateSchema


@dataclass
class KernelBindings:
    """Device tensors and leases owned by one kernel dispatch."""

    tensors: Mapping[str, Any]
    scratch_leases: tuple[DeviceLease, ...] = ()
    transient_leases: tuple[DeviceLease, ...] = ()
    consumed_leases: tuple[DeviceLease, ...] = ()
    output_tensors: Mapping[str, Any] = field(default_factory=dict)
    consumed: bool = False


class DeviceRuntime:
    """Execute serving kernels through one long-lived ChipWorker."""

    def __init__(
        self,
        *,
        platform: str,
        device_id: int,
        runtime_cfg: dict[str, Any] | None = None,
        keep_prefill_routed_staging: bool = False,
        worker_factory: Callable[[Any], Any] | None = None,
        run_config_factory: Callable[..., Any] | None = None,
    ) -> None:
        runtime_cfg = dict(runtime_cfg or {})
        if run_config_factory is None or worker_factory is None:
            from pypto.runtime import ChipWorker, RunConfig

            run_config_factory = run_config_factory or RunConfig
            worker_factory = worker_factory or ChipWorker
        self._run_config = run_config_factory(
            platform=platform,
            device_id=device_id,
            **runtime_cfg,
        )
        self._worker = worker_factory(self._run_config)
        self._pool = DeviceBufferPool(self._worker)
        self._state_store = DeviceStateStore(self._pool)
        self._compiled: dict[tuple[str, tuple[tuple[int, ...], ...], tuple[torch.dtype, ...]], Any] = {}
        self._fixed_weights: dict[RuntimeWeightKey, DeviceLease] = {}
        self._owned_leases: dict[int, DeviceLease] = {}
        self._active_upload_cache: dict[tuple[Any, ...], DeviceLease] = {}
        self._step_leases: dict[int, DeviceLease] = {}
        self._prefill_staging_leases: dict[int, DeviceLease] = {}
        self._keep_prefill_routed_staging = bool(keep_prefill_routed_staging)
        self._active_step: StepContext | None = None
        self._closed = False
        self.last_compile_seconds = 0.0
        self.last_run_seconds = 0.0
        self.last_compile_cache_hit = False

    @property
    def pool_stats(self) -> Any:
        return self._pool.stats

    def materialize(
        self,
        specs: list[TensorSpec],
        values: Mapping[str, Any],
    ) -> KernelBindings:
        self._require_active_step()
        tensors: dict[str, Any] = {}
        scratch_leases: list[DeviceLease] = []
        transient_leases: list[DeviceLease] = []
        consumed_leases: dict[int, DeviceLease] = {}
        output_tensors: dict[str, Any] = {}
        missing_required: list[str] = []
        for spec in specs:
            value = values.get(spec.name)
            if value is not None:
                tensor, transient = self._materialize_value(spec, value)
                tensors[spec.name] = tensor
                if transient is not None:
                    transient_leases.append(transient)
                consumed = self._consumed_intermediate(tensor)
                if consumed is not None and not spec.is_output:
                    consumed_leases[id(consumed)] = consumed
            elif spec.is_output or spec.init_value is None:
                category = AllocationCategory.INTERMEDIATE if spec.is_output else AllocationCategory.SCRATCH
                lease = self._pool.acquire(
                    spec.shape,
                    spec.dtype,
                    category=category,
                    reuse_key=spec.name,
                    init=spec.create_tensor(),
                )
                self._register_owned_lease(lease)
                self._track_step_lease(lease)
                tensors[spec.name] = lease.tensor
                if spec.is_output:
                    output_tensors[spec.name] = lease.tensor
                else:
                    scratch_leases.append(lease)
            else:
                missing_required.append(spec.name)
        if missing_required:
            raise KeyError(f"Missing runtime tensors for required inputs: {missing_required}")
        for spec in specs:
            if spec.is_output and spec.name not in output_tensors:
                output_tensors[spec.name] = tensors[spec.name]
        return KernelBindings(
            tensors=tensors,
            scratch_leases=tuple(scratch_leases),
            transient_leases=tuple(transient_leases),
            consumed_leases=tuple(consumed_leases.values()),
            output_tensors=output_tensors,
        )

    def run(
        self,
        case: KernelCase,
        specs: list[TensorSpec],
        bindings: KernelBindings,
    ) -> dict[str, Any]:
        self._require_active_step()
        if not isinstance(bindings, KernelBindings):
            raise TypeError(f"DeviceRuntime requires KernelBindings, got {type(bindings)!r}")
        if bindings.consumed:
            raise RuntimeError("KernelBindings have already been consumed")
        bindings.consumed = True

        start = time.perf_counter()
        compiled = self._compile(case, specs)
        self.last_compile_seconds = time.perf_counter() - start
        ordered_args = [bindings.tensors[spec.name] for spec in specs]
        start = time.perf_counter()
        try:
            self._worker.run(compiled, *ordered_args, config=self._run_config)
        finally:
            self.last_run_seconds = time.perf_counter() - start
            for lease in (*bindings.scratch_leases, *bindings.transient_leases, *bindings.consumed_leases):
                self._release_step_lease(lease)
        return dict(bindings.output_tensors)

    def begin_step(self, context: StepContext) -> None:
        self._require_open()
        if self._active_step is not None:
            raise RuntimeError(f"runtime step already active: {self._active_step.kind.value}")
        self._active_step = context

    def end_step(self) -> None:
        self._require_open()
        if self._active_step is None:
            raise RuntimeError("runtime step is not active")
        self._cleanup_step()

    def read_control(self, tensor: Any) -> torch.Tensor:
        host = self._copy_owned_to_host(tensor)
        self._release_tensor_if_step_owned(tensor)
        return host

    def export_output(self, tensor: Any) -> torch.Tensor:
        host = self._copy_owned_to_host(tensor)
        self._release_tensor_if_step_owned(tensor)
        return host.contiguous()

    def export_debug_tensor(self, tensor: Any) -> torch.Tensor:
        return self._copy_owned_to_host(tensor)

    def prepare_state(self, schemas: Sequence[LayerStateSchema]) -> None:
        self._require_open()
        self._state_store.prepare(schemas)

    def state_inputs(self, layer_id: int) -> dict[str, Any]:
        return self._state_store.inputs(layer_id)

    def state_outputs(self, layer_id: int) -> dict[str, Any]:
        return self._state_store.outputs(layer_id)

    def commit_state(self, layer_id: int, outputs: Mapping[str, Any]) -> None:
        self._state_store.commit(layer_id, outputs)

    def close(self) -> None:
        if self._closed:
            return
        if self._active_step is not None:
            self._cleanup_step()
        self._state_store.close()
        self._pool.close()
        self._fixed_weights.clear()
        self._owned_leases.clear()
        self._prefill_staging_leases.clear()
        self._compiled.clear()
        self._closed = True
        self._worker.close()

    def _materialize_value(self, spec: TensorSpec, value: Any) -> tuple[Any, DeviceLease | None]:
        if isinstance(value, RuntimeWeight):
            return self._materialize_fixed_weight(spec, value), None
        if isinstance(value, HostStagingTensor):
            lease = self._materialize_staging(spec, value)
            return lease.tensor, lease
        if isinstance(value, torch.Tensor):
            return self._materialize_host_tensor(spec, value), None
        self._validate_device_tensor(spec, value)
        return value, None

    def _materialize_staging(self, spec: TensorSpec, staging: HostStagingTensor) -> DeviceLease:
        host = self._validate_host_tensor(spec, staging.host_tensor, allow_cast=False)
        category = {
            StagingKind.PREFILL_ROUTED: AllocationCategory.STAGING_ROUTED,
            StagingKind.DECODE_SELECTED: AllocationCategory.STAGING_SELECTED,
        }[staging.kind]
        lease = self._pool.acquire(
            spec.shape,
            spec.dtype,
            category=category,
            reuse_key=staging.slot,
            init=host,
        )
        self._register_owned_lease(lease)
        self._track_step_lease(lease)
        if staging.kind is StagingKind.PREFILL_ROUTED:
            self._prefill_staging_leases[id(lease)] = lease
        return lease

    def _materialize_fixed_weight(self, spec: TensorSpec, weight: RuntimeWeight) -> Any:
        if weight.key.dtype != spec.dtype:
            raise TypeError(
                f"{spec.name} RuntimeWeight dtype mismatch: key has {weight.key.dtype}, spec requires {spec.dtype}"
            )
        lease = self._fixed_weights.get(weight.key)
        if lease is None:
            self._validate_host_tensor(spec, weight.host_tensor, allow_cast=False)
            lease = self._pool.allocate_persistent(
                spec.shape,
                spec.dtype,
                category=AllocationCategory.FIXED_WEIGHT,
                init=weight.host_tensor,
            )
            self._fixed_weights[weight.key] = lease
            self._register_owned_lease(lease)
        else:
            self._validate_lease(spec, lease)
        return lease.tensor

    def _materialize_host_tensor(self, spec: TensorSpec, tensor: torch.Tensor) -> Any:
        host = self._validate_host_tensor(spec, tensor, allow_cast=True)
        cache_key = (id(tensor), tensor.data_ptr(), tuple(spec.shape), spec.dtype)
        lease = self._active_upload_cache.get(cache_key)
        if lease is None:
            lease = self._pool.acquire(
                spec.shape,
                spec.dtype,
                category=AllocationCategory.ACTIVE_UPLOAD,
                reuse_key=spec.name,
                init=host,
            )
            self._active_upload_cache[cache_key] = lease
            self._register_owned_lease(lease)
            self._track_step_lease(lease)
        return lease.tensor

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
        compiled = case.fn.compile(*dummy_args, config=self._run_config)
        self._compiled[key] = compiled
        return compiled

    def _copy_owned_to_host(self, tensor: Any) -> torch.Tensor:
        self._require_active_step()
        lease = self._owned_leases.get(id(tensor))
        if lease is None or lease.tensor is not tensor:
            raise ValueError("tensor is not owned by this DeviceRuntime")
        return self._pool.copy_from(lease)

    def _release_tensor_if_step_owned(self, tensor: Any) -> None:
        lease = self._owned_leases.get(id(tensor))
        if lease is not None and lease.tensor is tensor:
            self._release_step_lease(lease)

    def _cleanup_step(self) -> None:
        first_error: BaseException | None = None
        for lease in list(self._step_leases.values()):
            try:
                self._release_step_lease(lease)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        self._active_upload_cache.clear()
        if not self._keep_prefill_routed_staging:
            for lease in list(self._prefill_staging_leases.values()):
                try:
                    self._pool.free(lease)
                    self._forget_owned_lease(lease)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
            self._prefill_staging_leases.clear()
        self._active_step = None
        if first_error is not None:
            raise first_error

    def _register_owned_lease(self, lease: DeviceLease) -> None:
        existing = self._owned_leases.get(id(lease.tensor))
        if existing is not None and existing is not lease:
            raise RuntimeError("DeviceTensor identity is already owned by another lease")
        self._owned_leases[id(lease.tensor)] = lease

    def _forget_owned_lease(self, lease: DeviceLease) -> None:
        if self._owned_leases.get(id(lease.tensor)) is lease:
            self._owned_leases.pop(id(lease.tensor))

    def _consumed_intermediate(self, tensor: Any) -> DeviceLease | None:
        lease = self._owned_leases.get(id(tensor))
        if (
            lease is not None
            and lease.tensor is tensor
            and lease.category is AllocationCategory.INTERMEDIATE
            and id(lease) in self._step_leases
        ):
            return lease
        return None

    def _track_step_lease(self, lease: DeviceLease) -> None:
        self._step_leases[id(lease)] = lease

    def _release_step_lease(self, lease: DeviceLease) -> None:
        if self._step_leases.pop(id(lease), None) is not None:
            self._pool.release(lease)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("DeviceRuntime is closed")

    def _require_active_step(self) -> None:
        self._require_open()
        if self._active_step is None:
            raise RuntimeError("runtime step is not active")

    @staticmethod
    def _validate_host_tensor(
        spec: TensorSpec,
        tensor: torch.Tensor,
        *,
        allow_cast: bool,
    ) -> torch.Tensor:
        if tensor.device.type != "cpu":
            raise ValueError(f"{spec.name} requires a CPU Host tensor, got {tensor.device}")
        if tuple(tensor.shape) != tuple(spec.shape):
            raise ValueError(f"{spec.name} shape mismatch: expected {tuple(spec.shape)}, got {tuple(tensor.shape)}")
        if tensor.dtype != spec.dtype:
            if not allow_cast:
                raise TypeError(f"{spec.name} dtype mismatch: expected {spec.dtype}, got {tensor.dtype}")
            tensor = tensor.to(dtype=spec.dtype)
        return tensor.contiguous()

    @staticmethod
    def _validate_lease(spec: TensorSpec, lease: DeviceLease) -> None:
        if lease.shape != tuple(spec.shape):
            raise ValueError(f"{spec.name} shape mismatch: expected {tuple(spec.shape)}, got {lease.shape}")
        if lease.dtype != spec.dtype:
            raise TypeError(f"{spec.name} dtype mismatch: expected {spec.dtype}, got {lease.dtype}")

    @staticmethod
    def _validate_device_tensor(spec: TensorSpec, tensor: Any) -> None:
        shape = getattr(tensor, "shape", None)
        dtype = getattr(tensor, "dtype", None)
        data_ptr = getattr(tensor, "data_ptr", None)
        data_ptr = data_ptr() if callable(data_ptr) else data_ptr
        if shape is None or dtype is None or not isinstance(data_ptr, int):
            raise TypeError(f"{spec.name} requires a Host tensor or DeviceTensor-compatible value")
        if tuple(shape) != tuple(spec.shape):
            raise ValueError(f"{spec.name} shape mismatch: expected {tuple(spec.shape)}, got {tuple(shape)}")
        if dtype != spec.dtype:
            raise TypeError(f"{spec.name} dtype mismatch: expected {spec.dtype}, got {dtype}")


__all__ = ["DeviceRuntime", "KernelBindings"]
