"""Serving backend construction."""

from typing import Any

from serving.backends.base import Backend, BackendName


def create_backend(
    name: BackendName,
    *,
    platform: str,
    device_id: int,
    runtime_cfg: dict[str, Any] | None = None,
    keep_prefill_routed_staging: bool = False,
) -> Backend:
    """Create a serving backend from its CLI-facing name."""
    if name == "worker":
        from serving.backends.worker_backend import WorkerBackend

        return WorkerBackend(
            platform=platform,
            device_id=device_id,
            runtime_cfg=runtime_cfg,
            keep_prefill_routed_staging=keep_prefill_routed_staging,
        )
    raise ValueError(f"unsupported backend: {name!r}")


__all__ = ["create_backend"]
