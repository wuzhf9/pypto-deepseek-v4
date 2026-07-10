"""Serving backend construction."""

from typing import Any

from serving.backends.base import Backend, BackendName
from serving.backends.direct_backend import DirectBackend


def create_backend(
    name: BackendName,
    *,
    platform: str,
    device_id: int,
    runtime_cfg: dict[str, Any] | None = None,
) -> Backend:
    """Create a serving backend from its CLI-facing name."""
    if name == "direct":
        return DirectBackend(platform=platform, device_id=device_id, runtime_cfg=runtime_cfg)
    if name == "worker":
        raise NotImplementedError(
            "worker backend was removed after profiling showed kernel runtime dominates; "
            "use backend='direct'"
        )
    raise ValueError(f"unsupported backend: {name!r}")


__all__ = ["create_backend"]
