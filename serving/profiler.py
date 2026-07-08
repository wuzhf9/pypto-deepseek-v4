"""Lightweight profiling helpers for serving runtime instrumentation."""

from contextlib import contextmanager
import time
from typing import Any, Iterator


def block_shape_from_kernel(kernel: str) -> str:
    name = kernel.removeprefix("block_").removesuffix("_fwd")
    for suffix in ("_prefill", "_decode"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def block_profile_fields(
    *,
    layer: int,
    mode: str,
    ratio: int,
    hash_route: bool,
    kernel: str,
) -> dict[str, Any]:
    return {
        "layer": layer,
        "mode": mode,
        "ratio": ratio,
        "hash_route": hash_route,
        "block_shape": block_shape_from_kernel(kernel),
        "kernel": kernel,
    }


class ProfileRecorder:
    """Print profile events without coupling formatting to runner logic."""

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = bool(enabled)

    @contextmanager
    def timer(self, name: str, **fields: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self._record(name, start, **fields)

    @contextmanager
    def backend_timer(self, name: str, backend: Any, **fields: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self._record_backend(name, start, backend, **fields)

    def record_weight_loader(self, name: str, weight_loader: Any, **fields: Any) -> None:
        if not self.enabled:
            return
        parts = [f"{key}={value}" for key, value in fields.items()]
        for stat_name, count, elapsed_ms in weight_loader.profile_summary():
            parts.append(f"{stat_name}={elapsed_ms:.3f}ms/{count}")
        print(f"[PROFILE] {name}: {' '.join(parts)}", flush=True)

    def _record(self, name: str, start: float, **fields: Any) -> None:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        details = _format_fields(fields)
        suffix = f" {details}" if details else ""
        print(f"[PROFILE] {name}: {elapsed_ms:.3f} ms{suffix}", flush=True)

    def _record_backend(self, name: str, start: float, backend: Any, **fields: Any) -> None:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        compile_ms = getattr(backend, "last_compile_seconds", 0.0) * 1000.0
        run_ms = getattr(backend, "last_run_seconds", 0.0) * 1000.0
        cache_hit = getattr(backend, "last_compile_cache_hit", False)
        details = _format_fields(
            {
                **fields,
                "compile_ms": f"{compile_ms:.3f}",
                "run_ms": f"{run_ms:.3f}",
                "cache_hit": cache_hit,
            }
        )
        print(f"[PROFILE] {name}: {elapsed_ms:.3f} ms {details}", flush=True)


def _format_fields(fields: dict[str, Any]) -> str:
    return " ".join(f"{key}={value}" for key, value in fields.items())


__all__ = ["ProfileRecorder", "block_profile_fields", "block_shape_from_kernel"]
