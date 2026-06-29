"""Small golden-test harness for standalone PyPTO kernel validation.

This module mirrors the useful parts of ``pypto-serving/pypto-lib/golden`` for
this repository: create tensors, compile a ``@pl.jit`` kernel, run it through
PyPTO, compute a PyTorch reference, and compare outputs.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class TensorSpec:
    """Runtime tensor description used by ``run_jit``."""

    name: str
    shape: list[int]
    dtype: torch.dtype
    init_value: int | float | torch.Tensor | Callable | None = None
    is_output: bool = False

    def create_tensor(self) -> torch.Tensor:
        if self.init_value is None:
            return torch.zeros(self.shape, dtype=self.dtype)
        if isinstance(self.init_value, (int, float)):
            return torch.full(self.shape, self.init_value, dtype=self.dtype)
        if isinstance(self.init_value, torch.Tensor):
            return self.init_value.to(dtype=self.dtype).clone()
        if callable(self.init_value):
            if self.init_value in (torch.randn, torch.rand, torch.zeros, torch.ones):
                return self.init_value(self.shape, dtype=self.dtype)
            result = self.init_value()
            return torch.as_tensor(result, dtype=self.dtype)
        raise TypeError(f"Unsupported init_value type {type(self.init_value)!r} for {self.name!r}")


@dataclass
class RunResult:
    passed: bool
    error: str | None = None
    execution_time: float | None = None
    work_dir: Path | None = None

    def __str__(self) -> str:
        time_str = f" ({self.execution_time:.2f}s)" if self.execution_time is not None else ""
        if self.passed:
            return "PASS" + time_str
        msg = "FAIL"
        if self.error:
            msg += f": {self.error}"
        return msg + time_str


class _Stage:
    def __init__(self, name: str) -> None:
        self._name = name
        self._start = 0.0

    def __enter__(self) -> "_Stage":
        print(f"[RUN] {self._name} ...", flush=True)
        self._start = time.time()
        return self

    def __exit__(self, *_exc: Any) -> bool:
        elapsed = time.time() - self._start
        print(f"[RUN] {self._name} done ({elapsed:.2f}s)", flush=True)
        return False


def ratio_allclose(
    atol: float | None = None,
    rtol: float | None = None,
    max_error_ratio: float = 0.005,
    max_show: int = 10,
) -> Callable:
    """Return an allclose comparator that allows a bounded outlier ratio."""

    if not 0.0 <= max_error_ratio <= 1.0:
        raise ValueError(f"max_error_ratio must be in [0, 1], got {max_error_ratio}")

    def compare(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        rtol: float,
        atol: float,
        **_kwargs: Any,
    ) -> tuple[bool, str]:
        eff_atol = atol if compare.atol_override is None else compare.atol_override
        eff_rtol = rtol if compare.rtol_override is None else compare.rtol_override

        actual_f = actual.detach().cpu().to(torch.float32)
        expected_f = expected.detach().cpu().to(torch.float32)

        nan_count = int(torch.isnan(actual_f).sum().item())
        inf_count = int(torch.isinf(actual_f).sum().item())
        if nan_count or inf_count:
            return False, f"illegal values in actual: NaN={nan_count} Inf={inf_count}"

        diff_abs = (actual_f - expected_f).abs()
        tolerance = eff_atol + eff_rtol * expected_f.abs()
        bad_mask = diff_abs > tolerance
        error_count = int(bad_mask.sum().item())
        numel = actual_f.numel()
        threshold = round(max_error_ratio * numel)

        if error_count <= threshold:
            return True, ""

        max_diff, flat_max_pos = torch.max(diff_abs.flatten(), dim=0)
        max_pos = tuple(int(i.item()) for i in torch.unravel_index(flat_max_pos, actual_f.shape))
        max_tol = float(tolerance[max_pos].item())

        bad_indices = torch.where(bad_mask.flatten())[0]
        flat_actual = actual_f.flatten()
        flat_expected = expected_f.flatten()
        flat_tol = tolerance.flatten()
        flat_diff = diff_abs.flatten()
        n_show = min(max_show, bad_indices.numel())
        shown = []
        for idx in bad_indices[:n_show]:
            i = int(idx.item())
            shown.append(
                f"    [{i}] actual={flat_actual[i].item():.8g}, "
                f"expected={flat_expected[i].item():.8g}, "
                f"diff={flat_diff[i].item():.4g}, tol={flat_tol[i].item():.4g}"
            )

        detail = (
            f"ratio_allclose fail: error_count={error_count}/{numel} "
            f"(ratio={error_count / numel:.4%}, allowed<={max_error_ratio:.4%}, "
            f"threshold={threshold} pts)\n"
            f"atol={eff_atol} rtol={eff_rtol}\n"
            f"max abs diff={max_diff.item():.6g} at {max_pos} (tol={max_tol:.6g})"
        )
        if shown:
            detail += "\nfirst mismatches:\n" + "\n".join(shown)
        return False, detail

    compare.atol_override = atol
    compare.rtol_override = rtol
    compare.__name__ = (
        f"ratio_allclose(atol={atol}, rtol={rtol}, max_error_ratio={max_error_ratio})"
    )
    return compare


def _save_tensors(dest_dir: Path, tensors: dict[str, torch.Tensor]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in tensors.items():
        torch.save(tensor, dest_dir / f"{name}.pt")


def _prepare_tensors(
    specs: list[TensorSpec],
    work_dir: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    tensors = {spec.name: spec.create_tensor() for spec in specs}
    input_snapshot = {
        spec.name: tensors[spec.name].clone()
        for spec in specs
        if not spec.is_output or spec.init_value is not None
    }
    _save_tensors(work_dir / "data" / "in", input_snapshot)
    return tensors, input_snapshot


def _compute_golden(
    specs: list[TensorSpec],
    input_snapshot: dict[str, torch.Tensor],
    work_dir: Path,
    golden_fn: Callable,
) -> dict[str, torch.Tensor]:
    with _Stage("compute golden"):
        scratch: dict[str, torch.Tensor] = {}
        for spec in specs:
            if spec.is_output and spec.init_value is None:
                scratch[spec.name] = torch.zeros(spec.shape, dtype=spec.dtype)
            else:
                scratch[spec.name] = input_snapshot[spec.name].clone()
        golden_fn(scratch)
        outputs = {spec.name: scratch[spec.name] for spec in specs if spec.is_output}
        _save_tensors(work_dir / "data" / "out", outputs)
        return outputs


def _validate_outputs(
    specs: list[TensorSpec],
    tensors: dict[str, torch.Tensor],
    golden_outputs: dict[str, torch.Tensor],
    rtol: float,
    atol: float,
    compare_fn: dict[str, Callable],
) -> None:
    with _Stage("validate"):
        input_tensors = {spec.name: tensors[spec.name] for spec in specs if not spec.is_output}
        actual_outputs = {spec.name: tensors[spec.name] for spec in specs if spec.is_output}
        failures: list[str] = []
        for name, expected in golden_outputs.items():
            actual = actual_outputs[name]
            comparator = compare_fn.get(name)
            if comparator is None:
                passed = torch.allclose(actual, expected, rtol=rtol, atol=atol)
                detail = "" if passed else _allclose_detail(actual, expected, rtol=rtol, atol=atol)
            else:
                passed, detail = comparator(
                    actual,
                    expected,
                    actual_outputs=actual_outputs,
                    expected_outputs=golden_outputs,
                    inputs=input_tensors,
                    rtol=rtol,
                    atol=atol,
                )
            if not passed:
                failures.append(f"{name}: {detail}")
        if failures:
            raise AssertionError("\n".join(failures))


def _allclose_detail(actual: torch.Tensor, expected: torch.Tensor, rtol: float, atol: float) -> str:
    actual_f = actual.detach().cpu().to(torch.float32)
    expected_f = expected.detach().cpu().to(torch.float32)
    diff_abs = (actual_f - expected_f).abs()
    max_diff, flat_max_pos = torch.max(diff_abs.flatten(), dim=0)
    max_pos = tuple(int(i.item()) for i in torch.unravel_index(flat_max_pos, actual_f.shape))
    tol = atol + rtol * expected_f[max_pos].abs()
    return (
        f"torch.allclose fail: atol={atol} rtol={rtol}; "
        f"max abs diff={max_diff.item():.6g} at {max_pos} (tol={float(tol.item()):.6g})"
    )


def _run_config(runtime_cfg: dict[str, Any]):
    from pypto.runtime import RunConfig

    return RunConfig(**runtime_cfg)


def run_jit(
    fn: Any,
    specs: list[TensorSpec],
    golden_fn: Callable | None = None,
    runtime_cfg: dict[str, Any] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    compare_fn: dict[str, Callable] | None = None,
    compile_only: bool = False,
) -> RunResult:
    """Compile a ``@pl.jit`` kernel, run it, and validate output tensors."""

    runtime_cfg = dict(runtime_cfg or {})
    compare_fn = compare_fn or {}
    start = time.time()
    work_dir: Path | None = None

    def fail(error: str) -> RunResult:
        return RunResult(False, error=error, execution_time=time.time() - start, work_dir=work_dir)

    try:
        config = _run_config(runtime_cfg)
    except Exception as exc:
        return fail(f"invalid runtime_cfg: {exc}")

    try:
        with _Stage("compile"):
            dummy_args = [torch.empty(spec.shape, dtype=spec.dtype) for spec in specs]
            compiled = fn.compile(*dummy_args, config=config)
            work_dir = Path(compiled.output_dir)
        if compile_only:
            total = time.time() - start
            print(f"[RUN] PASS ({total:.2f}s)", flush=True)
            return RunResult(True, execution_time=total, work_dir=work_dir)

        with _Stage("generate inputs"):
            tensors, input_snapshot = _prepare_tensors(specs, work_dir)

        golden_outputs = None
        if golden_fn is not None:
            golden_outputs = _compute_golden(specs, input_snapshot, work_dir, golden_fn)

        with _Stage("runtime"):
            ordered_args = [tensors[spec.name] for spec in specs]
            compiled(*ordered_args, config=config)

        if golden_outputs is None:
            total = time.time() - start
            print(f"[RUN] PASS ({total:.2f}s, validation skipped: no golden_fn)", flush=True)
            return RunResult(True, execution_time=total, work_dir=work_dir)

        _validate_outputs(specs, tensors, golden_outputs, rtol, atol, compare_fn)
    except Exception as exc:
        return fail(str(exc))

    total = time.time() - start
    print(f"[RUN] PASS ({total:.2f}s)", flush=True)
    return RunResult(True, execution_time=total, work_dir=work_dir)


__all__ = [
    "TensorSpec",
    "RunResult",
    "ratio_allclose",
    "run_jit",
]
