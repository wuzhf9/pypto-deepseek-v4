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


def ignore_output(*_args: Any, **_kwargs: Any) -> tuple[bool, str]:
    """Comparator for output buffers that are required by PyPTO but not validated."""
    return True, ""


def topk_indices_by_score(
    score_name: str,
    *,
    index_offset: int = 0,
    index_offset_name: str | None = None,
    invalid_index: int = -1,
    dim: int = -1,
    descending: bool = True,
    atol: float | None = None,
    rtol: float | None = None,
    max_show: int = 10,
) -> Callable:
    """Return a comparator for top-k index tensors backed by a score output.

    The score tensor should be validated separately. This comparator accepts
    small score-driven reordering, but still checks that actual indices are
    valid, unique, ordered by their actual paired scores, and no worse than the
    top-k boundary in the actual score row within tolerance.
    """

    if dim != -1:
        raise ValueError("topk_indices_by_score currently supports dim=-1 only")

    def compare(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        actual_outputs: dict[str, torch.Tensor],
        inputs: dict[str, torch.Tensor],
        rtol: float,
        atol: float,
        **_kwargs: Any,
    ) -> tuple[bool, str]:
        if score_name not in actual_outputs:
            return False, f"score output {score_name!r} is missing from actual outputs"

        eff_atol = atol if compare.atol_override is None else compare.atol_override
        eff_rtol = rtol if compare.rtol_override is None else compare.rtol_override
        offset = index_offset
        if index_offset_name is not None:
            if index_offset_name not in inputs:
                return False, f"index offset input {index_offset_name!r} is missing"
            offset = int(inputs[index_offset_name].detach().cpu().reshape(-1)[0].item())

        actual_i = actual.detach().cpu().to(torch.int64)
        expected_i = expected.detach().cpu().to(torch.int64)
        scores = actual_outputs[score_name].detach().cpu().to(torch.float32)

        if actual_i.shape != expected_i.shape:
            return False, f"topk shape mismatch: actual={tuple(actual_i.shape)} expected={tuple(expected_i.shape)}"
        if actual_i.shape[:-1] != scores.shape[:-1]:
            return False, f"topk/score leading shape mismatch: topk={tuple(actual_i.shape)} score={tuple(scores.shape)}"

        if torch.equal(actual_i, expected_i):
            return True, ""

        topk_rows = actual_i.reshape(-1, actual_i.shape[-1])
        expected_rows = expected_i.reshape(-1, expected_i.shape[-1])
        score_rows = scores.reshape(-1, scores.shape[-1])
        failures: list[str] = []

        def row_coord(row: int) -> tuple[int, ...]:
            if actual_i.dim() == 1:
                return ()
            shape = actual_i.shape[:-1]
            rem = row
            coords = []
            for size in reversed(shape):
                coords.append(rem % size)
                rem //= size
            return tuple(reversed(coords))

        for row in range(topk_rows.shape[0]):
            a_row = topk_rows[row]
            e_row = expected_rows[row]
            s_row = score_rows[row]
            valid_mask = e_row != invalid_index
            valid_count = int(valid_mask.sum().item())
            if valid_count == 0:
                if not torch.equal(a_row, e_row):
                    failures.append(f"row={row_coord(row)} expected all invalid, actual_prefix={a_row[:8].tolist()}")
                continue
            if not torch.all(valid_mask[:valid_count]).item() or torch.any(valid_mask[valid_count:]).item():
                failures.append(f"row={row_coord(row)} expected valid topk entries are not a prefix")
                continue
            if not torch.all(a_row[valid_count:] == invalid_index).item():
                failures.append(
                    f"row={row_coord(row)} invalid suffix mismatch: actual_suffix_prefix={a_row[valid_count:valid_count + 8].tolist()}"
                )
                continue

            a_valid = a_row[:valid_count]
            if torch.any(a_valid == invalid_index).item():
                failures.append(f"row={row_coord(row)} actual has invalid index inside valid prefix")
                continue
            local_idx = a_valid - offset
            if torch.any(local_idx < 0).item() or torch.any(local_idx >= s_row.shape[0]).item():
                failures.append(
                    f"row={row_coord(row)} actual index out of score range: actual_prefix={a_valid[:8].tolist()} offset={offset}"
                )
                continue
            if torch.unique(local_idx).numel() != local_idx.numel():
                failures.append(f"row={row_coord(row)} actual topk contains duplicate indices: {a_valid[:16].tolist()}")
                continue

            paired = s_row.gather(0, local_idx)
            if paired.numel() >= 2:
                left = paired[:-1]
                right = paired[1:]
                tol = eff_atol + eff_rtol * torch.maximum(left.abs(), right.abs())
                order_ok = left + tol >= right if descending else left <= right + tol
                if not torch.all(order_ok).item():
                    pos = int(torch.where(~order_ok)[0][0].item())
                    failures.append(
                        f"row={row_coord(row)} paired scores break order at {pos}: "
                        f"indices={a_valid[max(0, pos - 1):pos + 3].tolist()} "
                        f"scores={[float(x) for x in paired[max(0, pos - 1):pos + 3].tolist()]}"
                    )
                    continue

            boundary_vals = torch.topk(s_row, k=valid_count, largest=descending, sorted=True).values
            boundary = boundary_vals[-1]
            boundary_tol = eff_atol + eff_rtol * torch.maximum(paired.abs(), boundary.abs())
            if descending:
                boundary_ok = paired + boundary_tol >= boundary
            else:
                boundary_ok = paired <= boundary + boundary_tol
            if not torch.all(boundary_ok).item():
                pos = int(torch.where(~boundary_ok)[0][0].item())
                failures.append(
                    f"row={row_coord(row)} selected score outside topk boundary at {pos}: "
                    f"idx={int(a_valid[pos].item())} score={float(paired[pos].item()):.8g} "
                    f"boundary={float(boundary.item()):.8g}"
                )

            if len(failures) >= max_show:
                break

        if not failures:
            return True, ""

        mismatch_count = int((actual_i != expected_i).sum().item())
        return False, (
            f"topk_indices_by_score fail: mismatch_count={mismatch_count}, "
            f"score_name={score_name!r}, offset={offset}, invalid_index={invalid_index}\n"
            + "\n".join(failures)
        )

    compare.atol_override = atol
    compare.rtol_override = rtol
    compare.__name__ = (
        f"topk_indices_by_score(score_name={score_name!r}, atol={atol}, rtol={rtol})"
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
    "topk_indices_by_score",
    "run_jit",
]
