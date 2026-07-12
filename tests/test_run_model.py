"""Tests for device runtime composition in the runner smoke entrypoint."""

import pytest
import torch

from serving import run_model


class _FakeRuntime:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    return checkpoint


def test_run_model_creates_runtime_outside_runner_and_injects_it(monkeypatch, tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    runtime = _FakeRuntime()
    captured = {}

    def fake_device_runtime(*, platform, device_id, runtime_cfg, keep_prefill_routed_staging):
        captured["runtime"] = (platform, device_id, runtime_cfg, keep_prefill_routed_staging)
        return runtime

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            captured["runner_args"] = args
            captured["runner_kwargs"] = kwargs

        def prefill(self, _input_ids):
            return torch.zeros(1, 1, 4, 8, dtype=torch.bfloat16)

        def close(self):
            runtime.close()

    monkeypatch.setattr(run_model, "DeviceRuntime", fake_device_runtime)
    monkeypatch.setattr(run_model, "DeepSeekV4Runner", FakeRunner)

    result = run_model.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--platform",
            "a2a3",
            "--device",
            "2",
            "--max-layers",
            "0",
            "--no-head",
            "--enable-l2-swimlane",
        ]
    )

    assert result == 0
    assert captured["runtime"] == ("a2a3", 2, {"enable_l2_swimlane": True}, False)
    assert captured["runner_args"] == (str(checkpoint),)
    assert captured["runner_kwargs"]["runtime"] is runtime
    assert "platform" not in captured["runner_kwargs"]
    assert "device_id" not in captured["runner_kwargs"]
    assert runtime.close_calls == 1


def test_run_model_closes_runtime_when_runner_initialization_fails(monkeypatch, tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    runtime = _FakeRuntime()
    monkeypatch.setattr(run_model, "DeviceRuntime", lambda **_kwargs: runtime)

    class FailingRunner:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("runner init failed")

    monkeypatch.setattr(run_model, "DeepSeekV4Runner", FailingRunner)

    with pytest.raises(RuntimeError, match="runner init failed"):
        run_model.main(["--checkpoint", str(checkpoint)])

    assert runtime.close_calls == 1


def test_run_model_rejects_weight_index_argument() -> None:
    with pytest.raises(SystemExit):
        run_model.parse_args(["--weight-index", "index.json"])
