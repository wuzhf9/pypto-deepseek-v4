"""Tests for the backend composition in the runner smoke entrypoint."""

import pytest
import torch

from serving import run_model


class _FakeBackend:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def test_run_model_creates_backend_outside_runner_and_injects_it(monkeypatch) -> None:
    backend = _FakeBackend()
    captured = {}

    def fake_create_backend(name, *, platform, device_id):
        captured["factory"] = (name, platform, device_id)
        return backend

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            captured["runner_args"] = args
            captured["runner_kwargs"] = kwargs

        def prefill(self, _input_ids):
            return torch.zeros(1, 1, 4, 8, dtype=torch.bfloat16)

        def close(self):
            backend.close()

    monkeypatch.setattr(run_model, "create_backend", fake_create_backend)
    monkeypatch.setattr(run_model, "DeepSeekV4Runner", FakeRunner)

    result = run_model.main(
        [
            "--checkpoint",
            "checkpoint",
            "--backend",
            "direct",
            "--platform",
            "a2a3",
            "--device",
            "2",
            "--max-layers",
            "0",
            "--no-head",
        ]
    )

    assert result == 0
    assert captured["factory"] == ("direct", "a2a3", 2)
    assert captured["runner_args"] == ("checkpoint",)
    assert captured["runner_kwargs"]["backend"] is backend
    assert "platform" not in captured["runner_kwargs"]
    assert "device_id" not in captured["runner_kwargs"]
    assert backend.close_calls == 1


def test_run_model_closes_backend_when_runner_initialization_fails(monkeypatch) -> None:
    backend = _FakeBackend()
    monkeypatch.setattr(run_model, "create_backend", lambda *_args, **_kwargs: backend)

    class FailingRunner:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("runner init failed")

    monkeypatch.setattr(run_model, "DeepSeekV4Runner", FailingRunner)

    with pytest.raises(RuntimeError, match="runner init failed"):
        run_model.main(["--checkpoint", "checkpoint"])

    assert backend.close_calls == 1
