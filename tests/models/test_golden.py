"""Tests for the standalone PyPTO golden harness."""

import torch

from models.golden import TensorSpec, _prepare_tensors


def test_prepare_tensors_snapshots_inputs_by_default() -> None:
    specs = [TensorSpec("x", [2], torch.float32, init_value=1.0)]

    tensors, snapshot = _prepare_tensors(specs)

    assert torch.equal(snapshot["x"], tensors["x"])
    assert snapshot["x"].data_ptr() != tensors["x"].data_ptr()


def test_prepare_tensors_can_skip_input_snapshots() -> None:
    specs = [TensorSpec("x", [2], torch.float32, init_value=1.0)]

    tensors, snapshot = _prepare_tensors(specs, snapshot_inputs=False)

    assert torch.equal(tensors["x"], torch.ones(2))
    assert snapshot == {}
