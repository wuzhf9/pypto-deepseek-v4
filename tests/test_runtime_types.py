from dataclasses import FrozenInstanceError

import pytest
import torch

from serving.runtime_types import (
    HostStagingTensor,
    RuntimeWeight,
    RuntimeWeightKey,
    StagingKind,
    StepContext,
    StepKind,
)


def test_runtime_weight_key_separates_layout_metadata() -> None:
    base = RuntimeWeightKey("weight", torch.bfloat16, "linear_t")

    assert base == RuntimeWeightKey("weight", torch.bfloat16, "linear_t")
    assert base != RuntimeWeightKey("weight", torch.float32, "linear_t")
    assert base != RuntimeWeightKey("weight", torch.bfloat16, "identity")
    assert base != RuntimeWeightKey("weight", torch.bfloat16, "linear_t", layout_version=2)
    assert base != RuntimeWeightKey(
        "weight",
        torch.bfloat16,
        "linear_t",
        padding_profile="width=16",
    )


def test_runtime_values_preserve_host_tensor_identity_and_are_frozen() -> None:
    tensor = torch.ones(2, dtype=torch.bfloat16)
    key = RuntimeWeightKey("weight", tensor.dtype, "identity")
    weight = RuntimeWeight(key, tensor)
    staging = HostStagingTensor(tensor, StagingKind.PREFILL_ROUTED, "w1_t")

    assert weight.host_tensor is tensor
    assert staging.host_tensor is tensor
    assert staging.kind is StagingKind.PREFILL_ROUTED
    assert staging.slot == "w1_t"
    with pytest.raises(FrozenInstanceError):
        weight.key = key


def test_step_context_is_backend_neutral_value() -> None:
    context = StepContext(kind=StepKind.DECODE, seq_len=1, start_pos=7)

    assert context.kind is StepKind.DECODE
    assert context.seq_len == 1
    assert context.start_pos == 7
