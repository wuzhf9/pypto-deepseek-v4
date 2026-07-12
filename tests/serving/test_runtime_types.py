from dataclasses import FrozenInstanceError

import pytest
import torch

from serving.runtime_types import (
    HostStagingTensor,
    KernelCase,
    RuntimeWeight,
    RuntimeWeightKey,
    StagingKind,
    StepContext,
    StepKind,
)


def test_kernel_case_preserves_entrypoint_and_spec_builder() -> None:
    fn = object()
    spec_builder = object()
    case = KernelCase("block", fn, spec_builder)

    assert case.name == "block"
    assert case.fn is fn
    assert case.spec_builder is spec_builder
    with pytest.raises(FrozenInstanceError):
        case.name = "other"


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


def test_step_context_preserves_runtime_lifetime_fields() -> None:
    context = StepContext(kind=StepKind.DECODE, seq_len=1, start_pos=7)

    assert context.kind is StepKind.DECODE
    assert context.seq_len == 1
    assert context.start_pos == 7
