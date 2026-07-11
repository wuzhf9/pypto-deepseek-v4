"""Backend-neutral runtime value descriptors for serving execution."""

from dataclasses import dataclass
from enum import Enum

import torch


@dataclass(frozen=True)
class RuntimeWeightKey:
    """Stable identity for a tensor in its final kernel-facing layout."""

    name: str
    dtype: torch.dtype
    layout: str
    layout_version: int = 1
    padding_profile: str | None = None


@dataclass(frozen=True, eq=False)
class RuntimeWeight:
    """A fixed Host runtime layout with a backend-independent cache key."""

    key: RuntimeWeightKey
    host_tensor: torch.Tensor


class StagingKind(Enum):
    """Host tensors that must use a bounded backend staging path."""

    PREFILL_ROUTED = "prefill_routed"
    DECODE_SELECTED = "decode_selected"


@dataclass(frozen=True, eq=False)
class HostStagingTensor:
    """A transient Host tensor and its semantic device staging slot."""

    host_tensor: torch.Tensor
    kind: StagingKind
    slot: str


class StepKind(Enum):
    """Whole-model execution step kind."""

    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class StepContext:
    """Backend-neutral lifetime context for one prefill or decode step."""

    kind: StepKind
    seq_len: int
    start_pos: int


__all__ = [
    "HostStagingTensor",
    "RuntimeWeight",
    "RuntimeWeightKey",
    "StagingKind",
    "StepContext",
    "StepKind",
]
