"""Model-agnostic physical token layouts for sequence parallelism."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch


ForwardMode = Literal["prefill", "decode", "target_verify"]


@dataclass(frozen=True)
class SequenceParallelLayout:
    """One forward's logical/physical request and token layout.

    Padding requests and tokens are always appended after the logical tail.
    ``physical_tokens`` is divisible by ``world_size`` so every TP rank owns
    one equally-sized contiguous token shard.
    """

    mode: ForwardMode
    logical_requests: int
    physical_requests: int
    tokens_per_request: int
    logical_tokens: int
    physical_tokens: int
    local_tokens: int
    local_start: int
    local_valid_tokens: int
    graph_batch_size: int = 0

    @property
    def padding_requests(self) -> int:
        return self.physical_requests - self.logical_requests

    @property
    def padding_tokens(self) -> int:
        return self.physical_tokens - self.logical_tokens


def sequence_parallel_layout(
    *,
    mode: ForwardMode,
    logical_requests: int,
    physical_requests: int,
    tokens_per_request: int,
    logical_tokens: int,
    physical_tokens: int,
    world_size: int,
    rank: int,
    graph_batch_size: int = 0,
) -> SequenceParallelLayout:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    if logical_requests < 0 or physical_requests < logical_requests:
        raise ValueError(
            "request counts must satisfy 0 <= logical <= physical: "
            f"logical={logical_requests}, physical={physical_requests}"
        )
    if logical_tokens < 0 or physical_tokens < logical_tokens:
        raise ValueError(
            "token counts must satisfy 0 <= logical <= physical: "
            f"logical={logical_tokens}, physical={physical_tokens}"
        )
    if physical_tokens % world_size:
        raise ValueError(
            f"physical_tokens={physical_tokens} must be divisible by TP{world_size}"
        )
    if mode != "prefill" and tokens_per_request <= 0:
        raise ValueError(f"{mode} requires a positive tokens_per_request")
    if tokens_per_request > 0:
        if logical_tokens != logical_requests * tokens_per_request:
            raise ValueError("logical token/request counts disagree")
        if physical_tokens != physical_requests * tokens_per_request:
            raise ValueError("physical token/request counts disagree")

    local_tokens = physical_tokens // world_size
    local_start = rank * local_tokens
    local_valid_tokens = max(0, min(local_tokens, logical_tokens - local_start))
    return SequenceParallelLayout(
        mode=mode,
        logical_requests=logical_requests,
        physical_requests=physical_requests,
        tokens_per_request=tokens_per_request,
        logical_tokens=logical_tokens,
        physical_tokens=physical_tokens,
        local_tokens=local_tokens,
        local_start=local_start,
        local_valid_tokens=local_valid_tokens,
        graph_batch_size=graph_batch_size,
    )


def sequence_parallel_layout_from_attention_inputs(
    attention_inputs: Any,
    *,
    physical_tokens: int,
    world_size: int,
    rank: int,
) -> SequenceParallelLayout:
    """Build a physical token layout from framework attention metadata."""

    is_target_verify = bool(getattr(attention_inputs, "is_target_verify", False))
    if is_target_verify:
        mode: ForwardMode = "target_verify"
    elif attention_inputs.is_prefill:
        mode = "prefill"
    else:
        mode = "decode"

    physical_requests = int(attention_inputs.input_lengths.numel())
    logical_tokens = int(
        getattr(attention_inputs, "logical_token_count", 0) or physical_tokens
    )
    logical_requests = int(
        getattr(attention_inputs, "logical_request_count", 0) or physical_requests
    )
    tokens_per_request = 0
    if mode != "prefill":
        if physical_requests <= 0 or physical_tokens % physical_requests:
            raise ValueError(
                f"{mode} physical tokens must be uniform by request: "
                f"tokens={physical_tokens}, requests={physical_requests}"
            )
        tokens_per_request = physical_tokens // physical_requests

    return sequence_parallel_layout(
        mode=mode,
        logical_requests=logical_requests,
        physical_requests=physical_requests,
        tokens_per_request=tokens_per_request,
        logical_tokens=logical_tokens,
        physical_tokens=physical_tokens,
        world_size=world_size,
        rank=rank,
        graph_batch_size=(
            physical_requests
            if bool(getattr(attention_inputs, "is_cuda_graph", False))
            else 0
        ),
    )


def shard_physical_tokens(
    tensor: torch.Tensor,
    layout: SequenceParallelLayout,
) -> torch.Tensor:
    """Return this TP rank's contiguous view of an already-padded tensor."""

    if tensor.ndim == 0 or int(tensor.shape[0]) != layout.physical_tokens:
        raise ValueError(
            "physical token shard expects dim0 to equal physical_tokens: "
            f"shape={tuple(tensor.shape)}, physical={layout.physical_tokens}"
        )
    return tensor.narrow(0, layout.local_start, layout.local_tokens).contiguous()


__all__ = [
    "ForwardMode",
    "SequenceParallelLayout",
    "sequence_parallel_layout",
    "sequence_parallel_layout_from_attention_inputs",
    "shard_physical_tokens",
]
