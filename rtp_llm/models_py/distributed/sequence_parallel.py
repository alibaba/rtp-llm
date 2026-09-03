"""Model-agnostic physical token layouts for sequence parallelism."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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


# Compatibility helpers for callers that have not moved padding to the model
# boundary yet. New token-SP paths should use ``SequenceParallelLayout`` and
# ``shard_physical_tokens`` so no layer allocates padding storage.
@dataclass(frozen=True)
class TokenShardLayout:
    logical_tokens: int
    local_tokens: int
    local_start: int
    local_valid_tokens: int


def token_shard_layout(
    logical_tokens: int,
    world_size: int,
    rank: int,
) -> TokenShardLayout:
    if logical_tokens < 0:
        raise ValueError(f"logical_tokens must be non-negative, got {logical_tokens}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    local_tokens = (logical_tokens + world_size - 1) // world_size
    local_start = rank * local_tokens
    return TokenShardLayout(
        logical_tokens=logical_tokens,
        local_tokens=local_tokens,
        local_start=local_start,
        local_valid_tokens=max(0, min(local_tokens, logical_tokens - local_start)),
    )


def shard_tokens(tensor: torch.Tensor, layout: TokenShardLayout) -> torch.Tensor:
    if tensor.ndim == 0 or int(tensor.shape[0]) != layout.logical_tokens:
        raise ValueError(
            "token shard expects dim0 to equal logical_tokens: "
            f"shape={tuple(tensor.shape)}, logical={layout.logical_tokens}"
        )
    if layout.local_valid_tokens == layout.local_tokens:
        return tensor.narrow(0, layout.local_start, layout.local_tokens).contiguous()
    local = tensor.new_zeros((layout.local_tokens, *tensor.shape[1:]))
    if layout.local_valid_tokens:
        local.narrow(0, 0, layout.local_valid_tokens).copy_(
            tensor.narrow(0, layout.local_start, layout.local_valid_tokens)
        )
    return local


def shard_tokens_with_padding(
    tensor: torch.Tensor,
    logical_tokens: int,
    world_size: int,
    rank: int,
) -> tuple[torch.Tensor, int]:
    layout = token_shard_layout(logical_tokens, world_size, rank)
    return shard_tokens(tensor, layout), layout.local_valid_tokens


__all__ = [
    "ForwardMode",
    "SequenceParallelLayout",
    "TokenShardLayout",
    "sequence_parallel_layout",
    "shard_physical_tokens",
    "shard_tokens",
    "shard_tokens_with_padding",
    "token_shard_layout",
]
