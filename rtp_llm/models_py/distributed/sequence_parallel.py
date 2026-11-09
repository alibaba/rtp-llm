"""Model-agnostic token layouts for sequence parallelism."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenShardLayout:
    """Equal contiguous token shards with padding only after the logical tail."""

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
    """Build one equal token shard without materializing global padding."""

    if tensor.ndim == 0 or tensor.shape[0] != layout.logical_tokens:
        raise ValueError(
            "token shard expects dim0 to equal logical tokens: "
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
    """Return one equal padded token shard and its real row count."""

    layout = token_shard_layout(logical_tokens, world_size, rank)
    return shard_tokens(tensor, layout), layout.local_valid_tokens


__all__ = [
    "TokenShardLayout",
    "shard_tokens",
    "shard_tokens_with_padding",
    "token_shard_layout",
]
