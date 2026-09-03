"""Kimi K3 token-shard layout at the model boundary.

Padding is appended once after the logical token tail. Decoder layers keep
the equal local shard; attention projections temporarily recover logical rows
through AllGather/GEMM and return to the shard through GEMM/ReduceScatter.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class K3TokenLayout:
    logical_tokens: int
    local_tokens: int
    local_start: int
    local_valid_tokens: int


def make_k3_token_layout(
    logical_tokens: int,
    world_size: int,
    rank: int,
) -> K3TokenLayout:
    local_tokens = (logical_tokens + world_size - 1) // world_size
    local_start = rank * local_tokens
    return K3TokenLayout(
        logical_tokens=logical_tokens,
        local_tokens=local_tokens,
        local_start=local_start,
        local_valid_tokens=max(0, min(local_tokens, logical_tokens - local_start)),
    )


def shard_k3_tokens(
    tensor: torch.Tensor,
    layout: K3TokenLayout,
) -> torch.Tensor:
    """Create one equal shard, materializing only tail padding rows."""

    local = tensor.new_zeros((layout.local_tokens, *tensor.shape[1:]))
    if layout.local_valid_tokens:
        local.narrow(0, 0, layout.local_valid_tokens).copy_(
            tensor.narrow(0, layout.local_start, layout.local_valid_tokens)
        )
    return local


__all__ = ["K3TokenLayout", "make_k3_token_layout", "shard_k3_tokens"]
