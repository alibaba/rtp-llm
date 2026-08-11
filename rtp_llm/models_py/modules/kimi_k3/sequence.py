"""Packed-sequence metadata helpers for Kimi K3."""

from __future__ import annotations

from typing import Optional

import torch


def sequence_offsets(
    cu_seqlens: torch.Tensor,
    token_count: int,
    *,
    cu_seqlens_host: Optional[torch.Tensor] = None,
) -> list[tuple[int, int]]:
    """Validate packed prefix sums and return host-visible sequence ranges."""

    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a one-dimensional [batch + 1] tensor")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must use an integer dtype")
    source = (
        cu_seqlens_host
        if cu_seqlens_host is not None and cu_seqlens_host.numel()
        else cu_seqlens
    )
    offsets = [int(value) for value in source.detach().cpu().tolist()]
    if offsets[0] != 0 or offsets[-1] != token_count:
        raise ValueError(
            f"cu_seqlens must start at 0 and end at {token_count}, got {offsets}"
        )
    if any(left > right for left, right in zip(offsets, offsets[1:])):
        raise ValueError("cu_seqlens must be non-decreasing")
    return list(zip(offsets, offsets[1:]))


__all__ = ["sequence_offsets"]
