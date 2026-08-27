"""Compressed-indexer group geometry shared by GLM-5.3 model and MLA backends.

The indexer returns request-local *group* ids. Sparse MLA still consumes raw
token ids, so each selected group is expanded in score order and then lane
order. Keeping this transform separate from cache addressing makes it usable
by both BF16 and FP8 sparse MLA paths and gives us a CPU-testable reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class IndexerGroupingGeometry:
    selection_topk: int
    group_size: int
    attention_topk: int

    @property
    def tail_size(self) -> int:
        return self.group_size - 1 if self.group_size > 1 else 0

    @classmethod
    def from_attention_config(cls, attn_config: Any) -> "IndexerGroupingGeometry":
        selection_topk = int(attn_config.indexer_topk)
        group_size = int(getattr(attn_config, "indexer_compress_ratio", 1))
        configured_attention_topk = int(
            getattr(attn_config, "sparse_attention_topk", 0)
        )
        attention_topk = (
            configured_attention_topk
            if configured_attention_topk > 0
            else selection_topk * group_size + max(group_size - 1, 0)
        )
        geometry = cls(selection_topk, group_size, attention_topk)
        geometry.validate()
        return geometry

    def validate(self) -> None:
        if self.selection_topk <= 0:
            raise ValueError(
                f"selection_topk must be positive, got {self.selection_topk}"
            )
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")
        expected_attention_topk = self.selection_topk * self.group_size + self.tail_size
        if self.attention_topk != expected_attention_topk:
            raise ValueError(
                "sparse attention top-k must reserve expanded history groups and "
                "the incomplete tail: "
                f"{self.attention_topk} != {self.selection_topk} * "
                f"{self.group_size} + {self.tail_size}"
            )


def expand_indexer_group_indices(
    group_indices: torch.Tensor,
    group_size: int,
    *,
    raw_sequence_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Expand request-local group ids to request-local raw-token ids.

    ``[..., K]`` becomes ``[..., K * group_size]``. The output order is
    ``group score order`` then ``lane 0..group_size-1``. Negative group ids
    propagate to ``group_size`` copies of ``-1``. If per-row raw sequence
    lengths are supplied, lanes outside the request are also masked to ``-1``.

    The first implementation assumes non-overlapping groups
    ``group g -> [g*group_size, ..., g*group_size+group_size-1]``.
    """

    if group_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            "group_indices must use int32 or int64, got " f"{group_indices.dtype}"
        )
    if group_indices.dim() < 1:
        raise ValueError("group_indices must have at least one dimension")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if group_size == 1 and raw_sequence_lengths is None:
        return group_indices

    lanes = torch.arange(
        group_size, dtype=group_indices.dtype, device=group_indices.device
    )
    expanded = group_indices.unsqueeze(-1) * group_size + lanes
    expanded = torch.where(
        group_indices.unsqueeze(-1) >= 0,
        expanded,
        torch.full_like(expanded, -1),
    )

    if raw_sequence_lengths is not None:
        expected_shape = group_indices.shape[:-1]
        if tuple(raw_sequence_lengths.shape) != tuple(expected_shape):
            raise ValueError(
                "raw_sequence_lengths shape must match group_indices without "
                f"its top-k dimension: {tuple(raw_sequence_lengths.shape)} != "
                f"{tuple(expected_shape)}"
            )
        lengths = raw_sequence_lengths.to(device=expanded.device, dtype=expanded.dtype)
        expanded = torch.where(
            expanded < lengths.unsqueeze(-1).unsqueeze(-1),
            expanded,
            torch.full_like(expanded, -1),
        )

    return expanded.flatten(start_dim=-2)


def append_incomplete_tail_indices(
    expanded_history: torch.Tensor,
    raw_sequence_lengths: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Append the uncompressed tail required by GLM-5.3 KPool.

    Complete groups are scored through their compressed keys. The most recent
    ``length % group_size`` raw tokens have no pooled key yet, so they are
    always appended. The output reserves ``group_size - 1`` lanes and pads
    unused lanes with ``-1`` to keep a graph-stable sparse-attention width.
    """

    if expanded_history.dim() < 2:
        raise ValueError("expanded_history must have a row and top-k dimension")
    if group_size <= 1:
        return expanded_history
    expected_shape = expanded_history.shape[:-1]
    if tuple(raw_sequence_lengths.shape) != tuple(expected_shape):
        raise ValueError(
            "raw_sequence_lengths shape must match history rows: "
            f"{tuple(raw_sequence_lengths.shape)} != {tuple(expected_shape)}"
        )

    lengths = raw_sequence_lengths.to(
        device=expanded_history.device, dtype=expanded_history.dtype
    )
    tail_width = group_size - 1
    lanes = torch.arange(
        tail_width, device=expanded_history.device, dtype=expanded_history.dtype
    )
    tail_count = torch.remainder(lengths, group_size)
    tail_start = lengths - tail_count
    tail = tail_start.unsqueeze(-1) + lanes
    tail = torch.where(
        lanes < tail_count.unsqueeze(-1), tail, torch.full_like(tail, -1)
    )
    return torch.cat((expanded_history, tail), dim=-1)
