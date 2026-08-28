"""GLM-5.3-Flash compressed-indexer group geometry."""

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
        configured_topk = int(getattr(attn_config, "sparse_attention_topk", 0))
        attention_topk = (
            configured_topk
            if configured_topk > 0
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
        expected = self.selection_topk * self.group_size + self.tail_size
        if self.attention_topk != expected:
            raise ValueError(
                "sparse attention top-k must reserve expanded complete groups "
                f"and the incomplete tail: {self.attention_topk} != {expected}"
            )


def completed_group_lengths_i32(
    raw_positions: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Return completed compressed-group counts in the kernel ABI dtype.

    Decode position tensors are commonly int64, while DeepGEMM's paged MQA
    metadata and RTP-LLM's persistent top-k kernels require int32 lengths.
    Keeping this conversion at the grouping boundary prevents CUDA Graph
    dry-runs from forwarding an int64 ``context_lens`` tensor.
    """

    if raw_positions.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"raw_positions must use int32 or int64, got {raw_positions.dtype}"
        )
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    return torch.div(raw_positions + 1, group_size, rounding_mode="floor").to(
        dtype=torch.int32
    )


def expand_indexer_group_indices(
    group_indices: torch.Tensor,
    group_size: int,
    *,
    raw_sequence_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Expand request-local pooled ids to request-local raw-token ids.

    A pooled id is valid only when it is below ``raw_length // group_size``.
    This explicit completed-pool bound is important for short rows: an
    uninitialised/padded id equal to the first incomplete pool must never alias
    the raw tail, which is appended separately.
    """

    if group_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"group_indices must use int32 or int64, got {group_indices.dtype}"
        )
    if group_indices.dim() < 1:
        raise ValueError("group_indices must have at least one dimension")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if group_size == 1 and raw_sequence_lengths is None:
        return group_indices

    valid_groups = group_indices >= 0
    if raw_sequence_lengths is not None:
        expected_shape = group_indices.shape[:-1]
        if tuple(raw_sequence_lengths.shape) != tuple(expected_shape):
            raise ValueError(
                "raw_sequence_lengths shape must match group_indices without "
                f"top-k: {tuple(raw_sequence_lengths.shape)} != "
                f"{tuple(expected_shape)}"
            )
        lengths = raw_sequence_lengths.to(
            device=group_indices.device, dtype=group_indices.dtype
        )
        complete_group_count = torch.div(lengths, group_size, rounding_mode="floor")
        valid_groups = valid_groups & (
            group_indices < complete_group_count.unsqueeze(-1)
        )

    lanes = torch.arange(
        group_size, dtype=group_indices.dtype, device=group_indices.device
    )
    expanded = group_indices.unsqueeze(-1) * group_size + lanes
    expanded = torch.where(
        valid_groups.unsqueeze(-1), expanded, torch.full_like(expanded, -1)
    )
    return expanded.flatten(start_dim=-2)


def append_incomplete_tail_indices(
    expanded_history: torch.Tensor,
    raw_sequence_lengths: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Append up to ``group_size - 1`` raw tokens not yet pooled."""

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
