"""GLM-5.3-Flash compressed-indexer group geometry."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _expand_groups_append_tail_kernel(
        group_indices_ptr,
        raw_lengths_ptr,
        output_ptr,
        group_stride,
        output_stride,
        history_width,
        output_width,
        GROUP_SIZE: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        """Expand selected complete groups and append the live raw tail."""

        row = tl.program_id(0)
        tile = tl.program_id(1)
        cols = tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        mask = cols < output_width

        raw_len = tl.load(raw_lengths_ptr + row).to(tl.int64)
        complete_groups = raw_len // GROUP_SIZE
        tail_start = complete_groups * GROUP_SIZE
        tail_count = raw_len - tail_start

        is_history = cols < history_width
        group_col = cols // GROUP_SIZE
        lane = cols % GROUP_SIZE
        group_id = tl.load(
            group_indices_ptr + row * group_stride + group_col,
            mask=mask & is_history,
            other=-1,
        ).to(tl.int64)
        valid_group = (group_id >= 0) & (group_id < complete_groups)
        history_value = tl.where(valid_group, group_id * GROUP_SIZE + lane, -1).to(
            tl.int32
        )

        tail_lane = cols - history_width
        is_tail = (tail_lane >= 0) & (tail_lane < tail_count)
        tail_value = tl.where(is_tail, tail_start + tail_lane, -1).to(tl.int32)
        output = tl.where(is_history, history_value, tail_value)
        tl.store(output_ptr + row * output_stride + cols, output, mask=mask)


def _fused_group_expansion_enabled() -> bool:
    return os.environ.get("GLM53_INDEXER_GROUP_FUSED", "1") != "0"


def fused_expand_indexer_groups_with_tail(
    group_indices: torch.Tensor,
    raw_sequence_lengths: torch.Tensor,
    group_size: int,
) -> Optional[torch.Tensor]:
    """CUDA fast path for group expansion plus incomplete-tail append.

    Returns ``None`` on an applicability miss so callers can compose the
    framework reference functions.  The kernel preserves the strict completed
    group validity bound used by :func:`expand_indexer_group_indices`.
    """

    if (
        not _TRITON_AVAILABLE
        or not _fused_group_expansion_enabled()
        or not group_indices.is_cuda
        or group_indices.dim() != 2
        or group_size <= 1
        or not raw_sequence_lengths.is_cuda
        or raw_sequence_lengths.device != group_indices.device
        or group_indices.stride(1) != 1
    ):
        return None
    # The production persistent TopK ABI is int32 and the fused kernel writes
    # int32. Preserve the reference helper's dtype contract for any int64
    # caller by treating it as an applicability miss.
    if group_indices.dtype != torch.int32:
        return None
    if (
        raw_sequence_lengths.dim() != 1
        or raw_sequence_lengths.dtype not in (torch.int32, torch.int64)
        or raw_sequence_lengths.stride(0) != 1
    ):
        return None
    if int(raw_sequence_lengths.shape[0]) != int(group_indices.shape[0]):
        return None

    rows, groups = group_indices.shape
    history_width = int(groups) * int(group_size)
    output_width = history_width + int(group_size) - 1
    output = torch.empty(
        (rows, output_width), dtype=torch.int32, device=group_indices.device
    )
    if rows == 0:
        return output
    block_cols = 128
    grid = (rows, triton.cdiv(output_width, block_cols))
    _expand_groups_append_tail_kernel[grid](
        group_indices,
        raw_sequence_lengths,
        output,
        group_indices.stride(0),
        output.stride(0),
        history_width,
        output_width,
        GROUP_SIZE=group_size,
        BLOCK_COLS=block_cols,
    )
    return output


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
