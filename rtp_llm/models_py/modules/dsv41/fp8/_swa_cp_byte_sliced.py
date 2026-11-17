"""Shared CP byte-sliced SWA slot compaction metadata."""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import torch


class CPByteSlicedSlotCompaction(NamedTuple):
    unique_blocks: torch.Tensor
    compact_slots: torch.Tensor
    gather_lens_cpu: Tuple[int, ...] = ()
    contiguous_block_start: int = -1


def normalize_gather_row_offsets(out, row_offsets, batch):
    if row_offsets is None:
        return None
    if not (
        out.ndim == 2
        and out.shape[1] == 512
        and out.stride(1) == 1
        and out.stride(0) >= 512
        and isinstance(row_offsets, torch.Tensor)
        and row_offsets.device == out.device
        and row_offsets.dtype in (torch.int32, torch.int64)
        and row_offsets.shape == (batch,)
    ):
        raise ValueError("Packed SWA gather needs a row matrix and per-request offsets")
    return row_offsets.contiguous()


def build_cp_byte_sliced_slot_compaction(
    slot_mapping: torch.Tensor,
    full_entries_per_block: int,
    num_blocks: int,
    validation_site: str,
    negative_mode: str,
    gather_lens: Optional[torch.Tensor] = None,
) -> CPByteSlicedSlotCompaction:
    """Precompute the block compaction used by CP byte-sliced SWA kernels."""
    full_entries_per_block = int(full_entries_per_block)
    slots = (
        slot_mapping.reshape(-1)
        .to(dtype=torch.int64, device=slot_mapping.device)
        .contiguous()
    )
    gather_lens_cpu: Tuple[int, ...] = ()
    if gather_lens is not None:
        gather_lens_cpu = tuple(
            int(v)
            for v in gather_lens.to(device="cpu", dtype=torch.int32)
            .reshape(-1)
            .tolist()
        )

    valid = slots >= 0
    valid_slots = slots[valid]
    if valid_slots.numel() == 0:
        return CPByteSlicedSlotCompaction(
            unique_blocks=torch.empty((0,), dtype=torch.long, device=slots.device),
            compact_slots=torch.full_like(slot_mapping, -1, dtype=torch.long),
            gather_lens_cpu=gather_lens_cpu,
            contiguous_block_start=-1,
        )

    block_ids = valid_slots // full_entries_per_block
    block_offsets = valid_slots % full_entries_per_block
    unique_blocks, inverse = torch.unique(block_ids, sorted=True, return_inverse=True)
    compact_flat = torch.full_like(slots, -1)
    compact_flat[valid] = (
        inverse.to(torch.long) * full_entries_per_block + block_offsets
    )
    contiguous_block_start = -1
    if gather_lens is not None:
        # Only prefix reads consume the contiguous-block fast path. Writers do
        # not need a host copy of GPU metadata and must remain launch-ahead.
        unique_blocks_cpu = unique_blocks.detach().cpu().tolist()
        contiguous_block_start = int(unique_blocks_cpu[0])
        if int(unique_blocks_cpu[-1]) - contiguous_block_start + 1 != len(
            unique_blocks_cpu
        ):
            contiguous_block_start = -1
    return CPByteSlicedSlotCompaction(
        unique_blocks=unique_blocks,
        compact_slots=compact_flat.view_as(slot_mapping).contiguous(),
        gather_lens_cpu=gather_lens_cpu,
        contiguous_block_start=contiguous_block_start,
    )
