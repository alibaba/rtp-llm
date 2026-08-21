"""Shared CP byte-sliced SWA slot compaction metadata."""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import torch

from rtp_llm.models_py.modules.dsv4.fp8._trap_utils import validate_slot_mapping


class CPByteSlicedSlotCompaction(NamedTuple):
    unique_blocks: torch.Tensor
    compact_slots: torch.Tensor
    gather_lens_cpu: Tuple[int, ...] = ()


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
    validate_slot_mapping(
        validation_site,
        slots,
        block_size=full_entries_per_block,
        num_blocks=int(num_blocks),
        negative_mode=negative_mode,
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
        )

    block_ids = valid_slots // full_entries_per_block
    block_offsets = valid_slots % full_entries_per_block
    unique_blocks, inverse = torch.unique(block_ids, sorted=True, return_inverse=True)
    compact_flat = torch.full_like(slots, -1)
    compact_flat[valid] = (
        inverse.to(torch.long) * full_entries_per_block + block_offsets
    )
    return CPByteSlicedSlotCompaction(
        unique_blocks=unique_blocks,
        compact_slots=compact_flat.view_as(slot_mapping).contiguous(),
        gather_lens_cpu=gather_lens_cpu,
    )
