"""Destination slots for DSV4's optional prefill checkpoints.

The framework already copied immutable history into a new destination when
the checkpoint shares the restored block.  These mappings select only rows
produced by this forward.  The ordinary runtime mappings remain independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Iterator, Optional, Tuple

import torch


@dataclass(frozen=True)
class SparseSaveTargets:
    positions: torch.Tensor  # int64 [context batch, sparse points], -1 padding
    block_ids: torch.Tensor  # int32 [context batch, sparse points], -1 skip

    def __post_init__(self) -> None:
        if self.positions.dim() != 2 or self.block_ids.shape != self.positions.shape:
            raise ValueError(
                "sparse save positions and block IDs must have the same [B, S] shape"
            )
        if self.positions.dtype != torch.int64 or self.block_ids.dtype != torch.int32:
            raise TypeError("sparse save positions must be int64 and block IDs int32")

    @cached_property
    def _decoded_rows(self) -> Tuple[Tuple[int, int, int], ...]:
        # These are tiny pinned host metadata tensors from ModelInputGatherer.
        # Decode once per forward/tag. Every layer shares this instance, so
        # repeated .tolist() calls would cost CPU time across 40+ layers.
        points = self.positions.detach().cpu().tolist()
        blocks = self.block_ids.detach().cpu().tolist()
        rows = []
        for req, (point_row, block_row) in enumerate(zip(points, blocks)):
            for endpoint, block_id in zip(point_row, block_row):
                if endpoint >= 0 and block_id >= 0:
                    rows.append((req, int(endpoint), int(block_id)))
        return tuple(rows)

    def rows(self) -> Iterator[Tuple[int, int, int]]:
        return iter(self._decoded_rows)


def iter_sparse_save_slots(
    targets: Optional[SparseSaveTargets],
    positions: torch.Tensor,
    request_ids: torch.Tensor,
    *,
    entries_per_block: int,
    tokens_per_block: int,
    runtime_block_table: Optional[torch.Tensor] = None,
    cp_size: int = 1,
    cp_rank: int = 0,
    boundary_ratio: int = 1,
) -> Iterator[Tuple[int, torch.Tensor]]:
    """Yield ``(endpoint, slots[T])`` for each independent destination.

    ``positions`` and ``request_ids`` are in the producer's token domain:
    full global order after CP gather, or request-major order without CP.
    A destination owns only its endpoint block; prior blocks remain separate
    history bindings.  For fixed/ring pools, ``entries_per_block`` is the
    rank-local ring size and CP assigns rows by the full logical ring.
    FULL compressed KV pools use :func:`iter_sparse_full_kv_slots` below.
    """
    if targets is None:
        return
    if entries_per_block <= 0 or tokens_per_block <= 0 or cp_size <= 0:
        raise ValueError("sparse save block geometry must be positive")
    if not 0 <= cp_rank < cp_size or boundary_ratio <= 0:
        raise ValueError("invalid sparse save CP rank or boundary ratio")
    if positions.shape != request_ids.shape or positions.dim() != 1:
        raise ValueError(
            "sparse save token positions and request IDs must be flat and equal-sized"
        )

    pos = positions.to(torch.int64)
    req = request_ids.to(torch.int64)
    full_entries = entries_per_block * cp_size
    for row, endpoint, block_id in targets.rows():
        if row >= int(targets.positions.shape[0]) or endpoint <= 0:
            continue
        block_start = ((endpoint - 1) // tokens_per_block) * tokens_per_block
        first = max(block_start, endpoint - full_entries)
        valid = (req == row) & (pos >= first) & (pos < endpoint)
        if boundary_ratio != 1:
            valid = valid & (((pos + 1) % boundary_ratio) == 0)
        if cp_size > 1:
            ring_pos = pos % full_entries
            valid = valid & ((ring_pos // entries_per_block) == cp_rank)
            offset = ring_pos % entries_per_block
        else:
            offset = (
                (pos % tokens_per_block) // boundary_ratio
                if boundary_ratio != 1
                else pos % entries_per_block
            )

        if runtime_block_table is not None:
            # A stable runtime alias was already written by the ordinary
            # producer.  An independent destination must be materialized.
            cols = int(runtime_block_table.shape[1])
            if cols > 0:
                col = ((endpoint - 1) // tokens_per_block) % cols
                runtime_id = runtime_block_table[row, col]
                if runtime_id.device.type == "cpu":
                    if int(runtime_id.item()) == block_id:
                        continue
                else:
                    # Keep the alias decision on device. Calling .item() on
                    # the CUDA block table once per layer/endpoint serialized
                    # the prefill stream and was invalid under graph capture.
                    valid = valid & (runtime_id != block_id)
        slots = torch.where(valid, block_id * entries_per_block + offset, -1)
        yield endpoint, slots.contiguous()


def iter_sparse_full_kv_slots(
    targets: Optional[SparseSaveTargets],
    positions: torch.Tensor,
    request_ids: torch.Tensor,
    *,
    physical_tokens_per_block: int,
    kernel_tokens_per_block: int,
    entries_per_kernel_block: int,
    ratio: int,
    runtime_block_table: Optional[torch.Tensor] = None,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> Iterator[Tuple[int, torch.Tensor]]:
    """Map physical checkpoint IDs to the existing compressed KV layout.

    One physical ID expands to ``physical/kernel`` consecutive kernel IDs,
    exactly as ``BlockIds::updateKernelSlotAt`` does for the runtime table.
    CP FULL ownership is physical page round-robin, independent of state/SWA
    byte slicing.  The source copy supplies any restored history in this page.
    """
    if targets is None:
        return
    physical = int(physical_tokens_per_block)
    kernel = int(kernel_tokens_per_block)
    if physical <= 0 or kernel <= 0 or physical % kernel or ratio <= 0:
        raise ValueError("invalid compressed KV block geometry")
    if entries_per_kernel_block <= 0 or cp_size <= 0 or not 0 <= cp_rank < cp_size:
        raise ValueError("invalid compressed KV entries or CP rank")
    if positions.shape != request_ids.shape or positions.dim() != 1:
        raise ValueError(
            "compressed KV positions and request IDs must be flat and equal-sized"
        )

    pos = positions.to(torch.int64)
    req = request_ids.to(torch.int64)
    blocks_per_physical = physical // kernel
    for row, endpoint, block_id in targets.rows():
        if endpoint <= 0:
            continue
        physical_block = (endpoint - 1) // physical
        runtime_is_independent = None
        if runtime_block_table is not None:
            local_block = physical_block // cp_size
            first_kernel = local_block * blocks_per_physical
            if first_kernel < int(runtime_block_table.shape[1]):
                runtime_kernel = runtime_block_table[row, first_kernel]
                if runtime_kernel.device.type == "cpu":
                    if int(runtime_kernel.item()) // blocks_per_physical == block_id:
                        continue
                else:
                    runtime_is_independent = (
                        runtime_kernel // blocks_per_physical != block_id
                    )
        block_start = physical_block * physical
        valid = (req == row) & (pos >= block_start) & (pos < endpoint)
        valid = valid & (((pos + 1) % ratio) == 0)
        if runtime_is_independent is not None:
            valid = valid & runtime_is_independent
        if cp_size > 1:
            valid = valid & ((physical_block % cp_size) == cp_rank)
        kernel_offset = (pos - block_start) // kernel
        kernel_id = block_id * blocks_per_physical + kernel_offset
        slot = kernel_id * entries_per_kernel_block + (pos % kernel) // ratio
        yield endpoint, torch.where(valid, slot, -1).contiguous()
