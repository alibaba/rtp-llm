"""Fixed-shape device addressing for the native V4.1 packed cache pools.

The Triton launch geometry depends only on captured tensor shapes. Negative
slots are padding and never read or write the reserved native block zero.
The CPU implementations are format oracles for contract tests.
"""

from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace

import torch


@lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def position_slots(
        positions,
        table,
        output,
        count: tl.constexpr,
        columns_per_request: tl.constexpr,
        table_rows: tl.constexpr,
        table_columns: tl.constexpr,
        table_stride_row: tl.constexpr,
        table_stride_column: tl.constexpr,
        span: tl.constexpr,
        entries: tl.constexpr,
        ratio: tl.constexpr,
        full: tl.constexpr,
        pool_blocks: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        position = tl.load(positions + index, index < count, other=-1).to(tl.int64)
        request = index // columns_per_request
        logical_block = position // span
        present = (
            (index < count)
            & (position >= 0)
            & (request < table_rows)
            & (logical_block < table_columns)
        )
        block = tl.load(
            table + request * table_stride_row + logical_block * table_stride_column,
            present,
            other=0,
        ).to(tl.int64)
        if full:
            offset = (position % span) // ratio
        else:
            offset = position % entries
        valid = present & (block > 0) & (block < pool_blocks) & (offset < entries)
        slot = tl.where(valid, block * entries + offset, -1)
        tl.store(output + index, slot, index < count)

    @triton.jit
    def packed_rows(
        pool,
        slots,
        rows,
        count: tl.constexpr,
        width: tl.constexpr,
        entries: tl.constexpr,
        pool_blocks: tl.constexpr,
        pool_stride: tl.constexpr,
        planar: tl.constexpr,
        write: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        slot = tl.load(slots + row).to(tl.int64)
        block, offset = slot // entries, slot % entries
        if planar:
            physical_column = tl.where(
                column < 64,
                offset * 64 + column,
                entries * 64 + offset * 4 + column - 64,
            )
        else:
            physical_column = offset * width + column
        valid = (slot >= 0) & (block > 0) & (block < pool_blocks)
        pool_ptr = pool + block * pool_stride + physical_column
        rows_ptr = rows + row * width + column
        if write:
            value = tl.load(rows_ptr, column < width, other=0)
            tl.store(pool_ptr, value, valid & (column < width))
        else:
            value = tl.load(pool_ptr, valid & (column < width), other=0)
            tl.store(rows_ptr, value, column < width)

    return SimpleNamespace(
        triton=triton, position_slots=position_slots, packed_rows=packed_rows
    )


def position_slots(
    positions: torch.Tensor,
    table: torch.Tensor,
    *,
    span: int,
    entries: int,
    ratio: int,
    full: bool,
    pool_blocks: int,
) -> torch.Tensor:
    """Map batch-major raw token positions to packed-entry physical slots."""
    if positions.ndim < 1 or positions.shape[0] > table.shape[0]:
        raise ValueError("Graph positions must have a batch axis covered by the table")
    positions = positions.to(dtype=torch.int64).contiguous()
    result = torch.empty_like(positions)
    count = positions.numel()
    if count == 0:
        return result
    columns = count // positions.shape[0]
    if positions.device.type == "cuda":
        kernels = _kernels()
        kernels.position_slots[(kernels.triton.cdiv(count, 128),)](
            positions,
            table,
            result,
            count,
            columns,
            table.shape[0],
            table.shape[1],
            table.stride(0),
            table.stride(1),
            span,
            entries,
            ratio,
            full,
            pool_blocks,
            BLOCK=128,
            num_warps=4,
        )
        return result
    flat = positions.reshape(positions.shape[0], columns)
    logical = flat.div(span, rounding_mode="floor")
    requests = torch.arange(flat.shape[0], device=table.device)[:, None]
    block = table[requests, logical.clamp(0, table.shape[1] - 1)].long()
    offset = (
        flat.remainder(span).div(ratio, rounding_mode="floor")
        if full
        else flat.remainder(entries)
    )
    valid = (
        (flat >= 0)
        & (logical < table.shape[1])
        & (block > 0)
        & (block < pool_blocks)
        & (offset < entries)
    )
    result.copy_(torch.where(valid, block * entries + offset, -1).reshape_as(result))
    return result


def read_packed_rows(
    pool: torch.Tensor,
    slots: torch.Tensor,
    *,
    width: int,
    entries: int,
    planar: bool,
) -> torch.Tensor:
    """Gather rows without changing native physical stride or planar layout."""
    slots = slots.to(dtype=torch.int64).contiguous()
    rows = torch.empty((*slots.shape, width), dtype=pool.dtype, device=pool.device)
    if not slots.numel():
        return rows
    if pool.device.type == "cuda":
        kernels = _kernels()
        kernels.packed_rows[(slots.numel(), kernels.triton.cdiv(width, 128))](
            pool, slots, rows, slots.numel(), width, entries, pool.shape[0],
            pool.stride(0), planar, False, BLOCK=128, num_warps=4,
        )
        return rows
    flat = slots.reshape(-1)
    block, offset = flat.div(entries, rounding_mode="floor"), flat.remainder(entries)
    column = torch.arange(width, device=pool.device)[None, :]
    if planar:
        physical = torch.where(
            column < 64, offset[:, None] * 64 + column,
            entries * 64 + offset[:, None] * 4 + column - 64,
        )
    else:
        physical = offset[:, None] * width + column
    values = pool[block.clamp(0, pool.shape[0] - 1)[:, None], physical]
    valid = (flat >= 0) & (block > 0) & (block < pool.shape[0])
    rows.copy_(torch.where(valid[:, None], values, 0).reshape_as(rows))
    return rows


def write_packed_rows(
    pool: torch.Tensor,
    slots: torch.Tensor,
    rows: torch.Tensor,
    *,
    width: int,
    entries: int,
    planar: bool,
) -> None:
    """Scatter rows; negative slots and block zero have no side effects."""
    if tuple(rows.shape) != (*slots.shape, width):
        raise ValueError("Graph cache row shape does not match slots and packed width")
    slots = slots.to(dtype=torch.int64).contiguous()
    rows = rows.to(dtype=pool.dtype).contiguous()
    if not slots.numel():
        return
    if pool.device.type == "cuda":
        kernels = _kernels()
        kernels.packed_rows[(slots.numel(), kernels.triton.cdiv(width, 128))](
            pool, slots, rows, slots.numel(), width, entries, pool.shape[0],
            pool.stride(0), planar, True, BLOCK=128, num_warps=4,
        )
        return
    flat = slots.reshape(-1)
    block, offset = flat.div(entries, rounding_mode="floor"), flat.remainder(entries)
    valid = (flat >= 0) & (block > 0) & (block < pool.shape[0])
    column = torch.arange(width, device=pool.device)[None, :]
    off = offset[valid, None]
    physical = (
        torch.where(column < 64, off * 64 + column, entries * 64 + off * 4 + column - 64)
        if planar else off * width + column
    )
    pool[block[valid, None], physical] = rows.reshape(-1, width)[valid]
