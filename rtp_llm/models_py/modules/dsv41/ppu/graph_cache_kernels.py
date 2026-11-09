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

    @triton.jit
    def store_cp_prefill_rows(
        pool, table, rows, start, first,
        POOL_STRIDE: tl.constexpr, TABLE_STRIDE: tl.constexpr, WIDTH: tl.constexpr,
        ENTRIES: tl.constexpr, PAGE: tl.constexpr, OWNER: tl.constexpr,
        CP: tl.constexpr, RANK: tl.constexpr, PLANAR: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        owned = first + tl.program_id(0).to(tl.int64)
        logical = (owned // OWNER * CP + RANK) * OWNER + owned % OWNER
        block = tl.load(table + (owned // PAGE) * TABLE_STRIDE).to(tl.int64)
        offset = owned % PAGE
        column = tl.arange(0, BLOCK)
        if PLANAR:
            target = tl.where(column < 64, offset * 64 + column,
                              ENTRIES * 64 + offset * 4 + column - 64)
        else:
            target = offset * WIDTH + column
        value = tl.load(rows + (logical - start) * WIDTH + column, column < WIDTH, other=0)
        tl.store(pool + block * POOL_STRIDE + target, value, column < WIDTH)

    @triton.jit
    def store_prefill_csa(pool, plan, rows, scores, WIDTH: tl.constexpr, SEPARATE: tl.constexpr):
        item = tl.program_id(0).to(tl.int64)
        target = tl.load(plan + 2 * item)
        source = tl.load(plan + 2 * item + 1)
        if SEPARATE:
            column = tl.arange(0, WIDTH // 2)
            value = tl.load(rows + source * (WIDTH // 2) + column)
            score = tl.load(scores + source * (WIDTH // 2) + column)
            tl.store(pool + target + column, value)
            tl.store(pool + target + WIDTH // 2 + column, score)
        else:
            column = tl.arange(0, WIDTH)
            value = tl.load(rows + source * WIDTH + column)
            tl.store(pool + target + column, value)

    @triton.jit
    def assemble_cp_prefill_indices(
        metadata, positions, selected, indices, lengths, position_stride,
        selected_row_stride, selected_col_stride,
        TOPK: tl.constexpr, RATIO: tl.constexpr, WINDOW: tl.constexpr,
        WIDTH: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        column = tl.arange(0, BLOCK)
        position = tl.load(positions + row * position_stride)
        swa_base = tl.load(metadata + row * 4 + 2)
        swa_first = tl.load(metadata + row * 4 + 3)
        swa_position = position - WINDOW + 1 + column - TOPK
        swa_live = (column >= TOPK) & (column < TOPK + WINDOW) & (swa_position >= swa_first)
        values = tl.where(swa_live, swa_base + swa_position - swa_first, -1)
        if TOPK:
            global_base = tl.load(metadata + row * 4)
            count = tl.load(metadata + row * 4 + 1)
            entry = tl.load(selected + row * selected_row_stride + column * selected_col_stride,
                            column < TOPK, other=-1)
            global_live = (column < TOPK) & (column < (position + 1) // RATIO) & (entry >= 0) & (entry < count)
            values = tl.where(global_live, global_base + entry, values)
        values = values.to(tl.int32)
        live = values >= 0
        rank = tl.cumsum(live.to(tl.int32))
        length = tl.sum(live.to(tl.int32), 0)
        destination = tl.where(live, rank - 1, length + column - rank)
        tl.store(indices + row * WIDTH + destination, values, column < TOPK + WINDOW)
        tl.store(indices + row * WIDTH + column, -1, (column >= TOPK + WINDOW) & (column < WIDTH))
        tl.store(lengths + row, length)

    return SimpleNamespace(
        triton=triton, position_slots=position_slots, packed_rows=packed_rows,
        store_cp_prefill_rows=store_cp_prefill_rows, store_prefill_csa=store_prefill_csa,
        assemble_cp_prefill_indices=assemble_cp_prefill_indices,
    )


def _owned_before(end, span, owned_start, owned_count):
    cycles, tail = divmod(end, span)
    return cycles * owned_count + min(max(tail - owned_start, 0), owned_count)


def try_assemble_cp_prefill_indices(metadata, positions, selected, ratio, window, width):
    """Assemble and compact the stable global-then-SWA sparse row in one launch."""
    rows = positions.numel()
    if not (
        positions.is_cuda and positions.ndim == 1 and positions.dtype == torch.int64
        and metadata.shape == (rows, 4) and metadata.is_contiguous()
        and metadata.dtype == torch.int64 and metadata.device == positions.device
    ):
        return None
    topk = 0
    selected_strides = (0, 0)
    if selected is not None:
        if not (selected.ndim == 2 and selected.shape[0] == rows and ratio > 0
                and selected.device == positions.device and selected.dtype in (torch.int32, torch.int64)):
            return None
        topk = selected.shape[1]
        selected_strides = selected.stride()
    if not 0 < window <= topk + window <= width:
        return None
    indices = torch.empty((rows, 1, width), dtype=torch.int32, device=positions.device)
    lengths = torch.empty(rows, dtype=torch.int32, device=positions.device)
    if rows:
        kernels = _kernels()
        kernels.assemble_cp_prefill_indices[(rows,)](
            metadata, positions, selected, indices, lengths, positions.stride(0),
            *selected_strides, topk, ratio, window, width, kernels.triton.next_power_of_2(width),
        )
    return indices, lengths


def try_write_cp_prefill_rows(pool, placement, host_table, request, start, rows, *, planar):
    """Store locally owned entries directly into the scheduler's kernel pages."""
    ratio = placement.ratio
    if start % ratio or pool.span % ratio:
        return None
    start //= ratio
    page = pool.span // ratio
    owner = placement.physical_span // ratio
    if page > pool.entries or owner % page:
        return None
    cycle = owner * placement.size
    rank_start = owner * placement.rank
    first = _owned_before(start, cycle, rank_start, owner)
    stop = _owned_before(start + rows.shape[0], cycle, rank_start, owner)
    if first == stop:
        return 0
    page_end = (stop - 1) // page + 1
    if page_end > host_table.shape[1]:
        return None
    blocks = host_table[request, first // page:page_end].tolist()
    if len(set(blocks)) != len(blocks) or any(not 0 <= block < pool.tensor.shape[0] for block in blocks):
        return None
    kernels = _kernels()
    kernels.store_cp_prefill_rows[(stop - first,)](
        pool.tensor, pool.table[request], rows, start, first, pool.tensor.stride(0),
        pool.table.stride(1), rows.shape[1], pool.entries, page, owner, placement.size,
        placement.rank, planar, kernels.triton.next_power_of_2(rows.shape[1]),
    )
    return stop - first


def try_write_prefill_csa(pool, placement, host_table, request, start, length, rows, *, scores=None):
    """Deduplicate latest raw states by physical address on the host."""
    end = start + rows.shape[0]
    last_page = (end - 1) // pool.span + 1
    if end > length or last_page > host_table.shape[1]:
        return None
    entries = pool.entries
    owned_start = placement.rank * entries
    cycle = placement.full_entries
    first_page = start // pool.span
    width = rows.shape[1] if scores is None else 2 * rows.shape[1]
    plan = {}
    for page, block in enumerate(host_table[request, first_page:last_page].tolist(), first_page):
        begin = max(start, page * pool.span)
        stop = min(end, (page + 1) * pool.span)
        if block == -1 and stop <= length - length % 2:
            continue
        if not 0 <= block < pool.tensor.shape[0]:
            return None
        for offset in range(entries):
            last = stop - 1 - (stop - 1 - owned_start - offset) % cycle
            if last >= begin:
                target = block * pool.tensor.stride(0) + offset * width
                plan[target] = last - start
    if plan:
        indices = torch.tensor(tuple(plan.items()), dtype=torch.int64).to(pool.tensor.device, non_blocking=True)
        _kernels().store_prefill_csa[(len(plan),)](pool.tensor, indices, rows, scores, width, scores is not None)
    owned = _owned_before(end, cycle, owned_start, entries) - _owned_before(start, cycle, owned_start, entries)
    return owned, len(plan)


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
