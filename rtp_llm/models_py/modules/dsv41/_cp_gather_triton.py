"""Read only this CP rank's selected compact rows before the existing reduction."""

import triton
import triton.language as tl


@triton.jit
def gather_selected_kernel(
    pool,
    table,
    wanted,
    output,
    status,
    ROWS: tl.constexpr,
    ENTRIES: tl.constexpr,
    ENTRY_BYTES: tl.constexpr,
    POOL_PAGES: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    WANTED_STRIDE: tl.constexpr,
    RANK: tl.constexpr,
    CP: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    columns = tl.arange(0, BLOCK_BYTES)
    position = tl.load(wanted + rows * WANTED_STRIDE, rows < ROWS, other=-1).to(
        tl.int64
    )
    logical = tl.maximum(position, 0) // ENTRIES
    owned = (rows < ROWS) & (position >= 0) & (logical % CP == RANK)
    virtual = logical // CP
    in_table = owned & (virtual < TABLE_WIDTH)
    page = tl.load(table + virtual * TABLE_STRIDE, in_table, other=0).to(tl.int64)
    valid = in_table & (page > 0) & (page < POOL_PAGES)
    offset = (
        page[:, None] * PAGE_STRIDE
        + (tl.maximum(position, 0) % ENTRIES)[:, None] * ENTRY_BYTES
        + columns[None, :]
    )
    values = tl.load(
        pool + offset, valid[:, None] & (columns[None, :] < ENTRY_BYTES), other=0
    )
    tl.store(
        output + rows[:, None] * ENTRY_BYTES + columns[None, :],
        values,
        (rows[:, None] < ROWS) & (columns[None, :] < ENTRY_BYTES),
    )
    tl.store(status + tl.program_id(0), tl.sum((owned & ~valid).to(tl.int32), 0))
