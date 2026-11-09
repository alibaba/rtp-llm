"""Batch the readback of CP-sharded full pools without changing their layout."""

import torch
import triton.language as tl
from triton import cdiv, jit

from rtp_llm.models_py.modules.dsv41._profiler import record_function_range


_SCRATCH_BYTES = 128 * 1024**2


@jit
def _restore_rows(
    gathered,
    stripes,
    output,
    stripe_count,
    OWNER_ROWS: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    ENTRIES: tl.constexpr,
    WIDTH: tl.constexpr,
    PLANAR: tl.constexpr,
    TILE: tl.constexpr,
):
    stripe = tl.program_id(0)
    count = tl.load(stripes + stripe * 4 + 2)
    start = tl.load(stripes + stripe * 4 + 3)
    byte = tl.program_id(1) * TILE + tl.arange(0, TILE)
    row, column = byte // WIDTH, byte % WIDTH
    owner, owned_row = row // OWNER_ROWS, row % OWNER_ROWS
    page, entry = owned_row // PAGE_ROWS, owned_row % PAGE_ROWS
    if PLANAR:
        column = tl.where(
            column < 64, entry * 64 + column, ENTRIES * 64 + entry * 4 + column - 64
        )
    else:
        column = entry * WIDTH + column
    block = (owner * stripe_count + stripe).to(tl.int64)
    valid = row < count
    value = tl.load(
        gathered + block * BLOCK_BYTES + page * PAGE_BYTES + column, valid, other=0
    )
    tl.store(output + start * WIDTH + byte, value, valid)


def gather_pool(pool, placement, counts, *, width, planar):
    """Return packed request views in logical order using bounded CP all-gathers."""
    from rtp_llm.models_py.distributed.collective_torch import Group, all_gather

    size, ratio = placement.size, placement.ratio
    owner_rows, page_rows = placement.physical_span // ratio, pool.span // ratio
    if (
        pool.span % ratio
        or page_rows <= 0
        or pool.entries < page_rows
        or owner_rows != page_rows * placement.pages
    ):
        raise ValueError("CP readback requires complete compressed kernel pages")

    stripe_rows = size * owner_rows
    table = placement.table
    required_columns = cdiv(max(counts, default=0), stripe_rows)
    if table.shape[0] < len(counts) or table.shape[1] < required_columns:
        raise ValueError("native CP physical table is too short for batched readback")

    # One stripe is one physical block per rank. The same host lengths yield
    # the same groups on all ranks, including ranks with no live tail rows.
    stripes = []
    start = 0
    for request, count in enumerate(counts):
        for column, first in enumerate(range(0, count, stripe_rows)):
            stripes.append(
                (request, column, min(count - first, stripe_rows), start + first)
            )
        start += count

    output = pool.tensor.new_empty((start, width))
    page_bytes = pool.tensor.shape[1]
    block_bytes = placement.pages * page_bytes
    blocks = pool.tensor.view(-1, block_bytes)
    per_group = max(1, _SCRATCH_BYTES // ((size + 1) * block_bytes))
    for first in range(0, len(stripes), per_group):
        group = stripes[first : first + per_group]
        plan = torch.tensor(group, device=table.device, dtype=torch.int64)
        physical = table[plan[:, 0], plan[:, 1]]
        live = plan[:, 2] > placement.rank * owner_rows
        # Nonowners send a dummy block; their bytes are never consumed.
        physical = torch.where(live, physical, 0)
        local = blocks.index_select(0, physical)
        with record_function_range("dsv41.ppu.prefill.pool_blocks.batch"):
            gathered = all_gather(local, Group.TP)
        rows = max(item[2] for item in group)
        _restore_rows[(len(group), cdiv(rows * width, 1024))](
            gathered,
            plan,
            output,
            len(group),
            owner_rows,
            page_rows,
            block_bytes,
            page_bytes,
            pool.entries,
            width,
            planar,
            1024,
        )
        # Drop both buffers before allocating the next group's scratch.
        del local, gathered
    return output.split(counts)
