"""Batch the readback of CP-sharded full pools without changing their layout."""

import torch
import triton.language as tl
from triton import cdiv, jit

from rtp_llm.models_py.modules.dsv41._profiler import record_function_range
from rtp_llm.models_py.modules.dsv41.cp import cp_gather_rank_major
from rtp_llm.models_py.modules.dsv41.cp_cache import round_robin_coordinates_jit


_SCRATCH_BYTES = 128 * 1024**2


@jit
def _restore_rows(
    gathered,
    stripes,
    output,
    scale_bytes,
    stripe_count,
    OWNER_ROWS: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    CP_SIZE: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    ENTRIES: tl.constexpr,
    WIDTH: tl.constexpr,
    PLANAR: tl.constexpr,
    SPLIT_INDEXER: tl.constexpr,
    TILE: tl.constexpr,
):
    stripe = tl.program_id(0)
    count = tl.load(stripes + stripe * 4 + 2)
    start = tl.load(stripes + stripe * 4 + 3)
    byte = tl.program_id(1) * TILE + tl.arange(0, TILE)
    row, column = byte // WIDTH, byte % WIDTH
    owner, page, entry = round_robin_coordinates_jit(
        row, OWNER_ROWS, PAGE_ROWS, CP_SIZE
    )
    if PLANAR:
        page_offset = tl.where(
            column < 64, entry * 64 + column, ENTRIES * 64 + entry * 4 + column - 64
        )
    else:
        page_offset = entry * WIDTH + column
    block = (owner * stripe_count + stripe).to(tl.int64)
    valid = row < count
    value = tl.load(
        gathered + block * BLOCK_BYTES + page * PAGE_BYTES + page_offset, valid, other=0
    )
    if SPLIT_INDEXER:
        is_payload = column < 64
        out_row = start + row
        tl.store(output + out_row * 64 + column, value, valid & is_payload)
        tl.store(scale_bytes + out_row * 4 + column - 64, value, valid & ~is_payload)
    else:
        tl.store(output + start * WIDTH + byte, value, valid)


def _stripe_groups(counts, stripe_rows, per_group, shared_prefix_stripes=0):
    # One stripe is one physical block per rank. Host lengths produce the
    # same groups on every rank, including ranks with no live tail rows.
    stripes = []
    start = 0
    for request, count in enumerate(counts):
        for column, first in enumerate(range(0, count, stripe_rows)):
            if request and column < shared_prefix_stripes:
                continue
            stripes.append((request, column, min(count - first, stripe_rows), start + first))
        start += count
    for first in range(0, len(stripes), per_group):
        group = stripes[first : first + per_group]
        yield len(group), max(row[2] for row in group), group


def gather_pool(
    pool, placement, counts, *, width, planar, split_indexer=False, plans=None,
    receive=None, shared_prefix_stripes=0,
):
    """Return logical request views using bounded CP all-gathers.

    A split Indexer read writes contiguous FP4 payload and packed scale planes
    directly, returning (int8 [N, 64], int32 [N]) for each request.
    """
    if split_indexer and (not planar or width != 68 or pool.tensor.dtype != torch.uint8):
        raise ValueError("split Indexer readback requires planar 68-byte rows")

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
    shared_rows = shared_prefix_stripes * stripe_rows
    if shared_rows and (len(counts) < 2 or len(set(counts)) != 1 or shared_rows > counts[0]):
        raise ValueError("shared readback prefix requires uniform complete stripes")
    table = placement.table
    required_columns = cdiv(max(counts, default=0), stripe_rows)
    if table.shape[0] < len(counts) or table.shape[1] < required_columns:
        raise ValueError("native CP physical table is too short for batched readback")

    start = sum(counts)
    output = pool.tensor.new_empty((start, 64 if split_indexer else width))
    scale_bytes = pool.tensor.new_empty((start, 4)) if split_indexer else None
    page_bytes = pool.tensor.shape[1]
    block_bytes = placement.pages * page_bytes
    blocks = pool.tensor.view(-1, block_bytes)
    per_group = max(1, _SCRATCH_BYTES // ((size + 1) * block_bytes))
    if plans is None:
        groups = _stripe_groups(counts, stripe_rows, per_group, shared_prefix_stripes)
    else:
        from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_metadata import _upload

        key = ("pool_readback", tuple(counts), stripe_rows, per_group, shared_prefix_stripes, table.device)
        groups = plans.get(key)
        if groups is None:
            groups = tuple(
                (size, rows, _upload(group, table.device, torch.int64))
                for size, rows, group in _stripe_groups(counts, stripe_rows, per_group, shared_prefix_stripes)
            )
            plans[key] = groups
    for group_size, rows, group in groups:
        plan = torch.tensor(group, device=table.device, dtype=torch.int64) if plans is None else group
        physical = table[plan[:, 0], plan[:, 1]]
        live = plan[:, 2] > placement.rank * owner_rows
        # Nonowners send a dummy block; their bytes are never consumed.
        physical = torch.where(live, physical, 0)
        local = blocks.index_select(0, physical)
        with record_function_range("dsv41.ppu.prefill.pool_blocks.batch"):
            gathered = receive.gather(local) if receive is not None else cp_gather_rank_major(local, size)
        _restore_rows[(group_size, cdiv(rows * width, 1024))](
            gathered,
            plan,
            output,
            scale_bytes,
            group_size,
            owner_rows,
            page_rows,
            size,
            block_bytes,
            page_bytes,
            pool.entries,
            width,
            planar,
            split_indexer,
            1024,
        )
        # Drop the local packet and this view before constructing the next group.
        del local, gathered
    if shared_rows:
        # Keep each consumer's logical request layout and packed bytes.
        view = output.view(len(counts), counts[0], -1)
        view[1:, :shared_rows].copy_(view[:1, :shared_rows])
        if scale_bytes is not None:
            view = scale_bytes.view(len(counts), counts[0], 4)
            view[1:, :shared_rows].copy_(view[:1, :shared_rows])
    if scale_bytes is not None:
        return tuple(
            zip(
                output.view(torch.int8).split(counts),
                scale_bytes.view(torch.int32).reshape(-1).split(counts),
            )
        )
    return output.split(counts)
