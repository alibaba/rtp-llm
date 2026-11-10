"""Read and write the CP byte slices of native SWA Prefill pages."""

import torch
import triton.language as tl
from triton import cdiv, jit

from rtp_llm.models_py.modules.dsv41._profiler import record_function_range
from rtp_llm.models_py.modules.dsv41.cp import cp_gather_rank_major
from rtp_llm.models_py.modules.dsv41.cp_cache import (
    page_interval,
    rank_major_slice_offset_jit,
    ring_coordinates_jit,
    swa_prefix_start,
)
from rtp_llm.models_py.modules.dsv41.ppu.native_cache import _WIDTH


@jit
def _restore_tails(
    gathered,
    plan,
    out,
    packet_blocks,
    SPAN: tl.constexpr,
    ENTRIES: tl.constexpr,
    SLICE: tl.constexpr,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
):
    request = tl.program_id(0)
    start = tl.load(plan + request * 4)
    count = tl.load(plan + request * 4 + 1)
    first_block = tl.load(plan + request * 4 + 2)
    out_row = tl.load(plan + request * 4 + 3)
    byte = tl.program_id(1) * TILE + tl.arange(0, TILE)
    position = start + byte // WIDTH
    column, row = ring_coordinates_jit(position, SPAN, ENTRIES)
    block = first_block + column - start // SPAN
    logical_byte = row * WIDTH + byte % WIDTH
    address = rank_major_slice_offset_jit(block, logical_byte, packet_blocks, SLICE)
    valid = byte < count * WIDTH
    value = tl.load(gathered + address, valid, other=0)
    tl.store(out + out_row * WIDTH + byte, value, valid)


@jit
def _write_slice(
    pool,
    blocks,
    positions,
    packed,
    count,
    RANK: tl.constexpr,
    ENTRIES: tl.constexpr,
    SLICE: tl.constexpr,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
):
    byte = tl.program_id(0) * TILE + tl.arange(0, TILE)
    row = byte // WIDTH
    valid = row < count
    position = tl.load(positions + row, valid, other=0)
    block = tl.load(blocks + row, valid, other=0).to(tl.int64)
    local_byte = position % ENTRIES * WIDTH + byte % WIDTH - RANK * SLICE
    owned = valid & (local_byte >= 0) & (local_byte < SLICE)
    value = tl.load(packed + byte, owned, other=0)
    tl.store(pool + block * SLICE + local_byte, value, owned)


@jit
def _write_prefill_ring_range(
    pool,
    table,
    packed,
    start,
    end,
    block_count,
    TABLE_STRIDE: tl.constexpr,
    SPAN: tl.constexpr,
    ENTRIES: tl.constexpr,
    RANK: tl.constexpr,
    SLICE: tl.constexpr,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
):
    column = start // SPAN + tl.program_id(0)
    block = tl.load(table + column * TABLE_STRIDE).to(tl.int64)
    local_byte = tl.program_id(1) * TILE + tl.arange(0, TILE)
    logical_byte = RANK * SLICE + local_byte
    ring_row = logical_byte // WIDTH
    last = tl.minimum(end, (column + 1) * SPAN) - 1
    # Each ring offset maps to its newest position in this page checkpoint.
    position = last - (last % ENTRIES + ENTRIES - ring_row) % ENTRIES
    valid = (
        (local_byte < SLICE)
        & (ring_row < ENTRIES)
        & (position >= tl.maximum(start, column * SPAN))
        & (block >= 0)
        & (block < block_count)
    )
    source_byte = (position - start).to(tl.int64) * WIDTH + logical_byte % WIDTH
    value = tl.load(packed + source_byte, valid, other=0)
    tl.store(pool + block * SLICE + local_byte, value, valid)


@jit
def _read_staged(
    domain,
    packed,
    positions,
    indices,
    output,
    domain_size,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    index = tl.load(indices + row)
    position = tl.load(positions + row)
    in_range = index < domain_size
    found = tl.load(domain + index, in_range, other=-1)
    valid = in_range & (position == found)
    column = tl.arange(0, TILE)
    value = tl.load(packed + index * WIDTH + column, valid & (column < WIDTH), other=0)
    tl.store(output + row.to(tl.int64) * WIDTH + column, value, column < WIDTH)


def _tail_layout(prefixes, window, span):
    starts = tuple(swa_prefix_start(int(prefix), window) for prefix in prefixes)
    counts = tuple(int(prefix) - start for prefix, start in zip(prefixes, starts))
    requests = []
    pages = []
    out_row = 0
    for rid, (start, count) in enumerate(zip(starts, counts)):
        if not count:
            continue
        first, stop = page_interval(start, start + count, span)
        requests.append((start, count, len(pages), out_row))
        pages.extend((rid, column) for column in range(first, stop))
        out_row += count
    return counts, pages, requests


def gather_tails(pool, placement, prefixes, window, *, plans=None, host_table=None, receive=None):
    """Take one rank-major snapshot of the old SWA pages for the batch."""
    width = _WIDTH["swa_kv"]
    table = placement.table
    key = ("swa_tail", tuple(prefixes), window, pool.span, table.device)
    entry = plans.get(key) if plans is not None else None
    if entry is None:
        counts, pages, requests = _tail_layout(prefixes, window, pool.span)
        if plans is not None and pages:
            from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_metadata import _upload

            host_coords = torch.tensor(pages, dtype=torch.int64, device="cpu")
            coords = _upload(pages, table.device, torch.int64)
            plan = _upload(requests, table.device, torch.int64)
            plans[key] = counts, pages, requests, host_coords, coords, plan
    else:
        counts, pages, requests, host_coords, coords, plan = entry

    packed = pool.tensor.new_empty((sum(counts), width))
    if not pages:
        return packed.split(counts)

    if len(prefixes) > table.shape[0] or max(col for _, col in pages) >= table.shape[1]:
        raise ValueError("native CP SWA tail outside physical table")
    if plans is None:
        coords = torch.tensor(pages, dtype=torch.int64, device=table.device)
    blocks = table[coords[:, 0], coords[:, 1]]
    checked = (
        host_table[host_coords[:, 0], host_coords[:, 1]]
        if plans is not None and host_table is not None else blocks
    )
    if bool(((checked < 0) | (checked >= pool.tensor.shape[0])).any()):
        raise ValueError("native CP ring read refers to an unallocated physical block")
    local = pool.tensor.index_select(0, blocks)
    with record_function_range("dsv41.ppu.prefill.swa_tail.batch"):
        gathered = receive.gather(local) if receive is not None else cp_gather_rank_major(local, placement.size)
    if plans is None:
        plan = torch.tensor(requests, dtype=torch.int64, device=table.device)
    _restore_tails[(len(requests), cdiv(max(counts) * width, 512))](
        gathered,
        plan,
        packed,
        len(pages),
        pool.span,
        placement.full_entries,
        placement.slice_elements,
        width,
        512,
    )
    return packed.split(counts)


def write_rows(pool, placement, request, positions, packed):
    """Store each byte of the prevalidated SWA rows on its owning rank."""
    if not positions.numel():
        return
    width = _WIDTH["swa_kv"]
    blocks = pool.table[request, positions // pool.span]
    if bool(((blocks < 0) | (blocks >= pool.tensor.shape[0])).any()):
        raise ValueError("native CP SWA write refers to an unallocated physical block")
    _write_slice[(cdiv(positions.numel() * width, 512),)](
        pool.tensor,
        blocks,
        positions,
        packed,
        positions.numel(),
        placement.rank,
        placement.full_entries,
        placement.slice_elements,
        width,
        512,
    )


def write_prefill_range(pool, placement, host_table, request, start, end, window, packed):
    """Store a request slice and return its old-path absolute-position receipt."""
    if start == end:
        return []
    first = start // pool.span
    stop = (end - 1) // pool.span + 1
    if start < 0 or stop > host_table.shape[1]:
        raise ValueError(f"native Prefill {pool.framework_tag} position outside block table")

    written = []
    blocks = set()
    for column, block in enumerate(host_table[request, first:stop].tolist(), first):
        column_end = min(end, (column + 1) * pool.span)
        if block == -1:
            if column_end > end - window:
                raise ValueError(f"native Prefill {pool.framework_tag} is missing a live ring block")
            continue
        if not 0 <= block < pool.tensor.shape[0] or block in blocks:
            # Keep the old validation and write order for aliased columns.
            return None
        blocks.add(block)
        begin = max(start, column * pool.span, column_end - pool.entries)
        written.extend(range(begin, column_end))

    if written:
        width = _WIDTH["swa_kv"]
        _write_prefill_ring_range[(stop - first, cdiv(placement.slice_elements, 512))](
            pool.tensor,
            pool.table[request],
            packed,
            start,
            end,
            pool.tensor.shape[0],
            pool.table.stride(1),
            pool.span,
            placement.full_entries,
            placement.rank,
            placement.slice_elements,
            width,
            512,
        )
    return written


def read_staged_rows(domain, packed, positions):
    """Keep the searchsorted contract, copying hits and zeroing misses in place."""
    width = _WIDTH["swa_kv"]
    output = packed.new_empty((positions.numel(), width))
    if positions.numel():
        indices = torch.searchsorted(domain, positions)
        _read_staged[(positions.numel(),)](
            domain, packed, positions, indices, output, domain.numel(), width, 1024
        )
    return output
