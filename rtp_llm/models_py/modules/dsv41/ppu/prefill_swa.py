"""Read and write the CP byte slices of native SWA Prefill pages."""

import torch
import triton.language as tl
from triton import cdiv, jit

from rtp_llm.models_py.modules.dsv41._profiler import record_function_range
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
    block = first_block + position // SPAN - start // SPAN
    logical_byte = position % ENTRIES * WIDTH + byte % WIDTH
    owner, column = logical_byte // SLICE, logical_byte % SLICE
    valid = byte < count * WIDTH
    value = tl.load(
        gathered + (owner * packet_blocks + block) * SLICE + column, valid, other=0
    )
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


def gather_tails(pool, placement, prefixes, window):
    """Take one rank-major snapshot of the old SWA pages for the batch."""
    from rtp_llm.models_py.distributed.collective_torch import Group, all_gather

    width = _WIDTH["swa_kv"]
    counts = tuple(min(int(prefix), window - 1) for prefix in prefixes)
    packed = pool.tensor.new_empty((sum(counts), width))
    requests = []
    pages = []
    out_row = 0
    for rid, (prefix, count) in enumerate(zip(prefixes, counts)):
        if not count:
            continue
        start = int(prefix) - count
        requests.append((start, count, len(pages), out_row))
        pages.extend(
            (rid, column)
            for column in range(start // pool.span, (prefix - 1) // pool.span + 1)
        )
        out_row += count
    if not pages:
        return packed.split(counts)

    table = placement.table
    if len(prefixes) > table.shape[0] or max(col for _, col in pages) >= table.shape[1]:
        raise ValueError("native CP SWA tail outside physical table")
    coords = torch.tensor(pages, dtype=torch.int64, device=table.device)
    blocks = table[coords[:, 0], coords[:, 1]]
    if bool(((blocks < 0) | (blocks >= pool.tensor.shape[0])).any()):
        raise ValueError("native CP ring read refers to an unallocated physical block")
    local = pool.tensor.index_select(0, blocks)
    with record_function_range("dsv41.ppu.prefill.swa_tail.batch"):
        gathered = all_gather(local, Group.TP)
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
