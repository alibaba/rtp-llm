"""Integer-only draft lane mapping and pre-replay validation.

The validation flag is consumed on the host before any cache writer runs.
Host-derived expected positions remain independent of the device metadata.
"""

import torch
import triton as tr
import triton.language as tl


def is_supported(*tensors):
    return bool(tensors) and all(
        isinstance(t, torch.Tensor)
        and t.is_cuda
        and t.is_contiguous()
        and t.dtype in (torch.int32, torch.int64)
        and t.device == tensors[0].device
        for t in tensors
    )


@tr.jit
def _draft_rows(
    lengths,
    prefixes,
    cu,
    sources,
    inverse,
    valid,
    positions,
    visible,
    BATCH: tl.constexpr,
    LANES: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
    REQUEST_BLOCK: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    request = row // LANES
    lane = row % LANES
    length = tl.load(lengths + request, row < CAPACITY, other=0)
    prefix = tl.load(prefixes + request, row < CAPACITY, other=0)
    begin = tl.load(cu + request, row < CAPACITY, other=0).to(tl.int64)
    active = (row < CAPACITY) & (lane < length)
    tl.store(sources + row, tl.where(active, begin + lane, 0), row < CAPACITY)
    tl.store(visible + row, tl.where(active, prefix + lane + 1, 0), row < CAPACITY)

    batch_id = tl.arange(0, REQUEST_BLOCK)
    ends = tl.load(cu + batch_id + 1, batch_id < BATCH, other=2147483647)
    owner = tl.sum(
        ((row[:, None] >= ends[None, :]) & (batch_id[None, :] < BATCH)).to(tl.int32),
        axis=1,
    )
    owner = tl.minimum(owner, BATCH - 1)
    packed_begin = tl.load(cu + owner).to(tl.int64)
    local = row.to(tl.int64) - packed_begin
    total = tl.load(cu + BATCH)
    packed_valid = row < total
    packed_prefix = tl.load(prefixes + owner).to(tl.int64)
    tl.store(
        inverse + row, tl.where(packed_valid, owner * LANES + local, 0), row < CAPACITY
    )
    tl.store(valid + row, packed_valid, row < CAPACITY)
    tl.store(
        positions + row,
        tl.where(packed_valid, packed_prefix + local, 0),
        row < CAPACITY,
    )


def draft_rows(lengths, prefixes, cu, token_capacity):
    batch = lengths.numel()
    if (
        not is_supported(lengths, prefixes, cu)
        or lengths.ndim != 1
        or prefixes.shape != lengths.shape
        or cu.shape != (batch + 1,)
        or not 1 <= batch <= 32
        or token_capacity % batch
        or not 1 <= token_capacity // batch <= 4
    ):
        return None
    lanes = token_capacity // batch
    sources = torch.empty(token_capacity, device=cu.device, dtype=torch.int64)
    inverse = torch.empty_like(sources)
    valid = torch.empty(token_capacity, device=cu.device, dtype=torch.bool)
    positions = torch.empty_like(sources)
    visible = torch.empty((batch, lanes), device=cu.device, dtype=torch.int32)
    _draft_rows[(1,)](
        lengths,
        prefixes,
        cu,
        sources,
        inverse,
        valid,
        positions,
        visible,
        batch,
        lanes,
        token_capacity,
        tr.next_power_of_2(token_capacity),
        tr.next_power_of_2(batch),
        num_warps=4,
    )
    return sources, inverse, valid, positions, visible


@tr.jit
def _replay_invalid(
    lengths,
    prefixes,
    table,
    transported,
    expected,
    invalid,
    BATCH: tl.constexpr,
    COLUMNS: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    PAGE_SHIFT: tl.constexpr,
    LIVE_TOKENS: tl.constexpr,
    FACTOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    length = tl.load(lengths + row, row < BATCH, other=0)
    prefix = tl.load(prefixes + row, row < BATCH, other=0)
    active = (row < BATCH) & (length > 0)
    # Arithmetic shifts match floor division, including invalid negative IDs.
    aligned = (prefix >> 2) << 2
    first = aligned >> PAGE_SHIFT
    last = (prefix + length - 1) >> PAGE_SHIFT
    first_col = tl.minimum(tl.maximum(first, 0), COLUMNS - 1)
    last_col = tl.minimum(tl.maximum(last, 0), COLUMNS - 1)
    first_page = tl.load(table + row * COLUMNS + first_col, row < BATCH, other=0)
    last_page = tl.load(table + row * COLUMNS + last_col, row < BATCH, other=0)
    bad_state = active & (
        (first < 0)
        | (last >= COLUMNS)
        | (first_page <= 0)
        | (first_page >= POOL_SIZE)
        | (last_page <= 0)
        | (last_page >= POOL_SIZE)
    )
    actual = tl.load(transported + row, row < LIVE_TOKENS * FACTOR, other=0)
    wanted = tl.load(expected + row // FACTOR, row < LIVE_TOKENS * FACTOR, other=0)
    bad_position = (row < LIVE_TOKENS * FACTOR) & (actual != wanted)
    tl.store(invalid, tl.sum((bad_state | bad_position).to(tl.int32), 0) != 0)


def replay_invalid(
    lengths,
    prefixes,
    table,
    transported,
    expected,
    *,
    pool_size,
    page_size,
    token_capacity,
    position_factor
):
    batch = lengths.numel()
    if (
        not is_supported(lengths, prefixes, table, transported, expected)
        or lengths.ndim != 1
        or prefixes.shape != lengths.shape
        or lengths.dtype != torch.int32
        or prefixes.dtype != torch.int32
        or table.ndim != 2
        or table.shape[0] != batch
        or table.shape[1] <= 0
        or not 1 <= batch <= 32
        or not 0 <= expected.numel() <= token_capacity <= 128
        or expected.ndim != 1
        or position_factor not in (1, 3)
        or transported.numel() != token_capacity * position_factor
        or page_size <= 0
        or page_size > 1 << 30
        or page_size & (page_size - 1)
        or pool_size <= 0
    ):
        return None
    invalid = torch.empty((), device=lengths.device, dtype=torch.bool)
    _replay_invalid[(1,)](
        lengths,
        prefixes,
        table,
        transported,
        expected,
        invalid,
        batch,
        table.shape[1],
        pool_size,
        page_size.bit_length() - 1,
        expected.numel(),
        position_factor,
        tr.next_power_of_2(max(batch, expected.numel() * position_factor)),
        num_warps=4,
    )
    return invalid
