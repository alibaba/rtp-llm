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
    position_factor,
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


@tr.jit
def _replay_invalid_from_sources(
    source_lengths,
    source_prefixes,
    lengths,
    prefixes,
    table,
    transported,
    cu,
    invalid,
    BATCH: tl.constexpr,
    COLUMNS: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    PAGE_SHIFT: tl.constexpr,
    CAPACITY: tl.constexpr,
    FACTOR: tl.constexpr,
    REQUEST_BLOCK: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
):
    request = tl.arange(0, REQUEST_BLOCK)
    request_mask = request < BATCH
    # Source arithmetic must widen before both the scan and prefix + length.
    source_length = tl.load(source_lengths + request, request_mask, other=0).to(
        tl.int64
    )
    source_prefix = tl.load(source_prefixes + request, request_mask, other=0).to(
        tl.int64
    )
    end = tl.cumsum(source_length, 0)
    begin = end - source_length
    total = tl.sum(source_length, 0)
    active = request_mask & (source_length > 0)
    zeros = tl.cumsum((request_mask & (source_length == 0)).to(tl.int32), 0)
    captured_length = tl.load(lengths + request, request_mask, other=0).to(tl.int64)
    captured_prefix = tl.load(prefixes + request, request_mask, other=0).to(tl.int64)
    captured_cu = tl.load(cu + request, request_mask, other=0).to(tl.int64)
    bad_request = (
        (request_mask & ((source_length < 0) | (source_length > CAPACITY // BATCH)))
        | (active & (source_prefix <= 0))
        | (active & (zeros > 0))
        | (active & (source_prefix + source_length - 1 > 2147483647))
        | (request_mask & (captured_length != source_length))
        | (active & (captured_prefix != source_prefix))
        | (request_mask & (captured_cu != begin))
    )

    first = ((source_prefix >> 2) << 2) >> PAGE_SHIFT
    last = (source_prefix + source_length - 1) >> PAGE_SHIFT
    bad_request = bad_request | (
        active & ((first < 0) | (first >= COLUMNS) | (last < 0) | (last >= COLUMNS))
    )
    # For 1 <= length <= lanes <= 4, last_token - aligned_start <= 6.
    # A page holds >= 1 token, so the inclusive interval has <= 7 columns.
    # Invalid lengths already reject; restricting their loads keeps work bounded.
    column = first[:, None] + tl.arange(0, 8)[None, :]
    visible = (
        active[:, None]
        & (source_length[:, None] <= CAPACITY // BATCH)
        & (column >= 0)
        & (column < COLUMNS)
        & (column <= last[:, None])
    )
    safe_column = tl.minimum(tl.maximum(column, 0), COLUMNS - 1)
    safe_request = tl.minimum(request, BATCH - 1).to(tl.int64)
    physical = tl.load(
        table + safe_request[:, None] * COLUMNS + safe_column,
        visible,
        other=0,
    )
    bad_page = visible & ((physical <= 0) | (physical >= POOL_SIZE))
    bad_request = bad_request | (tl.sum(bad_page.to(tl.int32), 1) != 0)

    token = tl.arange(0, TOKEN_BLOCK)
    # Only the independent source partition determines the packed token owner.
    owner = tl.sum(
        ((token[:, None].to(tl.int64) >= end[None, :]) & request_mask[None, :]).to(
            tl.int32
        ),
        1,
    )
    owner = tl.minimum(tl.maximum(owner, 0), BATCH - 1)
    token_prefix = tl.load(source_prefixes + owner).to(tl.int64)
    token_begin = tl.gather(begin, owner, 0)
    wanted_position = token_prefix + token.to(tl.int64) - token_begin
    AXIS_BLOCK: tl.constexpr = 1 if FACTOR == 1 else 4
    axis = tl.arange(0, AXIS_BLOCK)
    position_mask = (token[:, None] < CAPACITY) & (axis[None, :] < FACTOR)
    actual_position = tl.load(
        transported + token[:, None] * FACTOR + axis[None, :],
        position_mask,
        other=0,
    ).to(tl.int64)
    bad_position = (
        position_mask
        & (token[:, None] < total)
        & (actual_position != wanted_position[:, None])
    )
    captured_total = tl.load(cu + BATCH).to(tl.int64)
    tl.store(
        invalid,
        (
            (tl.sum(bad_request.to(tl.int32), 0) != 0)
            | (tl.sum(tl.sum(bad_position.to(tl.int32), 1), 0) != 0)
            | (total < 0)
            | (total > CAPACITY)
            | (captured_total != total)
        ),
    )


def replay_invalid_from_sources(
    source_lengths,
    source_prefixes,
    lengths,
    prefixes,
    table,
    transported,
    cu,
    *,
    pool_size,
    page_size,
    token_capacity,
    position_factor,
):
    """Validate independent source snapshots in one launch without a host read.

    Admission owns snapshot independence, producer fences and replay lifetime.
    Unsupported tensor metadata or page geometry falls back via ``None``.
    """
    tensors = (
        source_lengths,
        source_prefixes,
        lengths,
        prefixes,
        table,
        transported,
        cu,
    )
    if not is_supported(*tensors) or any(t.dtype != torch.int32 for t in tensors):
        return None
    batch = source_lengths.numel()
    if (
        source_lengths.ndim != 1
        or not 1 <= batch <= 32
        or any(t.shape != (batch,) for t in (source_prefixes, lengths, prefixes))
        or cu.shape != (batch + 1,)
        or not isinstance(token_capacity, int)
        or not 0 < token_capacity <= 128
        or token_capacity % batch
        or not 1 <= token_capacity // batch <= 4
        or not isinstance(position_factor, int)
        or position_factor not in (1, 3)
        or transported.numel() != token_capacity * position_factor
        or table.ndim != 2
        or table.shape[0] != batch
        or table.shape[1] <= 0
        or not isinstance(pool_size, int)
        or pool_size <= 0
        or not isinstance(page_size, int)
        or not 0 < page_size <= 1 << 30
        or page_size & (page_size - 1)
    ):
        return None
    invalid = torch.empty((), device=source_lengths.device, dtype=torch.bool)
    _replay_invalid_from_sources[(1,)](
        source_lengths,
        source_prefixes,
        lengths,
        prefixes,
        table,
        transported,
        cu,
        invalid,
        batch,
        table.shape[1],
        min(pool_size, 1 << 31),
        page_size.bit_length() - 1,
        token_capacity,
        position_factor,
        tr.next_power_of_2(batch),
        tr.next_power_of_2(token_capacity),
        num_warps=4,
    )
    return invalid
