"""Raw-byte staging and metadata for the fixed V4.1 FlashMLA reader."""

import triton
import triton.language as tl


@triton.jit
def build_indices_kernel(
    request_ids,
    positions,
    floors,
    swa_page_ids,
    swa_starts,
    swa_ends,
    global_table,
    global_indices,
    main,
    main_lengths,
    extra,
    extra_lengths,
    status,
    query_valid,
    have_kv,
    REQUESTS: tl.constexpr,
    SWA_PAGES: tl.constexpr,
    SWA_ENTRIES: tl.constexpr,
    GLOBAL_PAGES: tl.constexpr,
    GLOBAL_ENTRIES: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    CAPACITY: tl.constexpr,
    EXTRA_WIDTH: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    request = tl.load(request_ids + row).to(tl.int64)
    position = tl.load(positions + row).to(tl.int64)
    floor = tl.load(floors + row).to(tl.int64)
    valid = (
        (position >= 0)
        & (position < 1048576)
        & (request >= 0)
        & (request < REQUESTS)
        & (floor >= 0)
        & (floor <= position)
    )
    error = ((position < -1) | ((position >= 0) & ~valid)).to(tl.int32) * 2
    selected = tl.minimum(tl.maximum(request, 0), max(REQUESTS - 1, 0))
    page = tl.load(swa_page_ids + selected, REQUESTS > 0, other=0).to(tl.int64)
    start = tl.load(swa_starts + selected, REQUESTS > 0, other=0).to(tl.int64)
    end = tl.load(swa_ends + selected, REQUESTS > 0, other=0).to(tl.int64)
    slots = tl.arange(0, 128)
    token = tl.maximum(tl.maximum(position - 127, floor), 0) + slots
    expected = valid & (token <= position)
    ring_valid = (start >= 0) & (end >= start) & (end - start <= SWA_ENTRIES)
    usable = (
        expected
        & ring_valid
        & (page > 0)
        & (page < SWA_PAGES)
        & (token >= start)
        & (token < end)
    )
    physical = page * SWA_ENTRIES + token % SWA_ENTRIES
    tl.store(main + row * 128 + slots, tl.where(usable, physical, -1))
    tl.store(main_lengths + row, tl.sum(expected.to(tl.int32), 0))
    error |= tl.sum((expected & ~usable).to(tl.int32), 0) > 0
    present = tl.sum(usable.to(tl.int32), 0) > 0
    if RATIO:
        cols = tl.arange(0, BLOCK)
        logical = tl.load(
            global_indices + row * CAPACITY + cols, cols < CAPACITY, other=-1
        ).to(tl.int64)
        previous = tl.load(
            global_indices + row * CAPACITY + cols - 1,
            (cols > 0) & (cols < CAPACITY),
            other=-1,
        ).to(tl.int64)
        malformed = (logical < -1) | (
            (cols > 0) & (logical >= 0) & ((previous < 0) | (logical <= previous))
        )
        error |= (valid & (tl.sum(malformed.to(tl.int32), 0) > 0)).to(tl.int32) * 2
        expected_extra = valid & (logical >= 0) & (logical < (position + 1) // RATIO)
        block = tl.maximum(logical, 0) // GLOBAL_ENTRIES
        safe_block = tl.minimum(block, max(TABLE_WIDTH - 1, 0))
        physical_page = tl.load(
            global_table + selected * TABLE_WIDTH + safe_block,
            (REQUESTS > 0) & (TABLE_WIDTH > 0),
            other=0,
        ).to(tl.int64)
        usable_extra = (
            expected_extra
            & (block < TABLE_WIDTH)
            & (physical_page > 0)
            & (physical_page < GLOBAL_PAGES)
        )
        physical_extra = (
            physical_page * GLOBAL_ENTRIES + tl.maximum(logical, 0) % GLOBAL_ENTRIES
        )
        tl.store(
            extra + row * EXTRA_WIDTH + cols,
            tl.where(usable_extra, physical_extra, -1),
            cols < EXTRA_WIDTH,
        )
        tl.store(extra_lengths + row, tl.sum(expected_extra.to(tl.int32), 0))
        error |= tl.sum((expected_extra & ~usable_extra).to(tl.int32), 0) > 0
        present |= tl.sum(usable_extra.to(tl.int32), 0) > 0
    tl.store(status + row, error)
    tl.store(query_valid + row, valid)
    tl.store(have_kv + row, present)


@triton.jit
def pack_selected_rows_kernel(
    source,
    indices,
    packed,
    remapped,
    ENTRIES: tl.constexpr,
    SOURCE_STRIDE: tl.constexpr,
    SLOTS: tl.constexpr,
    PAYLOAD: tl.constexpr,
    SCALES: tl.constexpr,
    PACKED_STRIDE: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
):
    page = tl.program_id(0).to(tl.int64)
    slot = tl.program_id(1) * BLOCK_SLOTS + tl.arange(0, BLOCK_SLOTS)
    physical = tl.load(
        indices + (page - 1) * SLOTS + slot, (page > 0) & (slot < SLOTS), other=-1
    ).to(tl.int64)
    valid = physical >= 0
    address = (physical // ENTRIES) * SOURCE_STRIDE + (physical % ENTRIES) * (
        PAYLOAD + SCALES
    )
    dims = tl.arange(0, PAYLOAD)
    data = tl.load(
        source + address[:, None] + dims[None, :],
        valid[:, None] & (slot[:, None] < SLOTS),
        other=0,
    )
    tl.store(
        packed + page * PACKED_STRIDE + slot[:, None] * PAYLOAD + dims[None, :],
        data,
        slot[:, None] < SLOTS,
    )
    scale_dims = tl.arange(0, SCALES)
    scales = tl.load(
        source + address[:, None] + PAYLOAD + scale_dims[None, :],
        valid[:, None] & (slot[:, None] < SLOTS),
        other=0,
    )
    tl.store(
        packed
        + page * PACKED_STRIDE
        + SLOTS * PAYLOAD
        + slot[:, None] * SCALES
        + scale_dims[None, :],
        scales,
        slot[:, None] < SLOTS,
    )
    tl.store(
        remapped + (page - 1) * SLOTS + slot,
        tl.where(valid, page * SLOTS + slot, -1),
        (page > 0) & (slot < SLOTS),
    )
