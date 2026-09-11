"""Compact V4.1 KV loads and bounded-workspace sparse attention kernels."""

import triton
import triton.language as tl


@triton.jit
def _ue8m0_to_float(byte):
    bits = byte.to(tl.uint32) << 23
    bits = tl.where(byte == 0, 0x00400000, bits)
    bits = tl.where(byte == 255, 0x7FC00000, bits)
    return bits.to(tl.uint32).to(tl.float32, bitcast=True)


@triton.jit
def _e2m1_to_float(code):
    magnitude = code & 7
    normal = tl.exp2((magnitude >> 1).to(tl.float32) - 1.0)
    normal *= 1.0 + (magnitude & 1).to(tl.float32) * 0.5
    value = tl.where(magnitude < 2, magnitude.to(tl.float32) * 0.5, normal)
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _load_compact_row(base, dims, valid, FORMAT: tl.constexpr):
    if FORMAT == 0:
        payload = tl.load(base + dims, mask=valid, other=0)
        scale = tl.load(base + 512 + dims // 32, mask=valid, other=127)
        return payload.to(tl.float8e4nv, bitcast=True).to(tl.float32) * _ue8m0_to_float(
            scale
        )
    else:
        packed = tl.load(base + dims // 2, mask=valid, other=0)
        code = (packed.to(tl.int32) >> ((dims % 2) * 4)) & 15
        if FORMAT == 1:
            scale = tl.load(base + 256 + dims // 16, mask=valid, other=0)
            scale_value = scale.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        else:
            scale = tl.load(base + 64 + dims // 32, mask=valid, other=127)
            scale_value = _ue8m0_to_float(scale)
        return _e2m1_to_float(code) * scale_value


@triton.jit
def gather_compact_kernel(
    pool,
    page_table,
    request_ids,
    positions,
    visible_lengths,
    output,
    status,
    NUM_REQUESTS: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    ROWS_PER_PAGE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    SLOT_COUNT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    FORMAT: tl.constexpr,
):
    query = tl.program_id(0)
    slot = tl.program_id(1)
    request = tl.load(request_ids + query).to(tl.int64)
    position = tl.load(positions + query * SLOT_COUNT + slot).to(tl.int64)
    visible = tl.load(visible_lengths + query).to(tl.int64)
    in_request = (request >= 0) & (request < NUM_REQUESTS)
    expected = (position >= 0) & (position < visible)
    block = tl.maximum(position, 0) // ROWS_PER_PAGE
    table_valid = in_request & (block < TABLE_WIDTH) & expected
    page = tl.load(
        page_table + request * TABLE_WIDTH + block, mask=table_valid, other=0
    ).to(tl.int64)
    valid = table_valid & (page > 0) & (page < NUM_PAGES)
    row = tl.maximum(position, 0) % ROWS_PER_PAGE
    dims = tl.arange(0, HEAD_DIM)
    base = pool + page * PAGE_STRIDE + row * ROW_BYTES
    values = _load_compact_row(base, dims, valid, FORMAT)
    tl.store(output + (query * SLOT_COUNT + slot) * HEAD_DIM + dims, values)
    error = tl.where(expected & ~valid, 1, 0)
    error |= tl.where((position < -1) | (visible < 0), 2, 0)
    tl.store(status + query * SLOT_COUNT + slot, error)


@triton.jit
def _accumulate_tile(
    q, values, valid, maximum, denominator, accumulator, SCALE: tl.constexpr
):
    scores = tl.sum(q[None, :] * values, axis=1) * SCALE
    scores = tl.where(valid, scores, float("-inf"))
    next_maximum = tl.maximum(maximum, tl.max(scores, axis=0))
    rescale = tl.exp(maximum - next_maximum)
    probabilities = tl.exp(scores - next_maximum)
    denominator = denominator * rescale + tl.sum(probabilities, axis=0)
    accumulator = accumulator * rescale + tl.sum(
        probabilities[:, None] * values, axis=0
    )
    return next_maximum, denominator, accumulator


@triton.jit
def compact_attention_kernel(
    query,
    request_ids,
    query_positions,
    replay_floors,
    swa_pool,
    swa_page_ids,
    swa_valid_starts,
    swa_valid_ends,
    global_pool,
    global_page_table,
    global_indices,
    sinks,
    output,
    lse,
    status,
    HEADS: tl.constexpr,
    NUM_REQUESTS: tl.constexpr,
    SWA_NUM_PAGES: tl.constexpr,
    SWA_ENTRIES: tl.constexpr,
    SWA_PAGE_STRIDE: tl.constexpr,
    GLOBAL_NUM_PAGES: tl.constexpr,
    GLOBAL_ENTRIES: tl.constexpr,
    GLOBAL_PAGE_STRIDE: tl.constexpr,
    GLOBAL_TABLE_WIDTH: tl.constexpr,
    GLOBAL_CAPACITY: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, 512)
    columns = tl.arange(0, BLOCK_N)
    request = tl.load(request_ids + row).to(tl.int64)
    position = tl.load(query_positions + row).to(tl.int64)
    floor = tl.load(replay_floors + row).to(tl.int64)
    active = position >= 0
    request_valid = (request >= 0) & (request < NUM_REQUESTS)
    query_valid = active & request_valid & (floor >= 0) & (floor <= position)
    q = tl.load(query + (row * HEADS + head) * 512 + dims).to(tl.float32)
    sink = tl.load(sinks + head)
    maximum = tl.maximum(sink, -1.0e30)
    denominator = tl.exp(sink - maximum)
    accumulator = tl.full((512,), 0.0, tl.float32)
    error = tl.where((position < -1) | (active & ~query_valid), 2, 0)

    swa_page = tl.load(swa_page_ids + request, mask=request_valid, other=0).to(tl.int64)
    swa_start = tl.load(swa_valid_starts + request, mask=request_valid, other=0).to(
        tl.int64
    )
    swa_end = tl.load(swa_valid_ends + request, mask=request_valid, other=0).to(
        tl.int64
    )
    swa_page_valid = (swa_page > 0) & (swa_page < SWA_NUM_PAGES)
    ring_valid = (
        (swa_start >= 0) & (swa_end >= swa_start) & (swa_end - swa_start <= SWA_ENTRIES)
    )
    for offset in range(0, 128, BLOCK_N):
        token = position - 127 + offset + columns
        expected = query_valid & (token >= 0) & (token >= floor) & (token <= position)
        valid = (
            expected
            & swa_page_valid
            & ring_valid
            & (token >= swa_start)
            & (token < swa_end)
        )
        error |= tl.where(tl.sum((expected & ~valid).to(tl.int32), axis=0) > 0, 1, 0)
        slots = tl.maximum(token, 0) % SWA_ENTRIES
        base = swa_pool + swa_page * SWA_PAGE_STRIDE + slots[:, None] * 528
        values = _load_compact_row(base, dims[None, :], valid[:, None], 0)
        maximum, denominator, accumulator = _accumulate_tile(
            q, values, valid, maximum, denominator, accumulator, SCALE
        )

    if COMPRESS_RATIO > 0:
        visible = (position + 1) // COMPRESS_RATIO
        for offset in range(0, GLOBAL_CAPACITY, BLOCK_N):
            candidate = offset + columns
            in_capacity = candidate < GLOBAL_CAPACITY
            index = tl.load(
                global_indices + row * GLOBAL_CAPACITY + candidate,
                mask=in_capacity,
                other=-1,
            ).to(tl.int64)
            previous = tl.load(
                global_indices + row * GLOBAL_CAPACITY + candidate - 1,
                mask=in_capacity & (candidate > 0),
                other=-1,
            ).to(tl.int64)
            malformed = in_capacity & (
                (index < -1)
                | (
                    (candidate > 0)
                    & (index >= 0)
                    & ((previous < 0) | (previous >= index))
                )
            )
            error |= tl.where(
                query_valid & (tl.sum(malformed.to(tl.int32), axis=0) > 0), 2, 0
            )
            expected = query_valid & in_capacity & (index >= 0) & (index < visible)
            block = tl.maximum(index, 0) // GLOBAL_ENTRIES
            in_table = expected & (block < GLOBAL_TABLE_WIDTH)
            page = tl.load(
                global_page_table + request * GLOBAL_TABLE_WIDTH + block,
                mask=in_table,
                other=0,
            ).to(tl.int64)
            valid = in_table & (page > 0) & (page < GLOBAL_NUM_PAGES)
            error |= tl.where(
                tl.sum((expected & ~valid).to(tl.int32), axis=0) > 0, 1, 0
            )
            slots = tl.maximum(index, 0) % GLOBAL_ENTRIES
            base = (
                global_pool + page[:, None] * GLOBAL_PAGE_STRIDE + slots[:, None] * 288
            )
            values = _load_compact_row(base, dims[None, :], valid[:, None], 1)
            maximum, denominator, accumulator = _accumulate_tile(
                q, values, valid, maximum, denominator, accumulator, SCALE
            )

    has_mass = query_valid & (denominator > 0)
    result = tl.where(has_mass, accumulator / denominator, 0.0)
    log_sum = tl.where(has_mass, tl.log(denominator) + maximum, float("-inf"))
    tl.store(output + (row * HEADS + head) * 512 + dims, result)
    tl.store(lse + row * HEADS + head, log_sum)
    tl.store(status + row * HEADS + head, error)
