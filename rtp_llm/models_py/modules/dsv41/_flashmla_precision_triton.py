"""Preserve the FP32 H64 guard while bounding partial extra-KV arithmetic."""

import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv41._compact_reader_triton import (
    _accumulate_tile,
    _load_compact_row,
    compact_attention_kernel,
)


@triton.jit
def _load_precision_global_row(base, dims, valid):
    packed = tl.load(base + dims // 2, mask=valid, other=0)
    code = (packed.to(tl.uint32) >> ((dims % 2) * 4)) & 15
    magnitude = code & 7
    # Every E2M1 value is exact in FP32; construct its exponent and mantissa.
    bits = tl.where(
        magnitude < 2,
        magnitude * 0x3F000000,
        ((magnitude // 2 + 126) << 23) | ((magnitude & 1) << 22),
    )
    value = (bits | ((code & 8) << 28)).to(tl.uint32).to(tl.float32, bitcast=True)
    scale = tl.load(base + 256 + dims // 16, mask=valid, other=0)
    return value * scale.to(tl.float8e4nv, bitcast=True).to(tl.float32)


# Keep the bounded calculation separate from the legacy precision path.
@triton.jit(noinline=True)
def _bounded_attention(
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
    HEAD_PAIR: tl.constexpr = False,
):
    row = tl.program_id(0)
    head = tl.program_id(1) * (2 if HEAD_PAIR else 1)
    dims = tl.arange(0, 512)
    columns = tl.arange(0, BLOCK_N)
    request = tl.load(request_ids + row).to(tl.int64)
    position = tl.load(query_positions + row).to(tl.int64)
    floor = tl.load(replay_floors + row).to(tl.int64)
    active = position >= 0
    request_valid = (request >= 0) & (request < NUM_REQUESTS)
    query_valid = (
        active
        & (position < 1048576)
        & request_valid
        & (floor >= 0)
        & (floor <= position)
    )
    error = tl.where((position < -1) | (active & ~query_valid), 2, 0)
    visible = (position + 1) // COMPRESS_RATIO
    candidates = tl.arange(0, triton.next_power_of_2(GLOBAL_CAPACITY))
    global_in_capacity = candidates < GLOBAL_CAPACITY
    indices = tl.load(
        global_indices + row * GLOBAL_CAPACITY + candidates,
        mask=global_in_capacity,
        other=-1,
    ).to(tl.int64)
    preceding = tl.load(
        global_indices + row * GLOBAL_CAPACITY + candidates - 1,
        mask=global_in_capacity & (candidates > 0),
        other=-1,
    ).to(tl.int64)
    malformed = global_in_capacity & (
        (indices < -1)
        | (
            (candidates > 0)
            & (indices >= 0)
            & ((preceding < 0) | (preceding >= indices))
        )
    )
    error |= tl.where(query_valid & (tl.sum(malformed.to(tl.int32), axis=0) > 0), 2, 0)
    expected_global = (
        query_valid & global_in_capacity & (indices >= 0) & (indices < visible)
    )
    blocks = tl.maximum(indices, 0) // GLOBAL_ENTRIES
    global_in_table = expected_global & (blocks < GLOBAL_TABLE_WIDTH)
    pages = tl.load(
        global_page_table + request * GLOBAL_TABLE_WIDTH + blocks,
        mask=global_in_table,
        other=0,
    ).to(tl.int64)
    valid_global = global_in_table & (pages > 0) & (pages < GLOBAL_NUM_PAGES)
    error |= tl.where(
        tl.sum((expected_global & ~valid_global).to(tl.int32), axis=0) > 0, 1, 0
    )
    # Scan every index before bounding arithmetic, including malformed tails/holes.
    global_limit = tl.max(tl.where(valid_global, candidates + 1, 0), axis=0)
    q = tl.load(query + (row * HEADS + head) * 512 + dims).to(tl.float32)
    sink = tl.load(sinks + head)
    maximum = tl.maximum(sink, -1.0e30)
    denominator = tl.exp(sink - maximum)
    accumulator = tl.full((512,), 0.0, tl.float32)
    if HEAD_PAIR:
        q_second = tl.load(query + (row * HEADS + head + 1) * 512 + dims).to(tl.float32)
        sink_second = tl.load(sinks + head + 1)
        maximum_second = tl.maximum(sink_second, -1.0e30)
        denominator_second = tl.exp(sink_second - maximum_second)
        accumulator_second = tl.full((512,), 0.0, tl.float32)

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
        if HEAD_PAIR:
            maximum_second, denominator_second, accumulator_second = _accumulate_tile(
                q_second,
                values,
                valid,
                maximum_second,
                denominator_second,
                accumulator_second,
                SCALE,
            )

    for offset in range(0, global_limit, BLOCK_N):
        candidate = offset + columns
        in_capacity = candidate < GLOBAL_CAPACITY
        index = tl.load(
            global_indices + row * GLOBAL_CAPACITY + candidate,
            mask=in_capacity,
            other=-1,
        ).to(tl.int64)
        expected = query_valid & in_capacity & (index >= 0) & (index < visible)
        block = tl.maximum(index, 0) // GLOBAL_ENTRIES
        in_table = expected & (block < GLOBAL_TABLE_WIDTH)
        page = tl.load(
            global_page_table + request * GLOBAL_TABLE_WIDTH + block,
            mask=in_table,
            other=0,
        ).to(tl.int64)
        valid = in_table & (page > 0) & (page < GLOBAL_NUM_PAGES)
        slots = tl.maximum(index, 0) % GLOBAL_ENTRIES
        base = global_pool + page[:, None] * GLOBAL_PAGE_STRIDE + slots[:, None] * 288
        values = _load_precision_global_row(base, dims[None, :], valid[:, None])
        maximum, denominator, accumulator = _accumulate_tile(
            q, values, valid, maximum, denominator, accumulator, SCALE
        )
        if HEAD_PAIR:
            maximum_second, denominator_second, accumulator_second = _accumulate_tile(
                q_second,
                values,
                valid,
                maximum_second,
                denominator_second,
                accumulator_second,
                SCALE,
            )

    has_mass = query_valid & (denominator > 0)
    result = tl.where(has_mass, accumulator / denominator, 0.0)
    log_sum = tl.where(has_mass, tl.log(denominator) + maximum, float("-inf"))
    tl.store(output + (row * HEADS + head) * 512 + dims, result)
    tl.store(lse + row * HEADS + head, log_sum)
    tl.store(status + row * HEADS + head, error)
    if HEAD_PAIR:
        has_mass_second = query_valid & (denominator_second > 0)
        result_second = tl.where(
            has_mass_second, accumulator_second / denominator_second, 0.0
        )
        log_sum_second = tl.where(
            has_mass_second,
            tl.log(denominator_second) + maximum_second,
            float("-inf"),
        )
        tl.store(output + (row * HEADS + head + 1) * 512 + dims, result_second)
        tl.store(lse + row * HEADS + head + 1, log_sum_second)
        tl.store(status + row * HEADS + head + 1, error)


@triton.jit
def partial_extra_attention_kernel(
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
    extra_lengths,
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
    SKIP_EMPTY_TILES: tl.constexpr = False,
    HEAD_PAIR: tl.constexpr = False,
):
    row = tl.program_id(0)
    extra_length = tl.load(extra_lengths + row)
    USE_PAIR: tl.constexpr = HEAD_PAIR and SKIP_EMPTY_TILES
    tl.static_assert(not USE_PAIR or HEADS % 2 == 0)
    if extra_length < 512 and (not USE_PAIR or tl.program_id(1) < HEADS // 2):
        if SKIP_EMPTY_TILES:
            _bounded_attention(
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
                HEADS=HEADS,
                NUM_REQUESTS=NUM_REQUESTS,
                SWA_NUM_PAGES=SWA_NUM_PAGES,
                SWA_ENTRIES=SWA_ENTRIES,
                SWA_PAGE_STRIDE=SWA_PAGE_STRIDE,
                GLOBAL_NUM_PAGES=GLOBAL_NUM_PAGES,
                GLOBAL_ENTRIES=GLOBAL_ENTRIES,
                GLOBAL_PAGE_STRIDE=GLOBAL_PAGE_STRIDE,
                GLOBAL_TABLE_WIDTH=GLOBAL_TABLE_WIDTH,
                GLOBAL_CAPACITY=GLOBAL_CAPACITY,
                COMPRESS_RATIO=COMPRESS_RATIO,
                SCALE=SCALE,
                BLOCK_N=BLOCK_N,
                HEAD_PAIR=USE_PAIR,
            )
        else:
            compact_attention_kernel(
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
                HEADS=HEADS,
                NUM_REQUESTS=NUM_REQUESTS,
                SWA_NUM_PAGES=SWA_NUM_PAGES,
                SWA_ENTRIES=SWA_ENTRIES,
                SWA_PAGE_STRIDE=SWA_PAGE_STRIDE,
                GLOBAL_NUM_PAGES=GLOBAL_NUM_PAGES,
                GLOBAL_ENTRIES=GLOBAL_ENTRIES,
                GLOBAL_PAGE_STRIDE=GLOBAL_PAGE_STRIDE,
                GLOBAL_TABLE_WIDTH=GLOBAL_TABLE_WIDTH,
                GLOBAL_CAPACITY=GLOBAL_CAPACITY,
                COMPRESS_RATIO=COMPRESS_RATIO,
                SCALE=SCALE,
                BLOCK_N=BLOCK_N,
            )
