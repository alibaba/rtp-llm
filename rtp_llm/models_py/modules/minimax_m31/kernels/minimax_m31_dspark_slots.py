"""MiniMax-M3.1 DSpark cache slot mapping kernels."""

import triton
import triton.language as tl


@triton.jit
def _commit_slots_cp1_kernel(
    REQUESTS,
    POSITIONS,
    TABLE,
    SLOTS,
    VALID,
    N: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    POS_STRIDE: tl.constexpr,
    TABLE_REQ_STRIDE: tl.constexpr,
    TABLE_BLOCK_STRIDE: tl.constexpr,
    TABLE_ROWS: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_bounds = row < N
    request = tl.load(REQUESTS + row * REQ_STRIDE, mask=in_bounds, other=-1).to(
        tl.int64
    )
    position = tl.load(POSITIONS + row * POS_STRIDE, mask=in_bounds, other=-1).to(
        tl.int64
    )
    block = position // PAGE_SIZE
    valid = (
        in_bounds
        & (request >= 0)
        & (request < TABLE_ROWS)
        & (position >= 0)
        & (block < TABLE_COLS)
    )
    physical = tl.load(
        TABLE + request * TABLE_REQ_STRIDE + block * TABLE_BLOCK_STRIDE,
        mask=valid,
        other=-1,
    ).to(tl.int64)
    valid &= physical >= 0
    slot = tl.where(valid, physical * PAGE_SIZE + position % PAGE_SIZE, -1)
    tl.store(SLOTS + row, slot, mask=in_bounds)
    tl.store(VALID + row, valid, mask=in_bounds)


@triton.jit
def _map_commit_slots_cp1_kernel(
    STARTS,
    LENGTHS,
    ENDS,
    TABLE,
    REQUESTS,
    POSITIONS,
    SLOTS,
    VALID,
    ROW_COUNT: tl.constexpr,
    BATCH: tl.constexpr,
    PAGE: tl.constexpr,
    TABLE_ROWS: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    SS: tl.constexpr,
    LS: tl.constexpr,
    ES: tl.constexpr,
    TRS: tl.constexpr,
    TCS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_bounds = row < ROW_COUNT
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), BATCH, tl.int32)
    for _ in tl.static_range(SEARCH_STEPS):
        active = lo < hi
        mid = (lo + hi) // 2
        start_mid = tl.load(
            STARTS + mid * SS, mask=in_bounds & active & (mid < BATCH), other=2147483647
        ).to(tl.int64)
        left = row.to(tl.int64) < start_mid
        hi = tl.where(active & left, mid, hi)
        lo = tl.where(active & ~left, mid + 1, lo)
    request = lo - 1
    safe_request = tl.maximum(request, 0)
    start = tl.load(STARTS + safe_request * SS, mask=in_bounds, other=0).to(tl.int64)
    length = tl.load(LENGTHS + safe_request * LS, mask=in_bounds, other=0).to(tl.int64)
    end = tl.load(ENDS + safe_request * ES, mask=in_bounds, other=0).to(tl.int64)
    row_valid = in_bounds & (request >= 0) & (row >= start) & (row < start + length)
    position = end - length + row.to(tl.int64) - start
    block = position // PAGE
    table_valid = (
        row_valid & (safe_request < TABLE_ROWS) & (position >= 0) & (block < TABLE_COLS)
    )
    physical = tl.load(
        TABLE + safe_request * TRS + block * TCS, mask=table_valid, other=-1
    ).to(tl.int64)
    slot_valid = table_valid & (physical >= 0)
    slot = tl.where(slot_valid, physical * PAGE + position % PAGE, -1)
    tl.store(REQUESTS + row, tl.where(row_valid, request, -1), mask=in_bounds)
    tl.store(POSITIONS + row, tl.where(row_valid, position, -1), mask=in_bounds)
    tl.store(SLOTS + row, slot, mask=in_bounds)
    tl.store(VALID + row, slot_valid, mask=in_bounds)
