"""Read replicated P-page windows from the live, possibly padded K-page view."""

import triton
import triton.language as tl


@triton.jit
def _window_slot(table, positions, requests, valid_queries, row, token,
                 TABLE_STRIDE: tl.constexpr, TABLE_COL_STRIDE: tl.constexpr,
                 PAGE: tl.constexpr, WINDOW: tl.constexpr):
    end = tl.load(positions + row)
    request = tl.load(requests + row)
    valid = tl.load(valid_queries + row)
    logical = tl.maximum(0, end - WINDOW + 1) + token
    valid = valid & (token < WINDOW) & (logical <= end)
    physical = tl.load(
        table + request * TABLE_STRIDE + (logical // PAGE) * TABLE_COL_STRIDE,
        valid, 0,
    )
    slot = physical.to(tl.int64) * PAGE + logical % PAGE
    return slot, valid & (physical > 0)


@triton.jit
def build_swa_indices(table, positions, requests, valid_queries, indices,
                       TABLE_STRIDE: tl.constexpr, TABLE_COL_STRIDE: tl.constexpr,
                       PAGE: tl.constexpr, WINDOW: tl.constexpr,
                       PAD_WINDOW: tl.constexpr, CACHE_PAGE: tl.constexpr,
                       CACHE_PAGE_STRIDE: tl.constexpr, CACHE_TOKEN_STRIDE: tl.constexpr,
                       INDEX_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    token = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot, valid = _window_slot(
        table, positions, requests, valid_queries, row, token,
        TABLE_STRIDE, TABLE_COL_STRIDE, PAGE, WINDOW,
    )
    offset = (slot // CACHE_PAGE) * CACHE_PAGE_STRIDE
    offset += (slot % CACHE_PAGE) * CACHE_TOKEN_STRIDE
    index = tl.where(valid, offset // INDEX_STRIDE, -1)
    tl.store(indices + row * PAD_WINDOW + token, index, token < PAD_WINDOW)
