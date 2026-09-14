"""Use the existing FP32 compact reader for partial H64 extra-topk rows."""

import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv41._compact_reader_triton import (
    compact_attention_kernel,
)


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
):
    row = tl.program_id(0)
    if tl.load(extra_lengths + row) < 512:
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
