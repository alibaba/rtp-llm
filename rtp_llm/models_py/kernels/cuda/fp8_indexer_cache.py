"""Raw FP8 Indexer layout views and bounded native-64 Decode repacking.

Persistent pages keep the framework's geometry. Both persistent and native
pages store all E4M3 K bytes before all FP32 scale bytes, never token records.
No quantization or dequantization occurs during this byte movement.
"""

import torch
import triton
import triton.language as tl


def canonical_indexer_views(raw):
    """Split a canonical K[N,128] || scale[N] workspace without a copy."""
    if (
        raw.dtype != torch.uint8
        or raw.ndim != 2
        or raw.shape[1] != 132
        or not raw.is_contiguous()
    ):
        raise ValueError("Canonical Indexer workspace must be contiguous UINT8[N,132]")
    tokens = raw.shape[0]
    flat = raw.view(-1)
    return (
        flat[: tokens * 128].view(torch.float8_e4m3fn).view(tokens, 128),
        flat[tokens * 128 :].view(torch.float32),
    )


@triton.jit(
    do_not_specialize=["START", "CACHE_BLOCKS", "TABLE_WIDTH", "TABLE_STRIDE", "PAGES"]
)
def _pack_native64(
    Cache,
    Table,
    Output,
    START,
    KERNEL_PAGE: tl.constexpr,
    CACHE_BLOCKS,
    TABLE_WIDTH,
    TABLE_STRIDE,
    PAGES,
    BLOCK: tl.constexpr,
):
    page = tl.program_id(0).to(tl.int64)
    request = page // PAGES
    local_page = page % PAGES
    byte = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    key_byte = byte < 64 * 128
    token = tl.where(key_byte, byte // 128, (byte - 64 * 128) // 4)
    feature = tl.where(key_byte, byte % 128, 128 + (byte - 64 * 128) % 4)
    position = START + local_page * 64 + token
    column = position // KERNEL_PAGE
    in_page = byte < 64 * 132
    block = tl.load(
        Table + request * TABLE_STRIDE + column,
        mask=in_page & (column < TABLE_WIDTH),
        other=0,
    ).to(tl.int64)
    tl.device_assert(
        (~in_page) | ((block >= 0) & (block < CACHE_BLOCKS)),
        "FP8 Indexer owner table has an invalid physical block",
    )
    offset = tl.where(
        feature < 128,
        position % KERNEL_PAGE * 128 + feature,
        KERNEL_PAGE * 128 + position % KERNEL_PAGE * 4 + feature - 128,
    )
    value = tl.load(
        Cache + block * KERNEL_PAGE * 132 + offset,
        mask=in_page & (column < TABLE_WIDTH) & (block > 0),
        other=0,
    )
    # Native block zero is reserved; each request receives disjoint pages.
    tl.store(Output + (page + 1) * 64 * 132 + byte, value, mask=in_page)


@triton.jit
def _prepare_direct_indexer_cache(
    Cache,
    Table,
    Context,
    CACHE_BLOCKS,
    PAGES,
    CHECK_TILES,
    CHECK_CTAS,
    QUERIES: tl.constexpr,
    PAGE: tl.constexpr,
    QUERY_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
    ZERO_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < CHECK_CTAS:
        request = pid // CHECK_TILES
        page = pid % CHECK_TILES * BLOCK + tl.arange(0, BLOCK)
        query = tl.arange(0, QUERY_BLOCK)
        length = tl.max(
            tl.load(Context + request * QUERIES + query, query < QUERIES, other=0),
            axis=0,
        )
        valid = (page < PAGES) & (page < tl.cdiv(length, PAGE))
        block = tl.load(Table + request * PAGES + page, valid, other=0)
        tl.device_assert(
            (~valid) | ((block >= 0) & (block < CACHE_BLOCKS)),
            "FP8 Indexer owner table has an invalid physical block",
        )
    else:
        # The native reader uses physical page zero for null blocks and short tiles.
        byte = (pid - CHECK_CTAS) * ZERO_BLOCK + tl.arange(0, ZERO_BLOCK)
        tl.store(Cache + byte, 0, byte < PAGE * 132)


def prepare_direct_indexer_cache(cache, table, context):
    """Keep native page zero zeroed and check the visible physical page IDs."""
    batch, pages = table.shape
    queries = context.size(1)
    page = cache.size(1)
    block, zero_block = 256, 1024
    check_tiles = triton.cdiv(pages, block)
    check_ctas = batch * check_tiles
    _prepare_direct_indexer_cache[(check_ctas + triton.cdiv(page * 132, zero_block),)](
        cache,
        table,
        context,
        cache.size(0),
        pages,
        check_tiles,
        check_ctas,
        queries,
        page,
        triton.next_power_of_2(queries),
        block,
        zero_block,
        num_warps=4,
        debug=True,
    )


def pack_indexer_pages64(cache, block_table, start, output):
    """Copy one planned owner-KV block into explicit native paged storage.

    ``output`` and the corresponding identity block table are prepared before
    Graph capture. Null source blocks are padding and never dereferenced.
    Actual per-query owner lengths mask them in the native score adapter.
    """
    if (
        cache.dtype != torch.uint8
        or cache.ndim != 3
        or cache.shape[2] != 132
        or not cache.is_contiguous()
        or cache.shape[1] % 64
    ):
        raise ValueError("Persistent Indexer cache requires whole 64-token subpages")
    if (
        block_table.ndim != 2
        or block_table.dtype != torch.int32
        or block_table.device != cache.device
        or block_table.stride(1) != 1
    ):
        raise ValueError("FP8 owner page table must use INT32 contiguous columns")
    if (
        output.dtype != torch.uint8
        or output.ndim != 4
        or output.shape[1:] != (64, 1, 132)
        or not output.is_contiguous()
        or output.device != cache.device
    ):
        raise ValueError("Native MQA output must be contiguous UINT8[pages,64,1,132]")
    batch = block_table.shape[0]
    if type(start) is not int or start < 0 or start % 64 or batch <= 0:
        raise ValueError(
            "Native MQA block starts must be nonnegative and 64-token aligned"
        )
    if (output.shape[0] - 1) % batch or output.shape[0] <= 1:
        raise ValueError(
            "Native MQA workspace needs one reserved page and equal request capacities"
        )
    pages = (output.shape[0] - 1) // batch
    if start + pages * 64 > block_table.shape[1] * cache.shape[1]:
        raise ValueError(
            "Native MQA workspace exceeds the planned owner table capacity"
        )
    output[0].zero_()
    _pack_native64[(batch * pages, triton.cdiv(64 * 132, 4096))](
        cache,
        block_table,
        output,
        start,
        cache.shape[1],
        cache.shape[0],
        block_table.shape[1],
        block_table.stride(0),
        pages,
        4096,
        num_warps=4,
        debug=True,
    )
