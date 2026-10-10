"""Shared MLA Page-RR cache movement and device coordinate kernels."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

_BLOCK_ELEMENTS = 4096


@dataclass
class _PrefixMetadata:
    # The descriptor owns staging until its async copy/graph replay is finished.
    host: torch.Tensor
    device: torch.Tensor
    ready: torch.cuda.Event
    producer_stream: int
    restore_tiles: int


def _metadata(descriptor, device):
    stream = torch.cuda.current_stream(device)
    metadata = descriptor._cuda_metadata
    if metadata is None or metadata.device.device != device:
        features = descriptor.feature_width
        tile_offsets = [0]
        for length in descriptor.prefix_lens:
            tile_offsets.append(
                tile_offsets[-1] + triton.cdiv(length * features, _BLOCK_ELEMENTS)
            )
        host = torch.tensor(
            (
                descriptor.request_indices + (0,),
                descriptor.global_page_starts + (0,),
                descriptor.prefix_lens + (0,),
                descriptor.local_page_offsets,
                descriptor.output_token_offsets,
                tuple(tile_offsets),
            ),
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        metadata = _PrefixMetadata(
            host,
            host.to(device=device, non_blocking=True),
            torch.cuda.Event(),
            stream.cuda_stream,
            tile_offsets[-1],
        )
        metadata.ready.record(stream)
        object.__setattr__(descriptor, "_cuda_metadata", metadata)
    elif metadata.producer_stream != stream.cuda_stream:
        stream.wait_event(metadata.ready)
        metadata.device.record_stream(stream)
    return metadata


@triton.jit
def _request_for_index(Offsets, index, REQUESTS: tl.constexpr):
    # upper_bound skips empty requests, including consecutive empty prefixes.
    lo = 0
    hi = REQUESTS
    while lo < hi:
        mid = (lo + hi) // 2
        end = tl.load(Offsets + mid + 1)
        before_end = index < end
        hi = tl.where(before_end, mid, hi)
        lo = tl.where(before_end, lo, mid + 1)
    return lo


@triton.jit
def _pack_prefix_kernel(
    Cache,
    Table,
    Metadata,
    Output,
    REQUESTS: tl.constexpr,
    PAGE_TOKENS: tl.constexpr,
    KERNEL_PAGE_TOKENS: tl.constexpr,
    SHARDS: tl.constexpr,
    RANK: tl.constexpr,
    FEATURES: tl.constexpr,
    CACHE_BLOCKS: tl.constexpr,
    CACHE_BLOCK_STRIDE: tl.constexpr,
    CACHE_TOKEN_STRIDE: tl.constexpr,
    CACHE_FEATURE_STRIDE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    TABLE_REQUEST_STRIDE: tl.constexpr,
    TABLE_PAGE_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    packed_page = tl.program_id(0).to(tl.int64)
    row_width: tl.constexpr = REQUESTS + 1
    request_indices = Metadata
    global_page_starts = Metadata + row_width
    prefix_lens = Metadata + 2 * row_width
    offsets = Metadata + 3 * row_width
    request = _request_for_index(offsets, packed_page, REQUESTS).to(tl.int64)
    packed_slot = packed_page - tl.load(offsets + request)
    prefix_len = tl.load(prefix_lens + request)
    global_page_start = tl.load(global_page_starts + request)
    first_relative_page = (RANK - global_page_start % SHARDS + SHARDS) % SHARDS
    relative_page = first_relative_page + packed_slot * SHARDS
    physical_pages = (prefix_len + PAGE_TOKENS - 1) // PAGE_TOKENS
    global_page = global_page_start + relative_page
    request_idx = tl.load(request_indices + request)
    # Ownership/payloads use physical pages; the live cache and table expose
    # kernel subpages. Mask the partial terminal page before reading its IDs.
    elements = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    token = elements // FEATURES
    feature = elements % FEATURES
    owned = (
        (relative_page < physical_pages)
        & (token < PAGE_TOKENS)
        & (relative_page * PAGE_TOKENS + token < prefix_len)
    )
    owner_local_page = global_page // SHARDS
    kernel_page = (
        owner_local_page * (PAGE_TOKENS // KERNEL_PAGE_TOKENS)
        + token // KERNEL_PAGE_TOKENS
    )
    in_table = kernel_page < TABLE_WIDTH
    block_id = tl.load(
        Table + request_idx * TABLE_REQUEST_STRIDE + kernel_page * TABLE_PAGE_STRIDE,
        mask=owned & in_table,
        other=0,
    ).to(tl.int64)
    valid_block = (block_id > 0) & (block_id < CACHE_BLOCKS) & in_table
    # debug=True is part of the launch contract: these checks remain enabled.
    tl.device_assert(
        (~owned) | valid_block,
        "packed owner page points to a null, reserved, or out-of-range block",
    )
    live = owned & valid_block
    values = tl.load(
        Cache
        + block_id * CACHE_BLOCK_STRIDE
        + (token % KERNEL_PAGE_TOKENS) * CACHE_TOKEN_STRIDE
        + feature * CACHE_FEATURE_STRIDE,
        mask=live,
        other=0.0,
    )
    tl.store(
        Output + packed_page * PAGE_TOKENS * FEATURES + elements,
        values,
        mask=elements < PAGE_TOKENS * FEATURES,
    )


@triton.jit
def _restore_prefix_kernel(
    Gathered,
    Metadata,
    Output,
    REQUESTS: tl.constexpr,
    PAGE_TOKENS: tl.constexpr,
    SHARDS: tl.constexpr,
    LOCAL_PAGES: tl.constexpr,
    FEATURES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int64)
    row_width: tl.constexpr = REQUESTS + 1
    global_page_starts = Metadata + row_width
    prefix_lens = Metadata + 2 * row_width
    local_page_offsets = Metadata + 3 * row_width
    output_token_offsets = Metadata + 4 * row_width
    tile_offsets = Metadata + 5 * row_width
    request = _request_for_index(tile_offsets, tile, REQUESTS)
    elements = (tile - tl.load(tile_offsets + request)) * BLOCK + tl.arange(0, BLOCK)
    position = elements // FEATURES
    feature = elements % FEATURES
    relative_page = position // PAGE_TOKENS
    global_page_start = tl.load(global_page_starts + request)
    global_page = global_page_start + relative_page
    owner = global_page % SHARDS
    first_relative_page = (owner - global_page_start % SHARDS + SHARDS) % SHARDS
    packed_slot = (relative_page - first_relative_page) // SHARDS
    packed_page = tl.load(local_page_offsets + request) + packed_slot
    source = (
        (owner * LOCAL_PAGES + packed_page) * PAGE_TOKENS + position % PAGE_TOKENS
    ) * FEATURES + feature
    live = position < tl.load(prefix_lens + request)
    values = tl.load(Gathered + source, mask=live, other=0.0)
    output_start = tl.load(output_token_offsets + request) * FEATURES
    tl.store(Output + output_start + elements, values, mask=live)


def pack_prefix_cuda(cache, table, descriptor, rank, output):
    with torch.cuda.device(cache.device):
        metadata = _metadata(descriptor, cache.device)
        features = descriptor.feature_width
        _pack_prefix_kernel[
            (
                descriptor.total_local_pages,
                triton.cdiv(descriptor.page_tokens * features, _BLOCK_ELEMENTS),
            )
        ](
            cache,
            table,
            metadata.device,
            output,
            descriptor.batch_size,
            descriptor.page_tokens,
            cache.shape[1],
            descriptor.shard_size,
            rank,
            features,
            cache.shape[0],
            cache.stride(0),
            cache.stride(1),
            cache.stride(2),
            table.shape[1],
            table.stride(0),
            table.stride(1),
            _BLOCK_ELEMENTS,
            num_warps=4,
            debug=True,
        )


def restore_prefix_cuda(gathered, descriptor, output):
    with torch.cuda.device(gathered.device):
        metadata = _metadata(descriptor, gathered.device)
        _restore_prefix_kernel[(metadata.restore_tiles,)](
            gathered,
            metadata.device,
            output,
            descriptor.batch_size,
            descriptor.page_tokens,
            descriptor.shard_size,
            descriptor.total_local_pages,
            descriptor.feature_width,
            _BLOCK_ELEMENTS,
            num_warps=4,
        )


@triton.jit
def _page_rr_cache_slot(
    position, batch, active, table, table_width, table_stride,
    page_size: tl.constexpr, kernel_page_size: tl.constexpr,
    cp_size: tl.constexpr, cp_rank: tl.constexpr,
    check_owner: tl.constexpr = False,
):
    # Ownership uses physical pages; tables and cache writers use kernel subpages.
    global_page = position // page_size
    offset = position % page_size
    local_page = global_page // cp_size * (page_size // kernel_page_size)
    local_page += offset // kernel_page_size
    owner = active & (position >= 0) & (global_page % cp_size == cp_rank)
    in_table = (local_page >= 0) & (local_page < table_width)
    physical_page = tl.load(
        table + batch * table_stride + local_page, owner & in_table, other=-1
    )
    if check_owner:
        tl.device_assert((~active) | (position >= 0), "positions must be non-negative")
        tl.device_assert((~owner) | in_table, "local block table width is insufficient for an owner page")
        tl.device_assert((~owner) | (~in_table) | (physical_page > 0), "owner page points to a null or unused local block")
    slot = physical_page.to(tl.int64) * kernel_page_size + offset % kernel_page_size
    # Page zero is reserved for padding, and non-owner writes are no-ops.
    return tl.where(owner & in_table & (physical_page > 0), slot, -1)


@triton.jit
def _prepare_prefill_metadata(
    padding_offset, prefix_lengths, table, slots, tokens, valid_tokens,
    table_width, table_stride,
    request_count: tl.constexpr, fixed_q_len: tl.constexpr, max_q_len: tl.constexpr,
    page_size: tl.constexpr, kernel_page_size: tl.constexpr,
    cp_size: tl.constexpr, cp_rank: tl.constexpr, block: tl.constexpr,
):
    row = tl.program_id(0) * block + tl.arange(0, block)
    valid = row < tokens
    # Use the framework's packed-to-padded map for ragged requests, as in RoPE.
    if fixed_q_len:
        batch = row // fixed_q_len
        offset = row % fixed_q_len
    else:
        padded = row + tl.load(padding_offset + row, valid, other=0)
        batch = padded // max_q_len
        offset = padded % max_q_len
    active = valid & (row < valid_tokens)
    in_batch = (batch >= 0) & (batch < request_count)
    tl.device_assert((~active) | in_batch, "batch index out of range")
    position = tl.load(prefix_lengths + batch, valid & in_batch, other=0) + offset
    slot = _page_rr_cache_slot(
        position, batch, active & in_batch, table, table_width, table_stride,
        page_size, kernel_page_size, cp_size, cp_rank, check_owner=True,
    )
    tl.store(slots + row, slot, valid)


@triton.jit
def _prepare_decode_metadata(
    sequence_base,
    capture_lengths,
    cu_seqlens,
    valid_token_mask,
    table,
    query_tables,
    positions,
    slots,
    local_lengths,
    rows,
    table_width,
    table_stride,
    queries: tl.constexpr,
    page_size: tl.constexpr,
    kernel_page_size: tl.constexpr,
    cp_size: tl.constexpr,
    cp_rank: tl.constexpr,
    multi_query_update: tl.constexpr,
    graph_decode: tl.constexpr,
    mtp_update: tl.constexpr,
    request_count: tl.constexpr,
    has_valid_mask: tl.constexpr,
    block: tl.constexpr,
):
    if mtp_update:
        # One program prepares a physical query and its complete page table.
        # Request boundaries remain device data, including zero-length slots.
        row = tl.program_id(0)
        valid = row < rows
        active = valid & (row < tl.load(cu_seqlens + request_count))
        if has_valid_mask:
            active = active & tl.load(valid_token_mask + row, valid, other=False)
        batch = tl.minimum(_request_for_index(cu_seqlens, row, request_count), request_count - 1)
        query_offset = row - tl.load(cu_seqlens + batch)
    else:
        row = tl.program_id(0) * block + tl.arange(0, block)
        valid = row < rows
        active = valid
        batch = row // queries
        query_offset = row % queries
    base = tl.load(sequence_base + batch, active, other=0)
    if graph_decode:
        # CudaGraphRunner's synthetic q1 capture leaves plus_1 at zero.
        # Live replay (including padded requests) instead supplies >= 1.
        capture_base = tl.load(capture_lengths + batch, valid, other=0) + 1
        base = tl.where(base == 0, capture_base, base)
    position = base + query_offset if multi_query_update else base - 1
    position = tl.where(active, position, 0)
    tl.store(positions + row, position, valid)

    # Count keys <= this query in the rank's physical-page stripes. Treating
    # B*Q as q1 rows avoids imposing a false *local* multi-query causal mask.
    end = tl.maximum(position + 1, 0)
    interval = page_size * cp_size
    length = end // interval * page_size + tl.minimum(
        tl.maximum(end % interval - cp_rank * page_size, 0), page_size
    )
    tl.store(local_lengths + row, tl.where(active, length, 0), valid)

    slot = _page_rr_cache_slot(
        position, batch, active, table, table_width, table_stride,
        page_size, kernel_page_size, cp_size, cp_rank,
    )
    tl.store(slots + row, slot, valid)

    if mtp_update:
        width = tl.maximum(table_width, 1)
        for start in range(0, width, block):
            column = start + tl.arange(0, block)
            page = tl.load(table + batch * table_stride + column, active & (column < table_width), other=0)
            tl.store(query_tables + row * width + column, page, column < width)
