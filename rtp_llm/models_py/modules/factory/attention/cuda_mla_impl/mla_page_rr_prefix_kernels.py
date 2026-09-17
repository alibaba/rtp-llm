"""Fused, backend-independent movement of raw MLA prefix pages."""

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
