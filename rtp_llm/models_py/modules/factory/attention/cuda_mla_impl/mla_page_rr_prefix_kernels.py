"""Fused, backend-independent movement of unquantized MLA prefix pages."""

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
        token_offsets, tile_offsets = [0], [0]
        for length in descriptor.prefix_lens:
            token_offsets.append(token_offsets[-1] + length)
            tile_offsets.append(
                tile_offsets[-1] + triton.cdiv(length * features, _BLOCK_ELEMENTS)
            )
        host = torch.tensor(
            (
                descriptor.prefix_lens + (0,),
                descriptor.local_page_offsets,
                tuple(token_offsets),
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
    offsets = Metadata + REQUESTS + 1
    request = _request_for_index(offsets, packed_page, REQUESTS).to(tl.int64)
    local_page = packed_page - tl.load(offsets + request)
    prefix_len = tl.load(Metadata + request)
    global_start = (local_page * SHARDS + RANK) * PAGE_TOKENS
    owned = global_start < prefix_len
    in_table = local_page < TABLE_WIDTH
    block_id = tl.load(
        Table + request * TABLE_REQUEST_STRIDE + local_page * TABLE_PAGE_STRIDE,
        mask=owned & in_table,
        other=0,
    ).to(tl.int64)
    valid_block = (block_id > 0) & (block_id < CACHE_BLOCKS) & in_table
    # debug=True is part of the launch contract: these checks remain enabled.
    tl.device_assert(
        (~owned) | valid_block,
        "packed owner page points to a null, reserved, or out-of-range block",
    )
    # Widen before multiplication: even masked lanes can overflow int32 when a
    # legal strided view has a large token or payload stride.
    elements = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    token = elements // FEATURES
    feature = elements % FEATURES
    live = (token < PAGE_TOKENS) & (global_start + token < prefix_len) & valid_block
    values = tl.load(
        Cache
        + block_id * CACHE_BLOCK_STRIDE
        + token * CACHE_TOKEN_STRIDE
        + feature * CACHE_FEATURE_STRIDE,
        mask=live,
        other=0,
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
    tile_offsets = Metadata + 3 * (REQUESTS + 1)
    request = _request_for_index(tile_offsets, tile, REQUESTS)
    elements = (tile - tl.load(tile_offsets + request)) * BLOCK + tl.arange(0, BLOCK)
    position = elements // FEATURES
    feature = elements % FEATURES
    global_page = position // PAGE_TOKENS
    owner = global_page % SHARDS
    local_page = global_page // SHARDS
    packed_page = tl.load(Metadata + REQUESTS + 1 + request) + local_page
    source = (
        (owner * LOCAL_PAGES + packed_page) * PAGE_TOKENS + position % PAGE_TOKENS
    ) * FEATURES + feature
    live = position < tl.load(Metadata + request)
    values = tl.load(Gathered + source, mask=live, other=0)
    output_start = tl.load(Metadata + 2 * (REQUESTS + 1) + request) * FEATURES
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
