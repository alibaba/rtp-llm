"""Portable DFlash attention over an already populated paged KV cache.

Context and query K/V share pages, but only the device-side logical sequence
length is visible.  One Triton launch evaluates each complete query block.
The caller owns the output and metadata buffers so graph replay can update
lengths/pages without reallocating or specializing on a particular KV length.
"""

from enum import IntEnum
from typing import Optional

import torch
import triton
import triton.language as tl


class DFlashCacheLayout(IntEnum):
    CUDA = 0
    AITER = 1
    AITER_VECTOR = 2


def is_supported(
    query: torch.Tensor,
    cache: torch.Tensor,
    query_width: int,
    *,
    cache_layout: DFlashCacheLayout = DFlashCacheLayout.CUDA,
) -> bool:
    """Shape/device gate; no device data reads or backend-specific imports."""
    if query.ndim != 3 or cache.ndim != 5 or cache.shape[1] != 2:
        return False
    if query.device.type != "cuda" or query.device != cache.device:
        return False
    if query.dtype != torch.bfloat16 or cache.dtype != query.dtype:
        return False
    if not cache.is_contiguous() or query.stride(-1) != 1:
        return False
    _, heads, dim = query.shape
    _, _, kv_heads, page_size, cache_dim = cache.shape
    if (
        query_width not in (1, 2, 3, 4, 5, 6, 7, 8, 16)
        or query.shape[0] % query_width
        or heads <= 0
        or kv_heads <= 0
        or heads % kv_heads
        or dim not in (64, 128, 256)
        or dim != cache_dim
        or page_size <= 0
        or cache_layout not in tuple(DFlashCacheLayout)
    ):
        return False
    return cache_layout != DFlashCacheLayout.AITER_VECTOR or page_size % 8 == 0


@triton.jit
def _cache_offsets(
    page,
    head,
    slot,
    dim,
    KV_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    plane = KV_HEADS * PAGE_SIZE * HEAD_DIM
    base = page.to(tl.int64) * (2 * plane) + head * PAGE_SIZE * HEAD_DIM
    if LAYOUT == 0:
        k = base + slot * HEAD_DIM + dim
        v = k + plane
    else:
        k = base + (dim // 8) * PAGE_SIZE * 8 + slot * 8 + dim % 8
        if LAYOUT == 2:
            v = base + plane + (slot // 8) * HEAD_DIM * 8 + dim * 8 + slot % 8
        else:
            v = base + plane + dim * PAGE_SIZE + slot
    return k, v


@triton.jit
def _dflash_attention_kernel(
    Q,
    Cache,
    Table,
    Lengths,
    Out,
    q_stride_row: tl.constexpr,
    q_stride_head: tl.constexpr,
    table_stride_row: tl.constexpr,
    table_stride_col: tl.constexpr,
    out_stride_row: tl.constexpr,
    out_stride_head: tl.constexpr,
    Q_WIDTH: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    CACHE_BLOCKS: tl.constexpr,
    LAYOUT: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    request = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // (Q_HEADS // KV_HEADS)
    length = tl.load(Lengths + request)
    # An invalid/empty logical row is neutral; never read pages outside its table.
    valid_length = (length > 0) & (length <= TABLE_COLS * PAGE_SIZE)
    length = tl.where(valid_length, length, 0)
    m = tl.arange(0, BLOCK_M)
    d = tl.arange(0, HEAD_DIM)
    positions = length - Q_WIDTH + m
    valid_query = (m < Q_WIDTH) & (positions >= 0) & (length > 0)
    q = tl.load(
        Q
        + (request * Q_WIDTH + m[:, None]) * q_stride_row
        + head * q_stride_head
        + d[None, :],
        mask=valid_query[:, None],
        other=0,
    )
    max_score = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    first = 0
    if WINDOW_SIZE > 0:
        first = tl.maximum(length - Q_WIDTH - WINDOW_SIZE + 1, 0)
        first = first // BLOCK_N * BLOCK_N
    for start in range(first, length, BLOCK_N):
        n = start + tl.arange(0, BLOCK_N)
        live = n < length
        page = tl.load(
            Table + request * table_stride_row + (n // PAGE_SIZE) * table_stride_col,
            mask=live,
            other=-1,
        )
        live = live & (page >= 0) & (page < CACHE_BLOCKS)
        k_offset, v_offset = _cache_offsets(
            page[:, None],
            kv_head,
            (n % PAGE_SIZE)[:, None],
            d[None, :],
            KV_HEADS,
            PAGE_SIZE,
            HEAD_DIM,
            LAYOUT,
        )
        key = tl.load(Cache + k_offset, mask=live[:, None], other=0)
        value = tl.load(Cache + v_offset, mask=live[:, None], other=0)
        scores = tl.dot(q, tl.trans(key)).to(tl.float32) * SCALE
        visible = valid_query[:, None] & live[None, :]
        if CAUSAL:
            visible = visible & (n[None, :] <= positions[:, None])
        if WINDOW_SIZE > 0:
            # HF DFlash: p-window < k; an inclusive window_left would be window-1.
            visible = visible & (n[None, :] > positions[:, None] - WINDOW_SIZE)
            if not CAUSAL:
                visible = visible & (n[None, :] < positions[:, None] + WINDOW_SIZE)
        scores = tl.where(visible, scores, -float("inf"))
        next_max = tl.maximum(max_score, tl.max(scores, 1))
        next_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        rescale = tl.exp2((max_score - next_max) * 1.4426950408889634)
        probabilities = tl.exp2((scores - next_max[:, None]) * 1.4426950408889634)
        accumulator = accumulator * rescale[:, None]
        # Retain the softmax probabilities' residual instead of rounding them
        # once to BF16; short rows can otherwise lose several output ulps.
        probabilities_hi = probabilities.to(value.dtype)
        probabilities_lo = (probabilities - probabilities_hi.to(tl.float32)).to(
            value.dtype
        )
        accumulator += tl.dot(probabilities_hi, value)
        accumulator += tl.dot(probabilities_lo, value)
        denominator = denominator * rescale + tl.sum(probabilities, 1)
        max_score = next_max
    result = accumulator / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(
        Out
        + (request * Q_WIDTH + m[:, None]) * out_stride_row
        + head * out_stride_head
        + d[None, :],
        result,
        mask=(m < Q_WIDTH)[:, None],
    )


def _check_metadata(cache, block_table, lengths, batch):
    if (
        block_table.ndim != 2
        or block_table.shape[0] != batch
        or block_table.dtype not in (torch.int32, torch.int64)
        or block_table.device != cache.device
        or lengths.ndim != 1
        or lengths.shape[0] != batch
        or not lengths.is_contiguous()
        or lengths.dtype not in (torch.int32, torch.int64)
        or lengths.device != cache.device
    ):
        raise ValueError("DFlash requires device block_table[B,pages] and lengths[B]")


def dflash_paged_attention(
    query: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_width: int,
    *,
    causal: bool,
    window_size: int = 0,
    cache_layout: DFlashCacheLayout = DFlashCacheLayout.CUDA,
    out: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Evaluate [B*Q,Hq,D] queries with prewritten context+query K/V.

    ``sequence_lengths`` includes the query block. The query positions are
    ``length-Q .. length-1``. A zero length denotes an inactive/padded request.
    ``window_size=2048`` means ``p-2048 < k <= p`` for a causal layer; zero
    disables the window. Cache backing storage is contiguous [pages,2,Hkv,P,D]
    with physical interpretation specified by ``cache_layout``. Supply ``out``
    allocated before graph capture for a fully allocation-free wrapper.
    """
    if not is_supported(query, cache, query_width, cache_layout=cache_layout):
        raise ValueError("unsupported DFlash paged-attention shape, dtype or layout")
    if window_size < 0:
        raise ValueError("DFlash window_size must be nonnegative")
    batch = query.shape[0] // query_width
    _check_metadata(cache, block_table, sequence_lengths, batch)
    if out is None:
        out = torch.empty_like(query, memory_format=torch.contiguous_format)
    if (
        out.shape != query.shape
        or out.dtype != query.dtype
        or out.device != query.device
        or out.stride(-1) != 1
    ):
        raise ValueError("DFlash output must match query shape, dtype and device")
    if batch == 0:
        return out
    _dflash_attention_kernel[(batch, query.shape[1])](
        query,
        cache,
        block_table,
        sequence_lengths,
        out,
        query.stride(0),
        query.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        out.stride(0),
        out.stride(1),
        Q_WIDTH=query_width,
        Q_HEADS=query.shape[1],
        KV_HEADS=cache.shape[2],
        HEAD_DIM=query.shape[2],
        PAGE_SIZE=cache.shape[3],
        TABLE_COLS=block_table.shape[1],
        CACHE_BLOCKS=cache.shape[0],
        LAYOUT=int(cache_layout),
        CAUSAL=causal,
        WINDOW_SIZE=window_size,
        SCALE=softmax_scale if softmax_scale is not None else query.shape[2] ** -0.5,
        BLOCK_M=16,
        BLOCK_N=64,
        num_warps=4,
        num_stages=1,
    )
    return out


@triton.jit
def _dflash_write_kv_kernel(
    Key,
    Value,
    Cache,
    Table,
    Requests,
    Positions,
    key_stride_row: tl.constexpr,
    key_stride_head: tl.constexpr,
    value_stride_row: tl.constexpr,
    value_stride_head: tl.constexpr,
    table_stride_row: tl.constexpr,
    table_stride_col: tl.constexpr,
    BATCH: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    CACHE_BLOCKS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    request = tl.load(Requests + row)
    position = tl.load(Positions + row)
    valid = (request >= 0) & (request < BATCH) & (position >= 0)
    valid = valid & (position < TABLE_COLS * PAGE_SIZE)
    page = tl.load(
        Table + request * table_stride_row + (position // PAGE_SIZE) * table_stride_col,
        mask=valid,
        other=-1,
    )
    valid = valid & (page >= 0) & (page < CACHE_BLOCKS)
    d = tl.arange(0, HEAD_DIM)
    k_offset, v_offset = _cache_offsets(
        page,
        head,
        position % PAGE_SIZE,
        d,
        KV_HEADS,
        PAGE_SIZE,
        HEAD_DIM,
        LAYOUT,
    )
    key = tl.load(
        Key + row * key_stride_row + head * key_stride_head + d, mask=valid, other=0
    )
    value = tl.load(
        Value + row * value_stride_row + head * value_stride_head + d,
        mask=valid,
        other=0,
    )
    tl.store(Cache + k_offset, key, mask=valid)
    tl.store(Cache + v_offset, value, mask=valid)


def dflash_write_paged_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    cache_layout: DFlashCacheLayout = DFlashCacheLayout.CUDA,
) -> None:
    """Write semantic [rows,Hkv,D] K/V into pages, skipping negative padding.

    Each valid (request,position) must occur once. K is already QK-normalized
    and rotated; V is unnormalized. Works for both context commit and query KV.
    """
    if not is_supported(key, cache, 1, cache_layout=cache_layout):
        raise ValueError("unsupported DFlash KV write shape, dtype or layout")
    if key.shape[1] != cache.shape[2] or value.shape != key.shape:
        raise ValueError("DFlash K/V head geometry must match the cache")
    if value.dtype != key.dtype or value.device != key.device or value.stride(-1) != 1:
        raise ValueError(
            "DFlash K/V dtype, device and last-dimension stride must match"
        )
    rows = key.shape[0]
    for metadata in (request_ids, positions):
        if (
            metadata.ndim != 1
            or metadata.shape[0] != rows
            or metadata.device != key.device
            or not metadata.is_contiguous()
            or metadata.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "DFlash KV write metadata must be contiguous device [rows]"
            )
    if (
        block_table.ndim != 2
        or block_table.device != key.device
        or block_table.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("DFlash KV write requires a device integer block table")
    if rows == 0:
        return
    _dflash_write_kv_kernel[(rows, key.shape[1])](
        key,
        value,
        cache,
        block_table,
        request_ids,
        positions,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        BATCH=block_table.shape[0],
        TABLE_COLS=block_table.shape[1],
        CACHE_BLOCKS=cache.shape[0],
        KV_HEADS=cache.shape[2],
        PAGE_SIZE=cache.shape[3],
        HEAD_DIM=cache.shape[4],
        LAYOUT=int(cache_layout),
        num_warps=4,
    )
