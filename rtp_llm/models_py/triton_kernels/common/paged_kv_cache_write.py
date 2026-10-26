"""Paged KV writes with a device-resident live-token bound for CUDA graphs."""

import torch
import triton
import triton.language as tl


@triton.jit
def _append_paged_kv_cache(
    key,
    value,
    cache,
    batch_indices,
    positions,
    page_indices,
    page_indptr,
    token_indptr,
    key_stride_n: tl.constexpr,
    key_stride_h: tl.constexpr,
    value_stride_n: tl.constexpr,
    value_stride_h: tl.constexpr,
    cache_stride_p: tl.constexpr,
    cache_stride_kv: tl.constexpr,
    cache_stride_h: tl.constexpr,
    cache_stride_t: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    # Graph replay retains the captured launch size. Padding metadata can still
    # name a previously active request, so check before reading any such entry.
    if token >= tl.load(token_indptr + BATCH_SIZE):
        return
    batch = tl.load(batch_indices + token)
    position = tl.load(positions + token)
    page_slot = tl.load(page_indptr + batch) + position // PAGE_SIZE
    page = tl.load(page_indices + page_slot).to(tl.int64)
    offset = tl.arange(0, BLOCK)
    dst = (
        page * cache_stride_p
        + head * cache_stride_h
        + (position % PAGE_SIZE) * cache_stride_t
        + offset
    )
    k = tl.load(
        key + token * key_stride_n + head * key_stride_h + offset,
        offset < HEAD_DIM,
        0.0,
    )
    v = tl.load(
        value + token * value_stride_n + head * value_stride_h + offset,
        offset < HEAD_DIM,
        0.0,
    )
    tl.store(cache + dst, k, offset < HEAD_DIM)
    tl.store(cache + dst + cache_stride_kv, v, offset < HEAD_DIM)


def append_paged_kv_cache_with_token_bound(
    key: torch.Tensor,
    value: torch.Tensor,
    cache: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    page_indices: torch.Tensor,
    page_indptr: torch.Tensor,
    token_indptr: torch.Tensor,
) -> None:
    """Append live rows to HND cache; leave all graph padding untouched."""
    assert key.stride(-1) == value.stride(-1) == cache.stride(-1) == 1
    assert token_indptr.is_cuda and token_indptr.dtype == torch.int32
    assert token_indptr.ndim == 1 and token_indptr.numel() > 0
    if key.element_size() == 1:
        # Copy FP8 storage bits without requiring FP8 arithmetic support from
        # the GPU or converting NaN encodings during a pure cache write.
        key = key.view(torch.uint8)
        value = value.view(torch.uint8)
        cache = cache.view(torch.uint8)
    _append_paged_kv_cache[(key.size(0), key.size(1))](
        key,
        value,
        cache,
        batch_indices,
        positions,
        page_indices,
        page_indptr,
        token_indptr,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        cache.stride(3),
        cache.size(3),
        key.size(2),
        token_indptr.numel() - 1,
        triton.next_power_of_2(key.size(2)),
    )
