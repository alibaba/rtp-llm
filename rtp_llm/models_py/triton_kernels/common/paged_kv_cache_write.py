"""Triton writes for paged MHA caches with independent K/V head dimensions."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.common.offset import linear_offset_64


@triton.jit(do_not_specialize=["num_tokens", "batch_size", "num_cache_pages"])
def _write_asymmetric_paged_kv_cache_kernel(
    key_ptr,
    value_ptr,
    k_cache_ptr,
    v_cache_ptr,
    batch_indices_ptr,
    positions_ptr,
    page_indices_ptr,
    page_indptr_ptr,
    stride_key_t,
    stride_key_h,
    stride_key_d,
    stride_value_t,
    stride_value_h,
    stride_value_d,
    stride_k_cache_p,
    stride_k_cache_h,
    stride_k_cache_s,
    stride_k_cache_d,
    stride_v_cache_p,
    stride_v_cache_h,
    stride_v_cache_s,
    stride_v_cache_d,
    num_tokens,
    batch_size,
    num_cache_pages,
    PAGE_SIZE: tl.constexpr,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """One program copies one token/head pair into its paged K and V slots."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    batch_idx = tl.load(batch_indices_ptr + token_idx)
    position = tl.load(positions_ptr + token_idx)
    valid_batch = (batch_idx >= 0) & (batch_idx < batch_size)

    page_start = tl.load(page_indptr_ptr + batch_idx, mask=valid_batch, other=0)
    page_end = tl.load(page_indptr_ptr + batch_idx + 1, mask=valid_batch, other=0)
    logical_page = position // PAGE_SIZE
    valid_page = (
        valid_batch
        & (position >= 0)
        & (logical_page >= 0)
        & (logical_page < page_end - page_start)
    )

    page_flat_idx = page_start + logical_page
    physical_page = tl.load(page_indices_ptr + page_flat_idx, mask=valid_page, other=-1)
    valid_physical_page = (
        valid_page & (physical_page >= 0) & (physical_page < num_cache_pages)
    )

    # SWA block tables leave pages outside the active window as -1. Match the
    # planner by redirecting those writes to reserved page 0; window masking
    # guarantees that its contents are never used as real attention values.
    physical_page = tl.where(valid_physical_page, physical_page, 0)
    slot = tl.where(position >= 0, position % PAGE_SIZE, 0)

    key_base = linear_offset_64(token_idx, stride_key_t) + linear_offset_64(
        head_idx, stride_key_h
    )
    k_cache_base = (
        linear_offset_64(physical_page, stride_k_cache_p)
        + linear_offset_64(head_idx, stride_k_cache_h)
        + linear_offset_64(slot, stride_k_cache_s)
    )
    k_offsets = tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K_DIM
    key = tl.load(
        key_ptr + key_base + k_offsets * stride_key_d,
        mask=k_mask,
    )
    tl.store(
        k_cache_ptr + k_cache_base + k_offsets * stride_k_cache_d,
        key,
        mask=k_mask,
    )

    value_base = linear_offset_64(token_idx, stride_value_t) + linear_offset_64(
        head_idx, stride_value_h
    )
    v_cache_base = (
        linear_offset_64(physical_page, stride_v_cache_p)
        + linear_offset_64(head_idx, stride_v_cache_h)
        + linear_offset_64(slot, stride_v_cache_s)
    )
    v_offsets = tl.arange(0, BLOCK_V)
    v_mask = v_offsets < V_DIM
    value = tl.load(
        value_ptr + value_base + v_offsets * stride_value_d,
        mask=v_mask,
    )
    tl.store(
        v_cache_ptr + v_cache_base + v_offsets * stride_v_cache_d,
        value,
        mask=v_mask,
    )


def write_asymmetric_paged_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    page_indices: torch.Tensor,
    page_indptr: torch.Tensor,
) -> None:
    """Write K/V into an HND paged cache without materializing index tensors.

    ``key`` and ``value`` are packed by token. ``batch_indices`` and
    ``positions`` identify each token's request and absolute sequence position;
    ``page_indptr`` and ``page_indices`` then map that position to a physical
    cache page. K and V may have different last dimensions and all four data
    tensors may be strided views.
    """
    assert (
        key.dim() == 3 and value.dim() == 3
    ), f"key/value must be [tokens, heads, dim], got {key.shape=} {value.shape=}"
    assert k_cache.dim() == 4 and v_cache.dim() == 4, (
        "k_cache/v_cache must be [pages, heads, page_size, dim], got "
        f"{k_cache.shape=} {v_cache.shape=}"
    )
    assert (
        key.shape[:2] == value.shape[:2]
    ), f"key/value token and head shapes differ: {key.shape=} {value.shape=}"
    assert (
        k_cache.shape[:3] == v_cache.shape[:3]
    ), f"K/V cache page layouts differ: {k_cache.shape=} {v_cache.shape=}"
    assert (
        key.shape[1] == k_cache.shape[1]
    ), f"key/cache head count differs: {key.shape[1]} != {k_cache.shape[1]}"
    assert (
        key.shape[2] == k_cache.shape[3]
    ), f"key/cache head dim differs: {key.shape[2]} != {k_cache.shape[3]}"
    assert (
        value.shape[2] == v_cache.shape[3]
    ), f"value/cache head dim differs: {value.shape[2]} != {v_cache.shape[3]}"
    assert key.dtype == k_cache.dtype and value.dtype == v_cache.dtype, (
        f"source/cache dtype differs: {key.dtype=}/{k_cache.dtype=}, "
        f"{value.dtype=}/{v_cache.dtype=}"
    )

    tensors = (
        key,
        value,
        k_cache,
        v_cache,
        batch_indices,
        positions,
        page_indices,
        page_indptr,
    )
    assert all(t.is_cuda for t in tensors), "paged KV write requires CUDA tensors"
    device = key.device
    assert all(
        t.device == device for t in tensors
    ), "all tensors must share one CUDA device"
    assert (
        batch_indices.dim()
        == positions.dim()
        == page_indices.dim()
        == page_indptr.dim()
        == 1
    )

    num_tokens = int(key.shape[0])
    if num_tokens == 0:
        return
    assert (
        batch_indices.numel() >= num_tokens and positions.numel() >= num_tokens
    ), "batch_indices and positions must cover every input token"
    assert page_indptr.numel() >= 2, "page_indptr must contain at least one request"
    assert (
        page_indices.numel() > 0
    ), "page_indices cannot be empty when tokens are written"
    assert k_cache.shape[0] > 0, "cache must contain reserved page 0"

    page_size = int(k_cache.shape[2])
    k_dim = int(key.shape[2])
    v_dim = int(value.shape[2])
    block_k = triton.next_power_of_2(k_dim)
    block_v = triton.next_power_of_2(v_dim)
    num_warps = 8 if max(block_k, block_v) > 256 else 4

    _write_asymmetric_paged_kv_cache_kernel[(num_tokens, key.shape[1])](
        key,
        value,
        k_cache,
        v_cache,
        batch_indices,
        positions,
        page_indices,
        page_indptr,
        *key.stride(),
        *value.stride(),
        *k_cache.stride(),
        *v_cache.stride(),
        num_tokens,
        page_indptr.numel() - 1,
        k_cache.shape[0],
        PAGE_SIZE=page_size,
        K_DIM=k_dim,
        V_DIM=v_dim,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        num_warps=num_warps,
    )
