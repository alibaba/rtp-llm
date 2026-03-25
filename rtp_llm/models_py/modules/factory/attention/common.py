"""Common utilities for attention implementations.

This module contains helper functions for FMHA implementations including:
- Cache store operations
- Parameter updates for CUDA graph
- KV cache offset management
"""

from typing import Any, Optional

import torch

from rtp_llm.models_py.modules.base.common.kvcache_store import WriteCacheStoreOp
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs


def create_write_cache_store_impl(
    attn_inputs: PyAttentionInputs,
) -> Optional[WriteCacheStoreOp]:
    """Create write cache store implementation if needed.

    Args:
        attn_inputs: Attention calculation input parameters

    Returns:
        WriteCacheStoreOp instance if cache store is needed, None otherwise
    """
    if attn_inputs.is_prefill and attn_inputs.cache_store_inputs:
        return WriteCacheStoreOp(
            attn_inputs.input_lengths,
            attn_inputs.prefix_lengths,
            attn_inputs.kv_cache_block_id_host,
            attn_inputs.cache_store_inputs,
        )
    return None


def apply_write_cache_store(
    write_cache_store_impl: Optional[WriteCacheStoreOp],
    attn_inputs: PyAttentionInputs,
    kv_cache: Optional[KVCache],
) -> None:
    """Apply write cache store operation if needed.

    Args:
        write_cache_store_impl: Write cache store implementation
        attn_inputs: Attention calculation input parameters
        kv_cache: KV Cache to write to
    """
    if (
        attn_inputs.is_prefill
        and attn_inputs.cache_store_inputs
        and write_cache_store_impl is not None
    ):
        write_cache_store_impl(kv_cache)


def copy_kv_cache_offset(old_offset: torch.Tensor, new_offset: torch.Tensor) -> None:
    """Copy KV Cache offset data.

    Used for CUDA graph parameter update scenarios.
    Copies new offset data into old offset tensor. If shapes match, copies directly,
    otherwise only copies the matching portion (slicing from the first dimension).

    Args:
        old_offset: Target offset tensor, data will be updated
        new_offset: Source offset tensor, provides new data
    """
    if new_offset.shape == old_offset.shape:
        old_offset.copy_(new_offset, non_blocking=True)
    else:
        # Build slice indices dynamically
        slice_indices = [
            slice(0, new_offset.size(dim)) for dim in range(new_offset.dim())
        ]
        target_slice = old_offset[tuple(slice_indices)]
        target_slice.copy_(new_offset, non_blocking=True)


def update_trt_params(
    fmha_impl: Any,
    rope_kvcache_impl: Any,
    fmha_params: Any,
    rope_params: Any,
    attn_inputs: PyAttentionInputs,
) -> None:
    """Update TRT-related parameters.

    Updates FMHA and RoPE parameters based on new input parameters, maintaining KV Cache offset consistency.
    Mainly used for CUDA graph parameter update scenarios.

    Args:
        fmha_impl: FMHA implementation object
        rope_kvcache_impl: RoPE KV Cache implementation object
        fmha_params: Current FMHA parameters
        rope_params: Current RoPE parameters
        attn_inputs: New attention calculation input parameters
    """
    new_fmha_params = fmha_impl.prepare(attn_inputs)
    new_offset = new_fmha_params.kv_cache_offset
    old_offset = fmha_params.kv_cache_offset
    copy_kv_cache_offset(old_offset, new_offset)

    new_rope_params = rope_kvcache_impl.prepare(attn_inputs)
    new_offset = new_rope_params.kv_cache_offset
    old_offset = rope_params.kv_cache_offset
    copy_kv_cache_offset(old_offset, new_offset)


def build_contiguous_block_table(
    cu_kv_seqlens: torch.Tensor,
    page_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a contiguous block table from cumulative KV sequence lengths.

    Given cu_kv_seqlens [batch+1] and a page_size, produces a [batch, max_pages]
    int32 tensor where entry [i, j] = start_page_of_request_i + j.

    This is used for workspace paged tensors where pages are laid out contiguously
    (no gaps), so the block table is simply a range per request.

    Args:
        cu_kv_seqlens: Cumulative KV lengths [batch_size + 1], int32, on device.
        page_size: Number of tokens per page/block.
        device: Target device.

    Returns:
        block_table: [batch_size, max_pages_per_req], int32, on device.
    """
    cu_kv = cu_kv_seqlens.cpu()
    batch_size = cu_kv.size(0) - 1
    if batch_size == 0:
        return torch.empty(0, 0, dtype=torch.int32, device=device)

    # Vectorized: compute start_page and n_pages per request
    starts = cu_kv[:-1]
    ends = cu_kv[1:]
    start_pages = starts // page_size
    n_pages = ((ends + page_size - 1) // page_size) - start_pages
    max_pages = int(n_pages.max().item())

    block_table = torch.zeros(batch_size, max_pages, dtype=torch.int32)
    cols = torch.arange(max_pages, dtype=torch.int32)
    for i in range(batch_size):
        np_i = int(n_pages[i].item())
        block_table[i, :np_i] = start_pages[i].int() + cols[:np_i]

    return block_table.to(device)
