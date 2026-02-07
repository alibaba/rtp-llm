"""Common helper functions for FMHA implementations.

This module contains shared logic extracted from FMHA implementations to avoid code duplication.
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
