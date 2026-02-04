"""CUDA attention implementations common utilities.

This module provides common utility functions used across different CUDA attention implementations.
"""
from typing import Any, Optional, Tuple

import torch

from rtp_llm.models_py.modules.base.common.kvcache_store import WriteCacheStoreOp
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs


def create_write_cache_store_op(attn_inputs: PyAttentionInputs) -> Optional[WriteCacheStoreOp]:
    """Create WriteCacheStoreOp if needed for cache storage.
    
    Args:
        attn_inputs: Attention input parameters
        
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


def copy_kv_cache_offset(old_offset: torch.Tensor, new_offset: torch.Tensor):
    """Copy KV cache offset tensor data.
    
    Handles both same-shape and different-shape tensors. For different shapes,
    only copies the data that fits in the target tensor.
    
    Args:
        old_offset: Target offset tensor to copy into
        new_offset: Source offset tensor to copy from
    """
    if new_offset.shape == old_offset.shape:
        old_offset.copy_(new_offset, non_blocking=True)
    else:
        slice_indices = [
            slice(0, new_offset.size(dim)) for dim in range(new_offset.dim())
        ]
        target_slice = old_offset[tuple(slice_indices)]
        target_slice.copy_(new_offset, non_blocking=True)


def apply_rope_and_kv_cache(
    qkv: torch.Tensor,
    kv_cache: Optional[KVCache],
    rope_kvcache_impl: Any,
    rope_params: Any,
    need_rope_kv_cache: bool = True,
) -> torch.Tensor:
    """Apply RoPE (Rotary Position Embedding) and KV cache operations.
    
    Args:
        qkv: Input QKV tensor
        kv_cache: KV cache object
        rope_kvcache_impl: RoPE and KV cache operation implementation
        rope_params: Parameters for RoPE operation
        need_rope_kv_cache: Whether to apply RoPE and KV cache
        
    Returns:
        Processed tensor ready for attention computation
    """
    if need_rope_kv_cache:
        return rope_kvcache_impl.forward(qkv, kv_cache, rope_params)
    return qkv


def write_to_cache_store(
    kv_cache: Optional[KVCache],
    attn_inputs: PyAttentionInputs,
    write_cache_store_impl: Optional[WriteCacheStoreOp],
):
    """Write KV cache to cache store if needed.
    
    Args:
        kv_cache: KV cache object
        attn_inputs: Attention input parameters
        write_cache_store_impl: Cache store operation implementation
    """
    if (
        attn_inputs.is_prefill
        and attn_inputs.cache_store_inputs
        and write_cache_store_impl is not None
    ):
        write_cache_store_impl(kv_cache)


def create_params(fmha_impl: Any, rope_kvcache_impl: Any, attn_inputs: PyAttentionInputs) -> Tuple[Any, Any]:
    """创建 FMHA 和 RoPE 的计算参数。
    
    Args:
        fmha_impl: FMHA 实现对象
        rope_kvcache_impl: RoPE 和 KV cache 实现对象
        attn_inputs: Attention 输入参数
        
    Returns:
        Tuple[Any, Any]: (fmha_params, rope_params)
    """
    fmha_params = fmha_impl.prepare(attn_inputs)
    rope_params = rope_kvcache_impl.prepare(attn_inputs)
    return fmha_params, rope_params


def update_params_for_cuda_graph(
    fmha_impl: Any,
    rope_kvcache_impl: Any,
    fmha_params: Any,
    rope_params: Any,
    attn_inputs: PyAttentionInputs,
) -> None:
    """更新 CUDA Graph 所需的参数。
    
    通过准备新的参数并复制 KV cache offset 来更新现有参数。
    
    Args:
        fmha_impl: FMHA 实现对象
        rope_kvcache_impl: RoPE 和 KV cache 实现对象
        fmha_params: 当前的 FMHA 参数
        rope_params: 当前的 RoPE 参数
        attn_inputs: Attention 输入参数
    """
    # Update FMHA params
    new_fmha_params = fmha_impl.prepare(attn_inputs)
    if hasattr(new_fmha_params, 'kv_cache_offset') and hasattr(fmha_params, 'kv_cache_offset'):
        copy_kv_cache_offset(fmha_params.kv_cache_offset, new_fmha_params.kv_cache_offset)

    # Update RoPE params
    new_rope_params = rope_kvcache_impl.prepare(attn_inputs)
    if hasattr(new_rope_params, 'kv_cache_offset') and hasattr(rope_params, 'kv_cache_offset'):
        copy_kv_cache_offset(rope_params.kv_cache_offset, new_rope_params.kv_cache_offset)
