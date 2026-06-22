"""Common utilities for attention implementations.

This module contains helper functions for FMHA implementations including:
- Workspace buffer pooling
- Cache store operations
- Parameter updates for CUDA graph
- KV cache offset management
"""

import threading
from typing import Any, Optional

import torch

from rtp_llm.models_py.modules.base.common.kvcache_store import WriteCacheStoreOp
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs


class WorkspaceBufferPool:
    """Thread-safe pool of reusable GPU workspace buffers.

    Each attention backend (XQA / TRT-LLM gen / FlashInfer) keeps its own pool
    sized for that backend. Buffers are uint8 scratch space handed to kernels;
    `get` reuses a pooled buffer or allocates one (outside the lock to minimize
    contention), `release` returns it to the pool for the next instance.
    """

    def __init__(self, size_mb: int) -> None:
        self._size_bytes = size_mb * 1024 * 1024
        self._pool: list[torch.Tensor] = []
        self._lock = threading.Lock()

    def get(self, device: str = "cuda") -> torch.Tensor:
        with self._lock:
            if self._pool:
                return self._pool.pop()
        return torch.zeros(self._size_bytes, dtype=torch.uint8, device=device)

    def release(self, buffer: torch.Tensor) -> None:
        with self._lock:
            self._pool.append(buffer)


def reshape_paged_kv_cache(
    paged_kv_cache: torch.Tensor,
    num_kv_heads: int,
    tokens_per_block: int,
    head_dim: int,
) -> torch.Tensor:
    """Reshape a raw 2D packed per-layer KV cache buffer into the 5D paged format.

    In hybrid cache mode the per-layer tensor arrives as a raw 2D buffer
    [block_num, kv_block_stride_elems].  The hybrid stride is
    max(full_attn, linear_attn), so we slice the prefix used by full-attention
    layers and reshape to [block_num, 2, num_kv_heads, tokens_per_block, head_dim].
    If the tensor is already multi-dimensional it is returned as-is.
    """
    if paged_kv_cache.dim() != 2:
        return paged_kv_cache
    block_num = paged_kv_cache.shape[0]
    expected_elems_per_block = 2 * num_kv_heads * tokens_per_block * head_dim
    if paged_kv_cache.shape[1] < expected_elems_per_block:
        raise ValueError(
            f"packed kv_cache_base has insufficient stride: "
            f"got stride={paged_kv_cache.shape[1]} elems, need={expected_elems_per_block} elems"
        )
    return paged_kv_cache[:, :expected_elems_per_block].reshape(
        block_num, 2, num_kv_heads, tokens_per_block, head_dim
    )


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
    kv_cache: Optional[LayerKVCache],
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
    """Copy new_offset into old_offset for CUDA graph parameter updates.

    If shapes match, copies directly. Otherwise zeros old_offset first and copies
    the overlapping region. The shape only mismatches on the block-count (last) dim,
    and only in benchmark/test harnesses that slice the page table to a shorter
    sequence; the batch dim never mismatches (a captured graph runs at a fixed batch
    — smaller real batches are zero-padded, not resized). Production
    cuda_graph_runner pre-allocates fixed-shape page tables, so this else branch is
    never taken there. Defensive hardening only — the current RoPE/XQA consumers read
    only blocks [0, nbPages), so the zeroed/truncated tail is never accessed.
    """
    if new_offset.shape == old_offset.shape:
        old_offset.copy_(new_offset, non_blocking=True)
    else:
        old_offset.zero_()
        slice_indices = [
            slice(0, min(new_offset.size(dim), old_offset.size(dim)))
            for dim in range(new_offset.dim())
        ]
        src_slice = new_offset[tuple(slice_indices)]
        dst_slice = old_offset[tuple(slice_indices)]
        dst_slice.copy_(src_slice, non_blocking=True)


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
