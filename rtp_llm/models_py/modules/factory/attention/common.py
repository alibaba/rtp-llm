"""Common utilities for attention implementations.

This module contains helper functions for FMHA implementations including:
- Cache store operations
- Parameter updates for CUDA graph
- KV cache offset management
"""

from typing import TYPE_CHECKING, Any, Optional

import torch

from rtp_llm.models_py.modules.base.common.kvcache_store import WriteCacheStoreOp
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs

if TYPE_CHECKING:
    from rtp_llm.ops.fused_rope_kvcache_op import (
        FusedRopeAttnParams,
        FusedRopeKVCachePrefillOpBase,
    )


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
            attn_inputs.kv_cache_block_id,
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


def refresh_fused_rope_params(
    rope_params: "FusedRopeAttnParams",
    rope_kvcache_impl: "FusedRopeKVCachePrefillOpBase",
    attn_inputs: PyAttentionInputs,
    *,
    captured_max_seq_len: int,
    captured_max_prefix_length: int,
) -> None:
    """Refresh captured fused RoPE/KV-cache params for a CUDA graph replay.

    A captured graph bakes max_seq_len and max_prefix_length in as launch
    scalars, and the kernel derives count_length and use_paged_fmha from
    max_prefix_length > 0, so replay values must stay within the immutable
    capture bounds and must not flip that predicate. kv_cache_offset is copied
    back into the captured buffer. The C++ runner is the authoritative replay
    gate; these checks are a secondary fail-fast boundary for direct callers.

    Args:
        rope_params: Params captured by the graph, updated in place
        rope_kvcache_impl: Fused RoPE/KV-cache implementation object
        attn_inputs: Replay inputs, refreshed in place by the graph runner
    """
    # prepare() converts the replay block table to the fused kernel's offset
    # layout, then the result is copied into the stable capture buffer below.
    # This extra conversion/allocation is currently the correctness-first
    # tradeoff on every replay; optimize it only with an in-place equivalent.
    new_params = rope_kvcache_impl.prepare(attn_inputs)

    if new_params.max_seq_len > captured_max_seq_len:
        raise RuntimeError(
            "CUDA graph replay exceeds the captured max_seq_len "
            f"(capture={captured_max_seq_len}, replay={new_params.max_seq_len})"
        )
    if new_params.max_prefix_length > captured_max_prefix_length:
        raise RuntimeError(
            "CUDA graph replay exceeds the captured max_prefix_length "
            f"(capture={captured_max_prefix_length}, "
            f"replay={new_params.max_prefix_length})"
        )
    if (new_params.max_prefix_length > 0) != (captured_max_prefix_length > 0):
        raise RuntimeError(
            "CUDA graph replay cannot flip whether a prefix is present, since "
            "count_length and use_paged_fmha are captured from it "
            f"(capture={captured_max_prefix_length}, "
            f"replay={new_params.max_prefix_length})"
        )

    captured_has_offset = rope_params.kv_cache_offset is not None
    replay_has_offset = new_params.kv_cache_offset is not None
    if captured_has_offset != replay_has_offset:
        raise RuntimeError(
            "CUDA graph replay cannot change whether a paged KV block table is "
            f"present (capture={captured_has_offset}, replay={replay_has_offset})"
        )
    if replay_has_offset:
        copy_kv_cache_offset(rope_params.kv_cache_offset, new_params.kv_cache_offset)

    # All other tensor addresses and scalar values are part of the captured
    # launch contract. The C++ runner refreshes their contents in place; a
    # direct caller that substitutes a tensor would otherwise appear to work
    # here while the graph keeps dereferencing the old capture-time address.
    for field in (
        "padding_offset",
        "position_ids",
        "cu_seqlens",
        "cu_kv_seqlens",
        "input_lengths",
        "prefix_lengths",
        "sequence_lengths",
    ):
        captured_tensor = getattr(rope_params, field)
        replay_tensor = getattr(new_params, field)
        if (captured_tensor is None) != (replay_tensor is None):
            raise RuntimeError(
                f"CUDA graph replay cannot change whether {field} is present"
            )
        if captured_tensor is not None and (
            captured_tensor.data_ptr() != replay_tensor.data_ptr()
            or captured_tensor.dtype != replay_tensor.dtype
            or captured_tensor.device != replay_tensor.device
        ):
            raise RuntimeError(
                f"CUDA graph replay must reuse the stable capture buffer for {field}"
            )


def update_attention_params(
    fmha_impl: Any,
    rope_kvcache_impl: Any,
    fmha_params: Any,
    rope_params: Any,
    attn_inputs: PyAttentionInputs,
) -> None:
    """Update attention and RoPE parameters for CUDA graph replay.

    Updates FMHA and RoPE parameters based on new input parameters, maintaining KV Cache offset consistency.
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
