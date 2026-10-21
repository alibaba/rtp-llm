"""Common utilities for attention implementations.

This module contains helper functions for FMHA implementations including:
- Cache store operations
- Parameter updates for CUDA graph
- KV cache offset management
"""

from math import isfinite
from typing import Any, Optional

import torch

from rtp_llm.models_py.modules.base.common.kvcache_store import WriteCacheStoreOp
from rtp_llm.models_py.modules.base.common.kvcache_store import (
    create_write_cache_store_impl as _create_write_cache_store_impl,
)
from rtp_llm.ops.compute_ops import KVCache, LayerKVCache, PyAttentionInputs


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
    kv_cache: Optional[KVCache] = None,
) -> Optional[WriteCacheStoreOp]:
    """Create write cache store implementation if needed.

    The op class is passed explicitly as this module's ``WriteCacheStoreOp``
    global, resolved on every call. That keeps this module the single seam for
    every attention backend: rebinding (or patching) ``WriteCacheStoreOp`` here
    changes what the shared factory instantiates, instead of being silently
    bypassed by the delegation.

    Args:
        attn_inputs: Attention calculation input parameters
        kv_cache: Whole-model KV cache handle, for multi-group (DSv4) callers

    Returns:
        WriteCacheStoreOp instance if cache store is needed, None otherwise
    """
    return _create_write_cache_store_impl(
        attn_inputs, kv_cache, op_cls=WriteCacheStoreOp
    )


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
    if attn_inputs.is_prefill and write_cache_store_impl is not None:
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


def update_attention_params(
    fmha_impl: Any,
    rope_kvcache_impl: Any,
    fmha_params: Any,
    rope_params: Any,
    attn_inputs: PyAttentionInputs,
) -> None:
    """Update attention and RoPE parameters for CUDA graph replay.

    Updates FMHA metadata, KV-cache offsets, and the decode RoPE sequence-length
    buffer. CUDA graph callers retain the original buffer identity.
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

    new_rope_params = rope_kvcache_impl.prepare(attn_inputs, forbid_reallocation=True)
    new_offset = new_rope_params.kv_cache_offset
    old_offset = rope_params.kv_cache_offset
    copy_kv_cache_offset(old_offset, new_offset)
    rope_params.sequence_lengths = new_rope_params.sequence_lengths


def embedding_graph_unsupported_reason(configs, backend) -> str:
    from importlib.util import find_spec

    from rtp_llm.ops import RopeStyle

    if configs.use_mla:
        return "cache-free embedding graph MLA is not implemented"
    if (
        getattr(configs, "is_sparse", False)
        or getattr(configs, "use_logn_attn", False)
        or getattr(configs, "q_scaling", 1.0) != 1.0
        or getattr(configs, "softmax_extra_scale", 1.0) != 1.0
    ):
        return (
            "embedding graph sparse/logn/custom attention scaling requires an adapter"
        )
    if configs.dtype not in (torch.float16, torch.bfloat16):
        return "embedding graph attention requires FP16 or BF16 activations"
    if configs.rope_config.style not in (RopeStyle.No, RopeStyle.Base):
        return "embedding graph supports no RoPE or base RoPE; other styles require an adapter"
    if configs.rope_config.style != RopeStyle.No and (
        configs.rope_config.dim <= 0
        or configs.rope_config.dim % 2
        or configs.rope_config.dim > configs.size_per_head
    ):
        return "invalid embedding graph rotary dimension"
    if configs.rope_config.offset != 0:
        return "embedding graph RoPE offsets require an adapter"
    if configs.rope_config.style == RopeStyle.Base and (
        not isfinite(configs.rope_config.scale) or configs.rope_config.scale < 1.0
    ):
        return "embedding graph base RoPE requires a finite scale >= 1"
    if (
        configs.head_num <= 0
        or configs.kv_head_num <= 0
        or configs.size_per_head <= 0
        or configs.head_num % configs.kv_head_num
    ):
        return "invalid embedding graph query/KV head counts"
    if find_spec(backend) is None:
        return f"embedding graph attention backend {backend} is not installed"
    return ""


def prepare_embedding_graph_lengths(
    attn_inputs,
    token_capacity,
    request_capacity,
    max_sequence_length,
    cu_seqlens_host,
    for_capture=False,
):
    """Update fixed host ragged boundaries, keeping padding in separate sequences."""
    lengths = attn_inputs.input_lengths
    if (
        lengths is None
        or lengths.is_cuda
        or lengths.dtype != torch.int32
        or lengths.ndim != 1
        or not lengths.is_contiguous()
    ):
        raise ValueError(
            "embedding graph planning requires contiguous host int32 input lengths"
        )
    if lengths.numel() != request_capacity:
        raise ValueError("embedding graph request capacity changed")
    prefix = attn_inputs.prefix_lengths
    if prefix is not None and prefix.numel() and bool(prefix.any()):
        raise ValueError("embedding graph does not support prefix KV")
    if bool((lengths < 0).any()):
        raise ValueError("embedding graph sequence lengths must be nonnegative")
    limit = max_sequence_length or token_capacity
    if not for_capture and bool((lengths > limit).any()):
        raise ValueError("embedding graph sequence exceeds position-table capacity")
    if int(lengths.sum()) > token_capacity:
        raise ValueError("embedding graph token capacity exceeded")
    if for_capture:
        lengths = torch.tensor(
            [
                min(limit, max(0, token_capacity - i * limit))
                for i in range(request_capacity)
            ],
            dtype=torch.int32,
        )
    real_tokens = int(lengths.sum())
    cu_seqlens_host[0] = 0
    torch.cumsum(lengths, 0, out=cu_seqlens_host[1 : request_capacity + 1])
    padding_slots = (token_capacity + limit - 1) // limit
    for index in range(padding_slots):
        cu_seqlens_host[request_capacity + index + 1] = min(
            token_capacity, real_tokens + (index + 1) * limit
        )
    return lengths


def create_embedding_graph_rope(configs, token_capacity, max_sequence_length):
    """Own each bucket's canonical RoPE table and fixed position storage."""
    rope = getattr(configs, "rope_config", None)
    if rope is None or int(rope.style) == 0:
        return None, None, None
    from rtp_llm.ops import get_rope_cache

    cache = get_rope_cache(
        rope, max_sequence_length or token_capacity, interleave=False
    )
    host = torch.zeros(token_capacity, dtype=torch.int64, pin_memory=True)
    return cache, host, torch.empty_like(host, device="cuda")


def prepare_embedding_graph_positions(attn_inputs, lengths, host, device):
    if host is None:
        return
    positions = getattr(attn_inputs, "combo_position_ids", None)
    if positions is not None and positions.numel():
        raise ValueError("explicit embedding graph positions require a model adapter")
    host.zero_()
    offset = 0
    for length in lengths.tolist():
        host[offset : offset + length].copy_(torch.arange(length, dtype=torch.int64))
        offset += length
    # Planning completes before reuse of the pinned source or Graph capture.
    device.copy_(host)


def apply_embedding_graph_rotary(q, k, rope_cache, positions):
    if rope_cache is None:
        return q, k
    # Canonical cache has non-interleaved [cos, sin] rows, matching the
    # existing ragged eager RoPE path. QK normalization remains in the layer.
    cache = rope_cache.index_select(0, positions).float()
    cos, sin = cache.chunk(2, dim=-1)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    rotary_dim = cache.size(-1)

    def rotate(tensor):
        left, right = tensor[..., :rotary_dim].float().chunk(2, dim=-1)
        rotated = torch.cat(
            (left * cos - right * sin, right * cos + left * sin), dim=-1
        )
        return torch.cat((rotated.to(tensor.dtype), tensor[..., rotary_dim:]), dim=-1)

    return rotate(q), rotate(k)
