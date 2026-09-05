"""Reference implementations for attention testing

This module provides reference attention implementations using flashinfer's
single_prefill_with_kv_cache and single_decode_with_kv_cache functions.
These can be used as ground truth for testing custom attention implementations.
"""

import math
from typing import List, Optional, Sequence

import torch
from flashinfer.decode import single_decode_with_kv_cache
from flashinfer.prefill import single_prefill_with_kv_cache

from rtp_llm.ops.compute_ops import LayerKVCache


def apply_base_rope_to_qkv_reference(
    qkv: torch.Tensor,
    input_lengths: Sequence[int],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_base: float = 10000.0,
    position_offsets: Optional[Sequence[int]] = None,
    rotary_dim: Optional[int] = None,
) -> torch.Tensor:
    """Apply non-interleaved Base RoPE to packed QKV in float32."""
    if position_offsets is None:
        position_offsets = [0] * len(input_lengths)
    if len(position_offsets) != len(input_lengths):
        raise ValueError("position_offsets and input_lengths must have equal size")
    if rotary_dim is None:
        rotary_dim = head_dim

    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q = qkv[:, :q_size].reshape(-1, num_q_heads, head_dim).float()
    k = qkv[:, q_size : q_size + kv_size].reshape(-1, num_kv_heads, head_dim).float()
    v = qkv[:, q_size + kv_size :]

    positions = torch.cat(
        [
            torch.arange(offset, offset + length, device=qkv.device)
            for offset, length in zip(position_offsets, input_lengths)
        ]
    ).float()
    inv_freq = 1.0 / (
        rope_base
        ** (
            torch.arange(0, rotary_dim, 2, device=qkv.device, dtype=torch.float32)
            / rotary_dim
        )
    )
    freqs = torch.outer(positions, inv_freq)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).unsqueeze(1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).unsqueeze(1)

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q = torch.cat([q_rot * cos + rotate_half(q_rot) * sin, q_pass], dim=-1)
    k = torch.cat([k_rot * cos + rotate_half(k_rot) * sin, k_pass], dim=-1)
    return torch.cat([q.flatten(1), k.flatten(1), v.float()], dim=-1).to(qkv.dtype)


def compute_paged_prefill_reference(
    query: torch.Tensor,
    kv_cache: LayerKVCache,
    page_table: torch.Tensor,
    sequence_lengths: Sequence[int],
    query_lengths: Sequence[int],
    causal: bool = True,
) -> torch.Tensor:
    """Compute a PyTorch prefill reference from an HND paged KV cache."""
    if len(sequence_lengths) != len(query_lengths):
        raise ValueError("sequence_lengths and query_lengths must have equal size")
    if sum(query_lengths) != query.shape[0]:
        raise ValueError("query_lengths must sum to the number of query tokens")

    cache = kv_cache.kv_cache_base
    page_size = cache.shape[3]
    num_kv_heads = cache.shape[2]
    num_q_heads = query.shape[1]
    head_dim = query.shape[2]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    outputs = []
    query_offset = 0
    scale = head_dim**-0.5
    for batch_idx, (kv_len, query_len) in enumerate(
        zip(sequence_lengths, query_lengths)
    ):
        page_count = math.ceil(kv_len / page_size)
        page_ids = page_table[batch_idx, :page_count].to(
            device=cache.device, dtype=torch.long
        )
        key = cache[page_ids, 0].permute(0, 2, 1, 3)
        value = cache[page_ids, 1].permute(0, 2, 1, 3)
        key = key.reshape(-1, num_kv_heads, head_dim)[:kv_len]
        value = value.reshape(-1, num_kv_heads, head_dim)[:kv_len]
        head_group_size = num_q_heads // num_kv_heads
        key = key.repeat_interleave(head_group_size, dim=1).float()
        value = value.repeat_interleave(head_group_size, dim=1).float()
        q = query[query_offset : query_offset + query_len].float()
        query_offset += query_len

        scores = torch.einsum("qhd,khd->hqk", q, key) * scale
        if causal:
            q_positions = (
                torch.arange(query_len, device=query.device) + kv_len - query_len
            )
            k_positions = torch.arange(kv_len, device=query.device)
            causal_mask = k_positions[None, :] <= q_positions[:, None]
            scores.masked_fill_(~causal_mask.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hqk,khd->qhd", probs, value))
    return torch.cat(outputs).to(query.dtype)


def compute_flashinfer_prefill_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Compute reference prefill attention output using flashinfer

    This function handles ragged tensor input by splitting into individual sequences
    and computing attention for each sequence separately using flashinfer's
    single_prefill_with_kv_cache as reference.

    Args:
        q: Query tensor [total_tokens, num_heads, head_dim]
        k: Key tensor [total_tokens, num_kv_heads, head_dim]
        v: Value tensor [total_tokens, num_kv_heads, head_dim]
        cu_seqlens: Cumulative sequence lengths [batch_size + 1]
        causal: Whether to use causal attention

    Returns:
        Attention output [total_tokens, num_heads, head_dim]

    Example:
        >>> batch_size = 2
        >>> num_heads = 32
        >>> num_kv_heads = 8
        >>> head_dim = 128
        >>> seq_lens = [100, 200]
        >>> total_tokens = sum(seq_lens)
        >>>
        >>> q = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float16, device="cuda")
        >>> k = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
        >>> v = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
        >>>
        >>> # Create cu_seqlens
        >>> cu_seqlens = torch.tensor([0, seq_lens[0], seq_lens[0] + seq_lens[1]], device="cuda")
        >>>
        >>> ref_output = compute_flashinfer_prefill_reference(
        ...     q, k, v, cu_seqlens, causal=True
        ... )
        >>> assert ref_output.shape == (total_tokens, num_heads, head_dim)
    """
    batch_size = cu_seqlens.size(0) - 1
    outputs = []

    # Process each sequence separately
    for i in range(batch_size):
        start_idx = cu_seqlens[i].item()
        end_idx = cu_seqlens[i + 1].item()

        q_seq = q[start_idx:end_idx]  # [seq_len, num_heads, head_dim]
        k_seq = k[start_idx:end_idx]  # [seq_len, num_kv_heads, head_dim]
        v_seq = v[start_idx:end_idx]  # [seq_len, num_kv_heads, head_dim]

        output_seq = single_prefill_with_kv_cache(
            q_seq,
            k_seq,
            v_seq,
            causal=causal,
            kv_layout="NHD",
        )
        outputs.append(output_seq)

    # Concatenate all outputs
    return torch.cat(outputs, dim=0)


def compute_flashinfer_decode_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sequence_lengths: List[int],
    block_id_list: List[List[int]],
    seq_size_per_block: int,
) -> torch.Tensor:
    """Compute reference decode attention outputs using flashinfer

    This function computes attention outputs for batched decode using flashinfer's
    single_decode_with_kv_cache as reference. It processes each sequence independently
    and stacks the results.

    Args:
        q: Query tensor [batch_size, num_heads, head_dim]
        k_cache: Key cache tensor [total_blocks, num_kv_heads, block_size, head_dim]
                 in HND (Head, Num_pages, Dim) layout
        v_cache: Value cache tensor [total_blocks, num_kv_heads, block_size, head_dim]
                 in HND (Head, Num_pages, Dim) layout
        sequence_lengths: List of sequence lengths for each batch element
        block_id_list: List of block ID lists for each sequence.
                       block_id_list[i] contains the block IDs used by sequence i.
                       This allows the caller to control block allocation logic.
        seq_size_per_block: Size of each block/page in the KV cache

    Returns:
        Reference attention output [batch_size, num_heads, head_dim]

    Example:
        >>> batch_size = 2
        >>> num_heads = 32
        >>> num_kv_heads = 8
        >>> head_dim = 128
        >>> seq_lens = [100, 200]
        >>> block_size = 64
        >>>
        >>> q = torch.randn(batch_size, num_heads, head_dim, dtype=torch.float16, device="cuda")
        >>> total_blocks = sum([math.ceil(s / block_size) for s in seq_lens])
        >>> k_cache = torch.randn(total_blocks, num_kv_heads, block_size, head_dim,
        ...                       dtype=torch.float16, device="cuda")
        >>> v_cache = torch.randn(total_blocks, num_kv_heads, block_size, head_dim,
        ...                       dtype=torch.float16, device="cuda")
        >>>
        >>> # Generate block ID list (sequential allocation in this example)
        >>> block_id_list = []
        >>> offset = 0
        >>> for seq_len in seq_lens:
        ...     num_blocks = math.ceil(seq_len / block_size)
        ...     block_id_list.append(list(range(offset, offset + num_blocks)))
        ...     offset += num_blocks
        >>>
        >>> ref_output = compute_flashinfer_decode_reference(
        ...     q, k_cache, v_cache, seq_lens, block_id_list, block_size
        ... )
        >>> assert ref_output.shape == (batch_size, num_heads, head_dim)
    """
    num_kv_heads = k_cache.shape[1]
    head_dim = q.shape[2]

    ref_outputs = []

    for i, seq_len in enumerate(sequence_lengths):
        # Get query for this batch element
        q_single = q[i]  # [num_heads, head_dim]

        # Get KV cache blocks for this sequence using provided block IDs
        block_ids = block_id_list[i]
        k_blocks = k_cache[
            block_ids
        ]  # [num_blocks, num_kv_heads, block_size, head_dim]
        v_blocks = v_cache[
            block_ids
        ]  # [num_blocks, num_kv_heads, block_size, head_dim]

        # Reshape to contiguous KV format [seq_len, num_kv_heads, head_dim]
        # HND layout: [blocks, num_kv_heads, block_size, head_dim]
        # Need to convert to NHD: [seq_len, num_kv_heads, head_dim]
        k_seq = (
            k_blocks.permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, head_dim)
            .permute(1, 0, 2)[:seq_len]
            .contiguous()
        )

        v_seq = (
            v_blocks.permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, head_dim)
            .permute(1, 0, 2)[:seq_len]
            .contiguous()
        )

        # Compute reference output using flashinfer
        ref_output = single_decode_with_kv_cache(
            q_single,
            k_seq,
            v_seq,
            kv_layout="NHD",
        )
        ref_outputs.append(ref_output)

    # Stack reference outputs [batch_size, num_heads, head_dim]
    ref_output_stacked = torch.stack(ref_outputs, dim=0)

    return ref_output_stacked
