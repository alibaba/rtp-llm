"""Reference implementations for attention testing

This module provides reference attention implementations using flashinfer's
single_decode_with_kv_cache function. These can be used as ground truth for
testing custom attention implementations.
"""

from typing import List

import torch
from flashinfer.decode import single_decode_with_kv_cache


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
