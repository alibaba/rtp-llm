from __future__ import annotations

import typing

import torch

__all__ = [
    "context_parallel_load_balance_split",
    "generate_qkv_restore_indices",
    "generate_qkv_padding_mask",
]

def context_parallel_load_balance_split(
    total_input_tokens: list[int],
    input_tokens: list[int],
    shuffle_indices: list[int],
    cp_rank: int,
    cp_size: int,
    cp_chunk_size: int,
    cp_padding_size: int,
) -> tuple[bool, list[int], list[int]]:
    """
    Distribute input tokens across context parallel ranks with load balancing.

    Args:
        total_input_tokens: Complete input token sequence before splitting
        input_tokens: Pre-allocated list for token chunk (will be resized)
        shuffle_indices: Pre-allocated list for shuffle indices (will be resized)
        cp_rank: Current rank ID in context parallel group (0-indexed)
        cp_size: Total number of ranks in context parallel group
        cp_chunk_size: Number of tokens assigned to current rank
        cp_padding_size: Padding tokens to add for alignment

    Returns:
        tuple of (success, input_tokens, shuffle_indices)
    """
    ...

def generate_qkv_restore_indices(
    prefill_cp_chunk_lengths: torch.Tensor, cp_size: int
) -> torch.Tensor:
    """
    Generate indices to restore original token order after parallel processing.

    Args:
        prefill_cp_chunk_lengths: Tensor of chunk lengths per rank
        cp_size: Number of context parallel ranks

    Returns:
        Restore indices tensor
    """
    ...

def generate_qkv_padding_mask(
    prefill_cp_chunk_lengths: torch.Tensor,
    prefill_cp_padding_lengths: torch.Tensor,
    cp_size: int,
) -> torch.Tensor:
    """
    Generate padding mask for QKV tensors in context parallel scenarios.

    Args:
        prefill_cp_chunk_lengths: Tensor of chunk lengths per rank
        prefill_cp_padding_lengths: Tensor of padding lengths per rank
        cp_size: Number of context parallel ranks

    Returns:
        Padding mask tensor
    """
    ...
