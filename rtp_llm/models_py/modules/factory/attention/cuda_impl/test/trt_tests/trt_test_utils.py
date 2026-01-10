"""Utility functions for TRT attention testing"""

from typing import List

import torch
import torch.nn.functional as F

from rtp_llm.ops.compute_ops import PyAttentionInputs


def compute_pytorch_prefill_reference(
    qkv: torch.Tensor,
    input_lengths: List[int],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Compute reference prefill attention outputs using PyTorch SDPA

    Args:
        qkv: QKV tensor [total_tokens, (num_q_heads + 2 * num_kv_heads) * head_dim]
        input_lengths: List of input sequence lengths for each batch element
        num_q_heads: Number of query heads
        num_kv_heads: Number of key/value heads
        head_dim: Dimension per head

    Returns:
        Reference attention output [total_tokens, num_q_heads * head_dim]
    """
    # Split QKV
    qkv_dim = (num_q_heads + 2 * num_kv_heads) * head_dim
    assert (
        qkv.shape[1] == qkv_dim
    ), f"QKV dimension mismatch: {qkv.shape[1]} vs {qkv_dim}"

    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    ref_outputs = []
    offset = 0

    for seq_len in input_lengths:
        # Extract QKV for this sequence
        qkv_seq = qkv[offset : offset + seq_len]  # [seq_len, qkv_dim]

        # Split into Q, K, V
        q = qkv_seq[:, :q_dim].reshape(seq_len, num_q_heads, head_dim)
        k = qkv_seq[:, q_dim : q_dim + kv_dim].reshape(seq_len, num_kv_heads, head_dim)
        v = qkv_seq[:, q_dim + kv_dim :].reshape(seq_len, num_kv_heads, head_dim)

        # Transpose to [num_heads, seq_len, head_dim] for SDPA
        q = q.transpose(0, 1)  # [num_q_heads, seq_len, head_dim]
        k = k.transpose(0, 1)  # [num_kv_heads, seq_len, head_dim]
        v = v.transpose(0, 1)  # [num_kv_heads, seq_len, head_dim]

        # Handle GQA: repeat K and V if needed
        if num_q_heads != num_kv_heads:
            assert (
                num_q_heads % num_kv_heads == 0
            ), "num_q_heads must be divisible by num_kv_heads"
            repeat_factor = num_q_heads // num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=0)
            v = v.repeat_interleave(repeat_factor, dim=0)

        # Compute attention
        # SDPA expects [batch, num_heads, seq_len, head_dim]
        q = q.unsqueeze(0)  # [1, num_q_heads, seq_len, head_dim]
        k = k.unsqueeze(0)  # [1, num_q_heads, seq_len, head_dim]
        v = v.unsqueeze(0)  # [1, num_q_heads, seq_len, head_dim]

        # Apply scaled dot product attention
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,  # Use causal masking for autoregressive generation
        )  # [1, num_q_heads, seq_len, head_dim]

        # Remove batch dimension and transpose back
        attn_output = attn_output.squeeze(0).transpose(
            0, 1
        )  # [seq_len, num_q_heads, head_dim]

        # Flatten to [seq_len, num_q_heads * head_dim]
        attn_output = attn_output.reshape(seq_len, num_q_heads * head_dim)

        ref_outputs.append(attn_output)
        offset += seq_len

    # Concatenate all outputs
    ref_output_concat = torch.cat(
        ref_outputs, dim=0
    )  # [total_tokens, num_q_heads * head_dim]

    return ref_output_concat


def print_attn_inputs_detail(attn_inputs: PyAttentionInputs, qkv: torch.Tensor = None):
    """Print detailed information about PyAttentionInputs for debugging

    Args:
        attn_inputs: The attention inputs object
        qkv: Optional QKV tensor to print
    """
    print(f"\n{'='*80}", flush=True)
    print(f"attn_inputs详细信息:", flush=True)
    print(f"{'='*80}", flush=True)

    if qkv is not None:
        print(f"qkv: {qkv}, qkv shape: {qkv.shape}", flush=True)

    print(f"is_prefill: {attn_inputs.is_prefill}", flush=True)

    print(
        f"prefix_lengths: shape={attn_inputs.prefix_lengths.shape}, dtype={attn_inputs.prefix_lengths.dtype}",
        flush=True,
    )
    print(f"  tensor: {attn_inputs.prefix_lengths}", flush=True)
    print(f"  values: {attn_inputs.prefix_lengths.cpu().tolist()}", flush=True)

    print(
        f"input_lengths: shape={attn_inputs.input_lengths.shape}, dtype={attn_inputs.input_lengths.dtype}",
        flush=True,
    )
    print(f"  tensor: {attn_inputs.input_lengths}", flush=True)
    print(f"  values: {attn_inputs.input_lengths.cpu().tolist()}", flush=True)

    print(
        f"cu_seqlens: shape={attn_inputs.cu_seqlens.shape}, dtype={attn_inputs.cu_seqlens.dtype}",
        flush=True,
    )
    print(f"  tensor: {attn_inputs.cu_seqlens}", flush=True)
    print(f"  values: {attn_inputs.cu_seqlens.cpu().tolist()}", flush=True)

    print(
        f"cu_kv_seqlens: shape={attn_inputs.cu_kv_seqlens.shape}, dtype={attn_inputs.cu_kv_seqlens.dtype}",
        flush=True,
    )
    print(f"  tensor: {attn_inputs.cu_kv_seqlens}", flush=True)
    print(f"  values: {attn_inputs.cu_kv_seqlens.cpu().tolist()}", flush=True)

    print(f"context_total_kv_length: {attn_inputs.context_total_kv_length}", flush=True)
    print(f"total_tokens: {attn_inputs.total_tokens}", flush=True)

    print(
        f"kv_cache_block_id_host: shape={attn_inputs.kv_cache_block_id_host.shape}, dtype={attn_inputs.kv_cache_block_id_host.dtype}",
        flush=True,
    )
    print(f"  tensor:\n{attn_inputs.kv_cache_block_id_host}", flush=True)
    print(f"  values: {attn_inputs.kv_cache_block_id_host.cpu().tolist()}", flush=True)

    print(
        f"kv_cache_block_id_device: shape={attn_inputs.kv_cache_block_id_device.shape}, device={attn_inputs.kv_cache_block_id_device.device}",
        flush=True,
    )
    print(f"  tensor:\n{attn_inputs.kv_cache_block_id_device}", flush=True)

    print(f"dtype: {attn_inputs.dtype}", flush=True)
    print(f"{'='*80}\n", flush=True)
