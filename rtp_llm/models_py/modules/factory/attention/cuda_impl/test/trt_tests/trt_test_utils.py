"""Utility functions for TRT attention testing"""

from typing import List, Optional, Tuple

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
    print(
        f"\n[compute_pytorch_prefill_reference] qkv.shape={qkv.shape}, input_lengths={input_lengths}, "
        f"num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}",
        flush=True,
    )

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


def print_original_kv_tensors(
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    seq_len: int,
    num_kv_heads: int,
    max_tokens_to_print: int = 2,
):
    """Print original K/V tensors before writing to cache

    Args:
        k_tensor: K tensor [seq_len, num_kv_heads, head_dim]
        v_tensor: V tensor [seq_len, num_kv_heads, head_dim]
        seq_len: Sequence length
        num_kv_heads: Number of KV heads
        max_tokens_to_print: Maximum number of tokens to print
    """
    print(
        f"\n  [ORIGINAL K/V] Before writing to cache (first {max_tokens_to_print} tokens, all KV heads, ALL dims):",
        flush=True,
    )
    for token_idx in range(min(seq_len, max_tokens_to_print)):
        print(f"    Token {token_idx}:", flush=True)
        for kv_head_idx in range(num_kv_heads):
            k_vals = k_tensor[token_idx, kv_head_idx, :]
            v_vals = v_tensor[token_idx, kv_head_idx, :]
            print(
                f"      KV_Head {kv_head_idx}: K={k_vals.tolist()}, V={v_vals.tolist()}",
                flush=True,
            )


def print_kv_cache_readback(
    kv_cache_tensor: torch.Tensor,
    block_offset: int,
    seq_len: int,
    num_kv_heads: int,
    seq_size_per_block: int,
    max_tokens_to_print: int = 2,
    max_dims_to_print: int = 4,
):
    """Print K/V values read back from cache

    Args:
        kv_cache_tensor: KV cache tensor [total_blocks, 2, num_kv_heads, seq_size_per_block, head_dim]
        block_offset: Offset into the cache
        seq_len: Sequence length
        num_kv_heads: Number of KV heads
        seq_size_per_block: Sequence size per block
        max_tokens_to_print: Maximum number of tokens to print
        max_dims_to_print: Maximum dimensions to print
    """
    print(
        f"\n  [READ BACK FROM CACHE] After writing (first {max_tokens_to_print} tokens, all KV heads):",
        flush=True,
    )
    for token_idx in range(min(seq_len, max_tokens_to_print)):
        block_idx = token_idx // seq_size_per_block
        pos_in_block = token_idx % seq_size_per_block

        print(
            f"    Token {token_idx} (Block {block_idx}, Pos {pos_in_block}):",
            flush=True,
        )
        for kv_head_idx in range(num_kv_heads):
            k_cached = kv_cache_tensor[
                block_offset + block_idx,
                0,
                kv_head_idx,
                pos_in_block,
                :max_dims_to_print,
            ]
            v_cached = kv_cache_tensor[
                block_offset + block_idx,
                1,
                kv_head_idx,
                pos_in_block,
                :max_dims_to_print,
            ]
            print(
                f"      KV_Head {kv_head_idx}: K={k_cached.tolist()}, V={v_cached.tolist()}",
                flush=True,
            )


def print_kv_cache_modifications(
    kv_cache_tensor: torch.Tensor, num_kv_heads: int, max_heads_to_print: int = 2
):
    """Print KV cache data after modifications

    Args:
        kv_cache_tensor: KV cache tensor [total_blocks, 2, num_kv_heads, seq_size_per_block, head_dim]
        num_kv_heads: Number of KV heads
        max_heads_to_print: Maximum number of heads to print
    """
    print(f"\n[MANUAL ATTENTION CALCULATION] Q_Head0 with KV_Head0", flush=True)
    print(f"KV cache data after modifications:", flush=True)
    print(
        f"  KV_Head0, Token0: K={kv_cache_tensor[0, 0, 0, 0, :4].tolist()}, V={kv_cache_tensor[0, 1, 0, 0, :4].tolist()}",
        flush=True,
    )
    print(
        f"  KV_Head0, Token1: K={kv_cache_tensor[0, 0, 0, 1, :4].tolist()}, V={kv_cache_tensor[0, 1, 0, 1, :4].tolist()}",
        flush=True,
    )
    if num_kv_heads > 1:
        print(
            f"  KV_Head1, Token0: K={kv_cache_tensor[0, 0, 1, 0, :4].tolist()}, V={kv_cache_tensor[0, 1, 1, 0, :4].tolist()}",
            flush=True,
        )


def compute_expected_attention_output(
    q_tensor: torch.Tensor,
    kv_cache_tensor: torch.Tensor,
    token_idx: int,
    q_head_idx: int,
    kv_head_idx: int,
    head_dim: int,
    max_tokens: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute expected attention output for debugging

    Args:
        q_tensor: Q tensor [total_tokens, head_num, head_dim]
        kv_cache_tensor: KV cache [total_blocks, 2, num_kv_heads, seq_size_per_block, head_dim]
        token_idx: Token index
        q_head_idx: Q head index
        kv_head_idx: KV head index
        head_dim: Head dimension
        max_tokens: Maximum tokens to attend to

    Returns:
        Tuple of (attention_scores, attention_weights, expected_output)
    """
    # Get Q for specified token and head
    q = q_tensor[token_idx, q_head_idx, :]  # [head_dim]

    # Get K and V from cache for specified KV head
    k = kv_cache_tensor[0, 0, kv_head_idx, :max_tokens, :]  # [max_tokens, head_dim]
    v = kv_cache_tensor[0, 1, kv_head_idx, :max_tokens, :]  # [max_tokens, head_dim]

    # Calculate attention: softmax(Q @ K^T / sqrt(d)) @ V
    scale = 1.0 / (head_dim**0.5)
    attn_scores = (
        torch.matmul(q.unsqueeze(0), k.transpose(0, 1)) * scale
    )  # [1, max_tokens]
    attn_weights = torch.softmax(attn_scores, dim=-1)
    expected_output = torch.matmul(attn_weights, v)[0]  # [head_dim]

    return attn_scores[0], attn_weights[0], expected_output


def print_expected_attention_output(
    q_tensor: torch.Tensor,
    kv_cache_tensor: torch.Tensor,
    head_dim: int,
    head_num: int,
    max_q_heads_to_print: int = 4,
):
    """Print expected attention outputs for diagnostic purposes

    Args:
        q_tensor: Q tensor [total_tokens, head_num, head_dim]
        kv_cache_tensor: KV cache [total_blocks, 2, num_kv_heads, seq_size_per_block, head_dim]
        head_dim: Head dimension
        head_num: Number of Q heads
        max_q_heads_to_print: Maximum Q heads to print
    """
    total_tokens = q_tensor.shape[0]
    if total_tokens < 2:
        return

    print(f"\n[DIAGNOSIS] Calculate attention for ALL Q heads @ KV_Head0", flush=True)
    print(
        f"  This helps identify which Q head TRT kernel is actually reading", flush=True
    )

    # Get K and V from cache for KV_Head 0
    k_kv_head0 = kv_cache_tensor[0, 0, 0, :2, :]  # [2, head_dim]
    v_kv_head0 = kv_cache_tensor[0, 1, 0, :2, :]  # [2, head_dim]

    print(f"\n  KV_Head0 data:", flush=True)
    print(f"    K[token0][:4] = {k_kv_head0[0, :4].tolist()}", flush=True)
    print(f"    K[token1][:4] = {k_kv_head0[1, :4].tolist()}", flush=True)
    print(f"    V[token0][:4] = {v_kv_head0[0, :4].tolist()}", flush=True)
    print(f"    V[token1][:4] = {v_kv_head0[1, :4].tolist()}", flush=True)

    # Calculate attention for Token1 with each Q head
    for q_head_idx in range(min(head_num, max_q_heads_to_print)):
        attn_scores, attn_weights, expected_output = compute_expected_attention_output(
            q_tensor, kv_cache_tensor, 1, q_head_idx, 0, head_dim
        )

        q_t1 = q_tensor[1, q_head_idx, :]
        print(f"\n  Token1 Q_Head{q_head_idx} @ KV_Head0:", flush=True)
        print(f"    Q[:4] = {q_t1[:4].tolist()}", flush=True)
        print(f"    Attn weights = {attn_weights.tolist()}", flush=True)
        print(f"    Expected output[:8] = {expected_output[:8].tolist()}", flush=True)


def print_q_tensor_info(
    q: torch.Tensor,
    head_num: int,
    max_tokens_to_print: int = 2,
    max_heads_to_print: int = 4,
    max_dims_to_print: int = 4,
):
    """Print Q tensor information before/after transpose

    Args:
        q: Q tensor [total_tokens, head_num, head_dim]
        head_num: Number of Q heads
        max_tokens_to_print: Maximum tokens to print
        max_heads_to_print: Maximum heads to print
        max_dims_to_print: Maximum dimensions to print
    """
    total_tokens = q.shape[0]

    print(
        f"\n[Q TENSOR] Before transpose (first {max_tokens_to_print} tokens, first {max_heads_to_print} Q heads):",
        flush=True,
    )
    for token_idx in range(min(total_tokens, max_tokens_to_print)):
        print(f"  Token {token_idx}:", flush=True)
        for q_head_idx in range(min(head_num, max_heads_to_print)):
            q_vals = q[token_idx, q_head_idx, :max_dims_to_print]
            print(f"    Q_Head {q_head_idx}: {q_vals.tolist()}", flush=True)


def print_q_tensor_final_layout(
    q: torch.Tensor,
    head_num: int,
    max_tokens_to_print: int = 2,
    max_heads_to_print: int = 4,
    max_dims_to_print: int = 4,
):
    """Print Q tensor final layout (after handling transpose)

    Args:
        q: Q tensor [total_tokens, head_num, head_dim]
        head_num: Number of Q heads
        max_tokens_to_print: Maximum tokens to print
        max_heads_to_print: Maximum heads to print
        max_dims_to_print: Maximum dimensions to print
    """
    total_tokens = q.shape[0]

    print(
        f"\n[Q TENSOR] Final layout [token, head, dim] (first {max_tokens_to_print} tokens, first {max_heads_to_print} heads):",
        flush=True,
    )
    for token_idx in range(min(total_tokens, max_tokens_to_print)):
        print(f"  Token {token_idx}:", flush=True)
        for q_head_idx in range(min(head_num, max_heads_to_print)):
            q_vals = q[token_idx, q_head_idx, :max_dims_to_print]
            print(f"    Q_Head {q_head_idx}: {q_vals.tolist()}", flush=True)


def print_per_token_per_head_comparison(
    output: torch.Tensor,
    ref_output: torch.Tensor,
    head_num: int,
    size_per_head: int,
    max_tokens_to_print: int = 4,
    max_heads_to_print: int = 4,
    max_dims_to_print: int = 4,
    rtol: float = 5e-3,
    atol: float = 5e-3,
):
    """Print detailed per-token per-head comparison

    Args:
        output: Model output [total_tokens, head_num * size_per_head]
        ref_output: Reference output [total_tokens, head_num * size_per_head]
        head_num: Number of heads
        size_per_head: Size per head
        max_tokens_to_print: Maximum tokens to print
        max_heads_to_print: Maximum heads to print
        max_dims_to_print: Maximum dimensions to print
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    total_tokens = output.shape[0]

    # Reshape outputs
    output_reshaped = output.view(total_tokens, head_num, size_per_head)
    ref_reshaped = ref_output.view(total_tokens, head_num, size_per_head)

    # Print match matrix
    print(
        f"\n[DEBUG] Per-token per-head comparison (ALL tokens, ALL heads):", flush=True
    )
    print(f"\n  Match Matrix (rows=tokens, cols=heads):", flush=True)

    # Print header
    header = "      "
    for h in range(min(head_num, 10)):
        header += f"H {h} "
    print(header, flush=True)

    # Print each token row
    for t in range(total_tokens):
        row = f"  T {t}: "
        for h in range(min(head_num, 10)):
            match = torch.allclose(
                output_reshaped[t, h, :], ref_reshaped[t, h, :], rtol=rtol, atol=atol
            )
            row += "✓  " if match else "✗  "
        print(row, flush=True)

    # Print detailed values
    print(
        f"\n[DEBUG] Detailed values (first {max_tokens_to_print} tokens, first {max_heads_to_print} heads):",
        flush=True,
    )
    print(flush=True)

    for token_idx in range(min(total_tokens, max_tokens_to_print)):
        print(f"  Token {token_idx}:", flush=True)
        for head_idx in range(min(head_num, max_heads_to_print)):
            out_vals = output_reshaped[token_idx, head_idx, :max_dims_to_print]
            ref_vals = ref_reshaped[token_idx, head_idx, :max_dims_to_print]

            diff = torch.abs(out_vals - ref_vals).max().item()
            match = torch.allclose(
                output_reshaped[token_idx, head_idx, :],
                ref_reshaped[token_idx, head_idx, :],
                rtol=rtol,
                atol=atol,
            )
            status = "✓" if match else "✗"

            print(
                f"    Head {head_idx}: out={out_vals.tolist()}, ref={ref_vals.tolist()}, diff={diff:.4f} {status}",
                flush=True,
            )
        print(flush=True)
