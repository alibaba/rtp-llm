import math
import os
import sys
from typing import List, Optional
from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs


def write_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: KVCache,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
):
    assert seq_lens.sum().item() == k.shape[0] and k.shape[0] == v.shape[0]
    batch_size = seq_lens.shape[0]
    k_cache = kv_cache.kv_cache_base.select(
        1, 0
    )  # [num_pages, num_kv_heads, page_size, head_dim]
    v_cache = kv_cache.kv_cache_base.select(
        1, 1
    )  # [num_pages, num_kv_heads, page_size, head_dim]
    page_size = k_cache.shape[2]
    max_seq_len = seq_lens.max().item()
    max_block_size = max_seq_len // page_size + 1
    token_idx = 0
    for i in range(batch_size):
        seq_len = seq_lens[i].item()
        num_blocks = (seq_len + page_size - 1) // page_size
        for block_idx in range(num_blocks):
            block_id = block_tables[i, block_idx].item()
            block_start = block_idx * page_size
            block_end = min(block_start + page_size, seq_len)
            block_len = block_end - block_start
            if block_len > 0:
                # Validate block_id is within cache bounds
                if block_id < 0 or block_id >= k_cache.shape[0]:
                    raise ValueError(
                        f"Invalid block_id {block_id} for sequence {i}, "
                        f"block {block_idx}. Cache has {k_cache.shape[0]} pages."
                    )
                k_cache[block_id, :, :block_len, :] = k[
                    token_idx : token_idx + block_len
                ].transpose(0, 1)
                v_cache[block_id, :, :block_len, :] = v[
                    token_idx : token_idx + block_len
                ].transpose(0, 1)
            token_idx += block_len


def attention_prefill_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool = True,
) -> torch.Tensor:
    assert (
        seq_lens.sum().item() == k.shape[0]
        and k.shape[0] == v.shape[0]
        and k.shape[0] == q.shape[0]
    )
    batch_size = seq_lens.shape[0]
    num_tokens = q.shape[0]
    dtype = q.dtype
    device = q.device
    # Scale factor
    scale = 1.0 / math.sqrt(head_dim)
    # Group query attention: repeat k and v if num_heads > num_kv_heads
    num_groups = num_heads // num_kv_heads
    if num_groups > 1:
        k = k.repeat_interleave(num_groups, dim=1)  # [num_tokens, num_heads, head_dim]
        v = v.repeat_interleave(num_groups, dim=1)  # [num_tokens, num_heads, head_dim]
    # Reshape for batch processing
    q = q.view(num_tokens, num_heads, head_dim)
    k = k.view(num_tokens, num_heads, head_dim)
    v = v.view(num_tokens, num_heads, head_dim)
    # Process each sequence in the batch
    token_idx = 0
    outputs = []
    for i in range(batch_size):
        seq_len = seq_lens[i].item()
        start_idx = token_idx
        end_idx = token_idx + seq_len
        token_idx = end_idx
        if seq_len == 0:
            continue
        q_seq = q[start_idx:end_idx].transpose(0, 1)
        k_seq = k[start_idx:end_idx].transpose(0, 1)
        v_seq = v[start_idx:end_idx].transpose(0, 1)
        scores = torch.einsum("hsd,htd->hst", q_seq, k_seq) * scale
        # Apply causal mask if needed
        if causal:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (seq_len, seq_len), fill_value=min_dtype, dtype=dtype, device=device
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            scores = scores + causal_mask.unsqueeze(0)
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)  # [num_heads, seq_len, seq_len]
        # Apply attention to values
        output = (
            torch.einsum("hst,htd->hsd", attn_weights, v_seq)
            .transpose(0, 1)
            .contiguous()
        )
        outputs.append(output)
    return torch.cat(outputs, dim=0)  # [num_tokens, num_heads, head_dim]


def gen_attention_inputs(
    page_size: int,
    num_pages: int,
    input_lengths: Optional[List[int]] = None,
    sequence_lengths: Optional[List[int]] = None,
) -> PyAttentionInputs:
    assert not (input_lengths is None and sequence_lengths is None)
    attention_inputs: PyAttentionInputs = PyAttentionInputs()
    batch_size: int = 0
    max_seq_len: int = 0
    if sequence_lengths is not None:
        batch_size = len(sequence_lengths)
        attention_inputs.sequence_lengths = torch.tensor(
            sequence_lengths, dtype=torch.int32, device=torch.device("cpu")
        ).pin_memory()
        max_seq_len = attention_inputs.sequence_lengths.max().item()
        attention_inputs.is_prefill = False
        attention_inputs.prefix_lengths = torch.zeros(batch_size, dtype=torch.int32)
        attention_inputs.input_lengths = attention_inputs.sequence_lengths
    if input_lengths is not None:
        batch_size = len(input_lengths)
        attention_inputs.input_lengths = torch.tensor(
            input_lengths, dtype=torch.int32, device=torch.device("cpu")
        ).pin_memory()
        attention_inputs.sequence_lengths = attention_inputs.input_lengths
        attention_inputs.prefix_lengths = torch.zeros(batch_size, dtype=torch.int32)
        attention_inputs.is_prefill = True
    cu_seqlens = torch.zeros(
        attention_inputs.input_lengths.numel() + 1,
        dtype=torch.int32,
        device=torch.device("cpu"),
    ).pin_memory()
    cu_seqlens[1:] = attention_inputs.input_lengths.cumsum(0)
    attention_inputs.cu_seqlens = cu_seqlens
    attention_inputs.cu_kv_seqlens = cu_seqlens
    max_seq_len = attention_inputs.input_lengths.max().item()
    max_block_size = max_seq_len // page_size + 1
    # Ensure we have enough pages in cache
    assert (
        batch_size * max_block_size <= num_pages
    ), f"Not enough pages: need {batch_size * max_block_size}, have {num_pages}"
    block_tables = (
        torch.arange(
            batch_size * max_block_size,
            dtype=torch.int32,
            device=torch.device("cpu"),
        )
        .view(batch_size, max_block_size)
        .pin_memory()
    )
    attention_inputs.kv_cache_block_id_device = block_tables
    attention_inputs.kv_cache_block_id_host = block_tables
    return attention_inputs
