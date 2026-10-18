"""Shared request layout for paged Indexer logits and its scheduler."""

import os
from typing import Any

import torch


def physical_block_table(attention_inputs: Any) -> torch.Tensor:
    # The kernel table has a stable device address across CUDA graph replays.
    kernel = getattr(attention_inputs, "kv_cache_kernel_block_id_device", None)
    if isinstance(kernel, torch.Tensor) and kernel.numel() > 0:
        return kernel
    return attention_inputs.kv_cache_block_id_device


def paged_mqa_context_lens(attention_inputs: Any, fmha_params: Any) -> torch.Tensor:
    """Keep MTP queries grouped by request on Blackwell.

    The logits call and schedule preparation must use the same [B, next_n]
    layout. Each query retains its own causal length, including padded rows.
    The environment override allows comparison with the flattened layout.
    """
    multi_token = bool(getattr(attention_inputs, "is_target_verify", False)) or bool(
        getattr(attention_inputs, "is_draft_extend", False)
    )
    if not multi_token:
        return fmha_params.kvlen_d.reshape(-1, 1)

    lengths = fmha_params.expanded_seq_lens
    if (
        lengths.numel() > 0
        and os.environ.get("RTP_LLM_INDEXER_NATIVE_MTP", "1") != "0"
        and lengths.is_cuda
        and torch.cuda.get_device_capability(lengths.device)[0] == 10
    ):
        batch_size = physical_block_table(attention_inputs).shape[0]
        if batch_size == 0 or lengths.numel() % batch_size:
            raise ValueError("MTP query lengths must contain equal rows per request")
        return lengths.view(batch_size, -1)
    return lengths.reshape(-1, 1)
