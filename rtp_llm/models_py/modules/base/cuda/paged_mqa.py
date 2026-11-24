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
    if adaptive_target_layout(attention_inputs):
        return lengths.reshape(-1, 1)
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


def paged_mqa_schedule_inputs(
    deep_gemm: Any, attention_inputs: Any, fmha_params: Any
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build matching context lengths and query mapping for 2.8.2 paged MQA."""
    multi_token = bool(getattr(attention_inputs, "is_target_verify", False)) or bool(
        getattr(attention_inputs, "is_draft_extend", False)
    )
    lengths = fmha_params.expanded_seq_lens if multi_token else fmha_params.kvlen_d
    if (
        not hasattr(deep_gemm, "fp8_paged_mqa_logits")
        and hasattr(deep_gemm, "fp8_fp4_paged_mqa_logits")
        and lengths.is_cuda
        and torch.cuda.get_device_capability(lengths.device)[0] == 10
    ):
        # The 2.8.2 kernel resolves each flattened query through indices.
        return lengths.reshape(-1, 1), torch.arange(
            lengths.numel(), dtype=torch.int32, device=lengths.device
        )
    return paged_mqa_context_lens(attention_inputs, fmha_params), None


def adaptive_target_layout(attention_inputs: Any) -> bool:
    return bool(getattr(attention_inputs, "is_target_verify", False)) and (
        os.environ.get("RTP_LLM_ADAPTIVE_VERIFICATION", "0") == "1"
    )
