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
    if _native_paged_mqa_enabled(lengths):
        batch_size = physical_block_table(attention_inputs).shape[0]
        if batch_size == 0 or lengths.numel() % batch_size:
            raise ValueError("MTP query lengths must contain equal rows per request")
        return lengths.view(batch_size, -1)
    return lengths.reshape(-1, 1)


def _native_paged_mqa_enabled(lengths: torch.Tensor) -> bool:
    return (
        lengths.numel() > 0
        and lengths.is_cuda
        and os.environ.get("RTP_LLM_INDEXER_NATIVE_MTP", "1") != "0"
        and torch.cuda.get_device_capability(lengths.device)[0] == 10
    )


def refresh_paged_mqa_request_ids(
    attention_inputs: Any,
    fmha_params: Any,
    lengths: torch.Tensor,
    deep_gemm: Any,
    forbid_realloc: bool,
) -> torch.Tensor | None:
    if not adaptive_target_layout(attention_inputs) or not _native_paged_mqa_enabled(
        lengths
    ):
        return None
    # Older wheels only support the original per-query API.
    if not all(
        "indices:" in (getattr(deep_gemm, name, None).__doc__ or "")
        for name in ("get_paged_mqa_logits_metadata", "fp8_paged_mqa_logits")
    ):
        return None

    batch_ids = fmha_params.batch_indice_d
    if batch_ids.shape != lengths.shape or batch_ids.dtype != torch.int32:
        raise ValueError("Paged MQA request IDs need one int32 value per query")
    ids = getattr(fmha_params, "paged_mqa_request_ids", None)
    if ids is None or ids.shape != batch_ids.shape or ids.device != batch_ids.device:
        if forbid_realloc:
            raise RuntimeError(
                "Paged MQA request ID storage changed during graph replay"
            )
        ids = torch.empty_like(batch_ids, memory_format=torch.contiguous_format)
        fmha_params.paged_mqa_request_ids = ids
    # DeepGEMM uses the final context length of each contiguous request group.
    # Graph padding carries batch ID 0. Give it its own group so its length 0
    # cannot replace the actual length of a preceding request with batch ID 0.
    ids.copy_(batch_ids)
    ids.masked_fill_(lengths <= 0, -1)
    return ids


def adaptive_target_layout(attention_inputs: Any) -> bool:
    return bool(getattr(attention_inputs, "is_target_verify", False)) and (
        os.environ.get("RTP_LLM_ADAPTIVE_VERIFICATION", "0") == "1"
    )
