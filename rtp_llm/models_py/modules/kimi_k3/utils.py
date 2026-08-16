"""Shared metadata and input helpers for Kimi K3 modeling."""

from __future__ import annotations

import os
from typing import Optional, Sequence

import torch

from rtp_llm.ops.compute_ops import PyAttentionInputs


def prefill_chunk_tokens() -> int:
    """Return the opt-in whole-model Prefill chunk size."""

    raw = os.environ.get("KIMI_K3_PREFILL_CHUNK_TOKENS", "0").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "KIMI_K3_PREFILL_CHUNK_TOKENS must be a non-negative integer, "
            f"got {raw!r}"
        ) from exc
    if value < 0:
        raise ValueError(
            "KIMI_K3_PREFILL_CHUNK_TOKENS must be non-negative, " f"got {value}"
        )
    return value


def fused_ag_workspace_global_tokens(
    max_seq_len: int,
    max_context_batch_size: int,
    chunk_tokens: int,
) -> int:
    """Bound the symmetric AG/GEMM workspace by one model invocation."""

    configured_tokens = int(max_seq_len) * int(max_context_batch_size)
    if chunk_tokens <= 0:
        return configured_tokens
    return min(
        configured_tokens,
        int(chunk_tokens) * int(max_context_batch_size),
    )


def mask_multimodal_token_ids(
    input_ids: torch.Tensor,
    multimodal_features: Sequence[torch.Tensor],
    multimodal_locs: torch.Tensor,
) -> torch.Tensor:
    """Zero the token ids that ``MultimodalEmbeddingInjector`` will overwrite.

    Multimodal rows do not hold vocab ids: ``MultimodalProcessor::expandTokenIds``
    replaces them with per-row feature hashes (``featureHashToTokenId``, an arbitrary
    ``int32`` that is routinely negative or ``>= vocab_size``) so the prefix cache can
    tell two images apart. Feeding those to the embedding op indexes out of bounds, so
    they must be masked before lookup. The zeroed rows are overwritten by the injector.
    """

    locs = multimodal_locs.to(device="cpu", dtype=torch.long).view(-1).tolist()
    masked_ids = input_ids.clone()
    for feature, loc in zip(multimodal_features, locs):
        if feature is None:
            continue
        # loc < 0 means the head rows already live in the reused KV prefix and only
        # the tail lands in this chunk, at token 0 -- same convention as the injector.
        offset = max(loc, 0)
        length = feature.size(0) - min(max(-loc, 0), feature.size(0))
        # Out-of-range spans are clipped rather than rejected here; the injector
        # raises the canonical IndexError after the embedding lookup.
        length = min(length, masked_ids.size(0) - offset)
        if length > 0:
            masked_ids.narrow(0, offset, length).fill_(0)
    return masked_ids


def sequence_offsets(
    cu_seqlens: torch.Tensor,
    token_count: int,
    *,
    cu_seqlens_host: Optional[torch.Tensor] = None,
) -> list[tuple[int, int]]:
    """Validate packed prefix sums and return host-visible sequence ranges."""

    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a one-dimensional [batch + 1] tensor")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must use an integer dtype")
    source = (
        cu_seqlens_host
        if cu_seqlens_host is not None and cu_seqlens_host.numel()
        else cu_seqlens
    )
    offsets = [int(value) for value in source.detach().cpu().tolist()]
    if offsets[0] != 0 or offsets[-1] != token_count:
        raise ValueError(
            f"cu_seqlens must start at 0 and end at {token_count}, got {offsets}"
        )
    if any(left > right for left, right in zip(offsets, offsets[1:])):
        raise ValueError("cu_seqlens must be non-decreasing")
    return list(zip(offsets, offsets[1:]))


def resolve_cu_seqlens(
    attention_inputs: PyAttentionInputs,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Resolve and validate the packed sequence boundaries for one invocation."""

    cu_seqlens = (
        attention_inputs.cu_seqlens
        if attention_inputs.is_prefill
        else attention_inputs.decode_cu_seqlens_d
    )
    if cu_seqlens is None or cu_seqlens.numel() == 0:
        cu_seqlens = (
            torch.tensor(
                [0, input_ids.numel()],
                dtype=torch.int32,
                device=input_ids.device,
            )
            if attention_inputs.is_prefill
            else torch.arange(
                input_ids.numel() + 1,
                dtype=torch.int32,
                device=input_ids.device,
            )
        )
    graph_decode = not attention_inputs.is_prefill and (
        bool(getattr(attention_inputs, "is_cuda_graph", False))
        or (input_ids.is_cuda and torch.cuda.is_current_stream_capturing())
    )
    if graph_decode:
        # Decode has exactly one packed token per request. Inspecting the CUDA
        # prefix sums on the host would make capture illegal and freeze replay
        # metadata; shape validation is sufficient here.
        if cu_seqlens.numel() != input_ids.numel() + 1:
            raise ValueError(
                "K3 CUDA Graph decode requires one cu_seqlens interval per token"
            )
    else:
        sequence_offsets(
            cu_seqlens,
            input_ids.numel(),
            cu_seqlens_host=getattr(attention_inputs, "cu_seqlens_host", None),
        )
    return cu_seqlens


__all__ = [
    "fused_ag_workspace_global_tokens",
    "mask_multimodal_token_ids",
    "prefill_chunk_tokens",
    "resolve_cu_seqlens",
    "sequence_offsets",
]
