"""Ascend implementation of Qwen3.5 recurrent Gated-DeltaNet decode."""

from __future__ import annotations

from typing import Optional

import torch

from rtp_llm.models_py.kernels.ascend.linear_attention import l2norm_fwd


def _get_ascendc_ops():
    try:
        from fla_npu.ops import ascendc
    except ImportError as exc:  # pragma: no cover - depends on the NPU image
        raise RuntimeError(
            "Qwen3.5 recurrent decode on Ascend requires the SoC-specific "
            "flash-linear-attention-npu wheel."
        ) from exc
    if not hasattr(ascendc, "npu_recurrent_gated_delta_rule"):
        raise RuntimeError(
            "The installed FLA-NPU wheel does not export "
            "npu_recurrent_gated_delta_rule; install the commit verified by "
            "the Qwen3.5 migration guide."
        )
    return ascendc


def _to_int_list(tensor: torch.Tensor) -> list[int]:
    return [int(value) for value in tensor.detach().cpu().tolist()]


_ACTUAL_SEQ_LENGTHS_CACHE: dict[tuple[str, int], torch.Tensor] = {}


def _decode_actual_seq_lengths(device: torch.device, batch: int) -> torch.Tensor:
    """``[0, 1, 1, ...]`` cuSeqlens-style metadata for single-token decode.

    Built from device-side fills and cached per (device, batch) so aclgraph
    capture records a stable tensor address (``torch.tensor`` from host values
    would H2D-sync, which is illegal during capture).
    """

    key = (str(device), batch)
    cached = _ACTUAL_SEQ_LENGTHS_CACHE.get(key)
    if cached is None:
        cached = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.ones(batch, dtype=torch.int32, device=device),
        ])
        _ACTUAL_SEQ_LENGTHS_CACHE[key] = cached
    return cached


def _fused_recurrent_decode_device(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    block_map: torch.Tensor,
    sequence_lengths: torch.Tensor,
    seq_size_per_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-token decode step with fully device-side metadata.

    Same AscendC operator as the host-metadata path below (single consumer):
    the read/write pages are gathered from ``block_map`` on device and the
    cross-block state seed uses block-level page-row copies, so the step is
    aclgraph-capturable and replayable while eager keeps the identical
    numerics.
    """

    from rtp_llm.models_py.kernels.ascend.paged_state import (
        decode_state_indices,
        seed_state_segment,
    )

    ascendc = _get_ascendc_ops()
    batch = q.shape[0]
    read_idx, write_idx = decode_state_indices(
        block_map, sequence_lengths, int(seq_size_per_block)
    )
    seed_state_segment(initial_state, read_idx, write_idx)
    result = ascendc.npu_recurrent_gated_delta_rule(
        q.reshape(batch, *q.shape[2:]).to(torch.bfloat16),
        k.reshape(batch, *k.shape[2:]).to(torch.bfloat16),
        v.reshape(batch, *v.shape[2:]).to(torch.bfloat16),
        initial_state,
        beta=beta.reshape(batch, beta.shape[-1]).to(torch.bfloat16),
        scale=float(scale),
        actual_seq_lengths=_decode_actual_seq_lengths(q.device, batch),
        ssm_state_indices=write_idx.to(torch.int32),
        g=g.reshape(batch, g.shape[-1]).float(),
    )
    out = result[0] if isinstance(result, (tuple, list)) else result
    return out.reshape(batch, 1, *out.shape[1:]).to(q.dtype), initial_state


def _resolve_state_pages(
    block_map: Optional[torch.Tensor],
    sequence_lengths: Optional[torch.Tensor],
    batch: int,
    token_count: int,
    seq_size_per_block: int,
) -> tuple[list[int], list[list[int]]]:
    """Return the read page and speculative write pages for each sequence.

    ``sequence_lengths`` is RTP's ``sequence_lengths_plus_1_d``: its value is
    the total sequence length after the first token in this invocation.  RTP's
    continuous-batching contract reads the state before that token from
    ``(length - 2) // block_size`` and stores every speculative token in a
    consecutive block-map entry beginning at ``(length - 1) // block_size``.
    The latter is intentionally *not* ordinary token-to-block placement.
    """
    if seq_size_per_block <= 0:
        raise ValueError("seq_size_per_block must be positive")
    if block_map is None:
        pages = [[batch_idx] * token_count for batch_idx in range(batch)]
        return list(range(batch)), pages
    if block_map.ndim != 2 or block_map.shape[0] != batch:
        raise ValueError("block_map must have shape [batch, max_blocks]")
    mapping = block_map.detach().cpu().tolist()
    first_lengths = (
        _to_int_list(sequence_lengths) if sequence_lengths is not None else [1] * batch
    )
    if len(first_lengths) != batch:
        raise ValueError("sequence_lengths must contain one value per batch")
    read_pages: list[int] = []
    write_pages: list[list[int]] = []
    for batch_idx, first_length in enumerate(first_lengths):
        if first_length <= 0:
            raise ValueError("decode sequence lengths must be positive")
        # The CUDA reference uses cal_block_idx(length - 1) for the load and
        # cal_block_idx(length) + token_idx for the writes, where
        # cal_block_idx(x) == (x - 1) // block_size.
        read_block_pos = max(first_length - 2, 0) // seq_size_per_block
        write_block_start = (first_length - 1) // seq_size_per_block
        if read_block_pos >= len(mapping[batch_idx]):
            raise ValueError("block_map does not cover the decode read position")
        read_page = int(mapping[batch_idx][read_block_pos])
        if read_page <= 0:
            raise ValueError(
                "non-positive decode state pages are not supported on Ascend yet"
            )
        read_pages.append(read_page)

        batch_pages: list[int] = []
        for token_idx in range(token_count):
            block_pos = write_block_start + token_idx
            if block_pos >= len(mapping[batch_idx]):
                raise ValueError("block_map does not cover the decode write position")
            page = int(mapping[batch_idx][block_pos])
            if page <= 0:
                raise ValueError(
                    "non-positive decode state pages are not supported on Ascend yet"
                )
            batch_pages.append(page)
        write_pages.append(batch_pages)
    return read_pages, write_pages


def _seed_first_write_pages(
    state: torch.Tensor,
    read_pages: list[int],
    write_pages: list[list[int]],
) -> None:
    for batch_idx, batch_pages in enumerate(write_pages):
        if not batch_pages:
            continue
        source = read_pages[batch_idx]
        destination = batch_pages[0]
        if source != destination:
            state[destination].copy_(state[source])


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    block_map: Optional[torch.Tensor] = None,
    seq_size_per_block=1,
    sequence_lengths: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cu_seqlens is not None:
        raise NotImplementedError(
            "cu_seqlens is a prefill interface; Ascend recurrent decode uses "
            "block_map and sequence_lengths"
        )
    if initial_state is None:
        raise ValueError("initial_state is required for recurrent decode")
    if not inplace_final_state:
        raise NotImplementedError(
            "Ascend recurrent decode currently requires inplace_final_state=True"
        )
    if initial_state.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(
            "FLA-NPU recurrent state must be bfloat16 or float32"
        )
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("q/k/v must have shapes [B,T,H,D]")
    batch, token_count = q.shape[:2]
    if v.shape[:2] != (batch, token_count):
        raise ValueError("q/k/v must share batch and token dimensions")
    if beta is not None and beta.shape != v.shape[:-1]:
        raise ValueError("beta must have shape [B,T,HV]")
    if g.shape != v.shape[:-1]:
        raise ValueError("g must have shape [B,T,HV]")
    if initial_state.ndim != 4 or initial_state.shape[1:] != (
        v.shape[2],
        v.shape[3],
        q.shape[3],
    ):
        raise ValueError("initial_state must have shape [pages,HV,DV,DK]")
    if initial_state.stride(-1) != 1:
        raise ValueError("initial_state DK dimension must be contiguous")
    if token_count > 8:
        raise ValueError(
            "FLA-NPU recurrent decode supports at most 8 tokens per sequence"
        )
    if beta is None:
        beta = torch.ones_like(v[..., 0])
    if scale is None:
        scale = k.shape[-1] ** -0.5
    elif scale <= 0:
        raise ValueError("scale must be positive")
    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    # Standard decode (one token per sequence): device-metadata path shared by
    # eager and aclgraph capture — no D2H, single AscendC consumer.  Multi-
    # token speculative decode (T <= 8) keeps the host-metadata path below,
    # which is eager-only (target-verify never runs inside the decode graph).
    if (
        token_count == 1
        and block_map is not None
        and sequence_lengths is not None
        and block_map.ndim == 2
        and block_map.shape[0] == batch
    ):
        return _fused_recurrent_decode_device(
            q, k, v, g, beta, scale, initial_state,
            block_map, sequence_lengths, int(seq_size_per_block),
        )

    read_pages, write_pages = _resolve_state_pages(
        block_map,
        sequence_lengths,
        batch,
        token_count,
        int(seq_size_per_block),
    )
    state = initial_state
    if token_count == 0:
        return v.new_empty(v.shape), state

    ascendc = _get_ascendc_ops()

    # The operator keeps the recurrence in FP32 across all tokens and uses one
    # state index per token for the BF16 snapshots.  Seed only the first output
    # page from the previously committed page; later pages are written by the
    # same launch, avoiding a BF16 reload between speculative tokens.
    _seed_first_write_pages(state, read_pages, write_pages)
    state_indices = [page for batch_pages in write_pages for page in batch_pages]
    actual_seq_lengths = torch.tensor(
        [0] + [token_count] * batch,
        dtype=torch.int32,
        device=q.device,
    )
    ssm_state_indices = torch.tensor(state_indices, dtype=torch.int32, device=q.device)
    result = ascendc.npu_recurrent_gated_delta_rule(
        q.reshape(-1, *q.shape[2:]).to(torch.bfloat16),
        k.reshape(-1, *k.shape[2:]).to(torch.bfloat16),
        v.reshape(-1, *v.shape[2:]).to(torch.bfloat16),
        state,
        beta=beta.reshape(-1, beta.shape[-1]).to(torch.bfloat16),
        scale=float(scale),
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=ssm_state_indices,
        g=g.reshape(-1, g.shape[-1]).float(),
    )
    out = result[0] if isinstance(result, (tuple, list)) else result
    return out.reshape(batch, token_count, *out.shape[1:]).to(q.dtype), state


__all__ = ["fused_recurrent_gated_delta_rule"]
