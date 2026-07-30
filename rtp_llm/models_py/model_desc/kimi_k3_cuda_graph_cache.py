from __future__ import annotations

from typing import Any

import torch


def cuda_graph_decode_block_ids(
    sequence_lengths_plus_one: torch.Tensor | None,
    block_map: torch.Tensor | None,
    page_size: int,
    cache_block_count: int,
    *,
    position_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve live KDA block IDs on device for CUDA Graph replay."""

    if (
        sequence_lengths_plus_one is None
        or sequence_lengths_plus_one.numel() == 0
        or not sequence_lengths_plus_one.is_cuda
    ):
        raise RuntimeError(
            "K3 CUDA Graph decode requires CUDA sequence_lengths_plus_1_d"
        )
    if block_map is None or block_map.numel() == 0 or not block_map.is_cuda:
        raise RuntimeError("K3 CUDA Graph decode requires a CUDA KDA block table")
    if block_map.ndim != 2:
        raise ValueError(
            "K3 CUDA Graph decode requires a two-dimensional KDA block table"
        )
    if block_map.shape[0] != sequence_lengths_plus_one.shape[0]:
        raise ValueError(
            "K3 CUDA Graph KDA block-table batch mismatch: "
            f"blocks={block_map.shape[0]} "
            f"lengths={sequence_lengths_plus_one.shape[0]}"
        )
    if page_size <= 0 or cache_block_count <= 0:
        raise ValueError(
            "K3 CUDA Graph decode requires positive cache/page dimensions"
        )

    positions = sequence_lengths_plus_one.to(torch.int64) + int(position_offset)
    page_indices = torch.div(
        positions.clamp_min(0),
        page_size,
        rounding_mode="floor",
    ).clamp_max(block_map.shape[1] - 1)
    rows = torch.arange(
        sequence_lengths_plus_one.shape[0],
        dtype=torch.int64,
        device=sequence_lengths_plus_one.device,
    )
    block_ids = block_map[rows, page_indices].to(torch.int64)
    valid = (
        (positions >= 0)
        & (block_ids > 0)
        & (block_ids < cache_block_count)
    )
    safe_block_ids = block_ids.clamp(min=0, max=cache_block_count - 1)
    return safe_block_ids, valid


def load_cuda_graph_decode_tensors(
    ssm_cache: torch.Tensor,
    conv_cache: torch.Tensor,
    sequence_lengths_plus_one: torch.Tensor | None,
    block_map: torch.Tensor | None,
    page_size: int,
    projection_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    block_ids, valid = cuda_graph_decode_block_ids(
        sequence_lengths_plus_one,
        block_map,
        page_size,
        ssm_cache.shape[0],
        position_offset=-2,
    )
    recurrent = ssm_cache.index_select(0, block_ids).transpose(-1, -2)
    packed_conv = conv_cache.index_select(0, block_ids).transpose(1, 2)
    q_state, k_state, v_state = torch.split(
        packed_conv, projection_size, dim=1
    )
    recurrent = torch.where(
        valid.view(-1, 1, 1, 1),
        recurrent,
        torch.zeros_like(recurrent),
    )
    conv_valid = valid.view(-1, 1, 1)
    return (
        torch.where(conv_valid, q_state, torch.zeros_like(q_state)),
        torch.where(conv_valid, k_state, torch.zeros_like(k_state)),
        torch.where(conv_valid, v_state, torch.zeros_like(v_state)),
        recurrent,
    )


def store_cuda_graph_decode_state(
    state: Any,
    ssm_cache: torch.Tensor,
    conv_cache: torch.Tensor,
    sequence_lengths_plus_one: torch.Tensor | None,
    block_map: torch.Tensor | None,
    page_size: int,
) -> None:
    # sequence_lengths_plus_one is past_length + one decode token. The final
    # state belongs to token position past_length.
    block_ids, valid = cuda_graph_decode_block_ids(
        sequence_lengths_plus_one,
        block_map,
        page_size,
        ssm_cache.shape[0],
        position_offset=-1,
    )
    recurrent = (
        state.recurrent_state.transpose(-1, -2)
        .to(dtype=ssm_cache.dtype)
        .contiguous()
    )
    packed_conv = (
        torch.cat(
            (
                state.q_conv_state,
                state.k_conv_state,
                state.v_conv_state,
            ),
            dim=1,
        )
        .transpose(1, 2)
        .to(dtype=conv_cache.dtype)
        .contiguous()
    )

    # Block 0 is RTP's invalid/synthetic-stream sentinel. Invalid rows still
    # participate in the fixed-shape scatter, so feed back the current cache
    # value to keep that sentinel unchanged across replay.
    current_recurrent = ssm_cache.index_select(0, block_ids)
    current_packed_conv = conv_cache.index_select(0, block_ids)
    recurrent = torch.where(
        valid.view(-1, 1, 1, 1),
        recurrent,
        current_recurrent,
    )
    packed_conv = torch.where(
        valid.view(-1, 1, 1),
        packed_conv,
        current_packed_conv,
    )
    ssm_cache.index_copy_(0, block_ids, recurrent)
    conv_cache.index_copy_(0, block_ids, packed_conv)
