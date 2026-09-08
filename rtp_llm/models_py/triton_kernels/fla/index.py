# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.fla.utils import tensor_cache


@dataclass
class FLAChunkMetadata:
    chunk_indices: torch.Tensor
    chunk_offsets: torch.Tensor
    token_capacity: int
    chunk_size: int


@triton.jit
def _prepare_chunk_offsets_kernel(
    cu_seqlens,
    chunk_offsets,
    N: tl.constexpr,
    BT: tl.constexpr,
):
    i_n = tl.program_id(0)
    offset = 0
    for n in range(N):
        bos = tl.load(cu_seqlens + n).to(tl.int32)
        eos = tl.load(cu_seqlens + n + 1).to(tl.int32)
        sequence_length = tl.maximum(eos - bos, 0)
        chunk_count = (sequence_length + BT - 1) // BT
        offset += tl.where(n < i_n, chunk_count, 0)
    tl.store(chunk_offsets + i_n, offset)


@triton.jit
def _prepare_chunk_indices_kernel(
    cu_seqlens,
    chunk_indices,
    N: tl.constexpr,
    BT: tl.constexpr,
):
    i_c = tl.program_id(0)
    selected_sequence = -1
    selected_chunk = 0
    chunk_start = 0
    for n in range(N):
        bos = tl.load(cu_seqlens + n).to(tl.int32)
        eos = tl.load(cu_seqlens + n + 1).to(tl.int32)
        sequence_length = tl.maximum(eos - bos, 0)
        chunk_count = (sequence_length + BT - 1) // BT
        is_selected = (i_c >= chunk_start) & (i_c < chunk_start + chunk_count)
        selected_sequence = tl.where(is_selected, n, selected_sequence)
        selected_chunk = tl.where(is_selected, i_c - chunk_start, selected_chunk)
        chunk_start += chunk_count
    tl.store(chunk_indices + i_c * 2, selected_sequence)
    tl.store(chunk_indices + i_c * 2 + 1, selected_chunk)


def prepare_chunk_graph_metadata(
    cu_seqlens: torch.Tensor,
    token_capacity: int,
    chunk_size: int,
    metadata: Optional[FLAChunkMetadata] = None,
) -> FLAChunkMetadata:
    sequence_count = cu_seqlens.numel() - 1
    chunk_capacity = triton.cdiv(token_capacity, chunk_size) + sequence_count
    if metadata is None:
        metadata = FLAChunkMetadata(
            chunk_indices=torch.empty(
                (chunk_capacity, 2), dtype=torch.int32, device=cu_seqlens.device
            ),
            chunk_offsets=torch.empty(
                sequence_count + 1, dtype=torch.int32, device=cu_seqlens.device
            ),
            token_capacity=token_capacity,
            chunk_size=chunk_size,
        )
    elif (
        metadata.token_capacity != token_capacity
        or metadata.chunk_size != chunk_size
        or metadata.chunk_offsets.numel() != sequence_count + 1
    ):
        raise ValueError("FLA graph chunk metadata capacity changed")

    _prepare_chunk_offsets_kernel[(sequence_count + 1,)](
        cu_seqlens,
        metadata.chunk_offsets,
        N=sequence_count,
        BT=chunk_size,
    )
    _prepare_chunk_indices_kernel[(chunk_capacity,)](
        cu_seqlens,
        metadata.chunk_indices,
        N=sequence_count,
        BT=chunk_size,
    )
    return metadata


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
