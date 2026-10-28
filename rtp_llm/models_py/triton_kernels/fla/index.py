# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton

from rtp_llm.models_py.triton_kernels.fla.utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
) -> torch.LongTensor:
    # Normalize omitted CPU mirrors so legacy two-argument callers share the
    # same cached indices as the recurrent wrappers' explicit None argument.
    return _prepare_chunk_indices(cu_seqlens, chunk_size, cu_seqlens_cpu)


@tensor_cache
def _prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: Optional[torch.Tensor],
) -> torch.LongTensor:
    lengths = prepare_lens(cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens)
    indices = torch.cat(
        [torch.arange(n) for n in triton.cdiv(lengths, chunk_size).tolist()]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
