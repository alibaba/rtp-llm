# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton

from rtp_llm.models_py.modules.dsv4.forward_metadata import cached_kda_chunk_indices
from rtp_llm.models_py.triton_kernels.fla.utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    # State-cache writes also consume this map after graph replay. Check the
    # forward-scoped host geometry before the legacy tensor-identity cache,
    # which cannot observe an input tensor refreshed in-place between steps.
    indices = cached_kda_chunk_indices(cu_seqlens, chunk_size)
    if indices is not None:
        return indices
    return _prepare_chunk_indices(cu_seqlens, chunk_size)


@tensor_cache
def _prepare_chunk_indices(
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
