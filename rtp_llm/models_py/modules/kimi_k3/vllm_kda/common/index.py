# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import torch

from rtp_llm.models_py.modules.kimi_k3.vllm_kda.compat import gpu_sync_allowed
from rtp_llm.models_py.modules.kimi_k3.vllm_kda.compat import async_tensor_h2d
from rtp_llm.models_py.modules.kimi_k3.vllm_kda.compat import triton

from .utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    with gpu_sync_allowed():
        chunk_counts = triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
    # Empty requests have no chunks but retain their sequence ordinal.
    # Counting chunk-zero entries would renumber every later request.
    pairs = [
        (sequence, chunk)
        for sequence, count in enumerate(chunk_counts)
        for chunk in range(count)
    ]
    chunk_indices = torch.tensor(pairs, dtype=cu_seqlens.dtype).reshape(-1, 2)
    return async_tensor_h2d(
        chunk_indices, device=cu_seqlens.device, dtype=cu_seqlens.dtype
    )


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return torch.cat(
        [cu_seqlens.new_zeros(1), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
