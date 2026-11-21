# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton

from rtp_llm.models_py.triton_kernels.fla.utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    host_lengths = getattr(cu_seqlens, "_host_lengths", None)
    if host_lengths is not None and len(host_lengths) > 0:
        counts = [triton.cdiv(n, chunk_size) for n in host_lengths]
        indices = torch.cat([torch.arange(n, dtype=cu_seqlens.dtype) for n in counts])
        host_out = torch.stack(
            [(indices.eq(0).cumsum(0) - 1).to(cu_seqlens.dtype), indices], 1
        )
        if cu_seqlens.is_cuda and host_out.numel() > 0:
            return host_out.pin_memory().to(
                device=cu_seqlens.device, non_blocking=True
            )
        return host_out.to(cu_seqlens)
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
    host_lengths = getattr(cu_seqlens, "_host_lengths", None)
    if host_lengths is not None:
        offsets = [0]
        for n in host_lengths:
            offsets.append(offsets[-1] + triton.cdiv(n, chunk_size))
        host_out = torch.tensor(offsets, dtype=cu_seqlens.dtype)
        if cu_seqlens.is_cuda and host_out.numel() > 0:
            return host_out.pin_memory().to(
                device=cu_seqlens.device, non_blocking=True
            )
        return host_out.to(cu_seqlens)
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)
