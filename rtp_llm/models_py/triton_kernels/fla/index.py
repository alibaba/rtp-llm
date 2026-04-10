# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/index.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton

from rtp_llm.models_py.triton_kernels.fla.utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def _cdiv_cpu(lens: torch.LongTensor, chunk_size: int):
    """Ceiling division on CPU to avoid GPU kernel launches for tiny tensors."""
    lens_cpu = lens.tolist()
    return [triton.cdiv(l, chunk_size) for l in lens_cpu]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    # Build indices entirely on CPU, then transfer to GPU once.
    # Each entry is (batch_idx, chunk_idx_within_seq).
    chunks_per_seq = _cdiv_cpu(prepare_lens(cu_seqlens), chunk_size)
    batch_ids = []
    chunk_ids = []
    for seq_idx, n_chunks in enumerate(chunks_per_seq):
        batch_ids.extend([seq_idx] * n_chunks)
        chunk_ids.extend(range(n_chunks))
    result = torch.tensor(
        list(zip(batch_ids, chunk_ids)),
        dtype=cu_seqlens.dtype,
        device=cu_seqlens.device,
    )
    return result


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    chunks = _cdiv_cpu(prepare_lens(cu_seqlens), chunk_size)
    # Cumsum on CPU (only batch+1 elements), then transfer to GPU once
    import itertools

    offsets = list(itertools.accumulate([0] + chunks))
    return torch.tensor(offsets, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
