# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched V4.1 decode selection and fused candidate metadata.

Candidate pooling/flags follow vLLM's attention/dsa/candidate_blocks.py.
Keep PyTorch's candidate selection algorithm: tied block scores can select
different tokens downstream. Only the unused ordering of the selected blocks
is removed. Token selection uses RTP's existing exact radix-select kernel.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4.fp8.indexer import _run_topk_v3, _topk_v3_enabled


def is_supported(logits: torch.Tensor, lengths: torch.Tensor, topk: int) -> bool:
    return (
        os.environ.get("DSV41_FUSED_DECODE_TOPK", "1") != "0"
        and _topk_v3_enabled()
        and logits.is_cuda
        and logits.dtype == torch.float32
        and logits.ndim == 2
        and logits.stride(1) == 1
        and logits.stride(0) >= logits.shape[1]
        and logits.shape[0] > 0
        and topk in (512, 1024, 2048)
        and logits.shape[1] >= topk
        and lengths.device == logits.device
        and lengths.dtype == torch.int32
        and lengths.is_contiguous()
        and lengths.numel() == logits.shape[0]
    )


@triton.jit
def _max_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _candidate_pool_kernel(
    logits,
    lengths,
    scores,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NBLOCKS: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.program_id(1) * TILE + tl.arange(0, TILE)
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    columns = blocks[:, None] * BLOCK_SIZE + offsets[None, :]
    visible = tl.load(lengths + row)
    values = tl.load(
        logits + row * STRIDE + columns,
        (blocks[:, None] < NBLOCKS)
        & (offsets[None, :] < BLOCK_SIZE)
        & (columns < WIDTH)
        & (columns < visible),
        other=-float("inf"),
    )
    pooled = tl.reduce(values, 1, _max_with_nan)
    pooled = tl.where(
        (visible > 0) & (blocks == (visible - 1) // BLOCK_SIZE),
        float("inf"),
        pooled,
    )
    tl.store(scores + row * NBLOCKS + blocks, pooled, blocks < NBLOCKS)


@triton.jit
def _candidate_store_flags_kernel(
    values,
    indices,
    candidates,
    flags,
    K: tl.constexpr,
    NBLOCKS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, 1024)
    for tile in range(tl.cdiv(NBLOCKS, 1024)):
        blocks = tile * 1024 + offsets
        tl.store(flags + row * NBLOCKS + blocks, 0, blocks < NBLOCKS)
    columns = tl.arange(0, triton.next_power_of_2(K))
    score = tl.load(values + row * K + columns, columns < K, other=-float("inf"))
    index = tl.load(indices + row * K + columns, columns < K, other=-1)
    valid = (columns < K) & (score > -float("inf"))
    # +inf pins the newest block; NaN and -inf remain invalid, as in torch.
    tl.store(candidates + row * K + columns, tl.where(valid, index, -1), columns < K)
    tl.debug_barrier()
    tl.store(flags + row * NBLOCKS + index, 1, valid)


@triton.jit
def _candidate_mask_kernel(
    logits,
    flags,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    NBLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * TILE + tl.arange(0, TILE)
    keep = tl.load(
        flags + row * NBLOCKS + columns // BLOCK_SIZE,
        columns < WIDTH,
        other=0,
    )
    tl.store(
        logits + row * STRIDE + columns, -float("inf"), (columns < WIDTH) & (keep == 0)
    )


@triton.jit
def _finite_topk_kernel(
    logits,
    lengths,
    output,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * TILE + tl.arange(0, TILE)
    length = tl.load(lengths + row)
    index = tl.load(output + row * K + columns, columns < K, other=-1)
    value = tl.load(
        logits + row * STRIDE + index,
        (columns < K) & (index >= 0) & (index < WIDTH),
        other=-float("inf"),
    )
    # topk_v3 already emits the required ascending short-row result. Long
    # rows can contain candidate-mask holes; lengths alone do not filter them.
    keep = (length <= K) | (tl.abs(value) < float("inf"))
    tl.store(output + row * K + columns, tl.where(keep, index, -1), columns < K)


def select_candidates(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    block_size: int,
    topk_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = logits.shape
    nblocks = triton.cdiv(width, block_size)
    count = min(topk_blocks, nblocks)
    scores = torch.empty((rows, nblocks), dtype=logits.dtype, device=logits.device)
    _candidate_pool_kernel[(rows, triton.cdiv(nblocks, 128))](
        logits, lengths, scores, width, logits.stride(0), block_size, nblocks, 128
    )
    values, indices = scores.topk(count, dim=-1, sorted=False)
    candidates = torch.empty((rows, count), dtype=torch.int32, device=logits.device)
    flags = torch.empty((rows, nblocks), dtype=torch.uint8, device=logits.device)
    _candidate_store_flags_kernel[(rows,)](
        values, indices, candidates, flags, count, nblocks
    )
    return candidates, flags


def mask_candidates(logits: torch.Tensor, flags: torch.Tensor, block_size: int) -> None:
    rows, width = logits.shape
    _candidate_mask_kernel[(rows, triton.cdiv(width, 1024))](
        logits, flags, width, logits.stride(0), flags.shape[1], block_size, 1024
    )


def select_tokens(
    logits: torch.Tensor, lengths: torch.Tensor, topk: int
) -> torch.Tensor:
    rows, width = logits.shape
    output = torch.empty((rows, topk), dtype=torch.int32, device=logits.device)
    if not _run_topk_v3(logits, lengths, output, topk, width):
        raise RuntimeError("V4.1 decode TopK support changed within a forward")
    _finite_topk_kernel[(rows, triton.cdiv(topk, 256))](
        logits, lengths, output, width, logits.stride(0), topk, 256
    )
    return output
