"""Fuse QSA selection bookkeeping while preserving native TopK tie ordering."""

import torch
import triton
import triton.language as tl


@triton.jit
def _mask_scores(
    scores,
    lengths,
    masked,
    N: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, BN)
    length = tl.load(lengths + row * LENGTH_STRIDE).to(tl.int64)
    complete = (length - tl.where(length < 0, 3, 0)) // 4
    values = tl.load(
        scores + row * ROW_STRIDE + column * COL_STRIDE, column < N, other=0
    )
    tl.store(
        masked + row * N + column,
        tl.where(column < complete, values, float("-inf")),
        column < N,
    )


@triton.jit
def _expand_tokens(
    values,
    blocks,
    lengths,
    output,
    K: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.arange(0, 4096)
    index = slot // 4
    value = tl.load(values + row * K + index, index < K, other=float("-inf"))
    block = tl.load(blocks + row * K + index, index < K, other=0)
    finite = (value == value) & (tl.abs(value) != float("inf"))
    token = tl.where((index < K) & finite, block * 4 + slot % 4, -1)
    length = tl.load(lengths + row * LENGTH_STRIDE).to(tl.int64)
    complete = (length - tl.where(length < 0, 3, 0)) // 4
    tail_offset = slot - 2048
    tail = tl.where(tail_offset < length - complete * 4, complete * 4 + tail_offset, -1)
    token = tl.where(slot >= 2048, tail, token)
    tl.store(output + row * 2051 + slot, token.to(tl.int32), slot < 2051)


def try_select_qsa_tokens(scores, lengths, *, compress_ratio, token_budget):
    if not (
        scores.is_cuda
        and torch.version.hip is None
        and scores.dtype == torch.float32
        and scores.ndim == 2
        and 0 < scores.shape[0] <= 8192
        and 0 < scores.shape[1] <= 8192
        and scores.stride(0) > 0
        and scores.stride(1) > 0
        and lengths.shape == (scores.shape[0],)
        and lengths.dtype == torch.int32
        and lengths.device == scores.device
        and lengths.stride(0) > 0
        and compress_ratio == 4
        and token_budget == 2048
        and torch.cuda.get_device_capability(scores.device)[0] >= 8
    ):
        return None
    rows, columns = scores.shape
    masked = torch.empty((rows, columns), device=scores.device, dtype=scores.dtype)
    _mask_scores[(rows,)](
        scores,
        lengths,
        masked,
        N=columns,
        ROW_STRIDE=scores.stride(0),
        COL_STRIDE=scores.stride(1),
        LENGTH_STRIDE=lengths.stride(0),
        BN=triton.next_power_of_2(columns),
        num_warps=4,
    )
    # Keep the original primitive and its sorting/tie behavior unchanged.
    values, blocks = torch.topk(masked, min(512, columns), dim=-1)
    output = torch.empty((rows, 2051), device=scores.device, dtype=torch.int32)
    _expand_tokens[(rows,)](
        values,
        blocks,
        lengths,
        output,
        K=values.shape[1],
        LENGTH_STRIDE=lengths.stride(0),
        num_warps=4,
    )
    return output
