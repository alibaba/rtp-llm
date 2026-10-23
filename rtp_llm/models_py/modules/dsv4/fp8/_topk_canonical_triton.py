"""Canonical sparse-index sets from an exact TopK score threshold."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["WIDTH", "SCORE_STRIDE", "OUT_STRIDE", "K"])
def _canonical_topk_kernel(
    scores,
    output,
    starts,
    ends,
    WIDTH,
    SCORE_STRIDE,
    OUT_STRIDE,
    K,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    FULL_WIDTH: tl.constexpr,
    FULL_K: tl.constexpr,
    SCORE_ALIGNED: tl.constexpr,
    OUT_ALIGNED: tl.constexpr,
    SCORE_CONTIGUOUS: tl.constexpr,
    OUT_CONTIGUOUS: tl.constexpr,
):
    if FULL_WIDTH:
        WIDTH = BLOCK_N
        if SCORE_CONTIGUOUS:
            SCORE_STRIDE = BLOCK_N
    if FULL_K:
        K = BLOCK_K
        if OUT_CONTIGUOUS:
            OUT_STRIDE = BLOCK_K
    if SCORE_ALIGNED:
        SCORE_STRIDE = tl.multiple_of(SCORE_STRIDE, 16)
    if OUT_ALIGNED:
        OUT_STRIDE = tl.multiple_of(OUT_STRIDE, 16)
    row = tl.program_id(0).to(tl.int64)
    start = tl.load(starts + row) if HAS_STARTS else 0
    end = tl.load(ends + row)
    k = tl.arange(0, BLOCK_K)
    in_k = FULL_K | (k < K)
    indices = tl.load(output + row * OUT_STRIDE + k, in_k, other=-1)
    selected_col = start + indices.to(tl.int64)
    selected_valid = (
        in_k
        & (indices >= 0)
        & (selected_col >= 0)
        & (selected_col < end)
        & (selected_col < WIDTH)
    )
    selected_scores = tl.load(
        scores + row * SCORE_STRIDE + selected_col,
        selected_valid,
        other=float("inf"),
    )
    cutoff = tl.min(selected_scores, 0)
    col = tl.arange(0, BLOCK_N)
    valid = (col >= start) & (col < end) & (FULL_WIDTH | (col < WIDTH))
    values = tl.load(scores + row * SCORE_STRIDE + col, valid, other=-float("inf"))
    above = valid & (values > cutoff)
    tied = valid & (values == cutoff)
    last = tl.load(output + row * OUT_STRIDE + K - 1).to(tl.int64)
    if (last >= 0) & (start >= 0) & (end <= WIDTH) & (last + start < end):
        # Exact TopK selections have a valid prefix followed by -1 padding.
        selected_count = K
    else:
        selected_count = tl.sum(selected_valid.to(tl.int32), 0)
    remaining = selected_count - tl.sum(above.to(tl.int32), 0)
    # WIDTH <= 32768 keeps the two 16-bit prefix counts independent.
    packed = tl.cumsum((above.to(tl.int32) << 16) + tied.to(tl.int32))
    tie_rank = packed & 65535
    above_rank = (packed.to(tl.uint32) >> 16).to(tl.int32)
    chosen = above | (tied & (tie_rank <= remaining))
    output_rank = above_rank + tl.minimum(tie_rank, remaining) - 1
    # Padding and selected entries occupy disjoint slots; no store barrier
    # is needed even though the native output is repaired in place.
    tl.store(output + row * OUT_STRIDE + k, -1, (k < K) & (k >= selected_count))
    tl.store(
        output + row * OUT_STRIDE + output_rank,
        col - start,
        chosen & (output_rank < K),
    )


def canonicalize_topk_if_supported(scores, output, ends, starts=None) -> bool:
    """Keep highest scores, break cutoff ties by lowest index, then sort indices.

    The input output tensor must contain an exact TopK selection over each
    [start, end) window with request-local indices and -1 padding. Valid score
    entries must not be NaN. No score perturbation or tolerance is introduced.
    """
    if not (
        scores.is_cuda
        and scores.dtype == torch.float32
        and scores.ndim == 2
        and output.is_cuda
        and output.dtype == torch.int32
        and output.ndim == 2
        and output.device == scores.device
        and output.shape[0] == scores.shape[0]
        and scores.stride(1) == 1
        and output.stride(1) == 1
        and 0 < scores.shape[1] <= 32768
        and 0 < output.shape[1] <= 2048
        and ends.device == scores.device
        and ends.is_contiguous()
        and ends.numel() == scores.shape[0]
        and ends.dtype in (torch.int32, torch.int64)
    ):
        return False
    if starts is not None and not (
        starts.device == scores.device
        and starts.is_contiguous()
        and starts.numel() == scores.shape[0]
        and starts.dtype in (torch.int32, torch.int64)
    ):
        return False
    if scores.shape[0] == 0:
        return True
    _canonical_topk_kernel[(scores.shape[0],)](
        scores,
        output,
        starts if starts is not None else ends,
        ends,
        WIDTH=scores.shape[1],
        SCORE_STRIDE=scores.stride(0),
        OUT_STRIDE=output.stride(0),
        K=output.shape[1],
        BLOCK_K=triton.next_power_of_2(output.shape[1]),
        BLOCK_N=triton.next_power_of_2(scores.shape[1]),
        HAS_STARTS=starts is not None,
        FULL_WIDTH=scores.shape[1] == triton.next_power_of_2(scores.shape[1]),
        FULL_K=output.shape[1] == triton.next_power_of_2(output.shape[1]),
        SCORE_ALIGNED=scores.stride(0) % 16 == 0,
        OUT_ALIGNED=output.stride(0) % 16 == 0,
        SCORE_CONTIGUOUS=scores.stride(0) == scores.shape[1],
        OUT_CONTIGUOUS=output.stride(0) == output.shape[1],
        num_warps=16 if scores.shape[1] > 8192 else 8,
    )
    return True
