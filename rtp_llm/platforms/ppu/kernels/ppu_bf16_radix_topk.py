"""Wide BF16 TopK: exact two-byte radix threshold, index-stable ties.

Histograms are partitioned over columns; no score sorting or fixed tie buffer.
Output indices are logical columns. Callers may canonicalize their order.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _keys(x):
    bits = x.to(tl.uint16, bitcast=True).to(tl.int32)
    bits = tl.where(x == 0, 0, bits)  # Treat signed zeros as the same score.
    return tl.where((bits & 32768) != 0, (~bits) & 65535, bits ^ 32768)


@triton.jit
def _histogram(S, Starts, Ends, Hist, Threshold, Stride,
               N, Parts, Fine: tl.constexpr,
               Block: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * Block + tl.arange(0, Block)
    valid = (col < N) & (col >= tl.load(Starts + row)) & (col < tl.load(Ends + row))
    score = tl.load(S + row.to(tl.int64) * Stride + col, valid, 0)
    key = _keys(score)
    if Fine:
        high = tl.load(Threshold + row * 3)
        valid = valid & ((key >> 8) == high)
        bins = key & 255
    else:
        bins = key >> 8
    counts = tl.histogram(bins, 256, mask=valid)
    tl.store(Hist + (row * Parts + part) * 256 + tl.arange(0, 256), counts)


@triton.jit
def _threshold(Hist, Starts, Ends, Threshold, Counts, Parts,
               PaddedParts: tl.constexpr, K: tl.constexpr, Fine: tl.constexpr):
    row = tl.program_id(0)
    part = tl.arange(0, PaddedParts)
    bins = tl.arange(0, 256)
    counts = tl.load(Hist + (row * Parts + part[:, None]) * 256 + bins[None, :],
                     part[:, None] < Parts, 0)
    partition_counts = counts
    counts = tl.sum(counts, axis=0)
    cumulative = tl.cumsum(counts, reverse=True)
    target = tl.minimum(K, tl.maximum(tl.load(Ends + row) - tl.load(Starts + row), 0))
    if Fine:
        target -= tl.load(Threshold + row * 3 + 1)
    selected = tl.max(tl.where(cumulative >= target, bins, 0), axis=0)
    partition_above = tl.sum(tl.where(bins[None, :] > selected, partition_counts, 0), axis=1)
    if Fine:
        coarse_above = tl.load(Counts + (row * Parts + part) * 2, part < Parts, 0)
        equal = tl.sum(tl.where(bins[None, :] == selected, partition_counts, 0), axis=1)
        tl.store(Counts + (row * Parts + part) * 2, coarse_above + partition_above, part < Parts)
        tl.store(Counts + (row * Parts + part) * 2 + 1, equal, part < Parts)
        tl.store(Threshold + row * 3 + 2, (tl.load(Threshold + row * 3) << 8) | selected)
    else:
        tl.store(Counts + (row * Parts + part) * 2, partition_above, part < Parts)
        above = tl.sum(tl.where(bins > selected, counts, 0), axis=0)
        tl.store(Threshold + row * 3, selected)
        tl.store(Threshold + row * 3 + 1, above)


@triton.jit
def _scatter(S, Starts, Ends, Threshold, Counts, Out, Stride,
             OutStride, N, Parts,
             PaddedParts: tl.constexpr, K: tl.constexpr, Block: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    parts = tl.arange(0, PaddedParts)
    gt_counts = tl.load(Counts + (row * Parts + parts) * 2, parts < Parts, 0)
    eq_counts = tl.load(Counts + (row * Parts + parts) * 2 + 1, parts < Parts, 0)
    total_gt = tl.sum(gt_counts)
    prefix_gt = tl.sum(tl.where(parts < part, gt_counts, 0))
    prefix_eq = tl.sum(tl.where(parts < part, eq_counts, 0))
    col = part * Block + tl.arange(0, Block)
    valid = (col < N) & (col >= tl.load(Starts + row)) & (col < tl.load(Ends + row))
    key = _keys(tl.load(S + row.to(tl.int64) * Stride + col, valid, 0))
    threshold = tl.load(Threshold + row * 3 + 2)
    gt = valid & (key > threshold)
    eq = valid & (key == threshold)
    gt_rank = prefix_gt + tl.cumsum(gt.to(tl.int32)) - 1
    eq_rank = total_gt + prefix_eq + tl.cumsum(eq.to(tl.int32)) - 1
    rank = tl.where(gt, gt_rank, eq_rank)
    tl.store(Out + row.to(tl.int64) * OutStride + rank, col, (gt | eq) & (rank < K))


@triton.jit
def _max_keys(S, Starts, Ends, Maxima, N, Stride, Parts, Block: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * Block + tl.arange(0, Block)
    valid = (col < N) & (col >= tl.load(Starts + row)) & (col < tl.load(Ends + row))
    key = _keys(tl.load(S + row.to(tl.int64) * Stride + col, valid, 0))
    tl.store(Maxima + row * Parts + part, tl.max(tl.where(valid, key, 0)))


@triton.jit
def _window_histogram(S, Starts, Ends, Maxima, Hist, Threshold, N, Stride, Parts,
                      PaddedParts: tl.constexpr, Block: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    ps = tl.arange(0, PaddedParts)
    base = tl.max(tl.load(Maxima + row * Parts + ps, ps < Parts, 0)) - 255
    if part == 0:
        tl.store(Threshold + row * 3, base)
    col = part * Block + tl.arange(0, Block)
    valid = (col < N) & (col >= tl.load(Starts + row)) & (col < tl.load(Ends + row))
    key = _keys(tl.load(S + row.to(tl.int64) * Stride + col, valid, 0))
    histogram = tl.histogram((key - base) & 255, 256, mask=valid & (key >= base))
    tl.store(Hist + (row * Parts + part) * 256 + tl.arange(0, 256), histogram)


@triton.jit
def _window_threshold(Hist, Starts, Ends, Threshold, Counts, Done, Parts,
                      PaddedParts: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    parts = tl.arange(0, PaddedParts)
    bins = tl.arange(0, 256)
    partitions = tl.load(Hist + (row * Parts + parts[:, None]) * 256 + bins[None, :],
                         parts[:, None] < Parts, 0)
    totals = tl.sum(partitions, axis=0)
    target = tl.minimum(K, tl.maximum(tl.load(Ends + row) - tl.load(Starts + row), 0))
    done = tl.sum(totals) >= target
    tl.store(Done + row, done)
    if done:
        cumulative = tl.cumsum(totals, reverse=True)
        selected = tl.max(tl.where(cumulative >= target, bins, 0), axis=0)
        above = tl.sum(tl.where(bins[None, :] > selected, partitions, 0), axis=1)
        equal = tl.sum(tl.where(bins[None, :] == selected, partitions, 0), axis=1)
        tl.store(Counts + (row * Parts + parts) * 2, above, parts < Parts)
        tl.store(Counts + (row * Parts + parts) * 2 + 1, equal, parts < Parts)
        tl.store(Threshold + row * 3 + 2, tl.load(Threshold + row * 3) + selected)


@triton.jit
def _fallback_histogram(S, Starts, Ends, Hist, Threshold, Done, Stride,
                        N, Parts, Fine: tl.constexpr, Block: tl.constexpr):
    if tl.load(Done + tl.program_id(0)) == 0:
        _histogram(S, Starts, Ends, Hist, Threshold, Stride, N, Parts, Fine, Block)


@triton.jit
def _fallback_threshold(Hist, Starts, Ends, Threshold, Counts, Done, Parts,
                        PaddedParts: tl.constexpr, K: tl.constexpr, Fine: tl.constexpr):
    if tl.load(Done + tl.program_id(0)) == 0:
        _threshold(Hist, Starts, Ends, Threshold, Counts, Parts, PaddedParts, K, Fine)


@triton.jit
def _prepare_rows(Out, Done, Active, OutStride, K: tl.constexpr, Selective: tl.constexpr):
    row = tl.program_id(0)
    active = True
    if Selective:
        active = tl.load(Active + row) != 0
    tl.store(Done + row, tl.where(active, 0, 1))
    if active:
        tl.store(Out + row.to(tl.int64) * OutStride + tl.arange(0, K), -1)


@triton.jit
def _selective_max_keys(S, Starts, Ends, Maxima, N, Stride, Parts, Block: tl.constexpr, Active, Selective: tl.constexpr):
    active = True
    if Selective:
        active = tl.load(Active + tl.program_id(0)) != 0
    if active:
        _max_keys(S, Starts, Ends, Maxima, N, Stride, Parts, Block)


@triton.jit
def _selective_window_histogram(S, Starts, Ends, Maxima, Hist, Threshold, N, Stride, Parts, PaddedParts: tl.constexpr, Block: tl.constexpr, Active, Selective: tl.constexpr):
    active = True
    if Selective:
        active = tl.load(Active + tl.program_id(0)) != 0
    if active:
        _window_histogram(S, Starts, Ends, Maxima, Hist, Threshold, N, Stride, Parts, PaddedParts, Block)


@triton.jit
def _selective_window_threshold(Hist, Starts, Ends, Threshold, Counts, Done, Parts, PaddedParts: tl.constexpr, K: tl.constexpr, Active, Selective: tl.constexpr):
    active = True
    if Selective:
        active = tl.load(Active + tl.program_id(0)) != 0
    if active:
        _window_threshold(Hist, Starts, Ends, Threshold, Counts, Done, Parts, PaddedParts, K)


@triton.jit
def _selective_scatter(S, Starts, Ends, Threshold, Counts, Out, Stride, OutStride, N, Parts, PaddedParts: tl.constexpr, K: tl.constexpr, Block: tl.constexpr, Active, Selective: tl.constexpr):
    active = True
    if Selective:
        active = tl.load(Active + tl.program_id(0)) != 0
    if active:
        _scatter(S, Starts, Ends, Threshold, Counts, Out, Stride, OutStride, N, Parts, PaddedParts, K, Block)


def bf16_radix_topk(scores, starts, ends, out, *, rows_to_compute=None):
    if (scores.ndim != 2 or scores.dtype != torch.bfloat16 or not scores.is_cuda
            or scores.stride(1) != 1 or out.dtype != torch.int32
            or out.ndim != 2 or out.stride(1) != 1 or out.shape[0] != scores.shape[0]
            or any(t.device != scores.device for t in (starts, ends, out))
            or any(t.dtype != torch.int32 or t.shape != (scores.shape[0],)
                   or not t.is_contiguous() for t in (starts, ends))):
        raise ValueError("BF16 radix TopK requires row-contiguous device scores and int32 bounds/output")
    rows, width = scores.shape
    if width > 262144 or out.shape[1] not in (512, 1024):
        raise ValueError("BF16 radix TopK supports up to 262144 columns and K=512/1024")
    if rows_to_compute is not None and (
            rows_to_compute.device != scores.device or rows_to_compute.dtype != torch.int32
            or rows_to_compute.shape != (rows,) or not rows_to_compute.is_contiguous()):
        raise ValueError("row selection must be a contiguous device int32 vector")
    if not rows or not width:
        if rows_to_compute is None:
            out.fill_(-1)
        elif rows:
            out.masked_fill_(rows_to_compute[:, None] != 0, -1)
        return out
    block = 4096
    parts = triton.cdiv(width, block)
    padded = triton.next_power_of_2(parts)
    hist = torch.empty((rows, parts, 256), dtype=torch.int32, device=scores.device)
    thresholds = torch.empty((rows, 3), dtype=torch.int32, device=scores.device)
    counts = torch.empty((rows, parts, 2), dtype=torch.int32, device=scores.device)
    maxima = torch.empty((rows, parts), dtype=torch.int32, device=scores.device)
    done = torch.empty((rows,), dtype=torch.int32, device=scores.device)
    active = starts if rows_to_compute is None else rows_to_compute
    selective = rows_to_compute is not None
    _prepare_rows[(rows,)](out, done, active, out.stride(0), out.shape[1], selective)
    _selective_max_keys[(rows, parts)](scores, starts, ends, maxima, width, scores.stride(0), parts, block, active, selective, num_warps=8)
    _selective_window_histogram[(rows, parts)](scores, starts, ends, maxima, hist, thresholds, width,
                                    scores.stride(0), parts, padded, block, active, selective, num_warps=8)
    _selective_window_threshold[(rows,)](hist, starts, ends, thresholds, counts, done, parts, padded, out.shape[1], active, selective, num_warps=8)
    for fine in (False, True):
        _fallback_histogram[(rows, parts)](scores, starts, ends, hist, thresholds, done,
                                           scores.stride(0), width, parts, fine, block, num_warps=8)
        _fallback_threshold[(rows,)](hist, starts, ends, thresholds, counts, done, parts,
                                     padded, out.shape[1], fine, num_warps=8)
    _selective_scatter[(rows, parts)](scores, starts, ends, thresholds, counts, out,
                          scores.stride(0), out.stride(0), width, parts,
                          padded, out.shape[1], block, active, selective, num_warps=8)
    return out
