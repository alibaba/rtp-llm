"""Wide BF16 TopK: exact two-byte radix threshold, index-stable ties.

Histograms are partitioned over columns; no score sorting or fixed tie buffer.
Output indices are logical columns. Callers may canonicalize their order.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _canonicalize_indices(Out, RowStride, K: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, K)
    ptr = Out + row.to(tl.int64) * RowStride + col
    indices = tl.load(ptr)
    ordered = tl.sort(tl.where(indices < 0, 2147483647, indices), descending=False)
    tl.store(ptr, tl.where(ordered == 2147483647, -1, ordered))


def canonicalize_topk_indices(out):
    """Keep the selected set, with logical indices ascending and padding last."""
    if out.dtype != torch.int32 or out.ndim != 2 or out.stride(1) != 1:
        raise ValueError("TopK indices require int32 rows with contiguous columns")
    if out.shape[1] not in (512, 1024):
        raise ValueError("TopK index ordering requires K=512/1024")
    if out.shape[0]:
        _canonicalize_indices[(out.shape[0],)](out, out.stride(0), out.shape[1])
    return out


@triton.jit
def _order_selected_scores(Scores, Out, ScoreStride, OutStride, N: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, K)
    ptr = Out + row.to(tl.int64) * OutStride + col
    index = tl.load(ptr)
    valid = (index >= 0) & (index < N)
    score = tl.load(Scores + row.to(tl.int64) * ScoreStride + index, valid, 0).to(tl.float32)
    score = tl.where(score == 0, 0.0, score)
    bits = score.to(tl.uint32, bitcast=True)
    key = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint64)
    packed = (key << 32) | (0xffffffff - index.to(tl.uint32)).to(tl.uint64)
    ordered = tl.sort(tl.where(valid, packed, 0), descending=True)
    result = (0xffffffff - (ordered & 0xffffffff).to(tl.uint32)).to(tl.int32)
    tl.store(ptr, tl.where(ordered == 0, -1, result))


def order_selected_by_score(scores, out):
    """Preserve the selected set, descending scores and index-stable ties."""
    if (scores.ndim != 2 or scores.dtype not in (torch.bfloat16, torch.float32)
            or scores.stride(1) != 1 or out.ndim != 2 or out.dtype != torch.int32
            or out.stride(1) != 1 or out.shape[0] != scores.shape[0]
            or out.shape[1] not in (512, 1024) or out.device != scores.device):
        raise ValueError("Score ordering requires matching BF16/FP32 score and int32 K=512/1024 rows")
    if out.shape[0]:
        _order_selected_scores[(out.shape[0],)](
            scores, out, scores.stride(0), out.stride(0), scores.shape[1], out.shape[1]
        )
    return out


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


@triton.jit
def _restore_short_order(S, L, O, WIDTH: tl.constexpr, SS: tl.constexpr, OS: tl.constexpr, K: tl.constexpr, STORE: tl.constexpr):
    row = tl.program_id(0)
    n = tl.load(L + row)
    if (n >= 0) & (n <= 32) & (n <= WIDTH):
        i = tl.arange(0, 32)
        value = tl.load(S + row * SS + i, i < n, other=0)
        bits = value.to(tl.uint16, bitcast=True).to(tl.int32)
        # Ascending BF16 radix key distinguishes -0 from +0. The network
        # below compares numbers, so signed zero ties there.
        key = tl.where((bits & 0x8000) != 0, bits ^ 0xffff, bits ^ 0x8000)
        floor = tl.min(tl.where(i < n, key, 0x10000), 0)
        first = (i < n) & (key > floor)
        last = (i < n) & (key == floor)
        dst = tl.where(first, tl.cumsum(first.to(tl.int32)) - 1,
                       tl.sum(first.to(tl.int32), 0) + tl.cumsum(last.to(tl.int32)) - 1)
        # Invert the stable gather permutation. Mask invalid ranks explicitly.
        ids = tl.min(tl.where((i[:, None] < n) & (dst[:, None] == i[None, :]),
                              i[:, None], 32), axis=0)
        vals = tl.gather(value.to(tl.float32), ids % 32, 0)
        for log_stage in tl.static_range(1, 6):
            for log_stride in tl.static_range(log_stage - 1, -1, -1):
                stride = 1 << log_stride
                other = i ^ stride
                peer_ids = tl.gather(ids, other, 0)
                peer_vals = tl.gather(vals, other, 0)
                lower = (i & stride) == 0
                a = tl.where(lower, ids, peer_ids)
                b = tl.where(lower, peer_ids, ids)
                av = tl.where(lower, vals, peer_vals)
                bv = tl.where(lower, peer_vals, vals)
                lane = ((i // (2 * stride)) * stride) + i % stride
                desc = (b == 32) | ((a != 32) & (av > bv))
                swap = desc == ((lane & (1 << (log_stage - 1))) != 0)
                ids = tl.where(swap, peer_ids, ids)
                vals = tl.where(swap, peer_vals, vals)
        col = tl.arange(0, STORE)
        tl.store(O + row * OS + col, -1, (col >= n) & (col < K))
        tl.store(O + row * OS + i, ids, i < n)


def restore_short_native_order(scores, lengths, out):
    """Finite BF16 valid scores, device int32 lengths, contiguous last dimension."""
    assert scores.ndim == out.ndim == 2 and lengths.shape == (scores.shape[0],)
    assert scores.dtype == torch.bfloat16 and lengths.dtype == out.dtype == torch.int32
    assert scores.device.type == 'cuda' and scores.device == lengths.device == out.device
    assert scores.stride(1) == out.stride(1) == lengths.stride(0) == 1
    assert scores.shape[0] == out.shape[0] and scores.shape[1] >= 32 and out.shape[1] >= 32
    if scores.shape[0]:
        _restore_short_order[(scores.shape[0],)](scores, lengths, out, scores.shape[1], scores.stride(0),
                                   out.stride(0), out.shape[1], triton.next_power_of_2(out.shape[1]))
    return out
