"""Numerical, selection, and timing helpers for prefill-score regression tests."""

import contextlib
import math
import statistics
import struct

import torch

_HEADS = 4
_HEAD_DIM = 128
_FP64_CHUNK = 1024


@contextlib.contextmanager
def tf32_disabled():
    """Explicit TF32-off for every FP32 baseline math gate, with restore."""
    matmul_before = torch.backends.cuda.matmul.allow_tf32
    cudnn_before = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_before
        torch.backends.cudnn.allow_tf32 = cudnn_before


def make_inputs(length, blocks=None, strided=False, seed=73, batch=2):
    if blocks is None:
        blocks = length // 4  # actual model uses N = floor(S/4)
    torch.manual_seed(seed)
    if strided:
        q = torch.randn(
            batch, length, 2 * _HEADS, _HEAD_DIM, device="cuda", dtype=torch.bfloat16
        )[:, :, ::2]
        k = torch.randn(
            batch, blocks * 2, _HEAD_DIM, device="cuda", dtype=torch.bfloat16
        )[:, ::2]
    else:
        q = torch.randn(
            batch, length, _HEADS, _HEAD_DIM, device="cuda", dtype=torch.bfloat16
        )
        k = torch.randn(batch, blocks, _HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    return q, k


def reference_fp64_chunked(q, block_keys, chunk=_FP64_CHUNK):
    """Independent FP64 reference, chunked over S only to bound peak memory."""
    b, s, h, d = q.shape
    n = block_keys.shape[1]
    out = torch.empty((b, s, n), device=q.device, dtype=torch.float64)
    kt = block_keys.double().transpose(-1, -2).unsqueeze(1)
    for start in range(0, s, chunk):
        end = min(start + chunk, s)
        part = torch.matmul(q[:, start:end].double(), kt).transpose(-1, -2)
        out[:, start:end] = torch.relu(part).sum(dim=-1) / math.sqrt(d)
    return out


def run_fp32_fallback(q, block_keys):
    """Production fallback math from indexer.py: FP32 matmul -> relu -> head sum."""
    scores = torch.matmul(q.float(), block_keys.float().transpose(-1, -2).unsqueeze(1))
    return torch.relu(scores.transpose(-1, -2)).sum(dim=-1) / math.sqrt(q.shape[-1])


def tail256_gate(a_indices, b_indices, a_scores, b_scores, visible_counts, n_blocks):
    """Compare selected blocks using the original sorted top-512 ranks.

    Permit at most three replacements per row, restricted to ranks 257..512
    on both sides. Rows with fewer than 512 visible blocks require exact sets.
    Invalid, duplicate, nonfinite, or missing selected entries are rejected.
    Scores must be values from the same topk call as the supplied indices.
    """
    if not (a_indices.shape == b_indices.shape == a_scores.shape == b_scores.shape):
        raise ValueError("ranked indices/scores shapes must agree")
    if a_indices.ndim != 2 or a_indices.shape[1] != min(512, n_blocks):
        raise ValueError("rank width must be min(512,N)")
    rows, width = a_indices.shape
    if visible_counts.shape != (rows,):
        raise ValueError("visible_counts must be metadata per row")
    meta_bad = (visible_counts < 0) | (visible_counts > n_blocks)
    expected = visible_counts.clamp(0, min(512, n_blocks))
    rank = torch.arange(1, width + 1, device=a_indices.device)[None, :]

    def validate(indices, scores):
        finite = torch.isfinite(scores)
        index_ok = (
            (indices >= 0) & (indices < visible_counts[:, None]) & (indices < n_blocks)
        )
        valid = finite & index_ok
        invalid = (
            (finite & ~index_ok) | torch.isnan(scores) | torch.isposinf(scores)
        ).any(-1)
        ids = torch.where(valid, indices, n_blocks).sort(-1).values
        duplicate = ((ids[:, 1:] == ids[:, :-1]) & (ids[:, 1:] != n_blocks)).any(-1)
        wrong_count = valid.sum(-1) != expected
        unsorted = (scores[:, :-1] < scores[:, 1:]).any(-1)
        return valid, invalid, duplicate, wrong_count, unsorted

    av, ai, ad, ac, ao = validate(a_indices, a_scores)
    bv, bi, bd, bc, bo = validate(b_indices, b_scores)

    def membership(src, src_valid, dst, dst_valid):
        if width == 0:
            return src_valid.clone()
        sorted_ids = torch.where(dst_valid, dst, n_blocks).sort(-1).values.contiguous()
        pos = torch.searchsorted(sorted_ids, src.contiguous()).clamp(max=width - 1)
        return src_valid & (sorted_ids.gather(-1, pos) == src)

    removed = av & ~membership(a_indices, av, b_indices, bv)
    added = bv & ~membership(b_indices, bv, a_indices, av)
    nr, na = removed.sum(-1), added.sum(-1)
    early_a, early_b = (removed & (rank <= 256)).any(-1), (added & (rank <= 256)).any(
        -1
    )
    strict_diff = (nr != 0) | (na != 0)
    short_bad = (visible_counts < 512) & strict_diff
    invalid = ai | bi | meta_bad
    duplicate, count_bad, unsorted = ad | bd, ac | bc, ao | bo
    violation = (
        invalid
        | duplicate
        | count_bad
        | unsorted
        | (nr != na)
        | (nr > 3)
        | early_a
        | early_b
        | short_bad
    )
    ah, bh = av & (rank <= 256), bv & (rank <= 256)
    head_common = membership(a_indices, ah, b_indices, bh).sum(-1)

    def count(mask):
        return int(mask.sum().item())

    def min_changed(mask):
        return (
            int(rank.expand_as(mask)[mask].min().item()) if bool(mask.any()) else None
        )

    result = {
        "contract": "tail256-v1",
        "rows": rows,
        "all_rows_pass": not bool(violation.any()),
        "violation_rows": count(violation),
        "strict_mismatch_rows": count(strict_diff),
        "max_replacements": int(torch.maximum(nr, na).max().item()) if rows else 0,
        "max_symmetric_difference": int((nr + na).max().item()) if rows else 0,
        "min_changed_rank_a": min_changed(removed),
        "min_changed_rank_b": min_changed(added),
        "head256_common_min": int(head_common.min().item()) if rows else 0,
        "head256_common_min_visible_ge512": (
            int(head_common[visible_counts >= 512].min().item())
            if bool((visible_counts >= 512).any())
            else None
        ),
        "strict_metric_scope": "exact equality of valid selected sets; historical strict test remains separate",
        "invalid_rows": count(invalid),
        "duplicate_rows": count(duplicate),
        "count_violation_rows": count(count_bad),
        "unsorted_rows": count(unsorted),
        "short_exact_violation_rows": count(short_bad),
        "early_a_rows": count(early_a),
        "early_b_rows": count(early_b),
        "details": [],
    }
    # Prioritize genuine violations, then accepted differences, at most 8 rows.
    selected = violation.nonzero().flatten()[:8].tolist()
    selected += [
        i for i in strict_diff.nonzero().flatten()[:8].tolist() if i not in selected
    ]
    for row in selected[:8]:

        def changes(indices, scores, mask):
            positions = mask[row].nonzero().flatten()[:8].tolist()
            return [
                {
                    "block": int(indices[row, p]),
                    "rank_1based": p + 1,
                    "score": float(scores[row, p]),
                    "float32_bits": "0x%08x"
                    % struct.unpack("<I", struct.pack("<f", float(scores[row, p])))[0],
                }
                for p in positions
            ]

        result["details"].append(
            {
                "row": row,
                "visible_count": int(visible_counts[row]),
                "expected_selected": int(expected[row]),
                "removed_count": int(nr[row]),
                "added_count": int(na[row]),
                "violation": bool(violation[row]),
                "removed": changes(a_indices, a_scores, removed),
                "added": changes(b_indices, b_scores, added),
                "head256_common": int(head_common[row]),
            }
        )
    return result


def _timed_step(fn, q, k, samples):
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(samples)
    ]
    for start, end in events:
        start.record()
        fn(q, k)
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in events]


def _profile_arm(fn, q, k, calls=3):
    try:
        from torch.autograd import DeviceType
        from torch.profiler import ProfilerActivity, profile
    except Exception as exc:
        return {"status": "NOT_RUN", "reason": f"profiler import failed: {exc}"}
    try:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(calls):
                fn(q, k)
            torch.cuda.synchronize()
    except Exception as exc:
        return {"status": "NOT_RUN", "reason": f"profiler failed: {exc}"}
    # Only device-side events (kernels/memcpys); CPU-op aggregates would
    # double-count the same device time through their child kernels.
    rows = []
    for event in prof.key_averages():
        if getattr(event, "device_type", None) != DeviceType.CUDA:
            continue
        total = getattr(event, "device_time_total", None)
        if total is None:
            total = getattr(event, "self_device_time_total", 0) or 0
        if total and total > 0:
            rows.append(
                {"kernel": event.key[:200], "count": event.count, "total_us": total}
            )
    if not rows:
        # A profiler pass that captured no device-side events is a failed
        # measurement, not a zero-duration result.
        return {
            "status": "NOT_RUN",
            "reason": "no device-side events captured by profiler",
        }
    rows.sort(key=lambda row: -row["total_us"])
    return {
        "status": "OK",
        "profiled_calls": calls,
        "device_kernel_time_sum_us": sum(row["total_us"] for row in rows),
        "device_kernel_time_per_call_us": sum(row["total_us"] for row in rows) / calls,
        "kernels": rows[:8],
        "note": "separate untimed profiler pass; excluded from timed samples",
    }


def _percentile_nearest_rank(ordered, quantile):
    if not ordered:
        return None
    rank = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[min(rank, len(ordered) - 1)]


def _summarize(samples, arm):
    arm_samples = [sample for sample in samples if sample["arm"] == arm]
    values = [sample["elapsed_ms"] for sample in arm_samples]
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    per_round = {}
    for sample in arm_samples:
        per_round.setdefault(sample["round"], []).append(sample["elapsed_ms"])
    return {
        "count": len(values),
        "min_ms": ordered[0],
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(ordered),
        "p90_ms": _percentile_nearest_rank(ordered, 0.9),
        "per_round_median_ms": {
            str(round_index): statistics.median(round_values)
            for round_index, round_values in sorted(per_round.items())
        },
    }
