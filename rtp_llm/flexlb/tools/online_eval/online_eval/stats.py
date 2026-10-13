"""Small dependency-free stats helpers for online evaluation."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Mapping, Sequence


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * p / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(ordered[lo])
    weight = rank - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def summarize_latencies(values: Sequence[float]) -> dict:
    return {
        "count": len(values),
        "p50": round(percentile(values, 50), 3),
        "p90": round(percentile(values, 90), 3),
        "p95": round(percentile(values, 95), 3),
        "p99": round(percentile(values, 99), 3),
        "max": round(max(values), 3) if values else 0.0,
        "mean": round(sum(values) / len(values), 3) if values else 0.0,
    }


def stddev(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    avg = sum(vals) / len(vals)
    return math.sqrt(sum((x - avg) ** 2 for x in vals) / len(vals))


def load_balance_summary(assignments: Iterable[str]) -> Mapping[str, object]:
    counts = Counter(x for x in assignments if x)
    if not counts:
        return {"counts": {}, "stddev": 0.0, "max_over_avg": 0.0}
    avg = sum(counts.values()) / len(counts)
    return {
        "counts": dict(counts),
        "stddev": round(stddev(counts.values()), 3),
        "max_over_avg": round(max(counts.values()) / avg, 3) if avg else 0.0,
    }
