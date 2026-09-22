"""Reusable numerical primitives; callers explicitly choose window semantics."""

import math


def select_window(rows, lower, upper, *, time, include_end=False):
    """Request cohorts use [lower, upper); counter endpoints may include upper."""
    return [
        row
        for row in rows
        if lower <= time(row)
        and (time(row) <= upper if include_end else time(row) < upper)
    ]


def counter_delta(values):
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values
    ):
        return None, "MISSING_COUNTER"
    if any(b < a for a, b in zip(values, values[1:])):
        return None, "COUNTER_RESET"
    if len(values) < 2:
        return None, "MISSING_COUNTER"
    return values[-1] - values[0], "AVAILABLE"


def percentile_nr(values, p, nd=1):
    """Pooled nearest rank, retaining the existing zero-sample caller contract."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
    return round(float(ordered[k]), nd)
