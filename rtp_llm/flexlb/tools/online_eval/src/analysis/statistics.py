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


def percentile_nr(values, p):
    """Exact nearest rank; no samples means unavailable, never a measured zero."""
    if type(p) not in (int, float) or not math.isfinite(p) or not 0 < p <= 1:
        raise ValueError("percentile must be finite and in (0, 1]")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("percentile requires finite numeric samples")
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(p * len(ordered)) - 1]
