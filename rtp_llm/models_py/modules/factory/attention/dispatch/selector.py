"""Pure-CPU selection policies for the attention backend dispatcher."""

from __future__ import annotations

import statistics
from typing import Callable, Dict, List, Optional

# select_stable thresholds
# Tunable at runtime via env: DYN_DECODE_THRESHOLD / DYN_DECODE_CLUSTER_MARGIN.
# Level 1: only switch if the winner's improvement meets this ratio.
STABLE_THRESHOLD = 0.05
# Level 2: survivors within this margin of each other -> tiebreak by registry order
STABLE_CLUSTER_MARGIN = 0.05


# kv grid: log2-equispaced; equal weighting is a log-uniform prior
_KV_GRID_FULL = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)


def kv_grid(max_seq_len: int) -> List[int]:
    """log-uniform kv grid, with the upper bound clipped to max_seq_len.

    Points exceeding max_seq_len are dropped; if max_seq_len falls between two
    buckets and is greater than the largest retained point, one max_seq_len point
    is appended to cover the production long tail. When max_seq_len is smaller
    than the smallest bucket, it degenerates to the single point [max_seq_len].
    """
    if max_seq_len < 1:
        raise ValueError(f"max_seq_len must be >= 1, got {max_seq_len}")
    grid = [kv for kv in _KV_GRID_FULL if kv <= max_seq_len]
    if not grid:
        return [max_seq_len]
    if grid[-1] < max_seq_len:
        grid.append(max_seq_len)  # append the long-tail upper-bound point
    return grid


# Aggregation/selection criteria
# selector: {backend name: [that backend's latency at each kv grid point]} -> selected backend name (empty matrix -> None).
Selector = Callable[[Dict[str, List[float]]], Optional[str]]


def select_stable(
    matrix: Dict[str, List[float]],
    registry_order: List[str],
    default_backend: Optional[str] = None,
    threshold: float = STABLE_THRESHOLD,
    cluster_margin: float = STABLE_CLUSTER_MARGIN,
) -> Optional[str]:
    """Two-level deterministic selection: threshold filter + registry-order tiebreaker.

    Level 1 (threshold filter): only keep backends that are significantly faster
    than the default_backend (improvement >= threshold). If none pass, return
    default_backend (no switch).

    Level 2 (cluster tiebreaker): among survivors, if mutual differences are
    within cluster_margin, pick the one with highest priority in registry_order
    (deterministic, noise-immune). Only when a backend is clearly faster than
    all others (gap > cluster_margin) does it win outright.

    Guarantee: as long as threshold is comfortably larger than the GPU bench
    noise band, the selection result is deterministic across runs on the same
    hardware. Concrete default ratios live in the STABLE_THRESHOLD /
    STABLE_CLUSTER_MARGIN constants (env-overridable), not here, so this doc
    stays correct if the values change.
    """
    if not matrix:
        return None

    means = {impl: statistics.fmean(lats) for impl, lats in matrix.items()}

    # --- Level 1: threshold filter vs default_backend ---
    baseline_cost = means.get(default_backend) if default_backend else None
    if baseline_cost is not None and baseline_cost > 0:
        survivors = {
            impl: cost
            for impl, cost in means.items()
            if (baseline_cost - cost) / baseline_cost >= threshold
        }
    else:
        # No baseline available: all candidates are survivors
        survivors = dict(means)

    if not survivors:
        return default_backend  # Nothing significantly better -> use default

    if len(survivors) == 1:
        return list(survivors.keys())[0]

    # --- Level 2: cluster + registry-order tiebreaker ---
    best_impl = min(survivors, key=survivors.get)
    best_cost = survivors[best_impl]

    # Cluster: backends within cluster_margin of the best are in the "same performance tier"
    cluster = [
        impl
        for impl, cost in survivors.items()
        if best_cost == 0 or (cost - best_cost) / best_cost < cluster_margin
    ]

    # Deterministic tiebreaker: pick the highest-priority one in registration order
    for name in registry_order:
        if name in cluster:
            return name

    return best_impl  # fallback (should not be reached)
