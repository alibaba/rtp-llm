"""Pure-CPU core of the attention backend dispatcher: kv grid, aggregation criteria, two-stage filtering orchestration, and capture bucket rules.

All GPU touchpoints are isolated behind injected functions (eligible_fn /
measure_fn / fallback_fn), so this module is unit-testable without a GPU.
Startup-time orchestration and real-machine micro-benchmark live in
backend_selector (GPU).
"""

from __future__ import annotations

import logging
import math
import statistics
from typing import Callable, Dict, List, Optional, Sequence

from rtp_llm.models_py.modules.factory.attention.dispatch.plan import Plan

logger = logging.getLogger(__name__)


# ─── capture bs bucket rules (single source of truth for the bench/plan path) ──
_KEY_BS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)
_STEP_FROM = 144
_STEP = 16


def capture_buckets(max_bs: int) -> List[int]:
    """Key-point bs buckets up to 128, then step-16 beyond.

    Covers the decode minibench range: fine-grained at small bs (latency-sensitive),
    key powers-of-2 and round points up to 128, coarser step-16 beyond that.
    If the engine passes decode_capture_batch_sizes directly, the engine's are used;
    this function only derives by rule when none are passed.
    """
    if max_bs < 1:
        raise ValueError(f"max_bs must be >= 1, got {max_bs}")
    bs: List[int] = [i for i in _KEY_BS if i <= max_bs]
    i = _STEP_FROM
    while i <= max_bs:
        bs.append(i)
        i += _STEP
    if not bs or bs[-1] != max_bs:
        bs.append(max_bs)
    return bs


# ─── kv grid: log2-equispaced; equal weighting is a log-uniform prior ───────────────────────
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


# ─── Aggregation/selection criteria ──────────────────────────────────────────────────────────
# selector: {backend name: [that backend's latency at each kv grid point]} -> selected backend name (empty matrix -> None).
Selector = Callable[[Dict[str, List[float]]], Optional[str]]


def select_min_mean(matrix: Dict[str, List[float]]) -> Optional[str]:
    """Default criterion: lowest equal-weighted expected latency across the kv grid (distribution-free; ties go to the first to appear)."""
    best: Optional[str] = None
    best_cost = math.inf
    for impl, lats in matrix.items():
        cost = statistics.fmean(lats)
        if cost < best_cost:
            best, best_cost = impl, cost
    return best


def select_minimax_regret(matrix: Dict[str, List[float]]) -> Optional[str]:
    """Alternative criterion: minimize the maximum relative regret (more robust to long-tail extreme points).

    At each grid point, regret = (this backend's latency - the point's best) / the
    point's best; pick the backend whose maximum regret across all points is
    smallest.
    """
    if not matrix:
        return None
    impls = list(matrix)
    n = len(matrix[impls[0]])
    best_per_kv = [min(matrix[i][k] for i in impls) for k in range(n)]
    best: Optional[str] = None
    best_worst = math.inf
    for i in impls:
        worst = 0.0
        for k in range(n):
            ref = best_per_kv[k]
            if ref > 0:
                worst = max(worst, (matrix[i][k] - ref) / ref)
        if worst < best_worst:
            best, best_worst = i, worst
    return best


# ─── Pure-CPU orchestration core ─────────────────────────────────────────────────────────
EligibleFn = Callable[
    [int], Sequence[str]
]  # (cap_bs) -> backend names in precision gate intersect support
MeasureFn = Callable[
    [str, int, int], Optional[float]
]  # (backend, cap_bs, kv) -> us|None
FallbackFn = Callable[
    [int], Optional[str]
]  # (cap_bs) -> fixed-priority backend name|None


def select_plan(
    buckets: Sequence[int],
    *,
    eligible_fn: EligibleFn,
    measure_fn: MeasureFn,
    fallback_fn: FallbackFn,
    grid: Sequence[int],
    selector: Selector = select_min_mean,
    note: str = "",
) -> Plan:
    """For each capture bucket, perform two-stage filtering + kv-grid aggregation comparison, producing a Plan (pure-CPU orchestration).

    If a backend's measurement at some (bucket, kv) raises an exception or returns
    None, that backend is marked N/A in that bucket and dropped from the
    candidates; if a whole bucket has no usable backend (eligible empty or all
    N/A), fall back to fixed priority and alert.
    """
    assignments: Dict[int, str] = {}
    for b in buckets:
        eligible = list(eligible_fn(b))
        matrix: Dict[str, List[float]] = {}
        for impl in eligible:
            lats: Optional[List[float]] = []
            for kv in grid:
                try:
                    lat = measure_fn(impl, b, kv)
                except (
                    Exception
                ) as e:  # general defense: a real exception is treated as N/A for that (bucket, kv)
                    logger.warning(
                        "[dispatcher] measure %s @ bs=%d kv=%d exception: %r",
                        impl,
                        b,
                        kv,
                        e,
                    )
                    lat = None
                if lat is None:
                    lats = None  # this impl has a gap in this bucket -> overall N/A
                    break
                lats.append(lat)
            if lats:
                matrix[impl] = lats

        choice = selector(matrix) if matrix else None
        if choice is not None:
            assignments[b] = choice
            continue

        # fallback: eligible empty or all N/A
        fb = fallback_fn(b)
        reason = (
            "eligible empty (precision gate intersect support)"
            if not eligible
            else "eligible all N/A"
        )
        logger.warning(
            "[dispatcher] bs=%d %s -> fall back to fixed priority %s",
            b,
            reason,
            fb or "(also failed, left empty)",
        )
        if fb is not None:
            assignments[b] = fb
    return Plan(assignments=assignments, note=note)
