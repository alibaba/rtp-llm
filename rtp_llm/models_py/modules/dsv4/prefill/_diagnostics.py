"""Opt-in prefill timing/reporting, separate from model orchestration."""

import os
from typing import Dict, Optional

import torch


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


# Opt-in CPU-wall accounting; disabled paths do not read allocator statistics.
_FWD_STATS = _env_flag("DSV4_FWD_STATS")
_FWD_STATS_MAX = int(os.environ.get("DSV4_FWD_STATS_MAX", "0") or 0)
_FWD_STATS_ROWS: list = []
_FWD_STATS_N = [0]

# GPU events are drained lazily without serializing measured forwards.
_FWD_GPU = _env_flag("DSV4_FWD_GPU")
if _FWD_GPU:
    _FWD_STATS = True
_FWD_GPU_PENDING: list = []
_FWD_GPU_ROWS: list = []

# Allocator counters that only move on a real driver call. A per-forward union
# buffer that the caching allocator recycles leaves all three at zero; one that
# does not is paying cudaMalloc/cudaFree (a cudaFree is a device sync).
_FWD_STATS_MEM_KEYS = (
    "num_device_alloc",
    "num_device_free",
    "num_alloc_retries",
)


def _fwd_stats_snap() -> Optional[tuple]:
    if not _FWD_STATS:
        return None
    ms = torch.cuda.memory_stats()
    return tuple(int(ms.get(k, 0)) for k in _FWD_STATS_MEM_KEYS)


def _fwd_gpu_drain(force: bool = False) -> None:
    """Drain completed GPU event pairs without blocking; force waits only during shutdown."""
    while _FWD_GPU_PENDING:
        n_tokens, cp_size, _ev0, ev1 = _FWD_GPU_PENDING[0]
        if force:
            ev1.synchronize()
        elif not ev1.query():
            return
        _FWD_GPU_PENDING.pop(0)
        gpu_ms = None
        try:
            gpu_ms = float(_ev0.elapsed_time(ev1))
        except RuntimeError:
            gpu_ms = None
        if gpu_ms is None:
            continue
        _FWD_GPU_ROWS.append((n_tokens, cp_size, gpu_ms))
        import sys

        print(
            "[FWDTG] rank=%d T=%d cp=%d gpu_ms=%.2f"
            % (
                (
                    torch.distributed.get_rank()
                    if torch.distributed.is_initialized()
                    else -1
                ),
                n_tokens,
                cp_size,
                gpu_ms,
            ),
            file=sys.stderr,
            flush=True,
        )


def _fwd_stats_report_row(
    *,
    n_tokens: int,
    cp_size: int,
    marks: Dict[str, float],
    mem_before: Optional[tuple],
    mem_after: Optional[tuple],
    ev0=None,
) -> None:
    """Emit one ``[FWDT]`` line of per-phase CPU wall times (ms)."""
    _FWD_STATS_N[0] += 1
    if _FWD_STATS_MAX and _FWD_STATS_N[0] > _FWD_STATS_MAX:
        return
    if _FWD_GPU and ev0 is not None:
        ev1 = torch.cuda.Event(enable_timing=True)
        ev1.record()
        _FWD_GPU_PENDING.append((n_tokens, cp_size, ev0, ev1))
        _fwd_gpu_drain()
    order = ("cpctx", "pos", "embed", "meta", "loop", "tail")
    parts = []
    prev = marks.get("entry")
    for name in order:
        cur = marks.get(name)
        if prev is None or cur is None:
            parts.append("%s=NA" % name)
            prev = cur if cur is not None else prev
            continue
        parts.append("%s=%.2f" % (name, (cur - prev) * 1e3))
        prev = cur
    total = 0.0
    if marks.get("entry") is not None and marks.get("tail") is not None:
        total = (marks["tail"] - marks["entry"]) * 1e3
    deltas = ""
    if mem_before is not None and mem_after is not None:
        deltas = " " + " ".join(
            "d_%s=%d" % (k, a - b)
            for k, b, a in zip(_FWD_STATS_MEM_KEYS, mem_before, mem_after)
        )
    import sys

    print(
        "[FWDT] rank=%d n=%d T=%d cp=%d total=%.2f %s%s"
        % (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
            _FWD_STATS_N[0],
            n_tokens,
            cp_size,
            total,
            " ".join(parts),
            deltas,
        ),
        file=sys.stderr,
        flush=True,
    )
    _FWD_STATS_ROWS.append((n_tokens, cp_size, total, tuple(parts)))


def _fwd_stats_flush() -> None:
    if _FWD_GPU:
        _fwd_gpu_drain(force=True)
    if not _FWD_STATS or not _FWD_STATS_ROWS:
        return
    import sys

    by_shape: Dict[tuple, list] = {}
    for n_tokens, cp_size, total, _parts in _FWD_STATS_ROWS:
        by_shape.setdefault((n_tokens, cp_size), []).append(total)
    print(
        "[FWDT] ---- summary over %d forwards ----" % len(_FWD_STATS_ROWS),
        file=sys.stderr,
        flush=True,
    )
    for (n_tokens, cp_size), totals in sorted(by_shape.items()):
        print(
            "[FWDT] T=%d cp=%d calls=%d mean_total=%.2f ms sum_total=%.1f ms"
            % (
                n_tokens,
                cp_size,
                len(totals),
                sum(totals) / len(totals),
                sum(totals),
            ),
            file=sys.stderr,
            flush=True,
        )
    if _FWD_GPU_ROWS:
        gpu_by_shape: Dict[tuple, list] = {}
        for n_tokens, cp_size, gpu_ms in _FWD_GPU_ROWS:
            gpu_by_shape.setdefault((n_tokens, cp_size), []).append(gpu_ms)
        for (n_tokens, cp_size), vals in sorted(gpu_by_shape.items()):
            print(
                "[FWDTG] T=%d cp=%d calls=%d mean_gpu=%.2f ms sum_gpu=%.1f ms"
                % (n_tokens, cp_size, len(vals), sum(vals) / len(vals), sum(vals)),
                file=sys.stderr,
                flush=True,
            )


if _FWD_STATS:
    import atexit

    atexit.register(_fwd_stats_flush)


# CUPTI captures are diagnostic, never clean TTFT measurements.
_FWD_PROFILE = _env_flag("DSV4_FWD_PROFILE")
_FWD_PROFILE_IDX = int(os.environ.get("DSV4_FWD_PROFILE_IDX", "20") or 20)
_FWD_PROFILE_ROWS = int(os.environ.get("DSV4_FWD_PROFILE_ROWS", "35") or 35)
# -1 = every rank profiles. CUPTI on all 8 ranks at once perturbs the pipeline
# and the CP collectives, so measurement legs normally pin one world rank.
_FWD_PROFILE_RANK = int(os.environ.get("DSV4_FWD_PROFILE_RANK", "-1") or -1)
_FWD_PROFILE_CT = [0]


def _fwd_profile_rank_ok() -> bool:
    if _FWD_PROFILE_RANK < 0:
        return True
    if not torch.distributed.is_initialized():
        return _FWD_PROFILE_RANK == 0
    return torch.distributed.get_rank() == _FWD_PROFILE_RANK


def _fwd_profile_start():
    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    )
    prof.__enter__()
    return prof


def _fwd_profile_dump(prof, fwd_idx: int = 0) -> None:
    import sys

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    # The C++ StepWindowProfiler's chrome traces come out with zero-timestamped
    # kernel events on this build; the python profiler's export carries real
    # per-kernel GPU times, so also dump one when DSV4_FWD_TRACE_DIR is set.
    trace_dir = os.environ.get("DSV4_FWD_TRACE_DIR", "")
    if trace_dir:
        try:
            path = os.path.join(trace_dir, f"fwd_rank{rank}_idx{fwd_idx}.json")
            prof.export_chrome_trace(path)
            print(
                "[FWDP] rank=%d chrome trace exported: %s" % (rank, path),
                file=sys.stderr,
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(
                "[FWDP] rank=%d chrome trace export failed: %r" % (rank, e),
                file=sys.stderr,
                flush=True,
            )
    ka = prof.key_averages()
    total_us = sum(float(e.self_device_time_total) for e in ka)
    print(
        "[FWDP] rank=%d ---- one forward: total self GPU %.2f ms ----"
        % (rank, total_us / 1e3),
        file=sys.stderr,
        flush=True,
    )
    print(
        ka.table(
            sort_by="self_device_time_total",
            row_limit=_FWD_PROFILE_ROWS,
            max_name_column_width=78,
        ),
        file=sys.stderr,
        flush=True,
    )
