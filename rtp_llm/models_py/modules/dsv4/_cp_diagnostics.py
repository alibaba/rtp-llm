"""Opt-in CP gather event accounting; no collective or tensor ownership."""

import os
import re
from typing import Optional

_DEFAULT_CP_PROFILE_NAME = "dsv4.cp.all_gather"

_CP_GATHER_STATS = os.environ.get("DSV4_CP_GATHER_STATS", "0") == "1"
_CP_GATHER_STATS_EVERY = int(os.environ.get("DSV4_CP_GATHER_STATS_EVERY", "200"))
_cp_gather_stats: dict = {
    "calls": 0,
    "launch_ms": 0.0,
    "restore_ms": 0.0,
    "total_ms": 0.0,
    "bytes": 0,
    "pending": [],
    "by_kind": {},
    "announced": False,
}
if _CP_GATHER_STATS:
    import atexit as _cp_atexit

    _cp_atexit.register(lambda: _cp_gather_stats_report())


def _cp_gather_kind(profile_name: Optional[str]) -> str:
    """Gather kind with the per-layer id stripped, so totals aggregate."""
    name = profile_name or _DEFAULT_CP_PROFILE_NAME
    return re.sub(r"\.L\d+\.", ".L*.", name)


def _cp_gather_record(kind: str, ev_start, ev_mid, ev_end, nbytes: int) -> None:
    """Accumulate one timed region. ``ev_mid`` may be None when the caller does
    not split launch from restore (the sync and async-wait paths)."""
    if not _cp_gather_stats["announced"]:
        _cp_gather_stats["announced"] = True
        import sys

        print(
            f"[CPGATHER] instrumentation live; first timed call kind={kind} "
            f"bytes={nbytes / 1e6:.2f}MB report_every={_CP_GATHER_STATS_EVERY}",
            file=sys.stderr,
            flush=True,
        )
    _cp_gather_stats["calls"] += 1
    _cp_gather_stats["pending"].append((kind, ev_start, ev_mid, ev_end, nbytes))
    if _cp_gather_stats["calls"] % _CP_GATHER_STATS_EVERY == 0:
        _cp_gather_stats_report()


def _cp_gather_stats_drain(force: bool = False) -> None:
    keep = []
    for name, ev0, ev1, ev2, nbytes in _cp_gather_stats["pending"]:
        if not (force or ev2.query()):
            keep.append((name, ev0, ev1, ev2, nbytes))
            continue
        if force:
            # elapsed_time raises "Both events must be completed" otherwise. Only
            # the forced (report/atexit) path pays this; the normal path uses the
            # non-blocking query() above so measuring never serializes a gather.
            ev2.synchronize()
        total = ev0.elapsed_time(ev2)
        if ev1 is not None:
            launch = ev0.elapsed_time(ev1)
            restore = ev1.elapsed_time(ev2)
            _cp_gather_stats["launch_ms"] += launch
            _cp_gather_stats["restore_ms"] += restore
        _cp_gather_stats["total_ms"] += total
        _cp_gather_stats["bytes"] += nbytes
        agg = _cp_gather_stats["by_kind"].setdefault(name, [0, 0.0, 0])
        agg[0] += 1
        agg[1] += total
        agg[2] += nbytes
    _cp_gather_stats["pending"] = keep


def _cp_gather_stats_report() -> None:
    import sys

    _cp_gather_stats_drain(force=True)
    s = _cp_gather_stats
    print(
        f"[CPGATHER] calls={s['calls']} launch={s['launch_ms']:.1f}ms "
        f"restore={s['restore_ms']:.1f}ms total={s['total_ms']:.1f}ms "
        f"bytes={s['bytes'] / 1e6:.1f}MB mean={s['total_ms'] / max(s['calls'], 1):.3f}ms",
        file=sys.stderr,
        flush=True,
    )
    for kind, (n, ms, nbytes) in sorted(s["by_kind"].items(), key=lambda kv: -kv[1][1]):
        print(
            f"[CPGATHER]   {kind}: n={n} total={ms:.1f}ms "
            f"mean={ms / max(n, 1):.3f}ms bytes={nbytes / 1e6:.1f}MB",
            file=sys.stderr,
            flush=True,
        )
