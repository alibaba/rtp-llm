from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...harness import AssertUtils
from ...registry import case
from ...support.priority import (
    PERF_SETTLE_S,
    _drain,
    _finally_hygiene,
    _fire,
    _lat_stats,
    _master_http,
    _prefill_names,
)


@case(
    "prio_low_no_starvation",
    category="priority",
    profiles=["single-nonbatch"],
    source="design §2.2 #4 — PR8(完成口径) + P6",
)
def prio_low_no_starvation(ctx: CaseContext):
    """Low-priority completion under non-saturated load (PR8 completion
    calibre + P6).  Two waves on the shared profile env (no inflight cap):
    each wave fires 30x4 FIRST (early low-priority arrivals) then 70x4 —
    8 requests against a capacity of thousands, no sustained saturation,
    so nothing queues and nothing preempts; the property under test is
    that the 30s still complete (rate 1.0).  With no explicit
    anti-starvation mechanism, non-suspension is the only mechanical
    protection (analysis report §3.7).  [Migration note: the source ran
    this on its 2P+4D production-shaped shared env; on this line the
    shared env is the single-nonbatch profile topology (1P+4D) — the
    non-saturation property is topology-independent, 8 requests remain
    far below capacity.]

    PR8's grade-registry entry is the deadline-ratio upper band (used by
    prio_queue_timeout_terminal); a completion rate cannot ride
    report.invariant("PR8") — the registry types PR8 as a band and
    invariant() rejects band ids — so the completion assertion folds
    into P6 with the rate spelled out in the detail (design §2.2's
    "PR8 完成口径" invariant intent).

    The 30-vs-70 terminal-latency split is recorded as calibration data
    only (non-saturated ratios are choreography-determined; no band).
    """
    ops = ctx.ops()
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    all_fires: list = []
    low_all: list = []
    high_all: list = []
    low_total = 0
    low_done = 0
    high_total = 0
    high_done = 0
    prefill_names: list = []
    try:
        # EV-1 phase-race guard (flake fix, 2026-08 third-round run): the
        # default FIRE_GAP_S=0.15s batch let consecutive submitters hit
        # the occupied-prefill-slot window inside the master's 1s
        # status-poll period — one wave request parked into the single
        # probe slot and hung on queueTimeout (inflight scheduler=1 for
        # 30s+).  Non-saturation is made DETERMINISTIC instead: fast
        # prefill (50ms) plus a 1.5s submit gap (> poll period +
        # completion visibility) means every submitter finds the queue
        # empty and dispatches directly.
        prefill_names = _prefill_names(ops)
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=50.0)
        time.sleep(PERF_SETTLE_S)
        for wave in range(2):
            wave_fires = []
            for i in range(8):
                rid = ops.next_request_id(base)
                prio = 30 if i < 4 else 70
                fr = _fire(ops, rid, priority=prio, input_len=2048, output_len=2)
                wave_fires.append(fr)
                time.sleep(1.5)
            all_fires.extend(wave_fires)
            low_fires = [fr for fr in wave_fires if fr.kwargs.get("priority") == 30]
            high_fires = [fr for fr in wave_fires if fr.kwargs.get("priority") == 70]
            low_all.extend(low_fires)
            high_all.extend(high_fires)
            outcomes = _drain(ops, wave_fires)
            ok_rids = {rid for (rid, ok, _c, _d) in outcomes if ok}
            low_total += len(low_fires)
            low_done += sum(1 for fr in low_fires if fr.rid in ok_rids)
            high_total += len(high_fires)
            high_done += sum(1 for fr in high_fires if fr.rid in ok_rids)
            clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
            if not clean_ok:
                report.invariant(
                    "P6", False, detail=f"wave{wave} inflight: {clean_detail}"
                )
                return report.finish(f"wave{wave} inflight dirty, early stop")
            time.sleep(2.0)  # quiet window between waves

        rate = low_done / low_total if low_total else 0.0
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        report.invariant(
            "P6",
            low_done == low_total and high_done == high_total and clean_ok,
            detail=(
                f"low completion {low_done}/{low_total} (rate {rate:.2f}), "
                f"high completion {high_done}/{high_total}, "
                f"low latency={_lat_stats(low_all)}, "
                f"high latency={_lat_stats(high_all)}, "
                f"inflight={'ok' if clean_ok else clean_detail}"
            ),
        )
        return report.finish(
            f"low={low_done}/{low_total}, high={high_done}/{high_total}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _finally_hygiene(ops, all_fires, prefill_names)
