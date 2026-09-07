from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from ...context import CaseContext, rid_base
from ...engine_ops import engine_inflight_clean
from ...grade import GradeReport
from ...harness import AssertUtils
from ...registry import case
from ...support.priority import (
    CODE_OK,
    CODE_YIELDED,
    FIRE_GAP_S,
    PERF_SETTLE_S,
    _all_engine_names,
    _engine_saw,
    _finally_hygiene,
    _master_http,
    _metric_lines,
    _metric_sum,
    _poll_engine_pending,
    _pq_live_spec,
    _prefill_names,
    _scrape_master_metrics,
)


@case(
    "atpm_preempt_prefill_queued_live",
    profiles=["single-batch", "batch-window"],
    source="preemption-stages audit (2026-09) — live PREFILL_QUEUED eviction",
    category="priority",
)
def atpm_preempt_prefill_queued_live(ctx: CaseContext):
    """LIVE PREFILL_QUEUED eviction: a higher-priority incoming replaces a
    queued low-priority victim in the master's prefill queue — the victim
    terminal is EXACTLY 8400 and no engine ever saw the rid.

    ENV (BATCH dispatcher — the decisive knob): under NON_BATCH the
    intake3 pull model parks every capacity-blocked submitter, so the
    queue-full replacement path has no trigger ([EV-1-FIXED]); BATCH
    puts the queue back in the master (WorkerBatcher + maxWaiting cap →
    AdmissionFallback → EvictionManager.tryAdmitByPrefillEviction).
    preemption={PREFILL_QUEUED} only, maxWaiting=2, 1P+4D, prefill
    slowed to 4s.  Decision axis: SINGLE on the single-batch lane (the
    original live family); FIXED_WINDOW maxRequests=32/wait 400ms on
    batch-window (production-isomorphic — audit #16; the 400ms window's
    timing effect on the queue-overflow choreography is exactly what
    the bw smoke run verifies).

    Choreography: a P50 placeholder dispatches first (occupying the
    engine's single prefill concurrency slot — the dispatch gate holds
    everything else MASTER_QUEUED_NOT_DISPATCHED), then two P30 victims
    queue (activeIndex=2=maxWaiting), then the P70 incoming overflows
    the queue: queueDeficit = queued(2) + 1 − maxWaiting(2) = 1
    (EvictionPlanner) → CANDIDATE_ORDER (priority asc, enqueuedAtMs
    desc — latest same-priority first) picks victim_b → master-local
    atomic replacement → 8400.

    Contract:
      * victim_b terminal is EXACTLY 8400 (yielded, retryable) and NO
        engine ever saw the rid (master-local transaction — the
        never-delivered proof);
      * victim_a / placeholder / incoming all complete 200 with real
        FetchResponse output;
      * metric plane: auto_tpm.victim.count{stage=prefill_queued} == 1
        with the 30←70 priority tags, and
        auto_tpm.priority_preempt.count{stage=prefill_queued} >= 1;
      * the ledger closes clean (master inflight + engine side) and
        recovery works.
    """
    env = ctx.env_manager.ensure(_pq_live_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    handles: dict = {}
    prefill_names: list = []
    try:
        prefill_names = _prefill_names(ops)
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=4000.0)
        time.sleep(PERF_SETTLE_S)

        ph = ops.next_request_id(base)
        ph_resp = ops.schedule(
            ph, priority=50, input_len=2048, output_len=2, timeout_s=90.0
        )
        if ph_resp.code != CODE_OK or not ph_resp.success:
            return False, f"placeholder schedule failed: {ph_resp.error_message}"
        if not _poll_engine_pending(ops, prefill_names[0], 1):
            return False, "placeholder never dispatched"

        va = ops.next_request_id(base)
        vb = ops.next_request_id(base)
        inc = ops.next_request_id(base)
        with ThreadPoolExecutor(max_workers=3) as pool:
            a_future = pool.submit(
                ops.schedule,
                va,
                priority=30,
                input_len=2048,
                output_len=2,
                timeout_s=90.0,
            )
            time.sleep(FIRE_GAP_S)
            b_future = pool.submit(
                ops.schedule,
                vb,
                priority=30,
                input_len=2048,
                output_len=2,
                timeout_s=90.0,
            )
            time.sleep(FIRE_GAP_S)
            inc_future = pool.submit(
                ops.schedule,
                inc,
                priority=70,
                input_len=2048,
                output_len=2,
                timeout_s=90.0,
            )
            a_resp = a_future.result(timeout=95.0)
            b_resp = b_future.result(timeout=95.0)
            inc_resp = inc_future.result(timeout=95.0)

        vb_evicted = b_resp.code == CODE_YIELDED and not b_resp.success
        va_ok = a_resp.code == CODE_OK and a_resp.success
        inc_ok = inc_resp.code == CODE_OK and inc_resp.success

        # Survivors: consume the FetchResponse streams (BATCH delivery —
        # the master owns the enqueue, the client stream is a
        # FetchResponse against the original prefill).
        completed = {}
        for rid, resp in ((ph, ph_resp), (va, a_resp), (inc, inc_resp)):
            if resp.code != CODE_OK or not resp.success:
                continue
            handle = ops.start_stream(resp, rid)
            handles[rid] = handle
            ended = handle.wait_end(45.0)
            completed[rid] = ended and handle.snap.completed
        survivors_completed = all(completed.get(r) for r in (ph, va, inc))
        vb_never_seen = not _engine_saw(ops, vb)

        samples = _scrape_master_metrics(ops)
        pq_victim = _metric_sum(
            samples, "auto_tpm_victim_count", {"stage": "prefill_queued"}
        )
        pq_victim_30_70 = _metric_sum(
            samples,
            "auto_tpm_victim_count",
            {
                "stage": "prefill_queued",
                "victim_priority": "30",
                "incoming_priority": "70",
            },
        )
        pq_preempt = _metric_sum(
            samples,
            "auto_tpm_priority_preempt_count",
            {"stage": "prefill_queued"},
        )

        report.invariant(
            "PR10",
            vb_evicted and va_ok and inc_ok and survivors_completed,
            context="live_prefill_queued_replacement",
            detail=(
                f"victim_b terminal={b_resp.code} (expected exactly "
                f"{CODE_YIELDED}), victim_a schedule={a_resp.code}, "
                f"incoming schedule={inc_resp.code}, FetchResponse "
                f"completed={ {r % 1_000_000: completed.get(r) for r in (ph, va, inc)} }"
            ),
        )
        report.invariant(
            "PR5",
            vb_never_seen and vb_evicted,
            context="live_victim_master_local_never_delivered",
            detail=(
                f"victim_b terminal={b_resp.code}, engine ever saw rid="
                f"{_engine_saw(ops, vb)} (master-local atomic replacement "
                f"— never-delivered proof; CANDIDATE_ORDER priority asc, "
                f"enqueuedAtMs desc picks the latest same-priority "
                f"victim_b over victim_a)"
            ),
        )
        report.invariant(
            "PR6",
            vb_evicted and pq_victim == 1.0 and (pq_preempt or 0.0) >= 1.0,
            context="live_prefill_queued_metrics",
            detail=(
                f"auto_tpm.victim.count{{stage=prefill_queued}}="
                f"{pq_victim} (expected 1), 30<-70 tagged sample="
                f"{pq_victim_30_70}, priority_preempt.count="
                f"{pq_preempt} (>=1), victim samples="
                f"{_metric_lines(samples, 'auto_tpm_victim_count')}"
            ),
        )
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        engine_clean, engine_detail = engine_inflight_clean(
            ops, _all_engine_names(ops), 30.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()
        report.invariant(
            "P6",
            clean_ok and engine_clean and recovery_ok,
            detail=(
                f"inflight={'ok' if clean_ok else clean_detail}, "
                f"engine={'ok' if engine_clean else engine_detail}, "
                f"recovery={recovery_msg}"
            ),
        )
        return report.finish(
            f"live prefill-queued eviction: victim_b={b_resp.code} "
            f"(never delivered), survivors completed="
            f"{survivors_completed}, metric victim.count={pq_victim}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        for handle in handles.values():
            handle.cancel()
        _finally_hygiene(ops, [], prefill_names)
