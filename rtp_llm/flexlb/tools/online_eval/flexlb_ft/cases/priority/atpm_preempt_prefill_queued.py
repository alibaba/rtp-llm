from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...harness import AssertUtils
from ...registry import case
from ...support.priority import (
    CODE_ENGINE_CANCELLED,
    CODE_YIELDED,
    PERF_SETTLE_S,
    _design_final_pattern,
    _drain,
    _finally_hygiene,
    _fire,
    _fire_batch,
    _master_http,
    _outcome_map,
    _poll_engine_pending,
    _prefill_names,
    _q2_spec,
)


@case(
    "atpm_preempt_prefill_queued",
    category="priority",
    profiles=["single-nonbatch"],
    source="design §2.3 #6 — PR10 + PR5 + PR6 + PR4",
)
def atpm_preempt_prefill_queued(ctx: CaseContext):
    """PREFILL_QUEUED preemption choreography under the intake3 pull model
    (PR10 + PR5 + PR6 + PR4, [EV-1-FIXED] design-final form).

    ENV-Q2: preemption allows PREFILL_QUEUED only, queueTimeout 60s,
    maxWaiting 8, inflight cap 1, single prefill.

    [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
    (6ad0315f10): capacity blocking parks EVERY submitter (pull-based
    WaitBucket, priority desc + FIFO tiebreak), so the queue-replacement
    choreography (a failed enqueue feeding AdmissionFallback → evict
    exactly one 30f → 8400) has no trigger — maxWaiting's
    enqueueUnderLock cap is a BATCH-path check the NON_BATCH pull model
    never reaches, no enqueue ever fails, and the eviction fallback
    never runs (zero victims across both waves; the deficit==1
    replacement exactness and multi-victim events migrate to the
    BATCH-profile white-box handover, design §2.5 row 11).

    Wave 1: a priority=50 placeholder parks the lease, then EIGHT
    requests + the incoming 70 queue up: 30a, 30b, 40a, 40b, 30c, 30d,
    30e, 30f, 70.  All nine park and complete 200; the dispatch order
    (design-final) is [30a (first parker — the wave's first submitter
    legitimately wins the first release slot), 70, 40a, 40b, 30b, 30c,
    30d, 30e, 30f] — after the first parker, strict priority desc +
    same-level FIFO.

    Wave 2 (same-priority infeasible shape): after the drain, a 70
    placeholder parks the lease; 70x8 + the incoming 90 all park.  The
    "no strictly-lower candidate → DECLINED" branch stays what the
    (never-triggered) fallback would see; zero victims holds trivially,
    and all nine complete 200 with the 90 dispatching FIRST among the
    wave (priority desc; first parker 70a keeps slot one).
    """
    env = ctx.env_manager.ensure(_q2_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    fires: list = []
    prefill_names: list = []
    try:
        prefill_names = _prefill_names(ops)
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=4000.0)
        time.sleep(PERF_SETTLE_S)

        # ---- wave 1: victim selection + deficit exactness --------------
        ph = ops.next_request_id(base)
        ph_fire = _fire(ops, ph, priority=50, input_len=2048, output_len=2)
        fires.append(ph_fire)
        if not ph_fire.ok:
            return False, f"wave1 placeholder failed: code={ph_fire.code}"
        if not _poll_engine_pending(ops, prefill_names[0], 1):
            return False, "wave1 placeholder never dispatched"

        tags = ["30a", "30b", "40a", "40b", "30c", "30d", "30e", "30f"]
        rids: dict = {}
        specs = []
        for tag in tags:
            rid = ops.next_request_id(base)
            rids[tag] = rid
            specs.append(
                (rid, {"priority": int(tag[:-1]), "input_len": 2048, "output_len": 2})
            )
        incoming = ops.next_request_id(base)
        specs.append((incoming, {"priority": 70, "input_len": 2048, "output_len": 2}))
        wave1 = _fire_batch(ops, specs)
        fires.extend(wave1)

        outcomes1 = _drain(ops, [ph_fire] + wave1)
        m1 = _outcome_map(outcomes1)
        # [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
        # (6ad0315f10): the whole wave parks — no enqueue ever fails, the
        # eviction fallback never runs, zero victims.  Design-final shape:
        # all nine settle 200 and the dispatch order is [30a (first
        # parker), 70, 40a, 40b, 30b..30f] (first submitter + priority
        # desc + same-level FIFO after it).
        zero_eviction_w1 = all(
            m1[rids[tag]][1] not in (CODE_YIELDED, CODE_ENGINE_CANCELLED)
            for tag in tags
        ) and m1[incoming][1] not in (CODE_YIELDED, CODE_ENGINE_CANCELLED)
        wave1_rids = [rids[t] for t in tags] + [incoming]
        prio1 = {rids[t]: int(t[:-1]) for t in tags}
        prio1[incoming] = 70
        first1, shape1, order1 = _design_final_pattern(
            ops, [ph_fire] + wave1, wave1_rids, prio1
        )
        all1_ok = all(m1[rids[t]][0] for t in tags) and m1[incoming][0]
        ph1_ok = m1[ph][0]

        report.invariant(
            "PR10",
            shape1 and zero_eviction_w1 and ph1_ok,
            context="deficit_exact_one_design_final",
            detail=(
                f"[EV-1-FIXED] baseline flipped at intake3 "
                f"PendingPlacementCoordinator (6ad0315f10): no enqueue ever "
                f"fails (maxWaiting is a BATCH-path cap under the NON_BATCH "
                f"pull model), the queue-full replacement has no trigger "
                f"and zero victims holds; dispatch shape ok={shape1}, "
                f"zero 8400/8429={zero_eviction_w1}, "
                f"codes={[(t, m1[rids[t]][1]) for t in tags]}, "
                f"incoming70={m1[incoming][1]}, "
                f"dispatch={[r % 1_000_000 for r in order1]}"
            ),
        )
        report.invariant(
            "PR5",
            zero_eviction_w1 and shape1,
            context="victim_determinism_design_final",
            detail=(
                "[EV-1-FIXED] baseline flipped at intake3 "
                "PendingPlacementCoordinator (6ad0315f10): eviction never "
                "triggers (no failed enqueue feeds the fallback) — zero "
                "victims across the wave; victim-selection determinism "
                "stays a white-box handover, the design-final dispatch "
                "shape carries the ordering evidence"
            ),
        )
        report.invariant(
            "PR6",
            all1_ok and shape1,
            context="prefill_queued_terminal_design_final",
            detail=(
                f"[EV-1-FIXED] baseline flipped at intake3 "
                f"PendingPlacementCoordinator (6ad0315f10): every parked "
                f"submitter (not just the first) is re-pulled on capacity "
                f"release — all wave codes 200, zero route-reject; "
                f"codes={[(t, m1[rids[t]][1]) for t in tags]}, "
                f"incoming70={m1[incoming][1]}, "
                f"dispatch={[r % 1_000_000 for r in order1]}"
            ),
        )
        report.invariant(
            "PR4",
            zero_eviction_w1 and ph1_ok and shape1 and all1_ok,
            context="strict_low_priority_victims_design_final",
            detail=(
                "[EV-1-FIXED] baseline flipped at intake3 "
                "PendingPlacementCoordinator (6ad0315f10): strictly-lower-"
                "priority victim selection stays white-box (the eviction "
                "fallback has no trigger under the pull model); the "
                "design-final form is zero victims + every queued request "
                "completing untouched in priority-desc dispatch order"
            ),
        )
        clean1_ok, clean1_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        report.invariant(
            "P6",
            shape1 and all1_ok and ph1_ok and clean1_ok,
            detail=(
                f"[EV-1-FIXED] wave1: all nine requests dispatched from the "
                f"park bucket and completed 200 (first parker 30a, then "
                f"priority desc + FIFO), "
                f"inflight={'ok' if clean1_ok else clean1_detail}"
            ),
        )
        if not (shape1 and all1_ok and ph1_ok and clean1_ok):
            return report.finish(f"wave1 incomplete, grades: {report.summary()}")

        # ---- wave 2: infeasible → zero eviction -------------------------
        ph2 = ops.next_request_id(base)
        ph2_fire = _fire(ops, ph2, priority=70, input_len=2048, output_len=2)
        fires.append(ph2_fire)
        if not ph2_fire.ok:
            return False, f"wave2 placeholder failed: code={ph2_fire.code}"
        if not _poll_engine_pending(ops, prefill_names[0], 1):
            return False, "wave2 placeholder never dispatched"

        w2_rids = [ops.next_request_id(base) for _ in range(8)]
        w2_specs = [
            (rid, {"priority": 70, "input_len": 2048, "output_len": 2})
            for rid in w2_rids
        ]
        inc90 = ops.next_request_id(base)
        w2_specs.append((inc90, {"priority": 90, "input_len": 2048, "output_len": 2}))
        wave2 = _fire_batch(ops, w2_specs)
        fires.extend(wave2)

        outcomes2 = _drain(ops, [ph2_fire] + wave2)
        m2 = _outcome_map(outcomes2)
        # [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
        # (6ad0315f10): the same-priority wave parks all eight 70s AND the
        # incoming 90 — the "no strictly-lower candidate -> DECLINED"
        # branch is what the (never-triggered) fallback would still see,
        # zero victims holds trivially, and the design-final form is all
        # nine completing 200 with the 90 dispatching FIRST among the wave
        # (priority desc; first parker 70a keeps slot one).
        zero_eviction = all(
            m2[rid][1] not in (CODE_YIELDED, CODE_ENGINE_CANCELLED)
            for rid in w2_rids + [ph2, inc90]
        )
        prio2 = {rid: 70 for rid in w2_rids}
        prio2[inc90] = 90
        first2, shape2, order2 = _design_final_pattern(
            ops, [ph2_fire] + wave2, w2_rids + [inc90], prio2
        )
        all2_ok = all(m2[rid][0] for rid in w2_rids) and m2[inc90][0]
        ph2_ok = m2[ph2][0]
        report.invariant(
            "PR10",
            zero_eviction and shape2 and all2_ok and ph2_ok,
            context="infeasible_no_partial_eviction_design_final",
            detail=(
                f"[EV-1-FIXED] zero eviction={zero_eviction} (no "
                f"strictly-lower candidate for the 90 — all-or-nothing, and "
                f"the fallback never triggers anyway), "
                f"90 terminal={m2[inc90][1]} (dispatched first among the "
                f"wave, priority desc), shape ok={shape2}, "
                f"dispatch={[r % 1_000_000 for r in order2]}, ph ok={ph2_ok}"
            ),
        )
        clean2_ok, clean2_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        report.invariant(
            "P6",
            shape2 and all2_ok and ph2_ok and clean2_ok,
            detail=(
                f"[EV-1-FIXED] wave2: all nine requests completed 200 (the "
                f"90 first among the wave by priority), "
                f"inflight={'ok' if clean2_ok else clean2_detail}"
            ),
        )
        return report.finish(
            f"wave1 shape={shape1} all-200 zero-victims, wave2 shape={shape2} "
            f"zero-eviction={zero_eviction}, grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _finally_hygiene(ops, fires, prefill_names)
