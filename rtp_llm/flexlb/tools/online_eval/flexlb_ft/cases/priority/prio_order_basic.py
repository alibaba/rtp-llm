from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...harness import AssertUtils
from ...registry import case
from ...support.priority import (
    CODE_OK,
    PERF_SETTLE_S,
    _design_final_pattern,
    _dispatch_order,
    _drain,
    _finally_hygiene,
    _fire,
    _fire_batch,
    _group_order_ok,
    _inversion_ratio,
    _master_http,
    _outcome_map,
    _poll_engine_pending,
    _prefill_names,
    _q1_spec,
)


@case(
    "prio_order_basic",
    category="priority",
    # Source ran on the dedicated priority-single-nonbatch profile; on
    # this line the SINGLE+NON_BATCH base IS single-nonbatch and the
    # PRIORITY axis arrives through the case-layer config injection
    # (_q1_spec / _prio_config).
    profiles=["single-nonbatch"],
    source="design §2.2 #1 — PR1(band) + PR2 + P6",
)
def prio_order_basic(ctx: CaseContext):
    """Priority-order fidelity (PR1 band + PR2 group FIFO + P6).

    Choreography (design §2.2 #1): ENV-Q1 — a single prefill so every
    request lands in the same queue (no routing ambiguity), no preemption,
    default queueTimeout (Java default 1h — nothing expires inside the
    window).  A priority=50 placeholder parks the inflight lease, then a
    mixed ladder is submitted LOW-first (30a, 30b, 50a, 50b, 70a, 70b,
    0.15s apart, 7 concurrent ≤ maxWaiting 8 — no route failure, no
    preemption).  [EV-1-FIXED] Under the intake3 pull-based coordinator
    (PendingPlacementCoordinator, 6ad0315f10) the whole wave parks and
    settles code=200; the design-final dispatch order is [ph, 30a (the
    wave's first submitter legitimately wins the FIRST release slot),
    70a, 70b, 50a, 50b, 30b] — every later release follows strict
    priority desc + same-level FIFO (probe evidence 2026-08-31, gaps
    ~3015ms, zero route-reject).

    Observation (design §3.3): engine request_lifecycle.running_ms
    ascending == dispatch order (the mock cluster is one JVM, clocks
    comparable); NON_BATCH parks capacity-blocked schedules, so the client
    settle order is a natural arbitration signal for same-millisecond
    conflicts — the risk-1 fallback the task brief pre-authorized.
    """
    env = ctx.env_manager.ensure(_q1_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    fires: list = []
    prefill_names: list = []
    try:
        prefill_names = _prefill_names(ops)
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=3000.0)
        time.sleep(PERF_SETTLE_S)

        ph = ops.next_request_id(base)
        ph_fire = _fire(ops, ph, priority=50, input_len=2048, output_len=2)
        fires.append(ph_fire)
        if not ph_fire.ok:
            return False, f"placeholder schedule failed: code={ph_fire.code}"
        if not _poll_engine_pending(ops, prefill_names[0], 1):
            return False, "placeholder never reached the prefill engine"

        tags = ["30a", "30b", "50a", "50b", "70a", "70b"]
        rids: dict = {}
        specs = []
        for tag in tags:
            rid = ops.next_request_id(base)
            rids[tag] = rid
            specs.append(
                (rid, {"priority": int(tag[:-1]), "input_len": 2048, "output_len": 2})
            )
        wave = _fire_batch(ops, specs)
        fires.extend(wave)

        outcomes = _drain(ops, fires)
        tag_of = {rid: tag for tag, rid in rids.items()}
        tag_of[ph] = "ph"
        priorities = {ph: 50}
        for tag, rid in rids.items():
            priorities[rid] = int(tag[:-1])

        order = _dispatch_order(ops, fires)
        order_tags = [tag_of.get(r, str(r)) for r in order]
        # [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
        # (6ad0315f10): the whole wave parks (pull-based WaitBucket, priority
        # desc + FIFO tiebreak) and every request settles code=200 — zero
        # route-reject.  Design-final dispatch shape: [ph, 30a (the wave's
        # first submitter legitimately wins the FIRST release slot), 70a,
        # 70b, 50a, 50b, 30b].  PR1 scores the REAL dispatch order
        # (running_ms asc + settle-rank arbitration) with the first parker
        # AND the pre-wave running placeholder excluded — the exclusion is
        # the designed pull-model behaviour, not an inversion amnesty.
        m = _outcome_map(outcomes)
        wave_rids = [rids[t] for t in tags]
        first_parker, shape_ok, wave_order = _design_final_pattern(
            ops, fires, wave_rids, priorities
        )
        wave_order_tags = [tag_of.get(r, str(r)) for r in wave_order]
        all_ok = m[ph][0] and all(m[rids[t]][0] for t in tags)

        report.check(
            "PR1",
            _inversion_ratio(order, priorities, exclude={ph, first_parker}),
            context="basic_order",
            detail=(
                f"[EV-1-FIXED] dispatch={order_tags} (design-final: first "
                f"parker 30a then priority desc; first parker and pre-wave "
                f"placeholder excluded from PR1 scoring)"
            ),
        )
        report.invariant(
            "PR2",
            shape_ok
            and _group_order_ok(order, [rids["70a"], rids["70b"]])
            and _group_order_ok(order, [rids["50a"], rids["50b"]])
            and _group_order_ok(order, [rids["30a"], rids["30b"]]),
            context="same_priority_fifo",
            detail=(
                f"[EV-1-FIXED] dispatch={order_tags} (shape covers "
                f"same-level FIFO inside every priority group)"
            ),
        )
        report.invariant(
            "PR6",
            shape_ok
            and all_ok
            and all(m[rids[t]][1] == CODE_OK for t in tags)
            and m[ph][1] == CODE_OK,
            context="design_final_dispatch",
            detail=(
                f"[EV-1-FIXED] baseline flipped at intake3 "
                f"PendingPlacementCoordinator (6ad0315f10): wave="
                f"{wave_order_tags} (first parker="
                f"{tag_of.get(first_parker, first_parker)} then priority "
                f"desc + same-level FIFO), all code=200 (zero route-reject), "
                f"codes={[(t, m[rids[t]][1]) for t in tags]}"
            ),
        )
        unfinished = [o for o in outcomes if not o[1]]
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        report.invariant(
            "P6",
            not unfinished and clean_ok,
            detail=(
                f"[EV-1-FIXED] issued=terminal no-loss: "
                f"{len(outcomes) - len(unfinished)}/7 completed, "
                f"unfinished={unfinished[:3] if unfinished else 'none'}, "
                f"inflight={'ok' if clean_ok else clean_detail}"
            ),
        )
        return report.finish(
            f"dispatched={order_tags} [EV-1-FIXED], grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _finally_hygiene(ops, fires, prefill_names)
