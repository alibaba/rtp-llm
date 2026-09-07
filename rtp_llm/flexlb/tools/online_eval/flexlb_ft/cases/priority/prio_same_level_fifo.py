from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...harness import AssertUtils
from ...registry import case
from ...support.priority import (
    PERF_SETTLE_S,
    _dispatch_order,
    _drain,
    _finally_hygiene,
    _fire_batch,
    _master_http,
    _outcome_map,
    _prefill_names,
    _q1_spec,
)


@case(
    "prio_same_level_fifo",
    category="priority",
    profiles=["single-nonbatch"],
    source="design §2.2 #2 — PR2 + P6",
)
def prio_same_level_fifo(ctx: CaseContext):
    """Same-priority FIFO (PR2 invariant + P6): seven priority=50 requests
    submitted with rid ascending (the first doubles as the placeholder) —
    the dispatch order must equal the submit order exactly.

    enqueueSeq is PriorityOrdering's second key (design §3.4 row 5);
    sequential submission with 0.15s gaps makes enqueuedAtMs order ==
    submit order, so the tie-break is exercised deterministically.  The
    requestId tie-break inside one arrival instant is NOT constructible
    from a single client (arrival order already decides) — white-box
    handover per design §2.5.
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

        rids = [ops.next_request_id(base) for _ in range(7)]
        specs = [
            (rid, {"priority": 50, "input_len": 2048, "output_len": 2}) for rid in rids
        ]
        fires.extend(_fire_batch(ops, specs))

        outcomes = _drain(ops, fires)
        order = _dispatch_order(ops, fires)
        # [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
        # (6ad0315f10): all seven same-priority peers park in the pull-based
        # coordinator and dispatch in pure submit order (enqueueSeq
        # tie-break, design §3.4 row 5) — rids[0] direct-dispatches against
        # the empty queue, rids[1] is the wave's first parker, and the
        # "dispatch == submit" equality on seven FIFO peers is now a REAL
        # observation object (was EV-1: only the first parker survived,
        # the rest route-rejected).
        m = _outcome_map(outcomes)
        all_ok = all(m[rid][0] for rid in rids)
        fifo_ok = order == rids and all_ok
        report.invariant(
            "PR2",
            fifo_ok,
            context="same_priority_fifo",
            detail=(
                f"[EV-1-FIXED] dispatch==submit:{order == rids}, "
                f"all 7 code=200={all_ok}, "
                f"dispatch={[r % 1_000_000 for r in order]}"
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
            f"fifo dispatch==submit:{order == rids} [EV-1-FIXED], "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _finally_hygiene(ops, fires, prefill_names)
