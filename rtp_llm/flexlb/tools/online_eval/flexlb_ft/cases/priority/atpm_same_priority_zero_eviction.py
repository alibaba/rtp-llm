from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...harness import AssertUtils
from ...registry import case
from ...support.priority import (
    CODE_ENGINE_CANCELLED,
    CODE_OK,
    CODE_YIELDED,
    PERF_SETTLE_S,
    ROUTE_REJECT_FAMILY,
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
    "atpm_same_priority_zero_eviction",
    category="priority",
    profiles=["single-nonbatch"],
    source="design §2.3 #8 — PR4 + AT3",
)
def atpm_same_priority_zero_eviction(ctx: CaseContext):
    """Same-priority never evicts (PR4 core + AT3, [EV-1-FIXED] design-final
    form): eight explicit priority=50 requests (Python-side per-request
    priority — the FORCE_PRIORITY semantics without the Java load
    client) plus the incoming 50 (the ninth) ALL park in the intake3
    PendingPlacementCoordinator (pull-based, priority desc + FIFO
    tiebreak — baseline flipped at 6ad0315f10); the (never-triggered)
    eviction fallback's strictly-lower-priority candidate filter would
    come up empty for a same-priority incoming, so ZERO victims are
    taken (no 8400/8429 anywhere) and the design-final shape is all
    nine completing 200 in pure submit FIFO order.
    ENV-Q2 is shared with atpm_preempt_prefill_queued (same
    fingerprint → same run, sequential order + finally hygiene)."""
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

        ph = ops.next_request_id(base)
        ph_fire = _fire(ops, ph, priority=50, input_len=2048, output_len=2)
        fires.append(ph_fire)
        if not ph_fire.ok:
            return False, f"placeholder failed: code={ph_fire.code}"
        if not _poll_engine_pending(ops, prefill_names[0], 1):
            return False, "placeholder never dispatched"

        queued_rids = [ops.next_request_id(base) for _ in range(8)]
        specs = [
            (rid, {"priority": 50, "input_len": 2048, "output_len": 2})
            for rid in queued_rids
        ]
        inc = ops.next_request_id(base)
        specs.append((inc, {"priority": 50, "input_len": 2048, "output_len": 2}))
        wave = _fire_batch(ops, specs)
        fires.extend(wave)

        outcomes = _drain(ops, [ph_fire] + wave)
        m = _outcome_map(outcomes)
        # [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
        # (6ad0315f10): every submitter parks — the incoming 50 no longer
        # receives the route-reject family, it completes like the rest; the
        # same-priority FIFO dispatch shape (pure submit order) is the
        # design-final observation object for "same priority never evicts".
        zero_eviction = all(
            m[rid][1] not in (CODE_YIELDED, CODE_ENGINE_CANCELLED)
            for rid in queued_rids + [inc]
        )
        prio = {rid: 50 for rid in queued_rids + [inc]}
        _first_sp, shape_sp, order_sp = _design_final_pattern(
            ops, [ph_fire] + wave, queued_rids + [inc], prio
        )
        all_ok = all(m[rid][0] for rid in queued_rids) and m[inc][0]
        report.invariant(
            "PR4",
            zero_eviction and shape_sp and all_ok,
            context="same_priority_zero_eviction_design_final",
            detail=(
                f"[EV-1-FIXED] zero 8400/8429={zero_eviction}, all nine 50s "
                f"completed={all_ok}, dispatch FIFO shape ok={shape_sp} "
                f"(pure submit order), "
                f"dispatch={[r % 1_000_000 for r in order_sp]}"
            ),
        )
        report.invariant(
            "AT3",
            m[inc][1] == CODE_OK,
            context="single_qos_incoming_design_final",
            detail=(
                f"[EV-1-FIXED] baseline flipped at intake3 "
                f"PendingPlacementCoordinator (6ad0315f10): the same-priority "
                f"incoming parks and completes 200 like every queued peer — "
                f"the original-error passthrough "
                f"({list(ROUTE_REJECT_FAMILY)}) is no longer reachable for "
                f"a capacity-blocked same-priority submitter, incoming 50 "
                f"terminal={m[inc][1]}"
            ),
        )
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        report.invariant(
            "P6",
            shape_sp and all_ok and clean_ok,
            detail=(
                f"[EV-1-FIXED] all nine 50s + ph dispatched and completed "
                f"(FIFO), inflight={'ok' if clean_ok else clean_detail}"
            ),
        )
        return report.finish(
            f"zero-eviction={zero_eviction}, incoming code={m[inc][1]}, "
            f"fifo shape={shape_sp}, grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _finally_hygiene(ops, fires, prefill_names)
