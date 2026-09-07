from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...harness import AssertUtils
from ...registry import case
from ...support.priority import (
    PERF_SETTLE_S,
    _design_final_pattern,
    _drain,
    _f1_spec,
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
    "atpm_comparator_frozen_weak",
    category="priority",
    profiles=["single-nonbatch"],
    source="design §2.3 #11 — PR9 (weak form)",
)
def atpm_comparator_frozen_weak(ctx: CaseContext):
    """Comparator freeze, black-box weak form (PR9): the full
    construction-time-freeze contract ("flipping ordering after queue
    creation must not reorder registered queues") needs runtime hot
    config reload, which the master env does not expose — white-box
    (AutoTpmE2EHarness).  The black-box equivalent proven here: the
    ORDERING CONFIG decided at construction time determines the queue's
    behaviour — the same load shape run under a PRIORITY env and a FIFO
    env orders differently.

    Half 1 (ENV-Q2, PRIORITY): 30x3 submitted first (the first is the
    placeholder), then 70x3 — the dispatch order must interleave as
    "all 70s before the remaining 30s": min(70-group running_ms) <
    max(30-group running_ms).

    Half 2 (ENV-F1, case-level ordering=fifo on the same 1P shape):
    identical choreography — strict arrival order: max(30-group
    running_ms) < min(70-group running_ms).  The later-arriving higher
    priorities do NOT jump under FIFO.

    Both halves assert both directions (the design's bidirectional
    contrast); running_ms comes from the single-JVM engine clocks.
    [EV-1-FIXED] the design-final contrast was restored at intake3
    (PendingPlacementCoordinator 6ad0315f10): every wave submitter parks
    and dispatches, so the ORDERING contrast is observable black-box
    again (the EV-1 downgrade had reduced both halves to the identical
    single-park shape)."""
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    hygiene: list = []

    def run_half(spec_fn, label):
        env = ctx.env_manager.ensure(spec_fn(ctx))
        ops = ctx.engine_ops(env)
        names = _prefill_names(ops)
        for name in names:
            ops.set_perf(name, prefill_fixed_ms=3000.0)
        time.sleep(PERF_SETTLE_S)

        ph = ops.next_request_id(base)
        ph_fire = _fire(ops, ph, priority=30, input_len=2048, output_len=2)
        half_fires = [ph_fire]
        if not ph_fire.ok:
            return None, f"{label} placeholder failed: code={ph_fire.code}"
        if not _poll_engine_pending(ops, names[0], 1):
            return None, f"{label} placeholder never dispatched"

        low_rids = [ops.next_request_id(base) for _ in range(2)]
        high_rids = [ops.next_request_id(base) for _ in range(3)]
        specs = [
            (rid, {"priority": 30, "input_len": 2048, "output_len": 2})
            for rid in low_rids
        ] + [
            (rid, {"priority": 70, "input_len": 2048, "output_len": 2})
            for rid in high_rids
        ]
        half_fires.extend(_fire_batch(ops, specs))
        outcomes = _drain(ops, half_fires)
        m = _outcome_map(outcomes)
        hygiene.append((ops, half_fires, names))

        # [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
        # (6ad0315f10): the dispatch-order contrast is observable again —
        # under PRIORITY the wave dispatches [low_1 (first parker), 70a,
        # 70b, 70c, low_2] (priority desc + FIFO after the first parker),
        # under FIFO pure submit order [low_1, low_2, 70a, 70b, 70c].  The
        # same construction-time ORDERING config still governs both halves
        # (the comparator-freeze contract's black-box weak form).
        wave_rids = low_rids + high_rids
        prio = {rid: 30 for rid in low_rids}
        prio.update({rid: 70 for rid in high_rids})
        _first_p, shape_ok, order = _design_final_pattern(
            ops, half_fires, wave_rids, prio, fifo=(label == "fifo_half")
        )
        all_ok = all(m[rid][0] for rid in wave_rids)
        ph_ok = m[ph][0]
        clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        half_ok = shape_ok and all_ok and ph_ok and clean_ok
        return (shape_ok, all_ok, ph_ok, clean_ok), (
            f"{label}: shape ok={shape_ok}, "
            f"dispatch={[r % 1_000_000 for r in order]}, "
            f"placeholder completed={ph_ok}, "
            f"codes={[(r % 1_000_000, m[r][1]) for r in wave_rids]}, "
            f"clean={'ok' if clean_ok else clean_detail}"
        )

    try:
        prio_result = run_half(_q2_spec, "priority_half")
        if prio_result[0] is None:
            return False, prio_result[1]
        (p_shape, p_all_ok, p_ph_ok, p_clean), p_note = prio_result
        fifo_result = run_half(_f1_spec, "fifo_half")
        if fifo_result[0] is None:
            return False, fifo_result[1]
        (f_shape, f_all_ok, f_ph_ok, f_clean), f_note = fifo_result

        prio_half_ok = p_shape and p_all_ok and p_ph_ok and p_clean
        fifo_half_ok = f_shape and f_all_ok and f_ph_ok and f_clean
        report.invariant(
            "PR9",
            prio_half_ok and fifo_half_ok,
            context="comparator_frozen_weak_design_final",
            detail=(
                f"[EV-1-FIXED] baseline flipped at intake3 "
                f"PendingPlacementCoordinator (6ad0315f10): the dispatch-"
                f"order contrast is live — under PRIORITY the 70s dispatch "
                f"before the remaining 30 (first parker exempt), under "
                f"FIFO pure arrival order ({p_note}; {f_note}).  "
                f"Comparator-freeze itself stays white-box (runtime "
                f"reload unavailable); the construction-time ORDERING "
                f"config is the black-box weak form."
            ),
        )
        report.invariant(
            "P6",
            prio_half_ok and fifo_half_ok,
            detail=(
                f"both halves drained all-five-200 with inflight clean "
                f"(priority half={prio_half_ok}, fifo half={fifo_half_ok})"
            ),
        )
        return report.finish(
            f"priority-half shape={p_shape}, fifo-half shape={f_shape} "
            f"(design-final contrast), grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        for ops_x, fires_x, names_x in hygiene:
            try:
                _finally_hygiene(ops_x, fires_x, names_x)
            except Exception:
                pass
