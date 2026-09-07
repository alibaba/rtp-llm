from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...harness import AssertUtils
from ...registry import case
from ...support.priority import (
    CODE_ADMISSION_TIMEOUT,
    CODE_ENGINE_CANCELLED,
    CODE_OK,
    CODE_SLO_EXPIRED,
    CODE_YIELDED,
    PERF_SETTLE_S,
    _drain,
    _finally_hygiene,
    _fire,
    _fire_batch,
    _master_http,
    _outcome_map,
    _poll_engine_pending,
    _prefill_names,
    _t1_spec,
)


@case(
    "atpm_preemption_disabled_zero_eviction",
    category="priority",
    profiles=["single-nonbatch"],
    source="design §2.3 #9 — AT2",
)
def atpm_preemption_disabled_zero_eviction(ctx: CaseContext):
    """Omitting the preemption block disables preemption entirely (AT2):
    PRIORITY ordering but no preemption config → EvictionManager's
    precondition rejects before any planning.  Two rounds: a saturated
    low-priority queue whose incoming 70 parks in the intake3
    PendingPlacementCoordinator ([EV-1-FIXED] baseline flipped at
    6ad0315f10 — the anti-overload mechanism under PRIORITY is now
    parking, not plain rejection; no fallback fires because no
    preemption config exists — zero 8400/8429/8430), then a saturated
    high-priority queue with an incoming 90 — the same park (high
    priority does not bypass capacity either; both incomings complete
    200 once capacity releases, or expire 8511 inside the queueTimeout
    window).

    ENV-T1 is shared with prio_queue_timeout_terminal (identical
    fingerprint): sequential execution + per-case finally hygiene.
    queueTimeout 8s means the deep tail of each saturated queue times out
    — that is a legal terminal (P6 = every request terminal, not every
    request completed)."""
    env = ctx.env_manager.ensure(_t1_spec(ctx))
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

        round_reports = []
        for label, fill_prio, inc_prio in (
            ("low_saturation", 30, 70),
            ("high_saturation", 70, 90),
        ):
            ph = ops.next_request_id(base)
            ph_fire = _fire(ops, ph, priority=fill_prio, input_len=2048, output_len=2)
            fires.append(ph_fire)
            if not ph_fire.ok:
                return False, f"{label} placeholder failed: code={ph_fire.code}"
            if not _poll_engine_pending(ops, prefill_names[0], 1):
                return False, f"{label} placeholder never dispatched"

            queued = [ops.next_request_id(base) for _ in range(8)]
            specs = [
                (rid, {"priority": fill_prio, "input_len": 2048, "output_len": 2})
                for rid in queued
            ]
            inc = ops.next_request_id(base)
            specs.append(
                (inc, {"priority": inc_prio, "input_len": 2048, "output_len": 2})
            )
            wave = _fire_batch(ops, specs)
            fires.extend(wave)
            outcomes = _drain(ops, [ph_fire] + wave)
            m = _outcome_map(outcomes)
            zero_preempt = all(
                m[rid][1]
                not in (CODE_YIELDED, CODE_ENGINE_CANCELLED, CODE_ADMISSION_TIMEOUT)
                for rid in queued + [ph]
            )
            # [EV-1-FIXED] baseline flipped at intake3
            # PendingPlacementCoordinator (6ad0315f10): the incoming parks
            # (no preemption fallback exists to fire — config omitted) and
            # completes 200 once capacity releases; an 8511 park expiry
            # inside the queueTimeout window is equally legal.
            inc_ok = m[inc][1] in (CODE_OK, CODE_SLO_EXPIRED)
            round_reports.append(
                (
                    label,
                    zero_preempt and inc_ok,
                    f"{label}: zero 8400/8429/8430={zero_preempt}, "
                    f"incoming{inc_prio} code={m[inc][1]} "
                    f"(park → 200 or 8511 expiry, [EV-1-FIXED])",
                )
            )
            clean_ok, clean_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
            if not clean_ok:
                round_reports[-1] = (
                    label,
                    False,
                    round_reports[-1][2] + f", inflight dirty: {clean_detail}",
                )
                break

        report.invariant(
            "AT2",
            all(ok for (_l, ok, _d) in round_reports),
            context="preemption_disabled",
            detail="; ".join(
                f"{l}={'ok' if ok else 'FAIL(' + d + ')'}" for l, ok, d in round_reports
            ),
        )
        report.invariant(
            "P6",
            all(ok for (_l, ok, _d) in round_reports),
            detail="every request reached a terminal (queue-timeout terminals included)",
        )
        return report.finish(
            f"rounds={[l for l, ok, _d in round_reports if ok]}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _finally_hygiene(ops, fires, prefill_names)
