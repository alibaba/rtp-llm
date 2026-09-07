from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...harness import AssertUtils
from ...registry import case
from ...support.priority import (
    CODE_SLO_EXPIRED,
    CODE_YIELDED,
    PERF_SETTLE_S,
    REASON_NAMES,
    REASON_UNSPECIFIED,
    _a1_spec,
    _drain,
    _finally_hygiene,
    _fire,
    _fire_batch,
    _master_http,
    _outcome_map,
    _poll_engine_pending,
    _prefill_names,
)


@case(
    "atpm_timeout_attribution",
    category="priority",
    profiles=["single-nonbatch"],
    source="design §2.3 #10 — PR7 (+PR8 deadline-no-extension)",
)
def atpm_timeout_attribution(ctx: CaseContext):
    """Admission-timeout expiry uniformity (PR7, [EV-1-FIXED] design-final
    form): under the intake3 PendingPlacementCoordinator (6ad0315f10) a
    capacity-blocked submitter parks with the schedule() RPC blocking
    until its queueTimeoutMs deadline, then terminals as plain 8511
    BATCH_SLO_EXPIRED with admission_reject_reason=UNSPECIFIED(0) — the
    attributed form (8430 + HIGHER_PRIORITY_AHEAD / 8431 +
    RESOURCE_EXHAUSTED) needs the AdmissionFailureClassifier to run at
    the queued-expiry decision, and that classifier has ZERO call sites
    in the intake3 master (Java-side observation gap — filed, not fixed
    here); every queued expiry rides the plain deadlineErrorType path
    (RequestLifecycleCoordinator.timeoutEntry fallback).

    ENV-A1: PREFILL_QUEUED preemption, queueTimeout 7s, maxWaiting 8.

    Wave 1 (mixed priorities): a 90a placeholder (12s prefill) parks the
    lease; eight 30s, the incoming 70, then 90b/90c all park.  Every
    queued member expires 8511/UNSPECIFIED at its own deadline inside
    the 12s window; 90a completes.  Zero 8400 victims (the eviction
    fallback never triggers — no failed enqueue under the pull model).

    Wave 2 (same shape, single client): a 70_early placeholder (10s
    prefill — it must OUTLAST the 70_late's ~8s deadline, the deadline
    cancels at delivery ACK), eight 30s, incoming 70_late — the 70_late
    expires 8511/UNSPECIFIED, 70_early completes.

    Deadline-no-extension (PR8, raw recording): the 70_late's terminal
    wall-time / queueTimeoutMs(7000) must stay ~1 — the park never
    restarts expiresAtMs (A9-3 keeps prio_queue_timeout_terminal as
    PR8's ONLY band consumer; the raw value rides the case finish
    detail)."""
    env = ctx.env_manager.ensure(_a1_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    fires: list = []
    prefill_names: list = []
    try:
        # ---- wave 1: 8430 + HIGHER_PRIORITY_AHEAD ----------------------
        prefill_names = _prefill_names(ops)
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=12000.0)
        time.sleep(PERF_SETTLE_S)

        ph90 = ops.next_request_id(base)
        ph90_fire = _fire(ops, ph90, priority=90, input_len=2048, output_len=2)
        fires.append(ph90_fire)
        if not ph90_fire.ok:
            return False, f"wave1 90a failed: code={ph90_fire.code}"
        if not _poll_engine_pending(ops, prefill_names[0], 1):
            return False, "wave1 90a never dispatched"

        low_rids = [ops.next_request_id(base) for _ in range(8)]
        specs = [
            (rid, {"priority": 30, "input_len": 2048, "output_len": 2})
            for rid in low_rids
        ]
        inc70 = ops.next_request_id(base)
        specs.append((inc70, {"priority": 70, "input_len": 2048, "output_len": 2}))
        q90b = ops.next_request_id(base)
        specs.append((q90b, {"priority": 90, "input_len": 2048, "output_len": 2}))
        q90c = ops.next_request_id(base)
        specs.append((q90c, {"priority": 90, "input_len": 2048, "output_len": 2}))
        wave1 = _fire_batch(ops, specs)
        fires.extend(wave1)
        inc70_fire = wave1[8]
        q90b_fire = wave1[9]
        q90c_fire = wave1[10]

        outcomes1 = _drain(ops, [ph90_fire] + wave1)
        m1 = _outcome_map(outcomes1)
        inc70_code = m1[inc70][1]
        inc70_reason = None
        if inc70_fire.resp is not None:
            inc70_reason = int(inc70_fire.resp.admission_reject_reason)
        victims8400 = [rid for rid in low_rids if m1[rid][1] == CODE_YIELDED]
        # [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
        # (6ad0315f10): every wave submitter parks; the attributed form
        # (8430 + HIGHER_PRIORITY_AHEAD) needs the AdmissionFailureClassifier
        # at the queued-expiry decision, which has ZERO call sites in the
        # intake3 master (Java-side observation gap, filed).  Observable
        # design-final form: every parked expiry is uniform plain
        # 8511 + UNSPECIFIED, zero 8400 victims, the placeholder completes.
        w1_wave = low_rids + [inc70, q90b, q90c]
        w1_expired = all(m1[rid][1] == CODE_SLO_EXPIRED for rid in w1_wave)
        w1_reasons_unspec = all(
            fr.reason == REASON_UNSPECIFIED for fr in wave1 if fr.resp is not None
        )
        w1_ok = w1_expired and w1_reasons_unspec and victims8400 == [] and m1[ph90][0]
        report.invariant(
            "PR7",
            w1_ok,
            context="higher_priority_ahead_design_final",
            detail=(
                f"[EV-1-FIXED] baseline flipped at intake3 "
                f"PendingPlacementCoordinator (6ad0315f10): incoming70 "
                f"terminal={inc70_code} "
                f"reason={REASON_NAMES.get(inc70_reason, inc70_reason)} "
                f"(park-expiry uniform: the 8430 attribution classifier has "
                f"zero call sites in the intake3 master — Java gap, filed), "
                f"all-wave expired 8511={w1_expired}, "
                f"reasons UNSPECIFIED={w1_reasons_unspec}, "
                f"victims8400={len(victims8400)}, "
                f"90b={m1[q90b][1]}/{q90b_fire.reason}, "
                f"90c={m1[q90c][1]}/{q90c_fire.reason}, "
                f"90a completed={m1[ph90][0]}"
            ),
        )
        # deadline-not-extended: under EV-1 the 70 has no queue residency
        # at all (fast route-reject) — the no-extension property needs an
        # admitted 70 (Java behaviour gap, filed with EV-1).
        # A9-3 (Mark P3-1): the fast-reject latency is NOT a deadline
        # observation; recording it under the PR8 band drifted the
        # property's calibre (prio_queue_timeout_terminal stays PR8's ONLY
        # band consumer).  Raw value carried in the case finish detail.
        inc70_wall_ms = (inc70_fire.settled_s - inc70_fire.submitted_s) * 1000.0
        clean1_ok, clean1_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        if not clean1_ok:
            return report.finish(
                f"wave1 inflight dirty: {clean1_detail}, " f"grades: {report.summary()}"
            )

        # ---- wave 2: attributed timeout, weak SAME/RESOURCE form -------
        # Placeholder prefill must outlast the 70_late's ~8.5s deadline
        # (deadline cancels at dispatch — see the case docstring).
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=10_000.0)
        time.sleep(PERF_SETTLE_S)

        ph70 = ops.next_request_id(base)
        ph70_fire = _fire(ops, ph70, priority=70, input_len=2048, output_len=2)
        fires.append(ph70_fire)
        if not ph70_fire.ok:
            return False, f"wave2 70_early failed: code={ph70_fire.code}"
        if not _poll_engine_pending(ops, prefill_names[0], 1):
            return False, "wave2 70_early never dispatched"

        low2_rids = [ops.next_request_id(base) for _ in range(8)]
        specs2 = [
            (rid, {"priority": 30, "input_len": 2048, "output_len": 2})
            for rid in low2_rids
        ]
        inc70l = ops.next_request_id(base)
        specs2.append((inc70l, {"priority": 70, "input_len": 2048, "output_len": 2}))
        wave2 = _fire_batch(ops, specs2)
        fires.extend(wave2)
        inc70l_fire = wave2[8]

        outcomes2 = _drain(ops, [ph70_fire] + wave2)
        m2 = _outcome_map(outcomes2)
        inc70l_code = m2[inc70l][1]
        inc70l_reason = (
            int(inc70l_fire.resp.admission_reject_reason)
            if inc70l_fire.resp is not None
            else None
        )
        victims2 = [rid for rid in low2_rids if m2[rid][1] == CODE_YIELDED]
        # [EV-1-FIXED] (see wave 1): the SAME/RESOURCE attribution branches
        # share the classifier's zero-call-site gap; the design-final
        # observable is the same uniform 8511/UNSPECIFIED park expiry.
        w2_wave = low2_rids + [inc70l]
        w2_expired = all(m2[rid][1] == CODE_SLO_EXPIRED for rid in w2_wave)
        w2_reasons_unspec = all(
            fr.reason == REASON_UNSPECIFIED for fr in wave2 if fr.resp is not None
        )
        w2_ok = w2_expired and w2_reasons_unspec and victims2 == [] and m2[ph70][0]
        report.invariant(
            "PR7",
            w2_ok,
            context="same_or_resource_weak_form_design_final",
            detail=(
                f"[EV-1-FIXED] baseline flipped at intake3 "
                f"PendingPlacementCoordinator (6ad0315f10): incoming70_late "
                f"terminal={inc70l_code} "
                f"reason={REASON_NAMES.get(inc70l_reason, inc70l_reason)} "
                f"(uniform park expiry — the SAME/RESOURCE attribution "
                f"branches share the classifier's zero-call-site gap, "
                f"Java-side, filed), all-wave expired 8511={w2_expired}, "
                f"reasons UNSPECIFIED={w2_reasons_unspec}, "
                f"victim8400={len(victims2)}, "
                f"70_early completed={m2[ph70][0]}"
            ),
        )
        clean2_ok, clean2_detail = AssertUtils.inflight_clean(_master_http(ops), 30.0)
        report.invariant(
            "P6",
            m1[ph90][0]
            and m2[ph70][0]
            and clean2_ok
            and victims8400 == []
            and victims2 == [],
            detail=(
                f"[EV-1-FIXED] placeholders completed, every parked wave "
                f"member expired 8511 (uniform), zero eviction victims, "
                f"inflight={'ok' if clean2_ok else clean2_detail}"
            ),
        )
        return report.finish(
            f"wave1 70={inc70_code}/{REASON_NAMES.get(inc70_reason)}, "
            f"wave2 70={inc70l_code}/{REASON_NAMES.get(inc70l_reason)}, "
            f"[EV-1-FIXED] inc70 park-expiry wall={inc70_wall_ms:.0f}ms "
            f"(deadline held, no extension — PR8 raw), "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _finally_hygiene(ops, fires, prefill_names)
