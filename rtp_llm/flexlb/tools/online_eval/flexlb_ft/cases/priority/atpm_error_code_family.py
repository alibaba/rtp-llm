from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...harness import AssertUtils
from ...registry import case
from ...support.priority import (
    CODE_ENGINE_CANCELLED,
    CODE_NO_DECODE,
    CODE_NO_PREFILL,
    CODE_OK,
    CODE_QUEUE_FULL,
    CODE_RESOURCE_EXHAUSTED,
    CODE_SLO_EXPIRED,
    CODE_YIELDED,
    PERF_SETTLE_S,
    REASON_NAMES,
    ROUTE_REJECT_FAMILY,
    _a1_spec,
    _c1_spec,
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
    "atpm_error_code_family",
    category="priority",
    profiles=["single-nonbatch"],
    source="design §2.4 #12 — AT4 + P6",
)
def atpm_error_code_family(ctx: CaseContext):
    """Error-code family separation (AT4): each admission failure code
    appears only under its own trigger condition, and the three segments
    never cross-contaminate.

    Segment 1 (8502 QUEUE_FULL, ENV-C1 maxOutstanding=2, G11b-isomorphic):
    two slow placeholders hold both global outstanding permits; two
    arrivals (priority 30 and 70 — the GLOBAL cap exempts no priority)
    fail submit's outstanding acquire → completeError(QUEUE_FULL) as a
    synchronous fast-reject.  The single-argument Response.error path
    leaves admission_reject_reason=UNSPECIFIED(0) (code-level finding;
    the actual pair is recorded for first-e2e calibration, per the
    design's "8502 vs 8431 presentation needs first-run calibration"
    note — the code-level expectation here is 8502 on the outstanding
    path).  After the placeholders drain, a sequential request succeeds
    (exact permit release).

    Segment 2 (capacity park, ENV-Q2 shared; [EV-1-FIXED] flipped at
    intake3 PendingPlacementCoordinator 6ad0315f10): a 70 placeholder
    parks the inflight lease, eight 70s + the incoming 90 ALL park —
    the {8402, 8510} route-reject family lost its capacity-blocked
    trigger (maxWaiting's enqueueUnderLock cap is a BATCH-path check
    the NON_BATCH pull model never reaches, so no enqueue ever fails
    and the tryFallback path to 8510 never runs).  Zero victims, all
    nine complete 200 with the 90 dispatching first among the wave
    (priority desc; first parker 70a keeps slot one); the explicit-cap
    rejection observation lives in segment 1's 8502.

    Segment 3 (expiry uniformity, ENV-A1 shared; [EV-1-FIXED] flipped
    at intake3): a 70_early placeholder (10s prefill — it must OUTLAST
    every queue deadline, the deadline-cancels-at-dispatch finding)
    parks the lease; 30a..30h + the incoming 90 all park and every one
    expires at its own 7s deadline as plain 8511 BATCH_SLO_EXPIRED +
    UNSPECIFIED (QUEUE_TIMEOUT 8503 is dead code; the 8431 +
    RESOURCE_EXHAUSTED attributed form needs the expiry-time
    classifier, which has zero call sites in the intake3 master —
    Java-side observation gap, filed).  The 70_early completes (its
    deadline cancelled at delivery ACK).

    Segment 4 (A4, Mark P1-2, SKELETON — BATCH dispatcher family,
    reserved not constructed): 8514 BATCH_TOKEN_CAPACITY / 8515
    SCHEDULER_PLAN_CONFLICT only fire on the BATCH dispatcher, which the
    current case base (SINGLE + NON_BATCH) never enters; filled
    when a priority-batch variant enables BATCH dispatch.

    Cross-segment isolation (AT4, per segment): segment-1 terminals
    contain no 8402/8403/8431/8400/8429/8511; segment-2 no
    8502/8403/8431/8400/8429/8511; segment-3 no 8502/8402/8403/8510.
    """
    # A4 (Mark P1-2): batch-dispatch caliber reservation, following the
    # dual-caliber paradigm (is_batch = ctx.batch_dispatch();
    # completion-duration caliber under BATCH, client-TTFT under NON_BATCH).
    # The 8514/8515 segment-4 codes below are BATCH-dispatcher-only, so
    # the whole segment is unreachable under this case's profile
    # (single-nonbatch = NON_BATCH base, PRIORITY axis injected at the case
    # layer); the arm is reserved so a priority-batch variant fills it
    # without touching the NON_BATCH segments below.
    if ctx.batch_dispatch():
        # TODO(A4): BATCH arm — segment 4 becomes live (8514 group token
        # capacity / 8515 plan conflict); fill when a priority-batch
        # variant enables BATCH dispatch.
        raise NotImplementedError(
            "atpm_error_code_family BATCH arm reserved — fill when a "
            "priority-batch variant enables BATCH dispatch"
        )
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    segs = []

    # ---- segment 1: 8502 QUEUE_FULL (ENV-C1) ---------------------------
    env1 = ctx.env_manager.ensure(_c1_spec(ctx))
    ops1 = ctx.engine_ops(env1)
    fires1: list = []
    names1: list = []
    try:
        names1 = _prefill_names(ops1)
        for name in names1:
            ops1.set_perf(name, prefill_fixed_ms=4000.0)
        time.sleep(PERF_SETTLE_S)

        ph_a = ops1.next_request_id(base)
        ph_b = ops1.next_request_id(base)
        ph_fires = _fire_batch(
            ops1,
            [
                (ph_a, {"priority": 50, "input_len": 2048, "output_len": 2}),
                (ph_b, {"priority": 50, "input_len": 2048, "output_len": 2}),
            ],
        )
        fires1.extend(ph_fires)
        if not all(f.ok for f in ph_fires):
            return False, f"seg1 placeholders failed: {[f.code for f in ph_fires]}"
        if not _poll_engine_pending(ops1, names1[0], 1):
            return False, "seg1 placeholders never dispatched"

        # Both outstanding permits are held from submit time; the next two
        # arrivals — low and high priority alike — must fast-reject 8502.
        rej_lo = ops1.next_request_id(base)
        rej_hi = ops1.next_request_id(base)
        rej_fires = _fire_batch(
            ops1,
            [
                (rej_lo, {"priority": 30, "input_len": 2048, "output_len": 2}),
                (rej_hi, {"priority": 70, "input_len": 2048, "output_len": 2}),
            ],
        )
        fires1.extend(rej_fires)

        m1 = _outcome_map(_drain(ops1, ph_fires + rej_fires))
        rej_codes = [m1[rej_lo][1], m1[rej_hi][1]]
        rej_fast = all(f.settled_s - f.submitted_s < 3.0 for f in rej_fires)
        rej_reasons = [f.reason for f in rej_fires]
        placeholders_ok = m1[ph_a][0] and m1[ph_b][0]
        isolated1 = all(
            c
            not in (
                CODE_NO_PREFILL,
                CODE_NO_DECODE,
                CODE_RESOURCE_EXHAUSTED,
                CODE_YIELDED,
                CODE_ENGINE_CANCELLED,
                CODE_SLO_EXPIRED,
            )
            for c in rej_codes + [m1[ph_a][1], m1[ph_b][1]]
        )

        for name in names1:
            ops1.set_perf(name, prefill_fixed_ms=100.0)
        recovery_ok, recovery_detail = ops1.verify_recovery()
        clean1_ok, clean1_detail = AssertUtils.inflight_clean(_master_http(ops1), 30.0)
        segs.append(
            (
                "s1_8502_outstanding",
                rej_codes == [CODE_QUEUE_FULL, CODE_QUEUE_FULL]
                and rej_fast
                and placeholders_ok
                and isolated1
                and recovery_ok
                and clean1_ok,
                (
                    f"rejected codes={rej_codes} (expected [8502, 8502]), "
                    f"reasons={rej_reasons} (expected [0, 0] UNSPECIFIED), "
                    f"fast={rej_fast}, placeholders completed={placeholders_ok}, "
                    f"isolated={isolated1}, recovery={recovery_ok}"
                    f"({recovery_detail[:60]}), "
                    f"inflight={'ok' if clean1_ok else clean1_detail}"
                ),
            )
        )
    finally:
        _finally_hygiene(ops1, fires1, names1)

    # ---- segment 2: {8402, 8510} route-reject family (ENV-Q2) ----------
    env2 = ctx.env_manager.ensure(_q2_spec(ctx))
    ops2 = ctx.engine_ops(env2)
    fires2: list = []
    names2: list = []
    try:
        names2 = _prefill_names(ops2)
        for name in names2:
            ops2.set_perf(name, prefill_fixed_ms=3000.0)
        time.sleep(PERF_SETTLE_S)

        ph2 = ops2.next_request_id(base)
        ph2_fire = _fire(ops2, ph2, priority=70, input_len=2048, output_len=2)
        fires2.append(ph2_fire)
        if not ph2_fire.ok:
            return False, f"seg2 placeholder failed: code={ph2_fire.code}"
        if not _poll_engine_pending(ops2, names2[0], 1):
            return False, "seg2 placeholder never dispatched"

        high_rids = [ops2.next_request_id(base) for _ in range(8)]
        specs2 = [
            (rid, {"priority": 70, "input_len": 2048, "output_len": 2})
            for rid in high_rids
        ]
        inc90 = ops2.next_request_id(base)
        specs2.append((inc90, {"priority": 90, "input_len": 2048, "output_len": 2}))
        wave2 = _fire_batch(ops2, specs2)
        fires2.extend(wave2)

        m2 = _outcome_map(_drain(ops2, [ph2_fire] + wave2))
        inc90_code = m2[inc90][1]
        zero_eviction = all(
            m2[rid][1] not in (CODE_YIELDED, CODE_ENGINE_CANCELLED)
            for rid in high_rids + [inc90]
        )
        # [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
        # (6ad0315f10): the route-reject family {8402, 8510} has no
        # capacity-blocked trigger left — every submitter parks, so the 90
        # completes 200 after the wave (priority desc; first parker 70a
        # keeps slot one).  The tryFallback path to 8510 needs a failed
        # enqueue, which the NON_BATCH pull model never produces
        # (maxWaiting is a BATCH-path cap — the equivalent explicit-cap
        # rejection observation lives in segment 1's 8502).  Zero
        # evictions; all nine complete.
        s2_wave = high_rids + [inc90]
        prio2 = {rid: 70 for rid in high_rids}
        prio2[inc90] = 90
        _s2_first, s2_shape, s2_order = _design_final_pattern(
            ops2, [ph2_fire] + wave2, s2_wave, prio2
        )
        s2_all_ok = all(m2[rid][0] for rid in high_rids) and m2[inc90][0]
        isolated2 = all(
            m2[rid][1]
            not in (
                CODE_QUEUE_FULL,
                CODE_NO_DECODE,
                CODE_RESOURCE_EXHAUSTED,
                CODE_YIELDED,
                CODE_ENGINE_CANCELLED,
                CODE_SLO_EXPIRED,
            )
            for rid in high_rids + [ph2, inc90]
        )
        clean2_ok, clean2_detail = AssertUtils.inflight_clean(_master_http(ops2), 30.0)
        segs.append(
            (
                "s2_capacity_park_design_final",
                inc90_code == CODE_OK
                and zero_eviction
                and s2_shape
                and s2_all_ok
                and m2[ph2][0]
                and isolated2
                and clean2_ok,
                (
                    f"[EV-1-FIXED] incoming90 terminal={inc90_code} "
                    f"(parks and completes — the "
                    f"{list(ROUTE_REJECT_FAMILY)} route-reject family lost "
                    f"its capacity-blocked trigger at intake3 "
                    f"PendingPlacementCoordinator 6ad0315f10; the explicit-"
                    f"cap rejection observation lives in s1's 8502), "
                    f"shape ok={s2_shape}, "
                    f"dispatch={[r % 1_000_000 for r in s2_order]}, "
                    f"zero 8400/8429={zero_eviction}, "
                    f"all nine completed={s2_all_ok}, "
                    f"placeholder completed={m2[ph2][0]}, "
                    f"isolated={isolated2}, "
                    f"inflight={'ok' if clean2_ok else clean2_detail}"
                ),
            )
        )
    finally:
        _finally_hygiene(ops2, fires2, names2)

    # ---- segment 3: 8431 + RESOURCE_EXHAUSTED (ENV-A1) -----------------
    env3 = ctx.env_manager.ensure(_a1_spec(ctx))
    ops3 = ctx.engine_ops(env3)
    fires3: list = []
    names3: list = []
    try:
        names3 = _prefill_names(ops3)
        # The placeholder must OUTLAST every queue deadline (deadline
        # cancels at delivery ACK — the queued items never dispatch).
        for name in names3:
            ops3.set_perf(name, prefill_fixed_ms=10_000.0)
        time.sleep(PERF_SETTLE_S)

        ph70 = ops3.next_request_id(base)
        ph70_fire = _fire(ops3, ph70, priority=70, input_len=2048, output_len=2)
        fires3.append(ph70_fire)
        if not ph70_fire.ok:
            return False, f"seg3 70_early failed: code={ph70_fire.code}"
        if not _poll_engine_pending(ops3, names3[0], 1):
            return False, "seg3 70_early never dispatched"

        low_rids = [ops3.next_request_id(base) for _ in range(8)]
        specs3 = [
            (rid, {"priority": 30, "input_len": 2048, "output_len": 2})
            for rid in low_rids
        ]
        inc90b = ops3.next_request_id(base)
        specs3.append((inc90b, {"priority": 90, "input_len": 2048, "output_len": 2}))
        wave3 = _fire_batch(ops3, specs3)
        fires3.extend(wave3)
        inc90b_fire = wave3[8]

        m3 = _outcome_map(_drain(ops3, [ph70_fire] + wave3))
        inc90b_code = m3[inc90b][1]
        inc90b_reason = (
            int(inc90b_fire.resp.admission_reject_reason)
            if inc90b_fire.resp is not None
            else None
        )
        victims8400 = [rid for rid in low_rids if m3[rid][1] == CODE_YIELDED]
        plain8511 = [rid for rid in low_rids if m3[rid][1] == CODE_SLO_EXPIRED]
        ph70_ok = m3[ph70][0]
        # [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
        # (6ad0315f10): the 90 parks (rather than route-rejecting) and
        # expires at its own 7s deadline — 8511 + UNSPECIFIED, the same
        # uniform park-expiry terminal as every 30.  The 8431 +
        # RESOURCE_EXHAUSTED attributed form needs the expiry-time
        # classifier, which has zero call sites in the intake3 master
        # (Java-side observation gap, filed); 8503 stays dead code.
        s3_wave = low_rids + [inc90b]
        s3_expired = all(m3[rid][1] == CODE_SLO_EXPIRED for rid in s3_wave)
        isolated3 = all(
            m3[rid][1]
            not in (
                CODE_QUEUE_FULL,
                CODE_NO_DECODE,
                CODE_YIELDED,
                CODE_ENGINE_CANCELLED,
            )
            for rid in low_rids + [ph70, inc90b]
        )
        clean3_ok, clean3_detail = AssertUtils.inflight_clean(_master_http(ops3), 30.0)
        segs.append(
            (
                "s3_expiry_uniformity_design_final",
                inc90b_code == CODE_SLO_EXPIRED
                and s3_expired
                and victims8400 == []
                and ph70_ok
                and isolated3
                and clean3_ok,
                (
                    f"[EV-1-FIXED] incoming90 terminal={inc90b_code} "
                    f"reason={REASON_NAMES.get(inc90b_reason, inc90b_reason)} "
                    f"(park expiry at its own 7s deadline — the 8431 + "
                    f"RESOURCE_EXHAUSTED attributed form needs the expiry-"
                    f"time classifier, zero call sites in the intake3 "
                    f"master, Java gap filed), all-wave expired 8511="
                    f"{s3_expired}, "
                    f"victim8400={len(victims8400)}, "
                    f"plain8511={len(plain8511)} (8503 stays dead code), "
                    f"70_early completed={ph70_ok}, isolated={isolated3}, "
                    f"inflight={'ok' if clean3_ok else clean3_detail}"
                ),
            )
        )
    finally:
        _finally_hygiene(ops3, fires3, names3)

    # ---- segment 4 (A4, Mark P1-2): BATCH dispatcher family — SKELETON --
    # Reserved, not constructed: CODE_BATCH_TOKEN_CAPACITY (8514, group
    # token capacity exceeded) and CODE_SCHEDULER_PLAN_CONFLICT (8515,
    # plan conflict) only fire on the BATCH dispatcher, which the
    # current case base (SINGLE + NON_BATCH) never enters.  Fill
    # when a priority-batch variant enables BATCH dispatch —
    # expected shape: saturate maxWaitingRequestsPerGroup so an incoming
    # over the group token budget rejects 8514; force a concurrent plan
    # mutation for 8515; keep the cross-segment isolation table growing
    # (segment-4 terminals must contain none of segments 1-3's codes).
    # Constants live at the module head next to the code-family block.

    try:
        report.invariant(
            "AT4",
            all(ok for (_l, ok, _d) in segs),
            context="error_code_family_separation",
            detail="; ".join(
                f"{l}={'ok' if ok else 'FAIL(' + d + ')'}" for l, ok, d in segs
            ),
        )
        report.invariant(
            "P6",
            all(ok for (_l, ok, _d) in segs),
            detail="every segment drained to terminals with inflight clean",
        )
        return report.finish(
            f"segments={[l for l, ok, _d in segs if ok]}, "
            f"grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
