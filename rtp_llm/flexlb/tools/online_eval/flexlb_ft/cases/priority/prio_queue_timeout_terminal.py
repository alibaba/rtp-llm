from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...grade import GradeReport
from ...registry import case
from ...support.priority import (
    CODE_SLO_EXPIRED,
    PERF_SETTLE_S,
    _drain,
    _finally_hygiene,
    _fire,
    _fire_batch,
    _poll_engine_pending,
    _prefill_names,
    _t1_spec,
)


@case(
    "prio_queue_timeout_terminal",
    category="priority",
    profiles=["single-nonbatch"],
    source="design §2.2 #5 — PR8(band) + P6",
)
def prio_queue_timeout_terminal(ctx: CaseContext):
    """queueTimeout as an absolute deadline (PR8 band + P6): sustained
    high-priority pressure must terminal the queued low-priority requests
    AT the deadline — never suspended past it (design §3.4 row 6,
    passive half: repeated juggling/queue-jumping never extends
    expiresAtMs; the active priorityAdmission half lives in
    atpm_timeout_attribution).

    ENV-T1: queueTimeout 8s, no preemption (the plain-timeout path — no
    priorityAdmission, so no 8430 attribution in this env), maxWaiting 8,
    inflight cap 1.

    Choreography (calibrated from the design's 70x3x4s sketch): 70a
    placeholder (10000ms) holds the lease; then 30a, 30b, 30c, 70b submit
    in one batch.  [EV-1-FIXED] Under the intake3 pull-based coordinator
    (PendingPlacementCoordinator, 6ad0315f10) the whole wave parks — and
    with prefill 10s > queueTimeout 8s > submit window ~0.7s, EVERY wave
    request's absolute deadline fires before the first lease release:
    all four settle 8511 BATCH_SLO_EXPIRED at enqueue+8s (30a ≈8.01s,
    70b ≈8.7s), zero route-reject.  The E10 calibration form (prefill
    deliberately beyond the deadline so the parked head provably expires
    AT its absolute deadline) carries over to every parked request.

    Assertions: PR8 band = max low-priority terminal wall-time / 8000ms
    (strict 1.25 — the latest submitter's deadline lands ≈1.07, the
    absolute-deadline proof: no suspension, no extension); low terminals
    all typed 8511 (implementation-period correction: the design's
    {8503, 8402, 8430} assumed QUEUE_TIMEOUT 8503 is the plain-path
    code, but 8503 is dead code in the master — the ordinary
    queued-expiry terminal is BATCH_SLO_EXPIRED 8511,
    RequestSlot.deadlineErrorType configured at registration); 70a
    succeeds; P6 every request reaches a terminal (no suspension).
    """
    env = ctx.env_manager.ensure(_t1_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "priority")
    fires: list = []
    prefill_names: list = []
    try:
        prefill_names = _prefill_names(ops)
        # E10 calibration: prefill 10s > queueTimeout 8s so every parked
        # request provably expires at its absolute deadline (probe E10:
        # 8511 at wall=8.01s).  [EV-1-FIXED] under the pull model the whole
        # wave parks: 30a/30b/30c and 70b all settle 8511 at their own
        # enqueue+8s deadlines before the t=10s lease release; 70a
        # (dispatched t=0) completes.
        for name in prefill_names:
            ops.set_perf(name, prefill_fixed_ms=10_000.0)
        time.sleep(PERF_SETTLE_S)

        h1 = ops.next_request_id(base)
        h1_fire = _fire(ops, h1, priority=70, input_len=2048, output_len=2)
        fires.append(h1_fire)
        if not h1_fire.ok:
            return False, f"70a schedule failed: code={h1_fire.code}"
        if not _poll_engine_pending(ops, prefill_names[0], 1):
            return False, "70a never dispatched"

        low_rids = [ops.next_request_id(base) for _ in range(3)]
        h2 = ops.next_request_id(base)
        specs = [
            (rid, {"priority": 30, "input_len": 2048, "output_len": 2})
            for rid in low_rids
        ]
        specs.append((h2, {"priority": 70, "input_len": 2048, "output_len": 2}))
        wave = _fire_batch(ops, specs)
        fires.extend(wave)
        low_fires = wave[:3]
        h2_fire = wave[3]

        outcomes = _drain(ops, fires)
        by_rid = {rid: (ok, code) for (rid, ok, code, _detail) in outcomes}

        # [EV-1-FIXED] baseline flipped at intake3 PendingPlacementCoordinator
        # (6ad0315f10): the whole wave parks (pull-based) and, with prefill
        # 10s > queueTimeout 8s > submit window ~0.7s, EVERY wave request's
        # absolute deadline fires before the first lease release — all
        # four settle 8511 BATCH_SLO_EXPIRED at enqueue+8s (30a ≈8.01s,
        # 70b ≈8.7s), none suspended past its deadline, zero route-reject.
        h1_ok = by_rid[h1][0]
        low_codes = [by_rid[rid][1] for rid in low_rids]
        wave_all_expired = (
            all(code == CODE_SLO_EXPIRED for code in low_codes)
            and by_rid[h2][1] == CODE_SLO_EXPIRED
        )
        max_low_s = max(fr.settled_s - fr.submitted_s for fr in low_fires)
        ratio = max_low_s / 8.0
        report.check(
            "PR8",
            ratio,
            context="queue_timeout_terminal",
            detail=(
                f"[EV-1-FIXED] all three 30s settle 8511 at their own "
                f"enqueue+8s deadlines (max wall {max_low_s * 1000:.0f}ms "
                f"/ 8000ms, absolute — no suspension, no extension), "
                f"wave codes="
                f"{[(rid % 1_000_000, c) for rid, c in zip(low_rids, low_codes)]}, "
                f"70b={by_rid[h2][1]} (parked; deadline before first release)"
            ),
        )
        report.invariant(
            "P6",
            h1_ok and wave_all_expired,
            detail=(
                f"[EV-1-FIXED] 70a ok={h1_ok}, whole wave 8511="
                f"{wave_all_expired} (park-to-deadline terminals, zero "
                f"route-reject), no suspension (queueTimeout absolute)"
            ),
        )
        return report.finish(
            f"ratio={ratio:.2f}, low codes={low_codes}, 70b={by_rid[h2][1]} "
            f"[EV-1-FIXED], grades: {report.summary()}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        _finally_hygiene(ops, fires, prefill_names)
