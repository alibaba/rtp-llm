from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...harness import AssertUtils, wait_for
from ...registry import case
from ...support.admission import (
    _await_tracked,
    _fire_tracked,
    _master_http,
    _prefill_names,
    _waiting_cap_spec,
)


@case(
    "admission_engine_waiting_batch_cap_reject",
    profiles=["batch-window", "single-batch"],
    requires=["enqueue_batch"],
    source="admission wave-3 B3: engine prefill waiting-queue cap (max_waiting_batches whole-batch backpressure reject — the non-waitable complement of W1's unbounded park)",
    category="admission",
)
def admission_engine_waiting_batch_cap_reject(ctx: CaseContext):
    """Engine waiting-batch cap gate: the cap is NOT a wait condition.

    Scenario: dedicated 1P+2D env (the W1 shape); prefill_fixed_ms=3000
    stretches each batch's execution window and /set_perf applies the
    runtime cap max_waiting_batches=1 (>0 = queued-batch ceiling, 0 =
    unbounded — the default).  Three requests are fired 0.4s apart (each
    its own batch, 40x the 10ms collection window): batch #1 is admitted
    running, batch #2 parks in prefillPendingQueue (waiting = 1 = cap —
    a snapshot poll proves the saturated state BEFORE the probe fires),
    and batch #3 finds waiting >= cap at schedulePrefillCompletion.

    Behaviour: the cap hit is a WHOLE-BATCH backpressure reject, not a
    park — schedulePrefillCompletion returns false before claiming any
    counter, every member of batch #3 is rolled back and the
    EnqueueBatch ack carries the batch-level error "prefill waiting
    queue full (backpressure): waiting=1 cap=1"; DefaultBatchDispatcher
    wraps it as EngineRejectedException ("EnqueueBatch rejected request
    N: prefill waiting queue full ...") and the master completes the
    request terminal — synchronous, typed, no queueing.  The cap counts
    QUEUED batches only (running is not charged), so the two in-flight
    occupants are untouched.

    Expected (contract): the probe terminates FAST (< 3s from fire,
    no park residence) with the backpressure error family ("prefill
    waiting queue full" + "backpressure") in its terminal error; the
    occupants (1 running + 1 queued) complete normally; the gate
    RECOVERS UNDER THE SAME PRESSURE — with batches #1/#2 still
    occupying the engine, /set_perf max_waiting_batches=0 (unbounded)
    lets a fourth fire park in the queue and run to completion; after
    the drain the engine park is empty, the master inflight ledger is
    clean and a fresh request succeeds (recovery).

    Prediction: expected to pass — the runtime override, the cap check
    (waiting >= cap rejects before claiming any counter) and the
    ack-error surface are covered by SetPerfMaxWaitingBatchesTest (4/4
    green); the only novel wiring is the master's
    EngineRejectedException-to-terminal path, already exercised by the
    enqueue-ack fault family.  Risk: batch coalescing collapsing the
    fires into fewer batches — mitigated by the 0.4s inter-fire gap and
    the pre-probe waiting>=1 observation (the case fails loudly rather
    than probing an unsaturated cap).
    """
    env = ctx.env_manager.ensure(_waiting_cap_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    fired: list = []
    try:
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=3000.0, max_waiting_batches=1)

        # Occupants: batch #1 running (3s window), batch #2 queued —
        # together they saturate waiting=1=cap.
        fire_errors = []
        for _ in range(2):
            rid = ops.next_request_id(base)
            err = _fire_tracked(ops, rid, fired, input_len=512, output_len=2)
            if err is not None:
                fire_errors.append((rid, err))
            time.sleep(0.4)  # >> maxCollectionWaitMs: one batch per fire
        if fire_errors:
            return False, f"occupant fire failed: {fire_errors[:1]}"

        # Cap-state proof BEFORE the probe: batch #2 sits in
        # prefillPendingQueue (waiting=1=cap).
        def cap_saturated() -> bool:
            snap = ops.snapshot_by_name()
            return all(
                int(snap.get(n, {}).get("prefill_waiting_batches", 0)) >= 1
                for n in names
            )

        cap_observed = wait_for(cap_saturated, 8.0, 0.1)
        if not cap_observed:
            return (
                False,
                "batch #2 never reached the waiting queue (cap never saturated)",
            )

        # Probe batch #3: whole-batch backpressure reject.  The master
        # surfaces the engine's EnqueueBatch reject SYNCHRONOUSLY on the
        # Schedule RPC — code 8510, "Delivery failed: EnqueueBatch rejected
        # request N: prefill waiting queue full (backpressure): waiting=1
        # cap=1" (DefaultBatchDispatcher wraps the ack error and the RPC
        # returns it to the caller) — so the probe accepts EITHER surface:
        # the RPC-level typed reject, or a successful fire whose stream
        # then terminates with the backpressure family.
        rid3 = ops.next_request_id(base)
        r3_t0 = time.monotonic()
        r3_err = ""
        try:
            resp3 = ops.schedule(rid3, input_len=512, output_len=2)
            if resp3.code != 200 or not resp3.success:
                r3_err = f"schedule failed ({resp3.code}): " f"{resp3.error_message}"
            else:
                handle3 = ops.start_stream(resp3, rid3)
                ended3 = handle3.wait_end(10.0)
                r3_err = str(handle3.snap.error or "") if ended3 else "no terminal"
        except Exception as exc:
            r3_err = repr(exc)
        reject_latency = time.monotonic() - r3_t0
        rejected = (
            "prefill waiting queue full" in r3_err.lower()
            and "backpressure" in r3_err.lower()
            and reject_latency < 3.0
        )

        # Gate recovery UNDER THE SAME PRESSURE: relax the cap while
        # batch #1 still runs and batch #2 still queues — the next fire
        # must park in the (now unbounded) queue and complete.
        for n in names:
            ops.set_perf(n, max_waiting_batches=0)
        snap_at_r4 = ops.snapshot_by_name()
        waiting_at_r4 = max(
            int(snap_at_r4.get(n, {}).get("prefill_waiting_batches", 0)) for n in names
        )
        rid4 = ops.next_request_id(base)
        fire_err4 = _fire_tracked(ops, rid4, fired, input_len=512, output_len=2)

        outcomes = _await_tracked(fired, wait_s=45.0)
        occupant_ok = len(outcomes) >= 2 and all(
            ok and err is None for _, _, _, ok, err in outcomes[:2]
        )
        # rid4 is fired[-1] whenever its fire succeeded; a synchronously
        # rejected probe (rid3) never enters `fired`, so index by identity,
        # not by ordinal.
        recovers_ok = (
            fire_err4 is None
            and len(fired) >= 3
            and outcomes[-1][0] == rid4
            and outcomes[-1][3]
            and outcomes[-1][4] is None
        )

        def park_empty() -> bool:
            snap = ops.snapshot_by_name()
            return all(
                int(snap.get(n, {}).get("prefill_waiting_batches", 0)) == 0
                and int(snap.get(n, {}).get("waiting", 0)) == 0
                for n in names
            )

        settled = wait_for(park_empty, 10.0, 0.2)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            cap_observed
            and rejected
            and occupant_ok
            and recovers_ok
            and settled
            and inflight_ok
            and recovery_ok
        )
        return passed, (
            f"cap_observed={cap_observed}, "
            f"probe_rejected={rejected} "
            f"(latency={reject_latency:.2f}s, err={r3_err[:80]}), "
            f"occupants_completed={occupant_ok}, "
            f"gate_recovers_under_pressure={recovers_ok} "
            f"(waiting_at_r4={waiting_at_r4}, fire_err4={fire_err4}), "
            f"park_settled_empty={settled}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            for n in names:
                ops.set_perf(n, prefill_fixed_ms=100.0, max_waiting_batches=0)
        except Exception:
            pass
