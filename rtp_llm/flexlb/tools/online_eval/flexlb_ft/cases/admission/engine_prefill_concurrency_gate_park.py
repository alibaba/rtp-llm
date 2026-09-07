from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...harness import AssertUtils, wait_for
from ...registry import case
from ...support.admission import (
    _drain_fired,
    _fire_request,
    _master_http,
    _prefill_names,
    _prefill_park_spec,
)


@case(
    "engine_prefill_concurrency_gate_park",
    profiles=["batch-window", "single-batch"],
    requires=["enqueue_batch"],
    source="admission wave-2 W1: engine prefill-concurrency gate (maxPrefillConcurrency park — wait condition, no reject)",
    category="admission",
)
def engine_prefill_concurrency_gate_park(ctx: CaseContext):
    """Engine prefill-concurrency gate: the gate is a WAIT condition.

    Scenario: dedicated 1P+2D env.  The single prefill engine runs with
    the production-locked maxPrefillConcurrency=1 (no /set_perf override
    — the lock is the point); prefill_fixed_ms=3000 stretches each
    batch's execution window.  Four requests are fired ~0.4s apart —
    far beyond maxCollectionWaitMs=10, so each arrives as its OWN batch,
    and the dispatcher's maxInflightBatchesPerPrefillWorker=4 window
    lets all four EnqueueBatch deliveries reach the engine while batch
    #1 is still running.

    Behaviour: JavaMockEngineCluster.schedulePrefillCompletion admits
    batch #1 (activePrefillBatches 0 -> 1) and parks batches #2-4 whole
    in the unbounded prefillPendingQueue (max_waiting_batches default
    0 = unlimited) — a WAIT, not a rejection: no request-level error is
    surfaced.  As each running batch goes terminal the parked queue
    head is admitted FIFO.

    Expected (contract): every fired request schedules successfully
    (zero rejections — the park absorbs the pressure); a snapshot poll
    observes prefill_waiting_batches >= 1 (the park is real); after the
    drain every fired request reached its terminal as a COMPLETED
    stream (FIFO order), the park is empty (waiting == 0 and
    prefill_waiting_batches == 0), the master inflight ledger is clean
    and a fresh request succeeds (recovery).

    Prediction: expected to pass — engine-side park semantics were
    ported with the v2 mock (schedulePrefillCompletion) and the batch
    lease window (4) strictly exceeds the concurrency window (1), so
    the park is deterministic.  Risk: FIXED_WINDOW coalescing —
    mitigated by the 0.4s inter-fire gap (40x the collection window).
    """
    env = ctx.env_manager.ensure(_prefill_park_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    fired: list = []
    try:
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=3000.0)

        fire_errors = []
        rids = []
        for _ in range(4):
            rid = ops.next_request_id(base)
            rids.append(rid)
            err = _fire_request(ops, rid, fired, input_len=512, output_len=2)
            if err is not None:
                fire_errors.append((rid, err))
            time.sleep(0.4)  # >> maxCollectionWaitMs: one batch per fire

        # Observe the park: batch #1 runs 3s, batches #2-4 sit in
        # prefillPendingQueue — poll before the first terminal.
        park_max = 0
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            snap = ops.snapshot_by_name()
            for n in names:
                info = snap.get(n, {})
                park_max = max(
                    park_max,
                    int(info.get("prefill_waiting_batches", 0)),
                    int(info.get("waiting", 0)),
                )
            if park_max >= 1:
                break
            time.sleep(0.2)

        outcomes = _drain_fired(ops, fired, wait_s=45.0)
        completed = [rid for rid, ok, _ in outcomes if ok]
        drain_errors = [(rid, err) for rid, ok, err in outcomes if not ok]

        # Park must settle back to empty after the drain.
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
            not fire_errors
            and park_max >= 1
            and len(completed) == len(rids)
            and not drain_errors
            and settled
            and inflight_ok
            and recovery_ok
        )
        return passed, (
            f"fired={len(rids)} (fire_errors={fire_errors[:1]}), "
            f"park_observed_max={park_max}, "
            f"completed={len(completed)}/{len(rids)} "
            f"(drain_errors={drain_errors[:1]}), "
            f"park_settled_empty={settled}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            for n in names:
                ops.set_perf(n, prefill_fixed_ms=100.0)
        except Exception:
            pass
