from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...harness import AssertUtils, wait_for
from ...registry import case
from ...support.admission import (
    BQ_PARK_REQUESTS,
    _await_tracked,
    _batcher_queue_spec,
    _decode_names,
    _fire_tracked,
    _master_http,
    _ParkedSampler,
    _prefill_names,
)


@case(
    "admission_batcher_queue_capacity_park",
    category="admission",
    profiles=["batch-window"],
    requires=["enqueue_batch"],
    source=(
        "admission wave-2 A5: master batcher-queue capacity gate "
        "(maxWaitingRequestsPerPrefillWorker park — waitable, no fast reject)"
    ),
)
def admission_batcher_queue_capacity_park(ctx: CaseContext):
    """Master batcher-queue capacity gate: the gate is a WAIT condition.

    Scenario: dedicated 1P+2D env with the batcher waiting-queue capacity
    tightened to 2 (scheduler.capacity.maxWaitingRequestsPerPrefillWorker
    — the Java default is 1024); prefill_fixed_ms=3000 stretches each
    batch.  Seven requests are fired 0.4s apart (each its own batch, 40x
    the 10ms collection window): the dispatcher lease window
    (maxInflightBatchesPerPrefillWorker=4) carries fires 1-4 onto the
    engine (1 running + 3 engine-side pending), fires 5-6 fill the master
    batcher queue to its capacity-2 ceiling, and fire 7 finds the queue
    full.

    Behaviour (new-B semantics): prefill admission is a Blocked WAIT —
    capacity trouble parks the request (coordinator placementWaiters)
    instead of rejecting it, and the parked retry rides the capacity-
    changed signal.  The whole wave therefore terminates successfully:
    the overflow requests queue/park, survive (the 60s queueTimeoutMs
    is far above the drain), and complete in FIFO order.

    Expected (contract): all seven schedules succeed (zero fast
    rejects — the waitable-gate contract); every request reaches its
    terminal as a COMPLETED stream with NON-DECREASING end times
    (FIFO); after the drain the engine park is empty, the master
    inflight ledger is clean and a fresh request succeeds (recovery).
    The master-side parked count (scheduler ledger minus engine-live)
    is a HARD assertion sampled by a background thread DURING the
    fire (verdict §3): fires 5-6 occupy the batcher queue and fire 7
    parks with its Schedule RPC open, so a 0.2s-cadence sampler that
    starts BEFORE the first fire must observe parked >= 1 while the
    wave is in flight — the 2026-09-04 run measured parked_max=0 only
    because its observation loop started AFTER the fires settled,
    structurally past the park window.  The drain-span number is an
    observation only: `ends` are the wait_end() return instants, so a
    wave that drains before the await starts collapses the measured
    span to ~0 regardless of the engine's internal serialization (the
    old >= 12s span proof only measured anything while the await raced
    a still-live drain).

    Prediction: 7/7 completed, zero rejects, parked_max >= 1 (queued
    fires 5-6 + parked fire 7 ride the master ledger while only the 4
    lease holders show on the engine), clean ledgers, recovery ok.
    Risk: none identified — the assertion now pins the invariant half
    that holds at any load level under Blocked admission PLUS the
    in-flight park observation the sampler window makes honest.
    """
    env = ctx.env_manager.ensure(_batcher_queue_spec(ctx, queue_timeout_ms=60_000))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    names = _prefill_names(ops)
    decode_names = _decode_names(ops)
    if not names:
        return False, "no prefill engines found"
    fired: list = []
    try:
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=3000.0)

        # The park window lives DURING the fire (verdict §3): fires 5-6
        # queue on the master batcher and fire 7 parks with its Schedule
        # RPC open, all of which drains once capacity frees — so the
        # sampler starts BEFORE the first fire and runs across the wave
        # and the drain; a post-fire observation loop structurally
        # misses the window (the 2026-09-04 parked_max=0 defect).
        sampler = _ParkedSampler(ops, names, decode_names).start()
        fire_errors = []
        try:
            for _ in range(BQ_PARK_REQUESTS):
                rid = ops.next_request_id(base)
                err = _fire_tracked(ops, rid, fired, input_len=512, output_len=2)
                if err is not None:
                    fire_errors.append((rid, err))
                time.sleep(0.4)  # >> maxCollectionWaitMs: one batch per fire
        finally:
            sampler.stop()

        outcomes = _await_tracked(fired, wait_s=45.0)
        completed = [rid for rid, _, _, ok, _ in outcomes if ok]
        failures = [(rid, err) for rid, _, _, ok, err in outcomes if not ok]
        ends = [end for _, _, end, _, _ in outcomes]
        # Batch-aware FIFO: same-batch members terminate together — order
        # is non-decreasing, not strictly increasing.  `ends` are the
        # wait_end() return instants, so a wave that drains before the
        # await starts collapses the span to ~0; the span/min_gap below
        # are observations only (see docstring).
        fifo_ordered = all(ends[i] <= ends[i + 1] for i in range(len(ends) - 1))
        drain_span = (max(ends) - min(ends)) if ends else 0.0

        def engine_park_empty() -> bool:
            snap = ops.snapshot_by_name()
            return all(
                int(snap.get(n, {}).get("prefill_waiting_batches", 0)) == 0
                and int(snap.get(n, {}).get("waiting", 0)) == 0
                for n in names
            )

        settled = wait_for(engine_park_empty, 10.0, 0.2)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        # Park evidence (hard, verdict §3): the in-flight sampler must
        # have seen the queued/parked wave on the master ledger while
        # the engines carried only the lease holders.
        parked_max = sampler.max_parked
        parked_proven = parked_max >= 1
        passed = (
            not fire_errors
            and parked_proven
            and len(completed) == BQ_PARK_REQUESTS
            and not failures
            and fifo_ordered
            and settled
            and inflight_ok
            and recovery_ok
        )
        return passed, (
            f"fired={BQ_PARK_REQUESTS} (fire_errors={fire_errors[:1]}), "
            f"master_side_parked_max={parked_max} "
            f"({sampler.max_detail}, samples={len(sampler.samples)}, "
            f"http_failures={sampler.http_failures}, in-flight sampled — "
            f"hard), "
            f"completed={len(completed)}/{BQ_PARK_REQUESTS} "
            f"(failures={failures[:1]}), "
            f"fifo_ordered={fifo_ordered} "
            f"(span={drain_span:.2f}s, "
            f"min_gap={min((ends[i + 1] - ends[i] for i in range(len(ends) - 1)), default=0.0):.2f}s, "
            f"observations), "
            f"engine_park_settled_empty={settled}, "
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
