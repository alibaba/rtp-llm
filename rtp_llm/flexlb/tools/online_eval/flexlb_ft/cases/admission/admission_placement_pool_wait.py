from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from ...context import CaseContext, rid_base
from ...engine_ops import engine_inflight_clean
from ...harness import AssertUtils, wait_for
from ...registry import case
from ...support.admission import (
    _await_tracked,
    _decode_names,
    _fire_tracked,
    _master_http,
    _ParkedSampler,
    _pool_wait_spec,
    _prefill_names,
)


@case(
    "admission_placement_pool_wait",
    profiles=["single-nonbatch", "window-nonbatch"],
    requires=["generate_stream"],
    source="admission wave-2 A4 (verdict §4.1 rebuild): prefill placement capacity wait — real backlog via delivery-lease cap 1 + queue cap 2, concurrent park sampling, serialized completion (name kept for history)",
    category="admission",
)
def admission_placement_pool_wait(ctx: CaseContext):
    """Prefill placement capacity wait: a REAL backlog construction
    (verdict §4.1; the case name keeps its historical "pool_wait" form
    from the pre-intake3 availability-filter contract).

    Scenario: dedicated 1P+2D env on the NON_BATCH base (single-nonbatch
    and window-nonbatch lanes — the request-level delivery lease is the
    same knob on both) with
    dispatcher.maxInflightRequestsPerPrefillWorker=1 and
    scheduler.capacity.maxWaitingRequestsPerPrefillWorker=2, all under
    prefill_fixed_ms=5000.  Request A is fired first; once A is
    OBSERVABLY RUNNING on the engine AND still live on the master
    ledger (both asserted — the precondition below), request B
    arrives.

    Mechanism under test (verdict §1.1/§2, the two-level master
    prefill park): the dispatcher delivery lease (1) is held by A, so
    B's route cannot dispatch — the planning frontier
    (availableBatchPublicationCredits = maxWaiting - activeIndex)
    blocks B from being selected while the WorkerBatcher ACTIVE queue
    holds its seats, and the capacity-blocked submitter parks as a
    Blocked request (BlockedRequestIndex.parkFrontier) with its
    Schedule RPC still OPEN.  The parked request stays on the master
    ledger (RequestRegistry liveRequestCount) and is absent from every
    engine snapshot until capacity frees (settleUnderLock ->
    signalPlacementCapacityChanged wakes the retry).

    Expected (contract): A completes; B's Schedule RPC parks OPEN —
    fire RPC duration > 0.5s (the parked-with-open-RPC evidence; the
    2026-09-04 false-green run measured 0.02s precisely because the
    no-op knob created no edge) — and the CONCURRENT background
    sampler observes master_side_parked >= 1 during B's park window
    (scheduler ledger carries B while no engine does: a HARD assertion
    now — sampling only after the fires settle structurally misses the
    park window, verdict §3); B completes AFTER A (end gap > 0 — B's
    prefill starts only at A's lease release; FIFO); no leakage
    (master + engine ledgers clean) and a fresh request succeeds
    (recovery).

    Prediction: A e2e ~5.5s (5s prefill + decode), B's fire RPC ~5s
    (parks until A's lease releases), parked_max=1 during the park
    window, B e2e ~10.5s (gap ~5s).  Risk: the pre-B precondition
    (A still holding the lease) remains the fragile link — the 5s
    prefill plus the explicit ledger+running check close the window,
    and a breach fails loudly instead of silently.
    """
    env = ctx.env_manager.ensure(_pool_wait_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    names = _prefill_names(ops)
    decode_names = _decode_names(ops)
    if not names:
        return False, "no prefill engines found"
    fired: list = []
    try:
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=5000.0)

        # A takes the single delivery lease and runs.
        rid_a = ops.next_request_id(base)
        fire_err_a = _fire_tracked(ops, rid_a, fired, input_len=512, output_len=2)
        if fire_err_a is not None:
            return False, f"request A fire failed: {fire_err_a}"

        def a_running() -> bool:
            snap = ops.snapshot_by_name()
            return any(int(snap.get(n, {}).get("running", 0)) >= 1 for n in names)

        if not wait_for(a_running, 10.0, 0.1):
            return False, "request A never reached RUNNING on prefill"

        # PRECONDITION (fail loudly, task #107 #18): B must arrive while
        # A REALLY still holds the lease — A running on the engine AND
        # live on the master ledger.  A fast snapshot that observed a
        # stale running fact (A already settled, pending 1 -> 0) would
        # let B route straight through — the old silent not-parking
        # failure.
        ledger_before_b = ops.master_scheduler_inflight()
        snap_before_b = ops.snapshot_by_name()
        a_running_now = any(
            int(snap_before_b.get(n, {}).get("running", 0)) >= 1 for n in names
        )
        if ledger_before_b < 1 or not a_running_now:
            engine_state = {
                n: (
                    snap_before_b.get(n, {}).get("running", -1),
                    snap_before_b.get(n, {}).get("waiting", -1),
                )
                for n in names
            }
            return False, (
                f"precondition failed before B fire: A must still hold the "
                f"lease (ledger={ledger_before_b}, engine_running="
                f"{a_running_now}, engines={engine_state}) — firing B now "
                f"would be vacuous (the lease is free)"
            )

        # B arrives while A runs: its Schedule RPC parks OPEN (Blocked
        # placement holds the RPC until the lease releases), so B's fire
        # runs on a worker thread while the main thread concurrently
        # samples the parked ledger — the park window lives DURING the
        # fire, and a post-fire sampler structurally misses it (verdict
        # §3, the 2026-09-04 parked_max=0 defect).
        rid_b = ops.next_request_id(base)
        sampler = _ParkedSampler(ops, names, decode_names).start()
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            t_b_call = time.monotonic()
            b_future = pool.submit(
                _fire_tracked,
                ops,
                rid_b,
                fired,
                timeout_s=60.0,
                input_len=512,
                output_len=2,
            )
            fire_err_b = b_future.result()  # settles at the lease release
        finally:
            pool.shutdown(wait=True)
            sampler.stop()
        b_fire_rpc_s = time.monotonic() - t_b_call

        outcomes = _await_tracked(fired, wait_s=30.0)
        if len(outcomes) != 2:
            return False, f"expected 2 tracked outcomes, got {len(outcomes)}"
        (_, t0a, end_a, ok_a, err_a) = outcomes[0]
        (_, t0b, end_b, ok_b, err_b) = outcomes[1]
        both_completed = ok_a and err_a is None and ok_b and err_b is None

        # Park evidence (hard): the sampler observed B on the master
        # ledger while no engine carried it, and B's Schedule RPC stayed
        # open past the decision path (> 0.5s — the parked-RPC proof;
        # an immediately-dispatched B settles in tens of ms).
        parked_max = sampler.max_parked
        b_parked_rpc = b_fire_rpc_s > 0.5
        # Serialization (FIFO): B's terminal strictly follows A's — B's
        # prefill cannot start before A's lease release, so a parallel
        # completion (gap <= 0) would falsify the capacity contract.
        b_after_a = end_b > end_a

        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        engine_clean, engine_detail = engine_inflight_clean(
            ops, names + decode_names, 15.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            fire_err_b is None
            and both_completed
            and parked_max >= 1
            and b_parked_rpc
            and b_after_a
            and inflight_ok
            and engine_clean
            and recovery_ok
        )
        return passed, (
            f"a_completed={ok_a and err_a is None} "
            f"(e2e={end_a - t0a:.2f}s), "
            f"b_completed={ok_b and err_b is None} "
            f"(e2e={end_b - t0b:.2f}s, fire_err={fire_err_b}, "
            f"fire_rpc={b_fire_rpc_s:.2f}s — parked-open-RPC), "
            f"b_after_a_gap={end_b - end_a:.2f}s (serialized), "
            f"master_side_parked_max={parked_max} "
            f"({sampler.max_detail}, samples={len(sampler.samples)}, "
            f"http_failures={sampler.http_failures}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"engine_clean={engine_clean}({engine_detail}), "
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
