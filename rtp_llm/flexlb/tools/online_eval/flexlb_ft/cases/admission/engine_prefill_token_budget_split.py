from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...harness import AssertUtils, wait_for
from ...registry import case
from ...support.admission import (
    _drain_fired,
    _fire_regroup_wave,
    _ledger_series_ok,
    _lifecycle_rows,
    _master_http,
    _park_settled,
    _prefill_batch_counters,
    _prefill_names,
    _regroup_spec,
    _shape_gate,
)


@case(
    "engine_prefill_token_budget_split",
    category="admission",
    profiles=["batch-window"],
    requires=["enqueue_batch"],
    source=(
        "engine regroup #8: in-engine dual-budget prefill regroup "
        "(over-budget master batch split + closed ledger)"
    ),
)
def engine_prefill_token_budget_split(ctx: CaseContext):
    """Engine-internal token-budget regroup: the VERDICT is the master's
    linkage to whatever shape the engine actually executed.

    Scenario: dedicated 1P+2D env, prefill.max_batch_tokens=1024 (the
    request dimension off), flat prefill.fixed_ms=3000.  Four
    512-token requests fire 10ms apart — all inside the master's 100ms
    collection window — so the engine receives ONE four-member master
    batch whose total logical tokens (4 x 512 = 2048, sum of
    computeTokens + hitTokens) is 2x the budget.

    Construction (gate, not verdict): the engine-side regroup composer
    (production FIFOScheduler.cc:371-481 semantics — the budget is a
    STOP, members join while admitted < budget) fills the execution
    batch with the first two arrivals, parks the tail members as one
    PrefillPendingBatch in prefillPendingQueue and admits them FIFO
    when the running batch drains — the executed-batch counters (delta
    over the pre-fire baseline) should grow by 2 batches / 4 requests
    with max size 2.  A shape MISS does not fail the case: the master
    assertions still run against the ACTUAL executed shape (that is the
    tested value); the deviation is recorded in the detail.

    Verdict (master linkage, the point of the case):
      * ledger identity — every member's request_lifecycle row still
        carries the SAME master batch_id (the split never rewrites it);
      * booking caliber — through the split window the master's
        inflight_batches peaks at exactly 1 (it books the ONE batch it
        dispatched; the engine-side split never multiplies master
        bookkeeping) with inflight_requests peaking at 4;
      * event-driven digestion — the member accounting steps DOWN
        through an intermediate plateau (prefix settles, tail still
        executing) instead of jumping to zero in one reconcile: the
        master tracks the engine's per-execution-batch completions
        live, not at TTL/expiry time;
      * scheduler_inflight never climbs mid-window (parked tail work
        executes inside the engine — the master must not re-admit or
        re-dispatch anything);
      * the park settles empty, the master inflight ledger drains clean
        and a fresh request succeeds (recovery).

    Prediction: expected to pass — the master's EnqueueBatch ledger is
    keyed to the batch it dispatched and reconciles engine completion
    facts per member (PrefillState.reconcileWorkerStatus), so the split
    is invisible in batch count and visible as a step-down in member
    count.
    """
    env = ctx.env_manager.ensure(_regroup_spec(ctx, 1024, 0))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        base_counters = _prefill_batch_counters(ops, names[0])
        rids, fired, fire_errors = _fire_regroup_wave(ops, base)

        # Park: while the prefix [r1, r2] executes its 3s window the
        # tail [r3, r4] sits in prefillPendingQueue.
        park_max = 0
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            snap = ops.snapshot_by_name()
            for n in names:
                park_max = max(
                    park_max,
                    int(snap.get(n, {}).get("prefill_waiting_batches", 0)),
                    int(snap.get(n, {}).get("waiting", 0)),
                )
            if park_max >= 1:
                break
            time.sleep(0.2)

        # Master ledger linkage — sample through the split window: the
        # prefix executes 3s, the tail another 3s, so ~8s of sampling
        # covers both completion steps (fire returned AFTER the
        # EnqueueBatch ACK, so the first sample already sees the full 4).
        samples, peak_batches = AssertUtils.inflight_batches_peak(
            _master_http(ops), "prefill", window_s=8.0, interval_s=0.2
        )
        ledger_ok, ledger_detail = _ledger_series_ok(
            samples, peak_batches, n_requests=len(rids), expect_intermediate=True
        )

        outcomes = _drain_fired(ops, fired, wait_s=45.0)
        completed = [rid for rid, ok, _ in outcomes if ok]
        drain_errors = [(rid, err) for rid, ok, err in outcomes if not ok]

        # Counters AFTER the drain, BEFORE verify_recovery's probe —
        # construction gate only (see docstring).
        after_counters = _prefill_batch_counters(ops, names[0])
        delta_batches = after_counters[0] - base_counters[0]
        delta_requests = after_counters[1] - base_counters[1]
        shape_ok, shape_detail = _shape_gate(
            delta_batches, delta_requests, after_counters[2], (2, 4, 2)
        )

        # Ledger identity: all four members still attribute to the ONE
        # master batch (EnqueueBatch-time batch_id is request-level —
        # the split never rewrites it).
        rows = _lifecycle_rows(ops, names[0], rids)
        member_batch_ids = {
            row.get("batch_id") if row else None for row in rows.values()
        }
        batch_id_ok = len(member_batch_ids) == 1 and next(
            iter(member_batch_ids), None
        ) not in (None, 0, -1)

        settled = wait_for(lambda: _park_settled(ops, names), 10.0, 0.2)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            not fire_errors
            and park_max >= 1
            and len(completed) == len(rids)
            and not drain_errors
            and batch_id_ok
            and ledger_ok
            and settled
            and inflight_ok
            and recovery_ok
        )
        return passed, (
            f"fired={len(rids)} (fire_errors={fire_errors[:1]}), "
            f"park_observed_max={park_max}, "
            f"completed={len(completed)}/{len(rids)} "
            f"(drain_errors={drain_errors[:1]}), "
            f"{shape_detail}, "
            f"master_linkage={ledger_ok}({ledger_detail}), "
            f"member_batch_ids={member_batch_ids}, "
            f"park_settled_empty={settled}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
