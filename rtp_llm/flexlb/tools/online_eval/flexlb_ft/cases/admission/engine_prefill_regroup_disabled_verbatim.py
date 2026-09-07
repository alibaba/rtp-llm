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
    "engine_prefill_regroup_disabled_verbatim",
    category="admission",
    profiles=["batch-window"],
    requires=["enqueue_batch"],
    source=(
        "engine regroup #8: in-engine dual-budget prefill regroup "
        "(0/0 disables regroup — legacy verbatim master batches)"
    ),
)
def engine_prefill_regroup_disabled_verbatim(ctx: CaseContext):
    """The 0/0 switch: regroup off, the master batch executes as-is —
    the VERDICT is the master's single-batch bookkeeping, the engine
    shape as gate.

    Scenario: prefill.max_batch_tokens=0 AND
    prefill.max_batch_requests=0 — the documented off switch: both
    dimensions zero disables the in-engine regroup entirely and the
    master batch executes verbatim (the pre-#8 behaviour).

    Construction (gate, not verdict): the executed-batch counters
    (delta over the pre-fire baseline) grow by exactly 1 batch /
    4 requests with max size 4 — the verbatim master shape (4x the
    tokens a 1024 budget would have split).  Gate misses are recorded
    in the detail, not the verdict; the master assertions still run
    against the ACTUAL executed shape.

    Verdict (master linkage, the point of the case):
      * booking caliber — through the execution window the master's
        inflight_batches peaks at exactly 1 and inflight_requests at
        exactly 4, with NO intermediate plateau: the verbatim batch
        settles ATOMICALLY (one reconcile step 4->0), and
        scheduler_inflight never climbs — the off-switch must not make
        the master re-dispatch or re-admit anything;
      * ledger identity — every member's lifecycle row attributes to
        the SAME master batch_id;
      * no park, and no master misbehaviour from it: a fresh request
        admits normally after the wave (recovery), the ledger drains
        clean (inflight_clean).

    Prediction: expected to pass — the disabled path is the preserved
    legacy code (prefillRegroupEnabled() == false), pinned in-JVM by
    PrefillBudgetRegroupTest.regroupOffReproducesVerbatimMasterBatch.
    """
    env = ctx.env_manager.ensure(_regroup_spec(ctx, 0, 0))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        base_counters = _prefill_batch_counters(ops, names[0])
        rids, fired, fire_errors = _fire_regroup_wave(ops, base)

        no_park = True
        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline:
            snap = ops.snapshot_by_name()
            if any(
                int(snap.get(n, {}).get("prefill_waiting_batches", 0)) > 0
                for n in names
            ):
                no_park = False
                break
            time.sleep(0.2)

        # Master ledger linkage — single verbatim batch: 3s execution
        # plus settle margin in ~5s of sampling; the atomic settle (no
        # intermediate member count) pins that the master books the ONE
        # batch it dispatched and releases it in one reconcile step.
        samples, peak_batches = AssertUtils.inflight_batches_peak(
            _master_http(ops), "prefill", window_s=5.0, interval_s=0.2
        )
        ledger_ok, ledger_detail = _ledger_series_ok(
            samples, peak_batches, n_requests=len(rids), expect_intermediate=False
        )

        outcomes = _drain_fired(ops, fired, wait_s=45.0)
        completed = [rid for rid, ok, _ in outcomes if ok]
        drain_errors = [(rid, err) for rid, ok, err in outcomes if not ok]

        after_counters = _prefill_batch_counters(ops, names[0])
        delta_batches = after_counters[0] - base_counters[0]
        delta_requests = after_counters[1] - base_counters[1]
        shape_ok, shape_detail = _shape_gate(
            delta_batches, delta_requests, after_counters[2], (1, 4, 4)
        )

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
            and no_park
            and len(completed) == len(rids)
            and not drain_errors
            and ledger_ok
            and batch_id_ok
            and settled
            and inflight_ok
            and recovery_ok
        )
        return passed, (
            f"fired={len(rids)} (fire_errors={fire_errors[:1]}), "
            f"no_park_through_window={no_park}, "
            f"completed={len(completed)}/{len(rids)} "
            f"(drain_errors={drain_errors[:1]}), "
            f"{shape_detail}, "
            f"master_linkage(atomic_1b4r)={ledger_ok}({ledger_detail}), "
            f"member_batch_ids={member_batch_ids}, "
            f"park_settled_empty={settled}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
