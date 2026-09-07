from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...harness import AssertUtils, _ttft_p50, wait_for
from ...registry import case
from ...support.admission import (
    _ledger_series_ok,
    _master_http,
    _park_settled,
    _prefill_batch_counters,
    _prefill_names,
    _regroup_spec,
    _shape_gate,
    _timed_request,
    _timed_wave_start,
)


@case(
    "engine_prefill_token_budget_boundary",
    category="admission",
    profiles=["batch-window"],
    requires=["enqueue_batch"],
    source=(
        "engine regroup #8: in-engine dual-budget prefill regroup "
        "(boundary — a batch exactly at the token budget is NOT split)"
    ),
)
def engine_prefill_token_budget_boundary(ctx: CaseContext):
    """Boundary: a batch exactly AT the token budget executes verbatim —
    the VERDICT is the master's single-batch bookkeeping + TTFT shape
    neutrality, the engine shape as gate.

    Scenario: prefill.max_batch_tokens=2048 — exactly the total of
    four 512-token requests coalesced into one master batch.

    Construction (gate, not verdict): production admission semantics
    (FIFOScheduler.cc:371-481) — members join while admitted < budget
    (strict), so the fourth member (admitted 1536 < 2048) still fits and
    the whole batch should execute as ONE (counters 1b/4r/max4, no park
    through the execution window).  Gate misses are recorded in the
    detail, not the verdict.

    Verdict (master linkage, the point of the case):
      * booking caliber — through the execution window the master's
        inflight_batches peaks at exactly 1 and inflight_requests at
        exactly 4, and the member accounting shows NO intermediate
        plateau: the verbatim batch settles ATOMICALLY (one reconcile
        step 4->0 — any intermediate value means the master split its
        own bookkeeping or lost members one by one);
      * TTFT shape neutrality — the batched wave's client-visible
        completion durations must not degrade beyond 50% against a
        single-request baseline on the same env (AssertUtils.
        ttft_degradation): batch shape must not cost latency;
      * scheduler_inflight never climbs mid-window;
      * the master inflight ledger drains clean and a fresh request
        succeeds (recovery).

    Prediction: expected to pass — the boundary condition is pinned
    in-JVM by PrefillBudgetRegroupTest.
    boundaryExactBudgetBatchDoesNotSplit.
    """
    env = ctx.env_manager.ensure(_regroup_spec(ctx, 2048, 0))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "admission")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        # Shape-neutral TTFT baseline: ONE 512-token request consumed to
        # terminal (its duration is the single-request reference the
        # batched wave is graded against).
        baseline_rid = ops.next_request_id(base)
        baseline_err, baseline_dur = _timed_request(
            ops, baseline_rid, input_len=512, output_len=2
        )
        if baseline_err:
            return False, f"baseline request failed: {baseline_err}"

        base_counters = _prefill_batch_counters(ops, names[0])

        # The timed wave: four 512-token requests fired 10ms apart (the
        # _fire_regroup_wave coalescing shape), each worker measuring its
        # OWN schedule->end duration concurrently — the streams open while
        # the batch still executes, so the durations are live TTFT
        # readings (a post-hoc drain would stamp the caller's consume
        # order instead of the completion instants).
        rids, pool, futures = _timed_wave_start(
            ops, base, 4, input_len=512, output_len=2
        )

        # No park: the single verbatim batch admits immediately —
        # poll through its 3s execution window.
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

        # Master ledger linkage — single verbatim batch: the 3s execution
        # plus settle margin fits in ~5s of sampling; expect_intermediate
        # is FALSE (atomic settle of the whole batch).
        samples, peak_batches = AssertUtils.inflight_batches_peak(
            _master_http(ops), "prefill", window_s=5.0, interval_s=0.2
        )
        ledger_ok, ledger_detail = _ledger_series_ok(
            samples, peak_batches, n_requests=len(rids), expect_intermediate=False
        )

        # Collect the timed wave — (err, duration_s) per member, each
        # worker's own schedule->end reading.
        wave_results = []
        for future in futures:
            try:
                wave_results.append(future.result())
            except Exception as exc:
                wave_results.append((repr(exc), None))
        pool.shutdown(wait=True)
        wave_errors = [err for err, _ in wave_results if err]
        wave_durs_ms = [
            dur * 1000.0 for err, dur in wave_results if not err and dur is not None
        ]

        after_counters = _prefill_batch_counters(ops, names[0])
        delta_batches = after_counters[0] - base_counters[0]
        delta_requests = after_counters[1] - base_counters[1]
        shape_ok, shape_detail = _shape_gate(
            delta_batches, delta_requests, after_counters[2], (1, 4, 4)
        )

        # TTFT shape neutrality (client-visible caliber — schedule ->
        # stream end; under BATCH dispatch the mock surfaces the first
        # output at fetch completion, so the duration IS the TTFT the
        # client sees): the verbatim 4-member wave graded against the
        # single-request baseline through the shared degradation gate.
        baseline_p50_ms = baseline_dur * 1000.0
        wave_p50_ms = _ttft_p50(wave_durs_ms)
        ttft_ok, ttft_detail = AssertUtils.ttft_degradation(
            baseline_p50_ms, wave_p50_ms
        )

        settled = wait_for(lambda: _park_settled(ops, names), 10.0, 0.2)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            not wave_errors
            and no_park
            and len(wave_durs_ms) == len(rids)
            and ledger_ok
            and ttft_ok
            and settled
            and inflight_ok
            and recovery_ok
        )
        return passed, (
            f"wave={len(rids)} (errors={wave_errors[:1]}), "
            f"no_park_through_window={no_park}, "
            f"completed={len(wave_durs_ms)}/{len(rids)}, "
            f"{shape_detail}, "
            f"master_linkage(atomic_1b4r)={ledger_ok}({ledger_detail}), "
            f"ttft_neutral={ttft_ok}({ttft_detail}), "
            f"park_settled_empty={settled}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
