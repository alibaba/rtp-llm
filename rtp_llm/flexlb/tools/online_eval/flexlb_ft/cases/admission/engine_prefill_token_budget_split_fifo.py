from __future__ import annotations

from ...context import CaseContext, rid_base
from ...harness import AssertUtils, wait_for
from ...registry import case
from ...support.admission import (
    _drain_fired_collect,
    _drain_fired_start,
    _fire_regroup_wave,
    _ledger_series_ok,
    _lifecycle_rows,
    _master_http,
    _park_settled,
    _prefill_batch_counters,
    _prefill_names,
    _regroup_spec,
    _shape_gate,
    _two_cluster_split,
)


@case(
    "engine_prefill_token_budget_split_fifo",
    category="admission",
    profiles=["batch-window"],
    requires=["enqueue_batch"],
    source=(
        "engine regroup #8: in-engine dual-budget prefill regroup "
        "(split preserves arrival order across execution batches)"
    ),
)
def engine_prefill_token_budget_split_fifo(ctx: CaseContext):
    """The split preserves arrival order — VERDICT on the client-visible
    completion chain + the master's linkage, engine order as gate.

    Scenario: identical config to engine_prefill_token_budget_split
    (1024-token budget, flat 3000ms prefill, four 512-token requests
    in one master batch) — the spec fingerprint matches, so ensure()
    reuses the very same env; this case pins the ORDER contract on a
    fresh wave of rids.

    Construction (gate, not verdict): the composer admits members while
    admitted < budget — the first two ARRIVALS form execution batch #1,
    the rest parks until that batch drains; the engine lifecycle end_ms
    must show TWO serial execution batches (two-cluster separation,
    _two_cluster_split — the concurrent wave's arrival order is
    nondeterministic, so the split is NOT rids[:2] vs rids[2:]).  The
    executed-batch counters should grow 2b/4r/max2.  Gate misses are
    recorded in the detail, not the verdict.

    Verdict (master/client linkage, the point of the case):
      * CLIENT two-batch chain — consumed CONCURRENTLY (each stamp is
        the true completion instant), the four client completion
        stamps cluster into the same TWO serial batches with ~3s
        separation: arrival-order execution must propagate to what
        the client sees, and a master re-dispatch / an engine
        reshuffle / an interleaved composition would each collapse
        the two-cluster structure;
      * the same master ledger linkage as the split case: peak
        inflight_batches == 1, member accounting stepping down through
        an intermediate plateau, scheduler_inflight never climbing;
      * the master inflight ledger is clean and a fresh request succeeds.

    Prediction: expected to pass — the serial two-batch structure is
    structural (the parked tail only starts after the running batch
    drains) and the ~3s execution gap dwarfs any scheduling jitter.
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

        # CLIENT FIFO needs completion timestamps stamped at the ACTUAL
        # completion instants, so consumption starts NOW — while the
        # streams are still executing (the prefix batch is ~3s from
        # done).  The workers block on their streams; the master-ledger
        # sampling below runs in the main thread in parallel.
        pool, futures = _drain_fired_start(ops, fired, wait_s=45.0)

        # Master ledger linkage through the split window (same window as
        # the split case: prefix 3s + tail 3s).
        samples, peak_batches = AssertUtils.inflight_batches_peak(
            _master_http(ops), "prefill", window_s=8.0, interval_s=0.2
        )
        ledger_ok, ledger_detail = _ledger_series_ok(
            samples, peak_batches, n_requests=len(rids), expect_intermediate=True
        )

        # Collect the concurrent consumers — done stamps are the true
        # completion instants, not the caller's consume order.
        outcomes = _drain_fired_collect(futures)
        pool.shutdown(wait=True)
        completed = [rid for rid, ok, _, _ in outcomes if ok]
        drain_errors = [(rid, err) for rid, ok, err, _ in outcomes if not ok]
        done_ts = {rid: done for rid, _, _, done in outcomes}
        # CLIENT two-batch chain (arrival-order robust): the four stamps
        # must cluster into two serial batches, NOT rids[:2] before
        # rids[2:] — the wave's arrival order is nondeterministic.
        client_two_batch_ok, client_two_batch_detail = _two_cluster_split(
            [done_ts.get(r) for r in rids], sep=1.0
        )

        after_counters = _prefill_batch_counters(ops, names[0])
        delta_batches = after_counters[0] - base_counters[0]
        delta_requests = after_counters[1] - base_counters[1]
        shape_ok, shape_detail = _shape_gate(
            delta_batches, delta_requests, after_counters[2], (2, 4, 2)
        )

        # ENGINE order (construction gate): the lifecycle end_ms must show
        # the same two serial execution batches (two-cluster separation —
        # same arrival-order caveat as the client leg).
        rows = _lifecycle_rows(ops, names[0], rids)
        end_ms = {
            rid: int(row.get("end_ms", 0)) if row else 0 for rid, row in rows.items()
        }
        engine_two_batch_ok, engine_two_batch_detail = _two_cluster_split(
            list(end_ms.values()), sep=1000.0
        )

        settled = wait_for(lambda: _park_settled(ops, names), 10.0, 0.2)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 30.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            not fire_errors
            and len(completed) == len(rids)
            and not drain_errors
            and client_two_batch_ok
            and ledger_ok
            and settled
            and inflight_ok
            and recovery_ok
        )
        return passed, (
            f"fired={len(rids)} (fire_errors={fire_errors[:1]}), "
            f"completed={len(completed)}/{len(rids)} "
            f"(drain_errors={drain_errors[:1]}), "
            f"client_two_batches={client_two_batch_ok}({client_two_batch_detail}), "
            f"{shape_detail}, "
            f"engine_two_batches(gate)={engine_two_batch_ok}"
            f"({engine_two_batch_detail}), "
            f"master_linkage={ledger_ok}({ledger_detail}), "
            f"park_settled_empty={settled}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
