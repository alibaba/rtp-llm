from __future__ import annotations

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, inject_type_all
from ...harness import TTL_DRAIN_TIMEOUT_S, wait_for
from ...registry import case
from ...support.status import (
    LONG_STREAM_TIMEOUT_S,
    _log_count,
    _master_ok,
    _prefill_batches_sum,
    _prefill_names,
    _run_requests,
    _status_spec,
    _timeout_typed,
    _ttl_anchor_deltas,
    _ttl_counter_observe,
    _wait_scheduler_zero,
)


@case(
    "status_prefill_suppress_all",
    category="status",
    profiles=["batch-window"],
    source="P0 status fault family: status_suppress_running+finished on every prefill",
)
def status_prefill_suppress_all(ctx: CaseContext):
    """Scenario: every prefill suppresses BOTH the running and the finished
    facts (status_suppress_running + status_suppress_finished) — the status
    channel goes fully silent for those tasks while the requests are live.

    Behaviour: with no ACTIVE fact the slot's lastWorkerStatusAtMs freezes,
    so the stale-inflight TTL (30s) is the ONLY ledger exit; the requests
    themselves terminate (success if the data plane stays up, or a
    timeout-class terminal otherwise — both are contract-acceptable; a
    non-timeout internal error or an infinite hang is not).

    Expectation (contract): master stays HTTP 200; every request ends with
    a legal terminal (ok or timeout-typed); scheduler_inflight AND the
    prefill inflight_batches both drain to zero within TTL(30s)+margin;
    TTL eviction anchors and the prefill endpoint TTL-eviction counter
    are observational — the counter only advances on the PASSIVE
    stale-inflight sweep, while these ledgers clear through the
    retire/settle completion paths (queueTimeout deadline / data-plane
    terminal), which never touch it; the counter channel must stay
    REACHABLE (UNREACHABLE = environment failure); after the injection
    is cleared a fresh batch recovers (verify_recovery).

    Grade: P0."""
    env = ctx.env_manager.ensure(_status_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
        anchors_before = (
            _log_count(env, "event=scheduler_inflight_ttl_eviction"),
            _log_count(env, "event=endpoint_inflight_ttl_eviction"),
        )
        # Event-channel baseline, before any eviction
        # this case can cause.  Unreachable = environment failure.
        ttl_before = ops.master_ttl_eviction_counts()
        if ttl_before is None:
            return False, (
                "master prometheus unreachable before injection — "
                "TTL-eviction observability missing (environment failure)"
            )
        inject_type_all(ops, names, "status_suppress_running")
        inject_type_all(ops, names, "status_suppress_finished")
        try:
            errs = _run_requests(
                ops, base, 4, concurrency=4, stream_timeout_s=LONG_STREAM_TIMEOUT_S
            )
            # Suppress stays ON so the TTL is the only cleanup path.
            sched_zero = _wait_scheduler_zero(ops)
            batches_zero = wait_for(
                lambda: _prefill_batches_sum(ops) == 0,
                TTL_DRAIN_TIMEOUT_S,
                2.0,
            )
            anchors_after = _ttl_anchor_deltas(env, anchors_before)
            # TTL counter OBSERVATION, not an assertion: the counter only
            # advances on the passive stale-inflight sweep, and these
            # ledgers clear through the retire/settle completion paths
            # (queueTimeout deadline / data-plane terminal), which never
            # touch it — the hard assertion is the drain itself
            # (sched_zero / batches_zero / final == 0).  The channel must
            # stay reachable: UNREACHABLE is an environment failure.
            ttl_channel_ok, ttl_channel_detail = _ttl_counter_observe(
                ops, ttl_before, "prefill"
            )
            # Scheduler-side delta rides the same 60s sweep — also
            # observational here; the hard scheduler assertion lives in
            # status_inflight_ttl_cleanup.
            sched_channel_ok, sched_channel_detail = _ttl_counter_observe(
                ops, ttl_before, "scheduler"
            )
        finally:
            clear_type_all(ops, names, "status_suppress_running")
            clear_type_all(ops, names, "status_suppress_finished")

        ok = sum(1 for e in errs if e is None)
        legal_terminal = all(e is None or _timeout_typed(e) for e in errs)
        bad_errs = [
            str(e)[:70] for e in errs if e is not None and not _timeout_typed(e)
        ]
        final_sched = ops.master_scheduler_inflight()
        final_batches = _prefill_batches_sum(ops)
        master_ok = _master_ok(ops)
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            legal_terminal
            and sched_zero
            and batches_zero
            and final_sched == 0
            and final_batches == 0
            and ttl_channel_ok
            and sched_channel_ok
            and master_ok
            and recovery_ok
        )
        return passed, (
            f"request_terminals: ok={ok}/4, "
            f"illegal_errors={bad_errs[:2]}, "
            f"scheduler_zero={sched_zero} (final={final_sched}), "
            f"prefill_batches_zero={batches_zero} (final={final_batches}), "
            f"ttl_anchors(sched,endp)={anchors_after}, "
            f"prefill_ttl_counter[{ttl_channel_detail}], "
            f"observability: {sched_channel_detail}, "
            f"master_200={master_ok}, recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_suppress_running")
        clear_type_all(ops, names, "status_suppress_finished")
