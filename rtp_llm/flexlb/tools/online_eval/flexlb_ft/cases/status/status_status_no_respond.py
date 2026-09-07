from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, inject_type_all
from ...harness import wait_for
from ...registry import case
from ...support.status import (
    MASTER_EVICT_S,
    _fire_and_forget,
    _log_count,
    _master_ok,
    _prefill_names,
    _status_spec,
    _ttl_anchor_deltas,
    _ttl_counter_observe,
    _wait_scheduler_zero,
)


@case(
    "status_status_no_respond",
    category="status",
    profiles=["batch-window"],
    source="P0 status fault family: status_no_respond — engine stops answering the status RPC",
)
def status_status_no_respond(ctx: CaseContext):
    """Scenario: prefills stop answering the WorkerStatus poll entirely
    (status_no_respond) while batches are live in their ledgers.

    Behaviour: the health poller accumulates strikes (3 consecutive
    failures) and demotes/retires the whole engine generation; the live
    slots freeze (no ACTIVE fact), so the stale TTL reclaims them.

    Expectation (contract): the alive count DROPS within the 3-strike
    window (generation retirement); master stays HTTP 200; the scheduler
    inflight drains to zero within TTL+margin; the TTL-eviction counter
    is observational here — the generation retirement itself clears the
    ledger through the retire path, which does not ride the TTL counter
    channel (the counter only advances on the passive stale-inflight
    sweep), but the channel must stay REACHABLE (UNREACHABLE =
    environment failure); after the injection is cleared the topology
    fully recovers (alive back to 2P) and a fresh request succeeds.

    Grade: P0."""
    env = ctx.env_manager.ensure(_status_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if len(names) < 2:
        return False, "need >=2 prefill engines"
    try:
        # Slow prefills widen the in-flight window so the injection lands
        # before the engines report their terminals.
        for n in names:
            ops.set_perf(n, prefill_fixed_ms=3000.0)
        rids, sched_err = _fire_and_forget(ops, base, 4)
        if sched_err:
            return False, f"could not stage live inflight: {sched_err}"

        anchors_before = (
            _log_count(env, "event=scheduler_inflight_ttl_eviction"),
            _log_count(env, "event=endpoint_inflight_ttl_eviction"),
        )
        # Event-channel baseline.  Unreachable =
        # environment failure.
        ttl_before = ops.master_ttl_eviction_counts()
        if ttl_before is None:
            return False, (
                "master prometheus unreachable before injection — "
                "TTL-eviction observability missing (environment failure)"
            )
        inject_type_all(ops, names, "status_no_respond")
        try:
            alive_dropped = wait_for(
                lambda: ops.master_alive_count("PREFILL") <= len(names) - 1,
                MASTER_EVICT_S,
                0.5,
            )
            all_retired = wait_for(
                lambda: ops.master_alive_count("PREFILL") == 0,
                MASTER_EVICT_S,
                0.5,
            )
            drained = _wait_scheduler_zero(ops)
            anchors_after = _ttl_anchor_deltas(env, anchors_before)
            # TTL counter OBSERVATION, not an assertion: the generation
            # retirement clears the ledger through the retire path, which
            # does not advance the counter (it only moves on the passive
            # stale-inflight sweep) — the hard assertion is the drain
            # itself (drained / final_sched == 0).  The channel must stay
            # reachable: UNREACHABLE is an environment failure.
            ttl_channel_ok, ttl_channel_detail = _ttl_counter_observe(
                ops, ttl_before, "scheduler"
            )
        finally:
            clear_type_all(ops, names, "status_no_respond")

        final_sched = ops.master_scheduler_inflight()
        master_ok = _master_ok(ops)
        alive_back = wait_for(
            lambda: ops.master_alive_count("PREFILL") >= len(names),
            MASTER_EVICT_S,
            0.5,
        )
        time.sleep(2.0)  # channel reconnect settle (crash_after precedent)
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            alive_dropped
            and all_retired
            and drained
            and final_sched == 0
            and ttl_channel_ok
            and master_ok
            and alive_back
            and recovery_ok
        )
        return passed, (
            f"generation_retired={all_retired} "
            f"(alive={ops.master_alive_count('PREFILL')}), "
            f"scheduler_zero={drained} (final={final_sched}), "
            f"ttl_anchors(sched,endp)={anchors_after}, "
            f"scheduler_ttl_counter[{ttl_channel_detail}], "
            f"master_200={master_ok}, topology_recovered={alive_back}, "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_no_respond")
        try:
            for n in names:
                ops.set_perf(n, prefill_fixed_ms=100.0)
        except Exception:
            pass
