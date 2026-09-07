from __future__ import annotations

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
    "status_version_regress",
    category="status",
    profiles=["batch-window"],
    source="P0 status fault family: status_version_regress — stale status version",
)
def status_version_regress(ctx: CaseContext):
    """Scenario: prefills keep answering the status RPC but with a
    REGRESSED version (status_version_regress) while batches are live.

    Behaviour: the master rejects the stale-version reports as invalid;
    sustained invalid reports accumulate into the health 3-strike, so the
    whole engine generation retires; the live slots freeze and the stale
    TTL reclaims them.

    Expectation (contract): the alive count DROPS (generation retirement);
    master stays HTTP 200; the scheduler inflight drains to zero within
    TTL+margin; the TTL-eviction counter is observational here — same
    rationale as status_status_no_respond (the retire path clears the
    ledger without advancing the passive-sweep counter), but the channel
    must stay REACHABLE (UNREACHABLE = environment failure).

    Grade: P0."""
    env = ctx.env_manager.ensure(_status_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "status")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    try:
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
        inject_type_all(ops, names, "status_version_regress")
        try:
            alive_dropped = wait_for(
                lambda: ops.master_alive_count("PREFILL") <= len(names) - 1,
                MASTER_EVICT_S,
                0.5,
            )
            drained = _wait_scheduler_zero(ops)
            anchors_after = _ttl_anchor_deltas(env, anchors_before)
            # TTL counter OBSERVATION, not an assertion — same rationale
            # as status_status_no_respond: the retire path clears the
            # ledger without advancing the passive-sweep counter; the
            # channel must stay reachable (UNREACHABLE = environment
            # failure).
            ttl_channel_ok, ttl_channel_detail = _ttl_counter_observe(
                ops, ttl_before, "scheduler"
            )
        finally:
            clear_type_all(ops, names, "status_version_regress")

        final_sched = ops.master_scheduler_inflight()
        master_ok = _master_ok(ops)
        # Post-clear topology recovery (observational — the spec asserts
        # retirement + drain; recovery is the shared-env hygiene proof).
        alive_back = wait_for(
            lambda: ops.master_alive_count("PREFILL") >= len(names),
            MASTER_EVICT_S,
            0.5,
        )

        passed = (
            alive_dropped
            and drained
            and final_sched == 0
            and master_ok
            and ttl_channel_ok
        )
        return passed, (
            f"generation_retired={alive_dropped} "
            f"(alive={ops.master_alive_count('PREFILL')}), "
            f"scheduler_zero={drained} (final={final_sched}), "
            f"ttl_anchors(sched,endp)={anchors_after}, "
            f"scheduler_ttl_counter[{ttl_channel_detail}], "
            f"master_200={master_ok}, topology_recovered={alive_back}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "status_version_regress")
        try:
            for n in names:
                ops.set_perf(n, prefill_fixed_ms=100.0)
        except Exception:
            pass
