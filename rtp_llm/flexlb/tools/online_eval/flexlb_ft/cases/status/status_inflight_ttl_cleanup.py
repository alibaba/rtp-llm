from __future__ import annotations

import time

from ...context import CaseContext, rid_base
from ...engine_ops import clear_type_all, inject_type_all
from ...harness import (
    OMIT,
    TTL_DRAIN_TIMEOUT_S,
    ConfigOverride,
    EnvSpec,
    _accepted,
    fault_env_perf,
    wait_for,
)
from ...registry import case
from ...support.status import _decode_names, _prefill_names, _ttl_eviction_events


@case(
    "status_inflight_ttl_cleanup",
    category="status",
    profiles=["batch-window"],
    source="flexlb_behavior_test.sh S1 (stuck inflight TTL cleanup)",
)
def inflight_ttl_cleanup(ctx: CaseContext):
    """S1 port, suppressed-alive form: the TTL sweep is the ONLY exit.

    Construction (follow-up fix): the legacy kill-engine form never
    reached the TTL sweep — the 3-strike engine-death verdict fires the
    fence / failure-terminal cleanup path first (Tara's forensics:
    ttl_eviction_delta=0, zero log anchors), so the
    staleInflightTimeoutMs machinery itself went untested.  The engines
    now stay ALIVE (no stop, no fence, no death verdict): the batch's
    rids are pre-generated and the suppression is armed BEFORE the
    requests are sent, on EVERY engine of BOTH roles (prefills AND
    decodes suppress all facts for those rids via status_suppress_rids
    — no RUNNING, no finished; without the D-side arm a decode
    finished fact would settle the slots and bypass the TTL exit
    again).  The requests really reach the engines (fire-and-forget
    Schedule + the prefills' accepted counters asserted >= the wave
    size up front — fail loudly otherwise) and the slow prefills (10s)
    stretch the in-flight window, but the ledger never sees ANY worker
    fact for them: no terminal, no engine-down signal — the only exit
    left is the scheduler slot's staleInflightTimeoutMs (30s) sweep,
    reported by the 60s maintainExpiration cycle.

    Expected (contract): the suppressed wave is accepted by the engines
    (precondition asserted BEFORE the observation — a suppression that
    never reached the engines is a construction failure, not a pass);
    while the suppression holds the scheduler inflight STAYS > 0 (no
    early cleanup: the engines are alive and silence alone must not
    settle anything); the ledger then drains to zero within the
    TTL-aware window (30s TTL + 60s sweep + margin); the
    scheduler-role TTL-eviction counter advances by >= the suppressed
    population (105s event window); the fleet still serves after the
    release (recovery).

    Profile semantics (v2): the env pins the legacy fault
    axes (PRIORITY + FIXED_WINDOW + BATCH, no queueTimeoutMs — the Java
    default 1h cannot expire these requests before the TTL; formerly
    harness.ttl_spec) — the
    declaration stays batch-window (label honesty + regression
    efficiency).  The fault-family override sets no queueTimeoutMs, so
    the deadline path is not an exit here.
    """
    env = ctx.env_manager.ensure(
        EnvSpec(
            label=f"fault_ttl_{ctx.profile}",
            n_prefill=2,
            n_decode=2,
            perf=fault_env_perf(),
            master_profile=ctx.profile,
            discovery="discovery_file",
            config_overrides=ConfigOverride(
                ordering="priority",
                decision="fixed_window",
                dispatcher="batch",
                queue_timeout_ms=OMIT,
            ),
        )
    )
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "status")
    pnames = _prefill_names(ops)
    dnames = _decode_names(ops)
    if not pnames or not dnames:
        return False, (f"engines missing (prefill={len(pnames)}, decode={len(dnames)})")
    all_names = pnames + dnames
    # Pre-generate the wave's rids so the suppression can be armed
    # BEFORE the requests are sent (rids are client-assigned) — no
    # RUNNING fact may leak into the ledger first.
    rids = [ops.next_request_id(base) for _ in range(6)]
    try:
        # Slow both prefills (10s) so the suppressed requests stay
        # genuinely in-flight at the engines: the in-flight window is
        # real, only its reporting is silenced.
        ops.set_perf("prefill-0", prefill_fixed_ms=10000.0)
        ops.set_perf("prefill-1", prefill_fixed_ms=10000.0)

        # Event-channel baseline, BEFORE any eviction
        # this case can cause.  Unreachable = environment failure, not a
        # pass reason.
        ttl_before = ops.master_ttl_eviction_counts()
        if ttl_before is None:
            return False, (
                "master prometheus unreachable before injection — "
                "TTL-eviction observability missing (environment failure)"
            )

        # Arm the suppression on every engine of both roles, then fire:
        # the requests travel for real, but neither role ever reports a
        # fact for them.
        inject_type_all(ops, all_names, "status_suppress_rids", rids=rids)

        # Fire-and-forget: schedule without consuming the response
        # streams — the observation target is the LEDGER.
        for rid in rids:
            resp = ops.schedule(rid, output_len=10)
            if resp.code != 200 or not resp.success:
                return False, f"schedule failed for rid={rid}: {resp.error_message}"

        # PRECONDITION (fail loudly): the suppressed wave must have
        # actually reached the engines — accepted >= 6 on the prefills —
        # otherwise the scenario degenerates into "nothing was ever
        # in-flight" and the TTL sweep would pass vacuously.
        accepted_ok = wait_for(
            lambda: _accepted(ops, "prefill-0") + _accepted(ops, "prefill-1") >= 6,
            15.0,
            0.5,
        )
        accepted_total = _accepted(ops, "prefill-0") + _accepted(ops, "prefill-1")
        if not accepted_ok:
            return False, (
                f"precondition failed: suppressed wave never reached the "
                f"engines (accepted={accepted_total} < 6 after 15s) — the "
                f"TTL scenario is vacuous, not proven"
            )
        inflight_before = ops.master_scheduler_inflight()

        # Observation window INSIDE the suppression: the engines are
        # alive and (unreported) RUNNING — worker silence alone must
        # settle nothing.  12s sits deep inside the 30s stale TTL and
        # past the 10s prefill horizon.
        time.sleep(12.0)
        inflight_held = ops.master_scheduler_inflight()

        # The ONLY exit: the TTL sweep drains the ledger (30s stale TTL
        # + 60s ExpirationTimer cycle + margin — the same worst-case
        # settle window as TTL_DRAIN_TIMEOUT_S).
        cleanup_ok = wait_for(
            lambda: ops.master_scheduler_inflight() == 0, TTL_DRAIN_TIMEOUT_S, 2.0
        )
        inflight_final = ops.master_scheduler_inflight()

        # event-channel assertion, SEPARATE from the
        # drain above: the scheduler-level TTL-eviction counter must
        # have advanced by at least the suppressed population (105s
        # window — the 60s maintenance sweep reports the eviction with
        # lag).
        ttl_events_ok, ttl_events_detail = _ttl_eviction_events(
            ops, ttl_before, "scheduler", len(rids)
        )

        # The engines were never stopped: release the suppression and
        # prove the fleet still serves.
        clear_type_all(ops, all_names, "status_suppress_rids")
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            inflight_held > 0
            and cleanup_ok
            and inflight_final == 0
            and ttl_events_ok
            and recovery_ok
        )
        return passed, (
            f"accepted={accepted_total}/6 (precondition), "
            f"inflight_before={inflight_before}, "
            f"inflight_held_after_12s={inflight_held} (no early cleanup), "
            f"ttl_cleanup_within_{TTL_DRAIN_TIMEOUT_S:.0f}s={cleanup_ok}, "
            f"inflight_final={inflight_final}, "
            f"scheduler_ttl_evictions[{ttl_events_detail}], "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            clear_type_all(ops, all_names, "status_suppress_rids")
            ops.set_perf("prefill-0", prefill_fixed_ms=100.0)
            ops.set_perf("prefill-1", prefill_fixed_ms=100.0)
        except Exception:
            pass
