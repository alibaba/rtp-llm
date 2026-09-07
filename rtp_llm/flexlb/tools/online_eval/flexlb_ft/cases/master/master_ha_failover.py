from __future__ import annotations

import time

from ...context import CaseContext
from ...registry import case
from ...support.ha import (
    HaRows,
    HaTrafficRunner,
    dual_spec_for_layout,
    ha_gate,
    instance_ops,
    restore_masters,
    rows_between,
)
from ...support.master import HA_STEADY_S, HA_SWITCH_S, _check_client_fields


@case(
    "master_ha_failover",
    category="master",
    profiles=["batch-window"],
    source="scenario 2 failover (brief p5/p6): sticky A, kill -9 A, "
    "same-request retry to B once, sticky moves to B",
)
def master_ha_failover(ctx: CaseContext):
    """Tier-2/3 by default (FLEXLB_FT_HA_LAYOUT, tier3), Tier-1 fallback.

    Six-step flow (brief p6):
      1. dual masters + client GRPC_TARGETS=A,B, sticky A; 100% served
         by A (master_target label verification);
      2. inject: kill -9 A;
      3. simplified switch: in-flight Schedule sees gRPC UNAVAILABLE
         (event 1, transport layer) -> SAME-REQUEST retry to B once ->
         success; sticky pointer advances to B (no probing thread, no
         state machine, no thresholds);
      4. assert-1: switch-window errors ~0 (same-request retry
         backstop) + post-switch master_target=B 100% + QPS non-zero;
      5. assert-2: B routes correctly (LOCAL_STANDALONE serves; Tier-1
         has no follower so no 8511);
      6. teardown reclaims everything (sticky stays on B for scenario 4).

    Tier-2 forwarding four-state matrix is NOT asserted here — JUnit
    territory (master_forward_matrix, see the HA group header).  The
    Tier-3 same-host distinct-IP layout is DEAD per the harness.py
    RULING (2026-09-02: localIp has no env override, wildcard bind,
    SELF_TARGET) — Tier-3 moves to the phase-2 dual-container topology;
    the 127.0.0.1/.2 wiring stays as the env-injection contract
    reference only.

    Two assertion-face cuts (documented, not lost): the real_master_host
    contract assertion is suspended until the optional p5 fix lands
    (Tier-1 has no ZK, so there is nothing to assert against); B's
    queue/inflight bookkeeping after the switch is approximated by the
    traffic assertions (post-switch ok-rate + master_target share), not
    mirrored per-request.
    """
    gate = ha_gate()
    if gate:
        return gate
    env = ctx.env_manager.ensure(dual_spec_for_layout(ctx))
    mgr = ctx.env_manager
    target_a = mgr.master_instance_target(env, "A")
    target_b = mgr.master_instance_target(env, "B")
    case_dir = ctx.case_dir("master_ha_failover")
    flow = HaTrafficRunner(
        ctx,
        env,
        case_dir,
        "ha_failover",
        targets=[target_a, target_b],  # sticky A first (brief p6)
        duration_s=60,
    )
    try:
        flow.start()
        time.sleep(HA_STEADY_S)
        t_kill = HaTrafficRunner.now()
        mgr.kill_master9_instance(env, "A")
        time.sleep(HA_SWITCH_S)
        t_switched = HaTrafficRunner.now()
        # B-side routing sanity (assert-2): B answers master/info.
        ops_b = instance_ops(ctx, env, "B")
        info_b = ops_b.master_info()
        b_ready = bool(info_b and info_b.get("ready"))
        flow.wait_finish()
        rows = flow.rows()
        guard = _check_client_fields(HaRows(rows))
        if guard:
            return guard
        steady = HaRows(rows_between(rows, None, t_kill))
        switch = HaRows(rows_between(rows, t_kill, t_switched))
        after = HaRows(rows_between(rows, t_switched, None))
        allr = HaRows(rows)

        steady_a = len(steady.rows) >= 10 and len(steady.target(target_a)) == len(
            steady.rows
        )
        # Same straddle-window lookup as _master_kill_dual: failover rows
        # keep their ORIGINAL (pre-kill) send timestamp, so the plain
        # switch window misses them by timing luck. A is healthy in the
        # lookback window, so any failover row there is kill-driven.
        failover_window = HaRows(rows_between(rows, t_kill - 10.0, t_switched))
        failover_seen = len(failover_window.failover_rows()) > 0
        switched_to_b = len(switch.target(target_b)) > 0
        switch_failed = len(switch.route("failed"))
        switch_bounded = (
            switch_failed <= max(1, int(0.05 * len(switch.rows)))
            if switch.rows
            else True
        )
        after_b_share = (
            len(after.target(target_b)) / len(after.rows) if after.rows else 0.0
        )
        after_ok = after.ok_rate() >= 0.90 if after.rows else False
        qps_nonzero = len(after.rows) >= 20
        no_dup = not allr.dup_rids()

        passed = (
            steady_a
            and failover_seen
            and switched_to_b
            and switch_bounded
            and after_b_share >= 0.95
            and after_ok
            and qps_nonzero
            and b_ready
            and no_dup
        )
        return passed, (
            f"steady_A={steady_a}({len(steady.rows)} rows), "
            f"failover_seen={failover_seen}, "
            f"switch: to_B={switched_to_b}, failed={switch_failed}/"
            f"{len(switch.rows)} (bounded={switch_bounded}), "
            f"after: B={after_b_share:.0%}, ok={after.ok_rate():.0%}, "
            f"qps_rows={len(after.rows)}, B_ready={b_ready}, "
            f"dup_rids={len(allr.dup_rids())}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        restore_masters(ctx, env)
