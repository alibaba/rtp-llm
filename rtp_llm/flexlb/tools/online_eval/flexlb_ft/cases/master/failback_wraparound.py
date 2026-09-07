from __future__ import annotations

import time

from ...context import CaseContext
from ...harness import AssertUtils
from ...registry import case
from ...support.ha import (
    HaRows,
    HaTrafficRunner,
    dual_spec_for_layout,
    ha_gate,
    instance_alive_full,
    instance_ops,
    recovery_rate,
    restore_masters,
    rows_between,
)
from ...support.master import HA_STEADY_S, HA_SWITCH_S, _check_client_fields


@case(
    "failback_wraparound",
    category="master",
    profiles=["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"],
    source="scenario 4 recovery (brief p9/p10): rebuild scenario-2 end "
    "state (sticky B, A dead), restart + converge A, kill B -> wrap "
    "back to A",
)
def failback_wraparound(ctx: CaseContext):
    """Tier-2/3 by default (FLEXLB_FT_HA_LAYOUT, tier3), Tier-1 fallback.

    Flow (brief p10):
      0. rebuild the scenario-2 end state: sticky A -> kill A -> sticky
         B (traffic keeps flowing on B);
      1. restart A (same argv/env): port ready -> /master/info ready ->
         topology re-converged within the 60s window (alive full);
      2. assert-1: A ready + alive full + a fresh 20-request recovery
         round >= 95% (RECOVERY_TIMEOUT_S judgement generalized per
         master);
      3. symmetric inject: kill -9 B (recovery and switch symmetry);
      4. wrap failback: in-flight connection failure -> the retry chain
         wraps around to the recovered A -> success; sticky pointer back
         to A (the simplest failback: no probing, no explicit switch);
      5. assert-2: symmetric-switch errors ~0 + master_target=A 100% +
         inflight clean + no 8511 storm.

    TODO(Tier-3, brief p10 notes 3/5): explicit failback following the
    real ZK leader (client re-polls real_master_host after A's
    re-election) and the pre_stop graceful variant (/hook/pre_stop ->
    leader handover <=30s + drain <=300s).  Tier-3 activation is NOT
    the same-host distinct-IP layout — that layout is DEAD per the
    harness.py RULING (2026-09-02) and moves to the phase-2
    dual-container topology (one network stack per container); both
    variants stay out of the wrap-around scope until then.
    """
    gate = ha_gate()
    if gate:
        return gate
    env = ctx.env_manager.ensure(dual_spec_for_layout(ctx))
    mgr = ctx.env_manager
    ops_a = instance_ops(ctx, env, "A")
    target_a = mgr.master_instance_target(env, "A")
    target_b = mgr.master_instance_target(env, "B")
    case_dir = ctx.case_dir("failback_wraparound")
    flow = HaTrafficRunner(
        ctx,
        env,
        case_dir,
        "failback_wrap",
        targets=[target_a, target_b],  # sticky A first
        duration_s=150,
    )
    try:
        flow.start()
        time.sleep(HA_STEADY_S)
        # -- step 0: rebuild scenario-2 end state (sticky B, A dead) --
        mgr.kill_master9_instance(env, "A")
        time.sleep(HA_SWITCH_S)
        # -- step 1: restart A, converge in the 60s window ------------
        mgr.restart_master_instance(env, "A")
        converged = instance_alive_full(ops_a, env, 60.0)
        # -- step 2: recovery round served by (or via) the new A ------
        rec_ok, rec_msg = recovery_rate(ops_a)
        time.sleep(8.0)
        t_kill_b = HaTrafficRunner.now()
        # -- step 3/4: symmetric kill B -> wrap-around back to A ------
        mgr.kill_master9_instance(env, "B")
        time.sleep(HA_SWITCH_S)
        t_switched = HaTrafficRunner.now()
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            mgr.master_instance_http(env, "A"), 10.0
        )
        time.sleep(5.0)
        flow.wait_finish()
        rows = flow.rows()
        guard = _check_client_fields(HaRows(rows))
        if guard:
            return guard
        switch = HaRows(rows_between(rows, t_kill_b, t_switched))
        after = HaRows(rows_between(rows, t_switched, None))
        allr = HaRows(rows)

        # Same straddle-window lookup as _master_kill_dual: failover rows
        # keep their ORIGINAL (pre-kill) send timestamp. The lookback
        # stays well clear of the step-0 kill-A failover rows (those sent
        # before THAT kill, >= 18s before t_kill_b).
        failover_window = HaRows(rows_between(rows, t_kill_b - 10.0, t_switched))
        failover_seen = len(failover_window.failover_rows()) > 0
        switched_to_a = len(switch.target(target_a)) > 0
        switch_failed = len(switch.route("failed"))
        switch_bounded = (
            switch_failed <= max(1, int(0.05 * len(switch.rows)))
            if switch.rows
            else True
        )
        # no 8511 storm: Tier-1 has no 8511 at all; on Tier-2/3 a storm
        # would surface as a flood of business-error rows in the window.
        storm_bounded = (
            len(switch.error_kind("business")) <= max(1, int(0.05 * len(switch.rows)))
            if switch.rows
            else True
        )
        after_a_share = (
            len(after.target(target_a)) / len(after.rows) if after.rows else 0.0
        )
        after_ok = after.ok_rate() >= 0.90 if after.rows else False
        no_dup = not allr.dup_rids()

        passed = (
            converged
            and rec_ok
            and failover_seen
            and switched_to_a
            and switch_bounded
            and storm_bounded
            and after_a_share >= 0.95
            and after_ok
            and inflight_ok
            and no_dup
        )
        return passed, (
            f"A_converged60s={converged}, {rec_msg}, "
            f"wrap: failover_seen={failover_seen}, to_A={switched_to_a}, "
            f"failed={switch_failed}/{len(switch.rows)} "
            f"(bounded={switch_bounded}), "
            f"8511_storm_bounded={storm_bounded}, "
            f"after: A={after_a_share:.0%}, ok={after.ok_rate():.0%}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"dup_rids={len(allr.dup_rids())}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        restore_masters(ctx, env)
