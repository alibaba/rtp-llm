from __future__ import annotations

import time

from ...context import CaseContext
from ...registry import case
from ...support.ha import (
    HaRows,
    HaTrafficRunner,
    ha_gate,
    restore_masters,
    rows_between,
    tier1_dual_spec,
)
from ...support.master import HA_STEADY_S, HA_SWITCH_S, _check_client_fields


@case(
    "fallback_direct",
    category="master",
    profiles=["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"],
    source="scenario 3 positive (brief p7/p8): kill A + kill B (all masters "
    "down) -> ENABLE_FALLBACK -> direct-to-engine streams",
)
def fallback_direct(ctx: CaseContext):
    """Tier-1 dual standalone, ENABLE_FALLBACK=true + ENDPOINTS_FILE.

    Phase-1 (positive, brief p8):
      0. steady: 100% master-routed (route_path=master);
      1. inject: kill -9 A then kill -9 B (all masters down);
      2. in-flight: UNAVAILABLE on A -> retry B UNAVAILABLE (double
         connection failure) -> direct fallback fires;
      3. runFallbackStream: RR engine pick from the endpoints.json
         static snapshot, generateStreamCall re-send (same request_id);
      4. assert-1: route_path=fallback success-rate healthy, master
         routing = 0 in the outage window, fallback <= 1 per request
         (no duplicate rids), errors bounded.
    """
    gate = ha_gate()
    if gate:
        return gate
    env = ctx.env_manager.ensure(tier1_dual_spec(ctx))
    mgr = ctx.env_manager
    target_a = mgr.master_instance_target(env, "A")
    target_b = mgr.master_instance_target(env, "B")
    case_dir = ctx.case_dir("fallback_direct")
    flow = HaTrafficRunner(
        ctx,
        env,
        case_dir,
        "fallback_direct",
        targets=[target_a, target_b],  # sticky A
        duration_s=60,
        enable_fallback=True,
    )
    try:
        flow.start()
        time.sleep(HA_STEADY_S)
        t_kill_a = HaTrafficRunner.now()
        mgr.kill_master9_instance(env, "A")
        time.sleep(0.5)  # brief transition: sticky may hop A -> B here
        t_kill_b = HaTrafficRunner.now()
        mgr.kill_master9_instance(env, "B")
        time.sleep(HA_SWITCH_S)
        t_outage_end = HaTrafficRunner.now()
        flow.wait_finish()
        rows = flow.rows()
        guard = _check_client_fields(HaRows(rows))
        if guard:
            return guard
        steady = HaRows(rows_between(rows, None, t_kill_a))
        outage = HaRows(rows_between(rows, t_kill_b, t_outage_end))
        allr = HaRows(rows)

        steady_master = len(steady.rows) >= 10 and len(steady.route("master")) == len(
            steady.rows
        )
        fb_rows = outage.route("fallback")
        fb_ok = [r for r in fb_rows if r.get("status") == "ok"]
        fb_rate = (len(fb_ok) / len(fb_rows)) if fb_rows else 0.0
        fb_share = (len(fb_rows) / len(outage.rows)) if outage.rows else 0.0
        master_leak = len(outage.route("master"))
        failed_rows = len(outage.route("failed"))
        failed_bounded = (
            failed_rows <= max(1, int(0.05 * len(outage.rows))) if outage.rows else True
        )
        no_dup = not allr.dup_rids()

        passed = (
            steady_master
            and len(fb_rows) >= 10
            and fb_rate >= 0.90
            and fb_share >= 0.80
            and master_leak == 0
            and failed_bounded
            and no_dup
        )
        return passed, (
            f"steady_master={steady_master}({len(steady.rows)} rows), "
            f"outage: fallback={len(fb_rows)}/{len(outage.rows)} "
            f"({fb_share:.0%}), fb_ok={fb_rate:.0%}, "
            f"master_leak={master_leak}, failed={failed_rows} "
            f"(bounded={failed_bounded}), dup_rids={len(allr.dup_rids())}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        restore_masters(ctx, env)
