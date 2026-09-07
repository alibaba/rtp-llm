"""Shared master scenario components. Resource and timing semantics live here."""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ..context import CaseContext, CaseDef, rid_base
from ..engine_ops import (
    StreamHandle,
    StreamSnapshot,
    clear_type_all,
    engine_inflight_clean,
    inject_type_all,
)
from ..harness import (
    OMIT,
    TTL_DRAIN_TIMEOUT_S,
    AssertUtils,
    ConfigOverride,
    EnvSpec,
    _cleanup_dynamic,
    _elastic_env,
    _run_batch,
    default_perf,
    fault_env_perf,
    wait_for,
)
from .ha import (
    HaRows,
    HaTrafficRunner,
    dual_spec_for_layout,
    ha_dual_enabled,
    ha_gate,
    instance_alive_full,
    instance_ops,
    recovery_rate,
    restore_masters,
    rows_between,
    tier1_dual_spec,
)

STREAM_TIMEOUT_S = 15.0
# 3-strike health marking + eviction window (fault-family precedent).
MASTER_EVICT_S = 30.0


def _master_http(ops) -> str:
    return f"http://127.0.0.1:{ops.master_http_port}"


# ===========================================================================
# Master HA group — kill -9 + restart (flexlb_behavior_test.sh ports)
# ===========================================================================


def _quota_spec(ctx: CaseContext) -> EnvSpec:
    """Quota-block env (S3): 1P+1D, maxInflightBatches=1 via config
    override (dispatcher.maxInflightBatchesPerPrefillWorker — the v1 env
    var FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES has no v2 consumer;
    formerly harness.quota_spec)."""
    return EnvSpec(
        label=f"fault_quota_{ctx.profile}",
        n_prefill=1,
        n_decode=1,
        perf=fault_env_perf(),
        master_profile=ctx.profile,
        discovery="discovery_file",
        config_overrides=ConfigOverride(
            ordering="priority",
            decision="fixed_window",
            dispatcher="batch",
            queue_timeout_ms=OMIT,
            max_inflight_batches=1,
        ),
    )


# ===========================================================================
# Cold-start burst — intake defect regression probe
# ===========================================================================


def _coldstart_spec(ctx: CaseContext) -> EnvSpec:
    """Cold-start probe env: mirrors the default topology (2P+4D, static
    file discovery, default config) but disables the master stability
    window so traffic hits the master during the first-connect storm
    (formerly harness.coldstart_spec)."""
    return EnvSpec(
        label=f"fault_coldstart_{ctx.profile}",
        n_prefill=2,
        n_decode=4,
        perf=default_perf(),
        master_profile=ctx.profile,
        master_stable_window_s=0.0,
    )


# ===========================================================================
# HA dual-master group (brief p1-p10; Tier attribution per case docstring)
# ===========================================================================
#
# master_forward_matrix (four-state forwarding matrix) is NOT implemented
# here: Tier-2 JUnit territory — the four-state matrix lives in flexlb-api
# src/test (ScheduleForwardMatrixTest), the embedded-ZK election layer in
# flexlb-sync src/test (ZkLeaderElectionTest).  This harness only
# orchestrates processes;
# the four states (LOCAL_MASTER / stale-forward-8511 / MASTER_NULL
# LOCAL_FALLBACK / transparent-forward) need deterministic election
# control the process-level harness cannot provide.


# Steady-state window before every fault injection (rows the sticky
# assertion reads from) and the failover observation window after it.
HA_STEADY_S = 12.0
HA_SWITCH_S = 10.0


def _prefill_engine_names(ops) -> list:
    try:
        snap = ops.snapshot()
    except Exception:
        return []
    return [
        e["name"]
        for e in snap.get("engines", [])
        if str(e.get("name", "")).startswith("prefill")
    ]


def _check_client_fields(all_rows: HaRows) -> Optional[tuple]:
    """Fail-closed guard: the HA observability contract fields must be on
    every row (delivered with the multi-target client)."""
    if all_rows.rows and all_rows.missing_fields:
        return False, (
            f"client_events rows missing HA fields {all_rows.missing_fields} "
            f"— stale JavaLoadClient build (HA contract: route_path "
            f"master|fallback|failed + master_target/failover/error_kind)"
        )
    return None


def _master_kill_dual(ctx: CaseContext):
    # Internal dual-master branch of master_kill (dispatched under the
    # FLEXLB_FT_HA_DUAL_MASTER gate) — NOT self-registered: the case name
    # "master_kill" belongs to the single public entry above.

    """Tier-1 dual-standalone generalized branch (brief p4 left lane).

    Start sticky-on-B (NOT A), kill -9 B (Mode 1 restart-zeroing fault):
    in-flight requests see UNAVAILABLE → same-request retry to A
    (event 1, no probing), sticky moves to A; restart B from the same
    argv/env and assert the cold-recovery contract generalized per
    master: inflight zeroed (orphans terminally visible), topology
    re-converged from the zero point within 60s, recovery >= 95% served
    by the surviving A.
    """
    env = ctx.env_manager.ensure(tier1_dual_spec(ctx))
    ops_a = instance_ops(ctx, env, "A")
    ops_b = instance_ops(ctx, env, "B")
    mgr = ctx.env_manager
    target_a = mgr.master_instance_target(env, "A")
    target_b = mgr.master_instance_target(env, "B")
    case_dir = ctx.case_dir("master_kill_dual")
    flow = HaTrafficRunner(
        ctx,
        env,
        case_dir,
        "master_kill_dual",
        targets=[target_b, target_a],  # sticky B first (brief p4)
        duration_s=90,
    )
    pid_b_before = None
    try:
        flow.start()
        pid_b_before = env.masters["B"].pid
        time.sleep(HA_STEADY_S)
        t_kill = HaTrafficRunner.now()
        mgr.kill_master9_instance(env, "B")
        time.sleep(HA_SWITCH_S)
        t_switched = HaTrafficRunner.now()
        # Mode 1 recovery: cold restart from the same argv/env.
        mgr.restart_master_instance(env, "B")
        pid_b_after = env.masters["B"].pid
        # Cold-recovery assertions generalized per master (brief p4 step 5).
        converged = instance_alive_full(ops_b, env, 60.0)
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            mgr.master_instance_http(env, "B"), 10.0
        )
        rec_ok, rec_msg = recovery_rate(ops_a)
        flow.wait_finish()
        rows = flow.rows()
        guard = _check_client_fields(HaRows(rows))
        if guard:
            return guard
        steady = HaRows(rows_between(rows, None, t_kill))
        switch = HaRows(rows_between(rows, t_kill, t_switched))
        after = HaRows(rows_between(rows, t_switched, None))
        allr = HaRows(rows)

        # Steady state = rows SENT before the kill that needed no rescue.
        # Kill-boundary rows (sent pre-kill, rescued post-kill by the
        # same-request retry: failover=True, landed on A) are the Mode-1
        # transition, not steady state — counting them against the
        # 100%-on-B check false-FAILs (remote evidence: 211 steady + 25
        # retry-rescued boundary rows + 8 other in-window rows).
        steady_plain = HaRows([r for r in steady.rows if not r.get("failover")])
        steady_b = len(steady_plain.rows) >= 10 and len(
            steady_plain.target(target_b)
        ) == len(steady_plain.rows)
        # Failover rows are same-request retries that keep their ORIGINAL
        # send timestamp — the victim was already in flight when B died,
        # so its row sends BEFORE t_kill and slicing by the [t_kill,
        # t_switched] switch window catches the retry only by timing luck
        # (run-1788363800: every kill-boundary retry-rescued row sent
        # before t_kill, failover_seen false-FAILed while the switch
        # itself was textbook). Look failover rows up across the in-flight
        # straddle instead: B is healthy throughout the lookback window,
        # so any failover row there can only be the kill-driven retry.
        failover_window = HaRows(rows_between(rows, t_kill - 10.0, t_switched))
        failover_seen = len(failover_window.failover_rows()) > 0
        switched_to_a = len(switch.target(target_a)) > 0
        switch_failed = len(switch.route("failed"))
        switch_bounded = (
            switch_failed <= max(1, int(0.05 * len(switch.rows)))
            if switch.rows
            else True
        )
        after_a_share = (
            len(after.target(target_a)) / len(after.rows) if after.rows else 0.0
        )
        after_ok = after.ok_rate() >= 0.90 if after.rows else False
        no_dup = not allr.dup_rids()
        pid_changed = pid_b_before != pid_b_after

        passed = (
            steady_b
            and failover_seen
            and switched_to_a
            and switch_bounded
            and after_a_share >= 0.95
            and after_ok
            and no_dup
            and pid_changed
            and converged
            and inflight_ok
            and rec_ok
        )
        return passed, (
            f"steady_B={steady_b}({len(steady_plain.rows)}/{len(steady.rows)} rows), "
            f"failover_seen={failover_seen}, "
            f"switch: to_A={switched_to_a}, failed={switch_failed}/"
            f"{len(switch.rows)} (bounded={switch_bounded}), "
            f"after_A={after_a_share:.0%} ok={after.ok_rate():.0%}, "
            f"pid_changed={pid_changed}, converged60s={converged}, "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"{rec_msg}, dup_rids={len(allr.dup_rids())}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            if env.masters.get("B") is None:
                ctx.env_manager.restart_master_instance(env, "B")
        except Exception:
            pass


def _prefill_names(ops) -> list[str]:
    snap = ops.snapshot()
    return [e["name"] for e in snap.get("engines", []) if e.get("role") == "prefill"]


# ===========================================================================
# Direct-path case (migrated from the legacy injection family,
# category reorg — rid_base family "chaos" -> "direct"; folded from
# the retired one-case direct module into master — the rid_base family
# stays "direct" so its id block keeps the sub-1M dedup-collision
# distance from the master block)
# ===========================================================================
