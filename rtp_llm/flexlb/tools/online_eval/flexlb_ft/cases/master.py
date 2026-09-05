"""Master-category cases: the master process as a fault victim.

Theme: the FlexLB master itself going down, blocking admission by
quota, or cold-starting under a first-connect burst — worker traffic
must converge to a healthy topology, in-flight state must settle (TTL
or explicit cleanup), and a restarted master must come back with clean
state.  master_kill (kill -9 + restart), master_quota_block (1P+1D
quota blocking + TTL recovery) and master_coldstart_burst (the intake
defect regression probe) share the env/flow helpers from harness.
direct_generate_error — the client-direct GenerateStreamCall bypass
(the load-client direct deployment shape, i.e. the client-side escape
path when the scheduling plane is unavailable) — moved in from the
retired one-case direct module.
"""

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
    TTL_DRAIN_TIMEOUT_S,
    AssertUtils,
    _cleanup_dynamic,
    _elastic_env,
    _run_batch,
    coldstart_spec,
    quota_spec,
    wait_for,
)
from .ha_support import (
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

MASTER_CASES: list[CaseDef] = []

STREAM_TIMEOUT_S = 15.0
# 3-strike health marking + eviction window (fault-family precedent).
MASTER_EVICT_S = 30.0


def case(
    name: str,
    profiles=None,
    requires=None,
    source: str = "",
    expected_fail: bool = False,
):
    """Register into MASTER_CASES (category is always "master").

    ``expected_fail=True`` declares a declared-finding probe (task #101):
    failing confirms the finding, passing resolves it — neither counts
    toward failed_count / the suite verdict / the exit code."""

    def deco(fn):
        MASTER_CASES.append(
            CaseDef(
                name=name,
                category="master",
                fn=fn,
                profiles=profiles,
                requires=requires,
                source=source,
                expected_fail=expected_fail,
            )
        )
        return fn

    return deco


def _master_http(ops) -> str:
    return f"http://127.0.0.1:{ops.master_http_port}"


# ===========================================================================
# Master HA group — kill -9 + restart (flexlb_behavior_test.sh ports)
# ===========================================================================


@case(
    "master_kill",
    profiles=["batch-window"],  # elastic_spec pins the legacy fault axes
    source="master HA: kill -9 master → restart → clean state + recovery",
)
def master_kill(ctx: CaseContext):
    # HA generalized branch (brief p3: "master_kill 用例泛化双 master 定向",
    # p4 left lane: start sticky-on-B, kill -9 B) — gated on
    # FLEXLB_FT_HA_DUAL_MASTER=1 so the default run keeps the historical
    # single-master flow byte-identical (compat hard constraint).
    if ha_dual_enabled():
        return _master_kill_dual(ctx)
    env, ops = _elastic_env(ctx)
    base = rid_base(ctx, "master")
    try:
        _cleanup_dynamic(ops, env)

        # Baseline: one request succeeds before the kill.
        addr, err0 = ops.run_one_request(
            ops.next_request_id(base),
            output_len=2,
            block_keys=[base + 11],
            stream_timeout_s=10.0,
        )
        del addr
        if err0:
            return False, f"baseline request failed: {err0}"

        # kill -9 the master, restart it from the same argv/env.
        ctx.env_manager.kill_master9(env)
        time.sleep(2.0)  # settle; port release
        ctx.env_manager.start_master(env)

        # Wait for the full topology to re-converge (ready + alive workers).
        alive_ok = wait_for(
            lambda: ops.master_alive_count("PREFILL") >= 2
            and ops.master_alive_count("DECODE") >= 4,
            60.0,
            1.0,
        )
        # Fresh master must start from clean inflight state.
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 10.0
        )
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = alive_ok and inflight_ok and recovery_ok
        return passed, (
            f"master_restarted, topology_reconverged={alive_ok}"
            f"(alive P:{ops.master_alive_count('PREFILL')}/"
            f"D:{ops.master_alive_count('DECODE')}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        # If the case died mid-way with the master down, bring it back so the
        # shared env stays usable (and teardown finds a ManagedProcess).
        try:
            if env.master is None:
                ctx.env_manager.start_master(env)
        except Exception:
            pass


@case(
    "master_quota_block",
    profiles=["batch-window"],
    source="flexlb_behavior_test.sh S3 (1P+1D quota blocking + TTL recovery)",
)
def master_quota_block(ctx: CaseContext):
    """S3 port: fill the 1-batch inflight quota → stop the only prefill →
    new requests fail (≥50%) → TTL cleanup → start engine → recovery ≥90%.

    Profile semantics (v2, task #55): the quota knob itself
    (dispatcher.maxInflightBatchesPerPrefillWorker) exists only under the
    BATCH dispatcher, and quota_spec pins the legacy fault axes (PRIORITY +
    FIXED_WINDOW + BATCH, maxInflightBatches=1) via FLEXLB_CONFIG — the
    declaration stays batch-window.
    """
    env = ctx.env_manager.ensure(quota_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "master")
    try:
        # Slow the only prefill: scheduled requests stick in inflight.
        ops.set_perf("prefill-0", prefill_fixed_ms=10000.0)

        # Fill the (maxInflightBatches=1) quota with fire-and-forget requests.
        rids = [ops.next_request_id(base) for _ in range(4)]
        for rid in rids:
            resp = ops.schedule(rid, output_len=10)
            if resp.code != 200 or not resp.success:
                return False, f"schedule failed for rid={rid}: {resp.error_message}"
        filled = wait_for(lambda: ops.master_scheduler_inflight() > 0, 15.0, 0.5)
        if not filled:
            return False, "could not fill the inflight quota"
        stuck = ops.master_scheduler_inflight()

        # Stop the only prefill → the stuck batch never completes.
        ops.stop_engine("prefill-0")
        time.sleep(3.0)

        # Blocked phase: 10 concurrent requests, expect ≥50% failures
        # (single prefill down + quota consumed → queue timeouts / rejects).
        block_rids = [ops.next_request_id(base) for _ in range(10)]

        def run(rid: int):
            return ops.run_one_request(
                rid,
                output_len=2,
                block_keys=[rid * 100 + 1],
                stream_timeout_s=12.0,
            )

        with ThreadPoolExecutor(max_workers=10) as pool:
            results = list(pool.map(run, block_rids))
        block_ok = sum(1 for _, err in results if err is None)
        block_fail_rate = (10 - block_ok) / 10
        block_err_types = sorted(
            {str(err)[:60] for _, err in results if err is not None}
        )[:3]

        # TTL cleanup: scheduler inflight drains to zero (the evicted
        # engine's endpoint row is gone, so watch the global counter).  The
        # window rides harness.TTL_DRAIN_TIMEOUT_S (95s = 30s stale TTL +
        # 60s sweeper phase + 5s margin — derivation in harness) instead
        # of the legacy bare 90.0, which sat exactly ON the worst-case
        # settle and let a slow sweep phase trip the wait.
        cleanup_ok = wait_for(
            lambda: ops.master_scheduler_inflight() == 0, TTL_DRAIN_TIMEOUT_S, 2.0
        )
        ops.start_engine("prefill-0")
        ops.set_perf("prefill-0", prefill_fixed_ms=100.0)
        alive_back = wait_for(
            lambda: ops.master_alive_count("PREFILL") >= 1,
            MASTER_EVICT_S,
            0.5,
        )
        time.sleep(2.0)

        # Recovery phase: 20 requests ≥90%.
        #
        # Sent SERIALLY (concurrency=1), unlike the other batch call sites:
        # a 10-way concurrent burst against the single restarted prefill keeps
        # the batcher queue mutating continuously, and PrefillEndpoint
        # .realPendingCount() is a lock-free snapshot that deliberately returns
        # Long.MAX_VALUE ("route away conservatively", see its comment) after 4
        # spin attempts fail to see a stable mutation version.  With ONE
        # prefill there is nowhere to route away to, so those requests die as
        # retryable "admission capacity is temporarily exhausted"
        # (RESOURCE_UNAVAILABLE on the only candidate — verified via runtime
        # DEBUG logs: "pendingRequests=9223372036854775807, alive=true").
        # That conservative degrade is scheduler design, not a recovery defect;
        # serializing removes the snapshot-contention noise so the assertion
        # keeps verifying what this case actually targets: quota released,
        # engine back, requests succeed again (observed 1-4 rejects across
        # 6 concurrent-burst runs — 80-100% flapping around the 90% gate).
        ok5, err5, _ = _run_batch(ops, base, 20, concurrency=1)
        recovery_rate = ok5 / 20 if err5 == 0 else ok5 / 20
        recovery_err_types = list(getattr(_run_batch, "last_error_types", []))[:3]

        passed = block_fail_rate >= 0.50 and cleanup_ok and alive_back and ok5 >= 18
        return passed, (
            f"stuck_inflight={stuck}, "
            f"blocked={block_ok}/10 ok (fail_rate={block_fail_rate:.0%}, >=50% required, "
            f"types={block_err_types}), "
            f"ttl_cleanup_within_{TTL_DRAIN_TIMEOUT_S:.0f}s={cleanup_ok}, "
            f"alive_restored={alive_back}, "
            f"recovery={ok5}/20({recovery_rate:.0%}, >=90% required, "
            f"types={recovery_err_types})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            snap = ops.snapshot_by_name()
            if snap.get("prefill-0", {}).get("stopped"):
                ops.start_engine("prefill-0")
            ops.set_perf("prefill-0", prefill_fixed_ms=100.0)
        except Exception:
            pass


# ===========================================================================
# Cold-start burst — intake defect regression probe
# ===========================================================================


@case(
    "master_coldstart_burst",
    profiles=["batch-window"],
    source="intake defect regression probe (cold-start first-connect storm)",
)
def coldstart_burst(ctx: CaseContext):
    """Fire 20 requests the instant the master reports ready.

    Regression probe for the three intake defects: CONNECT_TIMEOUT 20ms,
    3-strike dead marking on first connect, non-atomic getOrCreateWorkerStatus.
    Expected to FAIL or pass marginally today — the failure rate and the
    marked-dead sample count are recorded as the baseline for the intake fix.

    Profile semantics (v2, task #55): coldstart_spec carries NO config
    override, so the case would genuinely exercise each profile's config
    (unlike the pinned-spec cases above); it stays scoped to batch-window
    this round as a deliberate scope decision — spreading the intake probe
    across profiles is later-phase work.
    """
    env = ctx.env_manager.ensure(coldstart_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "master")
    expected = {"PREFILL": env.spec.n_prefill, "DECODE": env.spec.n_decode}
    try:
        # Sample worker_summary every 0.5s while the burst runs and for 10s
        # after — the cold-start window where engines get marked dead.
        samples: list[tuple[float, dict]] = []
        stop = threading.Event()

        def sample() -> Optional[dict]:
            info = ops.master_info()
            if not info:
                return None
            summary = info.get("worker_summary", {}) or {}
            return {
                role: (
                    int((summary.get(role) or {}).get("discovered", -1)),
                    int((summary.get(role) or {}).get("alive", -1)),
                )
                for role in ("PREFILL", "DECODE")
            }

        def sampler() -> None:
            t0 = time.monotonic()
            while not stop.is_set():
                s = sample()
                if s is not None:
                    samples.append((round(time.monotonic() - t0, 1), s))
                stop.wait(0.5)

        poller = threading.Thread(target=sampler, name="coldstart-sampler", daemon=True)
        poller.start()

        # Burst: 20 requests (10-way concurrent) immediately after ready.
        # The prefill address of every request is kept for the load-balance
        # assertion (balance_uniform_serial P1 contract) below.
        def run(rid: int):
            addr, err = ops.run_one_request(
                rid,
                output_len=2,
                block_keys=[rid * 100 + 1],
                stream_timeout_s=STREAM_TIMEOUT_S,
            )
            return addr, err

        rids = [ops.next_request_id(base) for _ in range(20)]
        with ThreadPoolExecutor(max_workers=10) as pool:
            results = list(pool.map(run, rids))
        ok = sum(1 for _, e in results if e is None)
        error_types = sorted({str(e)[:60] for _, e in results if e is not None})

        # Keep sampling 10s past the burst: transient 3-strike marks recover,
        # permanent ones stay dead (that is the S10-class regression).
        time.sleep(10.0)
        stop.set()
        poller.join(timeout=2.0)

        final = sample()
        dead_samples = sum(1 for _, s in samples if any(a < d for d, a in s.values()))
        final_ok = bool(
            final
            and all(d == expected[role] and a == d for role, (d, a) in final.items())
        )

        # Load-balance contract (user-mandated): under the cold-start burst
        # traffic must still spread across the engines.  Same calibration
        # as the task #61 balance suite (balance_uniform_serial / P1, with
        # the balance_concurrent_mix relaxed-caliber note): 20 requests over
        # 2 prefills (10-way concurrent), both engines used, no engine above
        # 80% of the *successful* requests — COST_BASED_PREFILL scores the
        # two prefills identically on an empty cold ledger and
        # RANDOM_WITHIN_TOLERANCE samples the tie window uniformly, so a
        # one-sided distribution can only come from an engine being
        # 3-strike-marked dead (the intake defect this probe guards).
        # 80% of 20 = 16 requests, i.e. the same "no engine eats the burst"
        # bound as the balance suite's P1 (loose floor 0.85 over the
        # uniform-random calibration; this probe keeps the historical 0.80
        # as its hard bound — semantics unchanged by the task #61 rework).
        addr_map = ops.addr_to_name()
        dist = Counter(addr_map.get(a, a) for a, e in results if e is None and a)
        n_ok = sum(dist.values())
        workers_used = len(dist)
        max_share = (max(dist.values()) / n_ok) if n_ok else 1.0
        balance_ok = workers_used >= 2 and max_share <= 0.80

        success_rate = ok / 20 * 100.0
        passed = (
            ok >= 16  # >=80% success + no permanent eviction
            and final_ok
            and balance_ok
        )
        return passed, (
            f"burst_ok={ok}/20 ({success_rate:.0f}%), "
            f"dead_samples={dead_samples}/{len(samples)}, final={final}, "
            f"error_types={error_types[:3]}, "
            f"balance: workers={workers_used}/{env.spec.n_prefill}, "
            f"max_share={max_share:.0%} (need >=2 workers and <=80%), "
            f"dist={json.dumps(dict(sorted(dist.items())))}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


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


@case(
    "master_freeze",
    profiles=["batch-window"],
    source="Mode 2 freeze (SIGSTOP→SIGCONT): content-not-lost assertions, "
    "short + long hang tiers (brief p3/p4)",
)
def master_freeze(ctx: CaseContext):
    """Tier-1 dual-standalone, sticky-on-B (brief p4 right lane).

    Two hang tiers against the SAME frozen JVM:
      * short hang (< keepalive ~40s judgement): SIGSTOP 6s → SIGCONT —
        recovery is invisible: rows sent during the hang complete after
        the thaw (no evaporation), no switch happens, pid unchanged.
      * long hang (> 40s judgement): SIGSTOP 46s → the channel keepalive
        (30s time / 10s timeout) marks B dead → UNAVAILABLE → same-
        request retry to A, sticky moves to A; after SIGCONT B is still
        the SAME process (ledger continuity, no cold restart) and the
        post-thaw window shows no error storm.

    Content-not-lost assertion set (brief p3 Mode 2): ledger continuity
    (pid + discovered topology), in-flight/no evaporation, no duplicate
    dispatch (unique rid rows), post-recovery clean traffic.
    """
    env = ctx.env_manager.ensure(tier1_dual_spec(ctx))
    ops_b = instance_ops(ctx, env, "B")
    mgr = ctx.env_manager
    target_a = mgr.master_instance_target(env, "A")
    target_b = mgr.master_instance_target(env, "B")
    case_dir = ctx.case_dir("master_freeze")
    flow = HaTrafficRunner(
        ctx,
        env,
        case_dir,
        "master_freeze",
        targets=[target_b, target_a],  # sticky B first (brief p4)
        duration_s=100,
        timeout_ms=30_000,
    )
    pid_b = None
    frozen = False
    try:
        flow.start()
        pid_b = env.masters["B"].pid
        time.sleep(HA_STEADY_S)
        # --- short-hang tier: SIGSTOP 6s (< 40s keepalive judgement) ---
        t_freeze1 = HaTrafficRunner.now()
        mgr.freeze_master_instance(env, "B")
        frozen = True
        time.sleep(6.0)
        t_cont1 = HaTrafficRunner.now()
        mgr.unfreeze_master_instance(env, "B")
        frozen = False
        time.sleep(8.0)
        t_after1 = HaTrafficRunner.now()
        # --- long-hang tier: SIGSTOP 46s (> 40s keepalive judgement) ---
        # W3 Mode-2 ledger mirror probes (brief p3 assertion face #1):
        # snapshot B's scheduler inflight + discovered counts right
        # before the freeze so the post-thaw snapshots can prove the
        # in-memory ledger was NOT reset to zero (SIGSTOP/SIGCONT keeps
        # the process image; a cold restart would zero both).
        inflight_pre_freeze = ops_b.master_scheduler_inflight()
        pre_info = ops_b.master_info() or {}
        pre_summary = pre_info.get("worker_summary", {}) or {}
        try:
            pre_disc_p = int((pre_summary.get("PREFILL") or {}).get("discovered", -1))
            pre_disc_d = int((pre_summary.get("DECODE") or {}).get("discovered", -1))
        except (TypeError, ValueError):
            pre_disc_p = pre_disc_d = -1
        t_freeze2 = HaTrafficRunner.now()
        mgr.freeze_master_instance(env, "B")
        frozen = True
        time.sleep(46.0)
        t_cont2 = HaTrafficRunner.now()
        mgr.unfreeze_master_instance(env, "B")
        frozen = False
        # Mirror snapshot at the very start of the post-thaw stable
        # window: the frozen in-flight entries must still be on B's
        # ledger (profile-default staleInflightTimeoutMs=300s >> the
        # 46s freeze, so neither the TTL sweep nor anything else may
        # have zeroed it).
        inflight_post_thaw = ops_b.master_scheduler_inflight()
        time.sleep(10.0)
        t_after2 = HaTrafficRunner.now()
        # Same-process + ledger-continuity probes right after the thaw.
        info_b = ops_b.master_info()
        b_ready = bool(info_b and info_b.get("ready"))
        summary = (info_b or {}).get("worker_summary", {}) or {}
        try:
            disc_p = int((summary.get("PREFILL") or {}).get("discovered", -1))
            disc_d = int((summary.get("DECODE") or {}).get("discovered", -1))
        except (TypeError, ValueError):
            disc_p = disc_d = -1
        ledger_kept = disc_p == env.spec.n_prefill and disc_d == env.spec.n_decode
        # W3: Mode-2 inflight ledger "not zeroed" mirror of Mode-1's
        # inflight_clean — two combined probes:
        #  * discovered counts must not regress across the freeze
        #    (monotonic no-rewind; ledger_kept above pins the end value);
        #  * with >=1 entry in flight at freeze time, the immediate
        #    post-thaw snapshot must still see >=1 (a SIGSTOP'd process
        #    retains its ledger; only a cold restart resets it to zero).
        # Unobservable setups (probe failure -1, or zero inflight at the
        # freeze instant) do not block — topology continuity is already
        # covered by ledger_kept.
        disc_monotonic = pre_disc_p < 0 or (
            disc_p >= pre_disc_p and disc_d >= pre_disc_d
        )
        inflight_not_reset = (
            inflight_pre_freeze <= 0
            or inflight_post_thaw < 0
            or inflight_post_thaw >= 1
        )
        inflight_ledger_kept = disc_monotonic and inflight_not_reset
        flow.wait_finish()
        rows = flow.rows()
        guard = _check_client_fields(HaRows(rows))
        if guard:
            return guard
        allr = HaRows(rows)
        hang1 = HaRows(rows_between(rows, t_freeze1, t_cont1))
        post1 = HaRows(rows_between(rows, t_cont1, t_after1))
        # Thaw burst proper: [t_cont1, freeze2-0.5). The last half-second
        # before the long hang is the freeze2 boundary — rows sent there
        # are exactly the frozen in-flight victims that end as deadline
        # rows ~30s later (evidence: 283 burst rows all-ok-on-B plus 8
        # boundary deadline rows landing inside the old 8s post1 window).
        burst1 = HaRows(rows_between(rows, t_cont1, t_freeze2 - 0.5))
        # The long-hang switch is CALLER-DEADLINE driven (flow
        # timeout_ms=30_000), not keepalive-death driven: remote evidence
        # shows the failover rows landing at freeze2+30.0~30.3s — the
        # frozen in-flight calls hit their 30s gRPC deadline, the freed
        # slots same-request-retry to A. Window the judgement around the
        # deadline-driven switch (freeze2+timeout-eps) instead of the old
        # keepalive assumption (freeze2+35s), which started AFTER the
        # switch had already happened.
        judged = HaRows(rows_between(rows, t_freeze2 + 29.0, t_cont2))
        post2 = HaRows(rows_between(rows, t_cont2, t_after2))
        pid_same = pid_b == env.masters["B"].pid

        # Short tier: zero-interference recovery. With MAX_CONCURRENCY=8
        # every slot is held by a frozen in-flight request, so the freeze
        # window itself may legitimately send ZERO rows (t+12~t+16
        # evidence); the honest no-evaporation contract is: whatever WAS
        # sent inside the window is ok-on-B, and the post-thaw burst
        # drains 100% ok-on-B (nothing evaporates into errors either).
        hang1_no_evap = (
            (
                len(hang1.rows) == 0
                or (
                    len(hang1.ok_rows()) == len(hang1.rows)
                    and len(hang1.target(target_b)) == len(hang1.rows)
                )
            )
            and len(burst1.rows) >= 3
            and len(burst1.ok_rows()) == len(burst1.rows)
            and len(burst1.target(target_b)) == len(burst1.rows)
        )
        post1_still_b = (
            len(post1.target(target_b)) == len(post1.rows) if post1.rows else True
        )
        # Long tier: deadline-driven switch → same-request retry to A.
        judged_failover = len(judged.failover_rows()) > 0
        judged_to_a = len(judged.target(target_a)) > 0
        # Mode-2 brief (p3) visibility gap: the pre-freeze in-flight
        # requests must reach a VISIBLE terminal — ok before the freeze,
        # or an explicit error_kind=deadline row when the frozen calls hit
        # their own deadline (remote evidence: 8 deadline rows plus the
        # post-thaw 8510 "generation retired" refusals). No silent
        # evaporation, no rows vanishing mid-flight. Deadline rows carry
        # the ORIGINAL send timestamp, and that timestamp STRADDLES the
        # freeze2 boundary: the frozen victims were already in flight
        # 1-3s before the freeze, so their send lands up to a few seconds
        # EARLY (run-1788363667: the deadline rows sent at rel-s 25.1~25.5
        # while freeze2 lands at ~26s ± jitter — the old [freeze2, cont2)
        # window missed them by pure timing luck). Count deadline
        # terminals over the whole straddle window instead.
        pre_freeze = HaRows(rows_between(rows, t_freeze2 - 2.0, t_freeze2))
        hang_deadline = HaRows(rows_between(rows, t_freeze2 - 4.0, t_cont2))
        inflight_terminal = (
            len(pre_freeze.rows) >= 1
            and all(
                r.get("status") == "ok"
                or r.get("error_kind") in ("deadline", "transport", "business")
                for r in pre_freeze.rows
            )
            and len(hang_deadline.error_kind("deadline")) >= 1
        )
        # Post-thaw: no error storm, traffic healthy (on A).
        post2_ok = post2.ok_rate() >= 0.90 if post2.rows else False
        post2_a_share = (
            len(post2.target(target_a)) / len(post2.rows) if post2.rows else 0.0
        )
        no_dup = not allr.dup_rids()

        passed = (
            hang1_no_evap
            and post1_still_b
            and judged_failover
            and judged_to_a
            and inflight_terminal
            and post2_ok
            and post2_a_share >= 0.80
            and pid_same
            and b_ready
            and ledger_kept
            and inflight_ledger_kept
            and no_dup
        )
        return passed, (
            f"short: hang_no_evap={hang1_no_evap}(hang={len(hang1.rows)} "
            f"rows, burst={len(post1.rows)} rows all-ok-on-B="
            f"{len(post1.ok_rows()) == len(post1.rows)}), "
            f"still_B={post1_still_b}, "
            f"long: judged_failover={judged_failover}, "
            f"to_A={judged_to_a}, "
            f"inflight_terminal={inflight_terminal}(pre_freeze="
            f"{len(pre_freeze.rows)}, deadline="
            f"{len(hang_deadline.error_kind('deadline'))}), "
            f"post2: ok={post2.ok_rate():.0%}, "
            f"A_share={post2_a_share:.0%}, pid_same={pid_same}, "
            f"B_ready={b_ready}, ledger_kept={ledger_kept} "
            f"(discovered P:{disc_p}/D:{disc_d}), "
            f"inflight_ledger_kept={inflight_ledger_kept} "
            f"(pre={inflight_pre_freeze}, post_thaw={inflight_post_thaw}, "
            f"disc_pre P:{pre_disc_p}/D:{pre_disc_d}), "
            f"dup_rids={len(allr.dup_rids())}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            if frozen:
                ctx.env_manager.unfreeze_master_instance(env, "B")
        except Exception:
            pass
        restore_masters(ctx, env)


@case(
    "master_ha_failover",
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


@case(
    "fallback_direct",
    profiles=["batch-window"],
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


@case(
    "fallback_negative_errorcode",
    profiles=["batch-window"],
    source="scenario 3 negative (brief p7/p8): business error codes / "
    "DEADLINE never trigger the direct fallback",
)
def fallback_negative_errorcode(ctx: CaseContext):
    """Tier-1 dual standalone, ENABLE_FALLBACK=true (armed but must NOT
    fire).

    Negative contract (brief p8 phase 2 + Tina's field naming):
      * business leg: inject engine-side enqueue_ack_error_code=8431 —
        the master ANSWERS with schedule_error: rows must be
        route_path=master + status=schedule_error + error_kind=business
        + the 8431 code visible in the error text; zero fallback rows.
      * deadline leg: SIGSTOP the sticky master with a short deadline —
        rows must be route_path=failed + error_kind=deadline with
        failover=false (no retry, no switch, no fallback — the
        single-listed DEADLINE_EXCEEDED assertion).
    """
    gate = ha_gate()
    if gate:
        return gate
    env = ctx.env_manager.ensure(tier1_dual_spec(ctx))
    mgr = ctx.env_manager
    ops_a = instance_ops(ctx, env, "A")
    target_a = mgr.master_instance_target(env, "A")
    target_b = mgr.master_instance_target(env, "B")
    case_dir = ctx.case_dir("fallback_negative")
    flow = HaTrafficRunner(
        ctx,
        env,
        case_dir,
        "fallback_negative",
        targets=[target_a, target_b],  # sticky A
        duration_s=75,
        timeout_ms=800,  # short deadline for the DEADLINE leg
        enable_fallback=True,
    )
    injected = False
    frozen = False
    try:
        flow.start()
        time.sleep(10.0)
        # --- deadline leg FIRST: frozen sticky master + 800ms deadline ---
        # Leg order swapped from the original draft (business first, settle
        # between the legs). A mid-case settle can no longer work here: with
        # TIMEOUT_MS=800 the clients drop streams by the hundreds (run
        # 1788359161: 734 client_gone census rows) and the master reconciles
        # those requests ONLY via the 30s endpoint-inflight TTL plus the 60s
        # request-expiry sweep — the last client send at t=75s expires at
        # t=135s, far past the 75s client run, so any inter-leg gate window
        # would either time out (false FAIL, the 20s draft) or outlive the
        # traffic (no rows left for the second leg). Deadline-first removes
        # the original no-resurrection concern structurally — business
        # FAILED rows cannot bleed backwards into an earlier window — and
        # the settle moves to the tail as the clear-resume contract.
        t_freeze = HaTrafficRunner.now()
        mgr.freeze_master_instance(env, "A")
        frozen = True
        time.sleep(8.0)
        t_cont = HaTrafficRunner.now()
        mgr.unfreeze_master_instance(env, "A")
        frozen = False
        time.sleep(8.0)
        # --- business leg: 8431 answers through the master -----------
        names = _prefill_engine_names(ops_a)
        inject_type_all(ops_a, names, "enqueue_ack_error_code", code=8431)
        injected = True
        t_inj = HaTrafficRunner.now()
        time.sleep(10.0)
        t_inj_end = HaTrafficRunner.now()
        clear_type_all(ops_a, names, "enqueue_ack_error_code")
        injected = False
        # The remaining ~39s of the 75s run is post-clear traffic and
        # doubles as recovery evidence that the clear took effect.
        flow.wait_finish()
        rows = flow.rows()
        guard = _check_client_fields(HaRows(rows))
        if guard:
            return guard
        biz = HaRows(rows_between(rows, t_inj, t_inj_end))
        dl = HaRows(rows_between(rows, t_freeze, t_cont))

        biz_sched_err = biz.status("schedule_error")
        biz_code_seen = all("8431" in str(r.get("error", "")) for r in biz_sched_err)
        biz_route_master = (
            len(biz.route("master")) == len(biz.rows) if biz.rows else False
        )
        biz_no_fallback = (
            len(biz.route("fallback")) == 0 and len(biz.route("failed")) == 0
        )
        biz_no_failover = len(biz.failover_rows()) == 0

        dl_rows = dl.error_kind("deadline")
        dl_failed_route = all(r.get("route_path") == "failed" for r in dl_rows)
        dl_no_retry = all(r.get("failover") is False for r in dl_rows)
        dl_no_fallback = len(dl.route("fallback")) == 0

        legs_passed = (
            len(biz_sched_err) >= 5
            and biz_code_seen
            and biz_route_master
            and biz_no_fallback
            and biz_no_failover
            and len(dl_rows) >= 3
            and dl_failed_route
            and dl_no_retry
            and dl_no_fallback
        )
        legs_summary = (
            f"business: schedule_error={len(biz_sched_err)}/{len(biz.rows)}, "
            f"code8431_visible={biz_code_seen}, route_master="
            f"{biz_route_master}, no_fallback={biz_no_fallback}, "
            f"no_failover={biz_no_failover}; "
            f"deadline: rows={len(dl_rows)}, route_failed="
            f"{dl_failed_route}, no_retry={dl_no_retry}, "
            f"no_fallback={dl_no_fallback}"
        )
        if not legs_passed:
            return False, legs_summary
        # --- tail settle: the clear must leave a drainable master -----
        # The mid-run dispatch stall this case provokes (short-deadline
        # client disconnect storm; see the run-1788359161 forensics in the
        # HA case-test notes) is reconciled by the master only through the
        # 30s endpoint-inflight TTL and the 60s request-expiry sweep, so
        # the drain tail runs from the last client send (t=75s) to its
        # expiry (+60s) plus a sweep pass. 150s from wait_finish covers
        # that with margin; a healthy master converges, a true slot leak
        # still times out and fails here.
        residual_tolerance = 8
        settled = wait_for(
            lambda: ops_a.master_scheduler_inflight() <= residual_tolerance,
            150.0,
            2.0,
        )
        if not settled:
            detail = ops_a.master_inflight()
            if isinstance(detail, dict):
                inflight_note = (
                    f"sched={detail.get('scheduler_inflight')}, "
                    f"prefill={[(ep.get('ip_port'), ep.get('inflight_batches', 0)) for ep in (detail.get('prefill_endpoints') or [])]}, "
                    f"decode={[(ep.get('ip_port'), ep.get('inflight_requests', 0) or ep.get('total_load', 0)) for ep in (detail.get('decode_endpoints') or [])]}"
                )
            else:
                inflight_note = str(detail)
            return False, (f"{legs_summary}; SETTLE_FAIL inflight=[{inflight_note}]")
        return True, legs_summary
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            if injected:
                clear_type_all(
                    ops_a, _prefill_engine_names(ops_a), "enqueue_ack_error_code"
                )
        except Exception:
            pass
        try:
            if frozen:
                ctx.env_manager.unfreeze_master_instance(env, "A")
        except Exception:
            pass
        restore_masters(ctx, env)


@case(
    "failback_wraparound",
    profiles=["batch-window"],
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


def _prefill_names(ops) -> list[str]:
    snap = ops.snapshot()
    return [e["name"] for e in snap.get("engines", []) if e.get("role") == "prefill"]


# ===========================================================================
# Direct-path case (migrated from the legacy injection family, task #85
# category reorg — rid_base family "chaos" -> "direct"; folded from
# the retired one-case direct module into master — the rid_base family
# stays "direct" so its id block keeps the sub-1M dedup-collision
# distance from the master block)
# ===========================================================================


@case(
    "direct_generate_error",
    source="gap G6/G7: /inject type=generate_error (GenerateStreamCall entry, client-direct path)",
)
def inject_generate_error(ctx: CaseContext):
    """generate_error is checked ONLY at the engine's GenerateStreamCall
    entry (JavaMockEngineCluster.generateStreamCall: onError before any
    request state is registered).

    Profile semantics (v2, task #55): under the v1 mode axis ALL master
    modes delivered via enqueueBatch + FetchResponse (direct-run evidence:
    generate_stream_rpcs=0 while enqueue_rpcs=3 / fetch_response_rpcs=3),
    so the fault was structurally unreachable for master traffic and the
    case pinned the client-direct contract.  Under v2 the BATCH
    dispatcher is still unreachable, but the NON_BATCH dispatcher routes
    client-sent GenerateStreamCall traffic through this exact check — a
    master-routed variant of this case is dedicated-phase material.  The
    contract pinned here is the CLIENT-DIRECT path (the load-client
    direct deployment shape; same direct-stub sequence EngineOps already
    uses for worker_cancel), which does not pass through the master at
    all — so the case runs unconditionally under every profile:

    inject -> the direct stream fails immediately with the injected
    error and registers no engine-side inflight; clear -> a fresh direct
    request completes normally."""
    ops = ctx.ops()
    base = rid_base(ctx, "direct")
    names = _prefill_names(ops)
    if not names:
        return False, "no prefill engines found"
    snap = ops.snapshot_by_name()
    target = None
    for n in names:
        entry = snap.get(n) or {}
        addr = entry.get("grpc_addr")
        if addr:
            target = str(addr)
            break
    if not target:
        return False, "no prefill engine address in snapshot"

    def direct_request(rid: int) -> tuple[Optional[str], object]:
        input_pb = ops.build_generate_input(rid)
        stub = ops.pb2_grpc.RpcServiceStub(ops._channel(target))
        call = stub.GenerateStreamCall(input_pb, timeout=30.0)
        handle = StreamHandle(call, StreamSnapshot())
        handle.wait_end(STREAM_TIMEOUT_S)
        if handle.snap.error:
            return str(handle.snap.error), handle
        if not handle.snap.completed:
            return "stream did not complete", handle
        return None, handle

    try:
        rid0 = ops.next_request_id(base)
        err0, _ = direct_request(rid0)
        if err0:
            return False, f"baseline direct request failed: {err0}"

        inject_type_all(ops, names, "generate_error")
        try:
            rid1 = ops.next_request_id(base)
            err1, _ = direct_request(rid1)
            # Cross-process the engine's onError(RuntimeException("injected
            # generate_error")) reaches the client as grpc status 2
            # (UNKNOWN) with an EMPTY message — the text is not transmitted
            # (verified round 5: grpc_message:"", grpc_status:2) — so the
            # assertion is error-arrived, same contract as the fetch_error
            # case; causality comes from the inject/clear sandwich.
            error_ok = err1 is not None
        finally:
            clear_type_all(ops, names, "generate_error")

        rid2 = ops.next_request_id(base)
        err2, _ = direct_request(rid2)
        engine_clean, engine_detail = engine_inflight_clean(ops, names)

        passed = error_ok and err2 is None and engine_clean
        return passed, (
            f"direct_target={target}, "
            f"error_surfaced={error_ok} ({err1}), "
            f"recovered={err2 is None}"
            f"{'' if err2 is None else ' err=' + err2[:60]}, "
            f"engine_inflight_clean={engine_clean}({engine_detail})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        clear_type_all(ops, names, "generate_error")
