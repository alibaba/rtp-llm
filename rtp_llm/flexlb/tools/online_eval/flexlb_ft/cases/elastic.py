"""Elastic-category cases: dynamic engine scale-out/in.

Theme: engines joining and leaving the cluster through the mock control
plane (/add_engine + /remove_engine) with the file-based dynamic
discovery chain enabled end to end — mock ``--discovery-file`` →
DiscoveryFileStore (atomic rewrite) → master ``FLEXLB_DISCOVERY_FILE``
→ FileServiceDiscovery (re-read per poll) → EngineSyncRunner →
EndpointRegistry → routing.  The master must converge to the new
topology (add ~26ms, remove ~1s per the verified flexlb-api behaviour,
FileDiscoveryDynamicScaleEndToEndTest), keep background traffic alive
across the transition, evict removed engines within the health window,
and survive concurrent add/remove storms.

Elastic scaling is a normal functional requirement (user ruling
2026-08), NOT a fault scenario — the cases pin the discovery/routing
contract, not any injected failure.  Convergence bounds assert at
second-scale timeouts (10-15s) to stay robust against slow CI machines.
"""

from __future__ import annotations

import json
import random
import threading
import time
from typing import Optional

from ..context import CaseContext, CaseDef, rid_base
from ..grade import GradeReport
from ..harness import (
    TTL_DRAIN_TIMEOUT_S,
    AssertUtils,
    EnvSpec,
    _accepted,
    _BackgroundFlow,
    _cleanup_dynamic,
    _discovery_entry_count,
    _discovery_has_http_port,
    _discovery_payload,
    _dynamic_engines,
    _elastic_env,
    _pump_until_accepted,
    _run_batch,
    _wait_master_alive,
    _wait_master_topology,
    fault_env_config,
    fault_env_perf,
    http_get_status,
    wait_for,
)

ELASTIC_CASES: list[CaseDef] = []

# File-discovery convergence caps (flexlb-api-verified: add ~26ms, remove
# ~1s; the caps below are deliberately loose, second-scale, for slow CI
# machines).
ADD_CONVERGENCE_S = 10.0
REMOVE_CONVERGENCE_S = 10.0
# Master eviction of a vanished discovery entry (sync 20ms + stale window) —
# generous cap so slow machines do not flake.
MASTER_EVICT_S = 30.0


def case(name: str, profiles=None, requires=None, source: str = ""):
    def deco(fn):
        ELASTIC_CASES.append(
            CaseDef(
                name=name,
                category="elastic",
                fn=fn,
                profiles=profiles,
                requires=requires,
                source=source,
            )
        )
        return fn

    return deco


def _master_http(ops) -> str:
    return f"http://127.0.0.1:{ops.master_http_port}"


# ===========================================================================
# Elastic cases (migrated from the legacy elastic group, task #85 category
# reorg — functional taxonomy, NOT fault scenarios)
# ===========================================================================


@case(
    "elastic_add_flow",
    profiles=["batch-window"],  # elastic_spec pins the legacy fault axes
    source="elastic acceptance: add under load (FileDiscoveryDynamicScaleEndToEndTest phase 2)",
)
def elastic_add_flow(ctx: CaseContext):
    env, ops = _elastic_env(ctx)
    base = rid_base(ctx, "elastic")
    flow: Optional[_BackgroundFlow] = None
    try:
        # Warm the initial topology: every case may run in a shared env, so
        # make sure no dynamic leftovers from a previous case exist.
        _cleanup_dynamic(ops, env)

        flow = _BackgroundFlow(ops, base, interval_s=0.2)
        flow.start()
        time.sleep(1.0)  # let the flow ramp up before the mutation

        status, body = ops.add_engine("prefill")
        if status != 200:
            flow.stop()
            return False, f"add_engine failed: {status} {body}"
        new_name = body["engine"]
        new_port = body["port"]

        converged = wait_for(
            lambda: _accepted(ops, new_name) > 0, ADD_CONVERGENCE_S, 0.2
        )
        time.sleep(1.0)  # flow keeps running a little past convergence
        total, ok = flow.stop()
        rate = ok / total if total else 0.0
        snap = ops.snapshot_by_name()
        passed = converged and rate >= 0.90
        return passed, (
            f"new_engine={new_name}(grpc={new_port}), "
            f"accepted={snap.get(new_name, {}).get('accepted', 0)}, "
            f"converged_within_{ADD_CONVERGENCE_S:.0f}s={converged}, "
            f"flow_success={ok}/{total}({rate:.1%}, >=90% required)"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if flow is not None:
            flow.stop()
        try:
            _cleanup_dynamic(ops, env)
        except Exception:
            pass


@case(
    "elastic_remove_flow",
    profiles=["batch-window"],  # elastic_spec pins the legacy fault axes
    source="elastic acceptance: remove under load (FileDiscoveryDynamicScaleEndToEndTest phase 3)",
)
def elastic_remove_flow(ctx: CaseContext):
    env, ops = _elastic_env(ctx)
    base = rid_base(ctx, "elastic")
    flow: Optional[_BackgroundFlow] = None
    try:
        _cleanup_dynamic(ops, env)

        status, body = ops.add_engine("prefill")
        if status != 200:
            return False, f"add_engine failed: {status} {body}"
        new_name = body["engine"]
        new_port = body["port"]

        # Wait for the discovery file + master to pick the new engine up.
        in_file = wait_for(
            lambda: _discovery_has_http_port(env, new_port - 1),
            ADD_CONVERGENCE_S,
            0.1,
        )
        alive3 = _wait_master_alive(ops, "PREFILL", 3, MASTER_EVICT_S)
        if not (in_file and alive3):
            return False, (
                f"engine {new_name} never converged: discovery_file={in_file}, "
                f"master_alive_prefill={ops.master_alive_count('PREFILL')}"
            )

        flow = _BackgroundFlow(ops, base, interval_s=0.2)
        flow.start()
        if not _pump_until_accepted(ops, new_name, base, 10.0):
            flow.stop()
            return False, "new engine did not accept any request before removal"

        accepted_at_removal = _accepted(ops, new_name)
        status, rm_body = ops.remove_engine(engine_name=new_name)
        if status != 200:
            flow.stop()
            return False, f"remove_engine failed: {status} {rm_body}"

        # Removal window: other engines keep serving; removed one must be gone
        # from the mock services map AND the discovery file.
        time.sleep(3.0)
        total, ok = flow.stop()
        rate = ok / total if total else 0.0

        gone_from_snapshot = new_name not in ops.snapshot_by_name()
        gone_from_file = wait_for(
            lambda: not _discovery_has_http_port(env, new_port - 1),
            REMOVE_CONVERGENCE_S,
            0.1,
        )
        # In-flight requests reach a terminal state: master inflight drains
        # (TTL_DRAIN_TIMEOUT_S — covers the 30s stale-inflight TTL plus the
        # 60s ExpirationTimer sweep; the legacy 90s cap sat below the
        # worst-phase settle and let residue poison later cases on this
        # shared env, task #87).
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), TTL_DRAIN_TIMEOUT_S
        )
        passed = rate >= 0.90 and gone_from_snapshot and gone_from_file and inflight_ok
        return passed, (
            f"removed={new_name}(grpc={new_port}, accepted_at_removal={accepted_at_removal}), "
            f"flow_success={ok}/{total}({rate:.1%}), "
            f"gone_from_snapshot={gone_from_snapshot}, "
            f"gone_from_discovery_file={gone_from_file}, "
            f"inflight_clean={inflight_ok}({inflight_detail})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if flow is not None:
            flow.stop()
        try:
            _cleanup_dynamic(ops, env)
        except Exception:
            pass


@case(
    "elastic_add_remove_cycle",
    profiles=["batch-window"],  # elastic_spec pins the legacy fault axes
    source="elastic acceptance: 3x add→verify→remove→verify cycle",
)
def elastic_add_remove_cycle(ctx: CaseContext):
    env, ops = _elastic_env(ctx)
    base = rid_base(ctx, "elastic")
    try:
        _cleanup_dynamic(ops, env)
        p_prefill, p_decode = _discovery_entry_count(env)
        if p_prefill < 0:
            return False, "discovery file unreadable before cycle"

        round_details = []
        all_ok = True
        for round_no in range(1, 4):
            status, body = ops.add_engine("prefill")
            if status != 200:
                all_ok = False
                round_details.append(f"r{round_no}: add failed {status}")
                break
            name = body["engine"]
            port = body["port"]

            file_ok = wait_for(
                lambda: _discovery_has_http_port(env, port - 1),
                ADD_CONVERGENCE_S,
                0.1,
            )
            alive_ok = _wait_master_alive(ops, "PREFILL", 3, MASTER_EVICT_S)
            traffic_ok = _pump_until_accepted(ops, name, base, 15.0)

            status_rm, _ = ops.remove_engine(engine_name=name)
            file_rm_ok = wait_for(
                lambda: not _discovery_has_http_port(env, port - 1),
                REMOVE_CONVERGENCE_S,
                0.1,
            )
            # File must stay parseable at every round boundary.
            parsable = _discovery_payload(env) is not None

            round_ok = (
                file_ok
                and alive_ok
                and traffic_ok
                and status_rm == 200
                and file_rm_ok
                and parsable
            )
            all_ok = all_ok and round_ok
            round_details.append(
                f"r{round_no}[{name}]: file={file_ok} alive={alive_ok} "
                f"traffic={traffic_ok} rm={status_rm} file_rm={file_rm_ok} "
                f"parsable={parsable}"
            )
            if not round_ok:
                break

        # Final sanity: routing back to normal on the initial topology.
        recovery_ok, recovery_msg = ops.verify_recovery()
        p_prefill_after, _ = _discovery_entry_count(env)
        topology_restored = p_prefill_after == p_prefill
        passed = all_ok and recovery_ok and topology_restored
        return passed, (
            f"rounds=[{'; '.join(round_details)}], "
            f"discovery_prefill_before={p_prefill}/after={p_prefill_after}, "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            _cleanup_dynamic(ops, env)
        except Exception:
            pass


@case(
    "elastic_rebalance",
    profiles=["batch-window"],  # elastic_spec pins the legacy fault axes
    source="elastic acceptance: cost-based rebalance after scale-out (share < 60%)",
)
def elastic_rebalance(ctx: CaseContext):
    env, ops = _elastic_env(ctx)
    base = rid_base(ctx, "elastic")
    try:
        _cleanup_dynamic(ops, env)

        # After the predecessor cases' remove_engine calls, the detached
        # engine stays ROUTABLE on the master until EngineSyncRunner evicts
        # it: the eviction threshold is max(3 × status poll interval, 1s)
        # measured from the engine's last successful status update, so the
        # dead endpoint leaves the routable set ~1-2s after the remove_engine
        # HTTP call returns (file rewrite + ≤1 sync tick + 1s threshold;
        # verified in sync.log: "[remove] engine ip changes").  Its empty
        # ledger makes it the LOWEST-score endpoint meanwhile, so an
        # immediate baseline burst routes straight onto the dead port —
        # requests die in batch-ack quarantine (BATCH_ACK_UNCERTAIN ×8) or
        # stopped-batcher rejects and the case FAILs without any scheduling
        # defect (verified: solo runs PASS 2/2, same-order sequence runs
        # FAIL 2/2; cancel storms hit the removed port from baseline
        # t+17ms).  Waiting on the ALIVE count is not enough — the health
        # 3-strike demotion lands ~0.5s BEFORE the endpoint eviction — so
        # wait for the discovered count (workerStatusMap size) to converge.
        converged = _wait_master_topology(
            ops, "PREFILL", env.spec.n_prefill, MASTER_EVICT_S
        )
        if not converged:
            info = ops.master_info() or {}
            entry = (info.get("worker_summary", {}) or {}).get("PREFILL") or {}
            return False, (
                f"prefill topology did not converge after cleanup: "
                f"discovered={entry.get('discovered', '?')} "
                f"alive={entry.get('alive', '?')} "
                f"(need discovered=alive={env.spec.n_prefill})"
            )

        # Phase 1 — baseline: 50 requests across the 2 initial prefills.
        p0_before = _accepted(ops, "prefill-0")
        p1_before = _accepted(ops, "prefill-1")
        ok1, err1, _ = _run_batch(ops, base, 50)
        p0_mid = _accepted(ops, "prefill-0") - p0_before
        p1_mid = _accepted(ops, "prefill-1") - p1_before
        if err1:
            return False, (
                f"baseline batch had {err1} errors, "
                f"types={_run_batch.last_error_types[:3]}"
            )

        # Phase 2 — scale out to 3 prefills.
        status, body = ops.add_engine("prefill")
        if status != 200:
            return False, f"add_engine failed: {status} {body}"
        new_name = body["engine"]
        in_file = wait_for(
            lambda: _discovery_has_http_port(env, body["port"] - 1),
            ADD_CONVERGENCE_S,
            0.1,
        )
        alive3 = _wait_master_alive(ops, "PREFILL", 3, MASTER_EVICT_S)
        if not (in_file and alive3):
            return False, (
                f"{new_name} never converged: file={in_file}, "
                f"alive={ops.master_alive_count('PREFILL')}"
            )

        # Phase 3 — another 50 requests; the new engine must take a share
        # (cost-aware rebalance, non-exclusive).
        new_before = _accepted(ops, new_name)
        ok2, err2, _ = _run_batch(ops, base, 50)
        p0_delta = _accepted(ops, "prefill-0") - p0_before - p0_mid
        p1_delta = _accepted(ops, "prefill-1") - p1_before - p1_mid
        new_delta = _accepted(ops, new_name) - new_before
        total_delta = p0_delta + p1_delta + new_delta
        share = (new_delta / total_delta) if total_delta else 0.0
        passed = new_delta > 0 and share < 0.60 and err2 == 0
        return passed, (
            f"new_engine={new_name}, "
            f"baseline_split=({p0_mid},{p1_mid}), "
            f"after_add_split=(p0+{p0_delta}, p1+{p1_delta}, {new_name}+{new_delta}), "
            f"new_share={share:.1%} (need >0% and <60%), "
            f"phase2_errors={err2}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            _cleanup_dynamic(ops, env)
        except Exception:
            pass


@case(
    "elastic_stop_after_add",
    profiles=["batch-window"],  # elastic_spec pins the legacy fault axes
    source="elastic acceptance: add → traffic → /stop_engine (3-fail evict) → /start_engine recovery",
)
def elastic_stop_after_add(ctx: CaseContext):
    env, ops = _elastic_env(ctx)
    base = rid_base(ctx, "elastic")
    try:
        _cleanup_dynamic(ops, env)

        status, body = ops.add_engine("prefill")
        if status != 200:
            return False, f"add_engine failed: {status} {body}"
        new_name = body["engine"]
        new_port = body["port"]
        in_file = wait_for(
            lambda: _discovery_has_http_port(env, new_port - 1),
            ADD_CONVERGENCE_S,
            0.1,
        )
        alive3 = _wait_master_alive(ops, "PREFILL", 3, MASTER_EVICT_S)
        if not (in_file and alive3):
            return False, (
                f"{new_name} never converged: file={in_file}, "
                f"alive={ops.master_alive_count('PREFILL')}"
            )

        # Make sure the new engine really serves traffic before we stop it.
        if not _pump_until_accepted(ops, new_name, base, 15.0):
            return False, "new engine accepted no traffic before stop"
        accepted_before_stop = _accepted(ops, new_name)

        # HTTP-stop the engine (single JVM: /stop_engine, not a process kill).
        # Master health checks fail → consecutive-failure eviction (alive 3→2).
        ops.stop_engine(new_name)
        evicted = wait_for(
            lambda: ops.master_alive_count("PREFILL") <= 2,
            MASTER_EVICT_S,
            0.5,
        )
        # While it is down: requests still succeed on the surviving 2 prefills.
        addr, err = ops.run_one_request(
            ops.next_request_id(base),
            output_len=2,
            block_keys=[base + 7],
            stream_timeout_s=10.0,
        )
        del addr

        # Bring it back and confirm re-discovery + traffic resumption.
        ops.start_engine(new_name)
        alive_back = _wait_master_alive(ops, "PREFILL", 3, MASTER_EVICT_S)
        resumed = _pump_until_accepted(ops, new_name, base, 20.0)
        accepted_after = _accepted(ops, new_name)

        passed = (
            evicted
            and err is None
            and alive_back
            and resumed
            and accepted_after > accepted_before_stop
        )
        return passed, (
            f"engine={new_name}(grpc={new_port}), "
            f"accepted_before_stop={accepted_before_stop}, "
            f"evicted_after_stop={evicted}(alive={ops.master_alive_count('PREFILL')}), "
            f"during_downtime_request={'ok' if err is None else err}, "
            f"alive_restored={alive_back}, "
            f"accepted_after_restart={accepted_after}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        # Restore: ensure the engine is started again (harmless if already up)
        # and drop dynamic engines so the shared env returns to 2P+4D.
        try:
            snap = ops.snapshot_by_name()
            for name in _dynamic_engines(ops, env):
                if snap.get(name, {}).get("stopped"):
                    try:
                        ops.start_engine(name)
                    except Exception:
                        pass
            _cleanup_dynamic(ops, env)
        except Exception:
            pass


@case(
    "elastic_concurrent_ops",
    profiles=["batch-window"],  # elastic_spec pins the legacy fault axes
    source="elastic acceptance: concurrent add/remove storm, master stays healthy",
)
def elastic_concurrent_ops(ctx: CaseContext):
    env, ops = _elastic_env(ctx)
    base = rid_base(ctx, "elastic")
    try:
        _cleanup_dynamic(ops, env)

        duration_s = 10.0
        added_lock = threading.Lock()
        added_ports: list[int] = []
        op_counts = {"add_ok": 0, "add_fail": 0, "rm_ok": 0, "rm_fail": 0}
        counts_lock = threading.Lock()

        def adder(worker_id: int) -> None:
            deadline = time.monotonic() + duration_s
            while time.monotonic() < deadline:
                role = "prefill" if worker_id % 2 == 0 else "decode"
                try:
                    status, body = ops.add_engine(role)
                except Exception:
                    status, body = 0, None
                with counts_lock:
                    if status == 200:
                        op_counts["add_ok"] += 1
                        with added_lock:
                            added_ports.append(body["port"])
                    else:
                        op_counts["add_fail"] += 1
                time.sleep(0.25 + 0.15 * worker_id)

        def remover(worker_id: int) -> None:
            deadline = time.monotonic() + duration_s
            rnd = random.Random(worker_id * 977)
            while time.monotonic() < deadline:
                with added_lock:
                    candidates = list(added_ports)
                if candidates:
                    port = rnd.choice(candidates)
                    try:
                        status, _ = ops.remove_engine(port=port)
                    except Exception:
                        status = 0
                    with counts_lock:
                        if status == 200:
                            op_counts["rm_ok"] += 1
                        else:
                            op_counts["rm_fail"] += 1
                    # Remove also from the candidate pool (either this thread
                    # or a racing sibling may have taken it down).
                    with added_lock:
                        if port in added_ports:
                            added_ports.remove(port)
                time.sleep(0.4 + 0.15 * worker_id)

        threads = [
            threading.Thread(target=adder, args=(0,), daemon=True),
            threading.Thread(target=adder, args=(1,), daemon=True),
            threading.Thread(target=remover, args=(0,), daemon=True),
            threading.Thread(target=remover, args=(1,), daemon=True),
        ]
        for t in threads:
            t.start()

        # Main thread: 1 health request + master-200 probe per second.
        health_ok = 0
        health_fail = 0
        health_err_types: set = set()
        master_200 = True
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            rid = ops.next_request_id(base)
            _, err = ops.run_one_request(
                rid,
                output_len=2,
                block_keys=[rid * 100 + 1],
                stream_timeout_s=10.0,
            )
            if err is None:
                health_ok += 1
            else:
                health_fail += 1
                health_err_types.add(str(err)[:60])
            status_code = http_get_status(
                f"{_master_http(ops)}/rtp_llm/inflight_status", timeout=5
            )
            if status_code != 200:
                master_200 = False
            time.sleep(1.0)
        for t in threads:
            t.join(15.0)

        # After the storm: the file must parse and agree with the mock services.
        payload = _discovery_payload(env)
        parsable = payload is not None
        if parsable:
            p_n, d_n = _discovery_entry_count(env)
            snap = ops.snapshot()
            snap_p = sum(
                1 for e in snap.get("engines", []) if e.get("role") == "prefill"
            )
            snap_d = sum(
                1 for e in snap.get("engines", []) if e.get("role") == "decode"
            )
            counts_match = (p_n == snap_p) and (d_n == snap_d)
        else:
            p_n = d_n = snap_p = snap_d = -1
            counts_match = False

        health_total = health_ok + health_fail
        passed = (
            master_200
            and parsable
            and counts_match
            and health_total > 0
            # Spec hard assertions are master HTTP 200 + discovery/file
            # consistency; the per-second health probe is observational.
            # Keep only a conservative floor (≥50%) so a total blackout
            # still fails the case while add/remove storms legitimately
            # degrade availability below 100%.
            and health_ok / health_total >= 0.5
        )
        return passed, (
            f"ops={json.dumps(op_counts)}, "
            f"health={health_ok}/{health_total} "
            f"(err_types={sorted(health_err_types)[:2]}), "
            f"master_200={master_200}, "
            f"discovery_parsable={parsable}, "
            f"entries=(prefill {p_n} vs snapshot {snap_p}, decode {d_n} vs {snap_d})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            _cleanup_dynamic(ops, env)
        except Exception:
            pass


# ===========================================================================
# Scale-in pending-drain protection (user-identified coverage gap, 2026-09)
# ===========================================================================
#
# The elastic family above only removes engines whose requests are all
# DISPATCHED (BackgroundFlow requests are short and in flight).  The gap:
# what happens to requests the master has already QUEUED for the victim's
# WorkerBatcher but never EnqueueBatch'd (its inflight-batch leases are
# full) when the scale-in event lands?  They must not sit silently until
# queueTimeout (Java default 1h) — that is a silent hour-long loss.

# Contract deadline for a stranded request's VISIBLE terminal state after
# remove_engine: stale-inflight TTL (30s) + margin.  The OTHER caliber in
# the task brief (short queueTimeout + margin) is deliberately NOT used:
# queueTimeoutMs stays at its 1h Java default so a master that parks the
# stranded set until queueTimeout FAILS this case as a finding instead of
# having the wait shortened into compliance.
PENDING_DRAIN_TERMINAL_S = 40.0
# Master accounting cleanup cap: stale window (statusStaleAfterMs=10s +
# 3s cleaner period) + generous margin, but far below queueTimeout (1h).
PENDING_DRAIN_CLEAN_S = 50.0
# Wave shaping: serial sends at ~30x the 10ms FIXED_WINDOW collection
# window so every request forms its own batch — a fast burst collapses
# into one batch, dispatches wholesale behind ONE lease and strands
# nothing on the master side.
PENDING_DRAIN_WAVE_INTERVAL_S = 0.3
PENDING_DRAIN_WAVE_MAX = 14
# Fail-fast floors for the scenario construction (the case is meaningless
# unless the stranded set is proven non-empty before the removal).
PENDING_DRAIN_VICTIM_MIN = 3
PENDING_DRAIN_VICTIM_TARGET = 4
# Slow prefill: 8s batches hold both inflight-batch leases for the whole
# wave + removal window (first completion at t+8s; the wave finishes at
# ~t+5s).
PENDING_DRAIN_SLOW_MS = 8000.0
# Shape classification thresholds (observation-only, no hard band):
#   fast_fail  — terminal within ~the engine-death window (streams cut by
#                shutdownNow almost immediately after the remove call)
#   stale_window_fail — terminal in the 10s statusStale + 3s cleaner +
#                margin band (the expected fail-closed BATCH_DISPATCH_FAILED
#                shape from WorkerBatcher.stopAndDrain)
#   slow_fail  — terminal only near/after the 30s stale-inflight TTL
#                (worst acceptable shape; finding candidate)
PENDING_DRAIN_FAST_FAIL_S = 5.0
PENDING_DRAIN_STALE_WINDOW_S = 16.0


def _pending_drain_spec(ctx: CaseContext) -> EnvSpec:
    """Dedicated env for the pending-drain case: 2P+2D, dynamic file
    discovery, legacy fault axes with maxInflightBatchesPerPrefillWorker=2.

    Two reasons the case does NOT reuse elastic_spec: (a) 2 inflight
    batches per worker is the production-aligned lease cap, giving exactly
    two dispatched batches before queue residency; (b) the fingerprint
    differs from every other spec (elastic_spec=4, quota_spec=1), so the
    INITIAL-engine victim (prefill-0, permanently removed — a removed
    initial engine never comes back on its port/name) never poisons a
    shared env: this spec owns a private one.  queueTimeoutMs is
    intentionally left at the Java default (1h) — see
    PENDING_DRAIN_TERMINAL_S."""
    return EnvSpec(
        label=f"fault_pending_drain_{ctx.profile}",
        n_prefill=2,
        n_decode=2,
        perf=fault_env_perf(),
        master_profile=ctx.profile,
        discovery="discovery_file",
        master_env={"FLEXLB_CONFIG": fault_env_config(max_inflight_batches=2)},
    )


@case(
    "elastic_remove_pending_drain",
    profiles=["batch-window"],  # elastic family: BATCH dispatcher + fault axes
    source="user-identified gap: scale-in protection for requests queued-but-undispatched on the removed engine",
)
def elastic_remove_pending_drain(ctx: CaseContext):
    """Scale-in must not strand requests already QUEUED at the master for
    the removed engine (user-identified coverage gap, 2026-09).

    Scenario: victim = prefill-0 (initial engine) on a private 2P+2D env,
    removed through the production scale-in chain (/remove_engine ->
    discovery-file rewrite -> master FileServiceDiscovery loss).  Both
    prefills run at 8s so the victim's two inflight-batch leases stay
    occupied while a serial wave (one request per 300ms — each its own
    FIXED_WINDOW batch) keeps landing requests on it: after the first two
    single-request batches dispatch, every further victim-routed request
    sits in the master-side WorkerBatcher queue — accepted by Schedule,
    never EnqueueBatch'd.  A pre-assertion proves the stranded set is
    non-empty (victim-routed > engine-side waiting+running) BEFORE the
    removal fires.

    Behaviour: remove_engine(victim) while both batch leases are occupied
    and the stranded requests are parked in the master queue.

    Expected (CONTRACT — the behaviour the system SHOULD have, not
    necessarily what it has today):
      1. (invariant P6) every victim-routed request reaches a VISIBLE
         terminal state — completed, or an explicit error on its stream —
         within stale-TTL 30s + margin, i.e. PENDING_DRAIN_TERMINAL_S = 40s
         after the removal.  Deadline caliber: the stale-TTL scale (see
         PENDING_DRAIN_TERMINAL_S for why queueTimeout is left at 1h).
      2. (observation, no hard band) WHICH shape the terminal takes:
         completed elsewhere / re-routed (best), fast explicit failure
         (acceptable), failure only at the stale/TTL window (worst —
         finding candidate).  Reported as a per-request type + latency
         distribution plus the master accounting-cleanup latency.
      3. Master accounting returns to baseline (inflight_clean within
         PENDING_DRAIN_CLEAN_S) and the survivor keeps serving (recovery
         batch >= 95%).
      4. Topology convergence: victim gone from the mock services map and
         the discovery file, master prefill alive count drops to 1.

    Prediction (current master, from the code walk): remove_engine stops
    the engine immediately (setStopped + drainAndShutdown + server
    shutdownNow), so the parked FetchResponse streams break within ~1s
    with a transport error — a fast explicit failure.  The master-side
    cleanup is NOT immediate: the dead engine stops reporting
    WorkerStatus, and after statusStaleAfterMs (10s in this config)
    EngineSyncRunner / ExpirationCleaner retire the generation,
    PrefillEndpoint.closeEndpoint runs WorkerBatcher.stopAndDrain which
    fail-closes every queued item to BATCH_DISPATCH_FAILED ("Worker
    scheduling queue rejected request").  No re-routing exists on that
    path.  Expected observed shape: client-visible terminal ~0-2s (engine
    death), master accounting clean ~10-15s — the case passes in the
    fast-fail shape.  If the retirement chain fails to drain the queue,
    the stranded items wait out queueTimeout (1h) -> inflight_clean(50s)
    FAILS -> finding.
    """
    env = ctx.env_manager.ensure(_pending_drain_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "elastic")
    victim = "prefill-0"
    survivor = "prefill-1"
    # (rid, response, stream handle, routed engine name)
    fired: list[tuple[int, object, object, str]] = []
    try:
        snap = ops.snapshot_by_name()
        if victim not in snap or survivor not in snap:
            # The spec owns a private env, but a same-process rerun would
            # reuse it with prefill-0 already gone — fail fast and say why.
            return False, (
                f"{victim}/{survivor} missing from the private env (one-shot "
                f"victim: a rerun needs a fresh process); "
                f"engines={sorted(snap)}"
            )
        _cleanup_dynamic(ops, env)  # no dynamic leftovers in a private env
        addr_map = ops.addr_to_name()
        victim_http_port = int(snap[victim]["grpc_addr"].rsplit(":", 1)[1]) - 1

        # -- slow BOTH prefills: symmetric 8s ledgers keep ESTIMATED_TTFT
        #    splitting the wave across both engines (a slow-only victim is
        #    priced out and receives no traffic at all) while every 8s
        #    batch holds a lease for the whole wave + removal window.
        for name in (victim, survivor):
            ops.set_perf(name, prefill_fixed_ms=PENDING_DRAIN_SLOW_MS)
        time.sleep(1.5)  # master perf sync

        # -- serial wave: each request its own FIXED_WINDOW batch; the
        #    victim's first two batches take both leases, everything routed
        #    there afterwards parks in the master-side WorkerBatcher queue.
        wave = 0
        victim_routed = 0
        schedule_rejects = 0
        while (
            victim_routed < PENDING_DRAIN_VICTIM_TARGET
            and wave < PENDING_DRAIN_WAVE_MAX
        ):
            wave += 1
            rid = ops.next_request_id(base)
            try:
                resp = ops.schedule(
                    rid,
                    input_len=1024,
                    output_len=2,
                    block_keys=[rid * 100 + j for j in range(3)],
                )
            except Exception:
                schedule_rejects += 1
                continue
            if resp.code != 200 or not resp.success:
                schedule_rejects += 1
                continue
            route = addr_map.get(ops.role_addr(resp, "PREFILL"), "")
            handle = ops.start_stream(resp, rid)  # FetchResponse parked
            fired.append((rid, resp, handle, route))
            if route == victim:
                victim_routed += 1
            time.sleep(PENDING_DRAIN_WAVE_INTERVAL_S)

        # -- PRE-ASSERTION: the stranded set is non-empty.  Requests the
        #    master accepted for the victim but the ENGINE never received
        #    (no EnqueueBatch) are exactly the master-side queue residents
        #    under test.
        vsnap = ops.snapshot_by_name().get(victim, {})
        engine_inflight = vsnap.get("waiting", 0) + vsnap.get("running", 0)
        stranded = victim_routed - engine_inflight
        if victim_routed < PENDING_DRAIN_VICTIM_MIN or stranded < 1:
            return False, (
                f"scenario construction failed: victim_routed={victim_routed} "
                f"(need >={PENDING_DRAIN_VICTIM_MIN}), engine waiting+running="
                f"{engine_inflight}, stranded={stranded} (need >=1), "
                f"wave={wave}, schedule_rejects={schedule_rejects}"
            )

        # -- THE SCALE-IN EVENT (production chain: /remove_engine ->
        #    discovery rewrite; the engine dies, the master learns via
        #    the file, the stranded requests keep waiting).
        t_remove = time.monotonic()
        status, rm_body = ops.remove_engine(engine_name=victim)
        if status != 200:
            return False, f"remove_engine failed: {status} {rm_body}"
        rm_body = rm_body or {}
        waiting_at_removal = rm_body.get("waiting_at_removal")
        running_at_removal = rm_body.get("running_at_removal")

        # -- terminal-state collection for every fired request.  Latency is
        #    the stream's own terminated_s timestamp minus t_remove, so the
        #    serial collection order cannot distort it.
        outcomes = []  # (rid, route, kind, latency_s, err)
        for rid, resp, handle, route in fired:
            ended = handle.wait_end(PENDING_DRAIN_TERMINAL_S + 5.0)
            snap_e = handle.snap
            latency = (
                snap_e.terminated_s - t_remove
                if snap_e.terminated_s is not None
                else time.monotonic() - t_remove
            )
            if snap_e.completed and not snap_e.error:
                kind, err = "completed", None
            elif ended and snap_e.error:
                kind, err = "error", str(snap_e.error)[:80]
            elif ended:
                # Stream ended with neither completion nor error — an empty
                # close is a SILENT loss, not a visible terminal state.
                kind, err = "empty", "stream closed without terminal frame"
            else:
                kind, err = "hang", "no terminal state"
            outcomes.append((rid, route, kind, latency, err))

        victim_out = [o for o in outcomes if o[1] == victim]
        survivor_out = [o for o in outcomes if o[1] != victim]

        # -- shape classification (observation, no hard band).
        def shape_of(kind: str, latency: float) -> str:
            if kind == "completed":
                return "completed"
            if kind == "error":
                if latency <= PENDING_DRAIN_FAST_FAIL_S:
                    return "fast_fail"
                if latency <= PENDING_DRAIN_STALE_WINDOW_S:
                    return "stale_window_fail"
                return "slow_fail"
            return "no_terminal"

        shapes = [shape_of(k, lat) for _, _, k, lat, _ in victim_out]
        shape_counts = {s: shapes.count(s) for s in sorted(set(shapes))}
        latencies = sorted(lat for _, _, _, lat, _ in victim_out)
        lat_med = latencies[len(latencies) // 2] if latencies else float("nan")
        lat_max = latencies[-1] if latencies else float("nan")

        # -- master accounting cleanup latency (observation + cap).
        t_clean = time.monotonic()
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), PENDING_DRAIN_CLEAN_S
        )
        clean_latency = time.monotonic() - t_clean

        # -- survivor keeps serving: restore fast perf, then a 20-request
        #    recovery batch on the remaining prefill (>= 95%).
        try:
            ops.set_perf(survivor, prefill_fixed_ms=100.0)
        except Exception:
            pass
        ok_n, _err_n, _ = _run_batch(ops, base, 20)
        recovery_rate = ok_n / 20.0

        # -- topology convergence (same assertions as elastic_remove_flow).
        gone_from_snapshot = victim not in ops.snapshot_by_name()
        gone_from_file = wait_for(
            lambda: not _discovery_has_http_port(env, victim_http_port),
            REMOVE_CONVERGENCE_S,
            0.1,
        )
        alive_1 = _wait_master_alive(ops, "PREFILL", 1, MASTER_EVICT_S)

        # -- CONTRACT ASSERTIONS -------------------------------------------
        # P6 #1: every victim-routed request reached a VISIBLE terminal
        # state (completed / explicit error — never a hang, never an empty
        # close) within the stale-TTL+margin deadline.  Anything else is a
        # completeness violation at every grade.
        violations = [
            (rid, kind, round(lat, 1), err)
            for rid, _r, kind, lat, err in victim_out
            if kind not in ("completed", "error") or lat > PENDING_DRAIN_TERMINAL_S
        ]
        report.invariant(
            "P6",
            not violations,
            context="pending_drain",
            detail=f"violations={violations[:3]}",
        )
        # P6 #2: no accounting leak — the master's inflight/ledger entries
        # for the stranded set return to baseline well before queueTimeout.
        report.invariant(
            "P6",
            inflight_ok,
            context="master_accounting",
            detail=(
                f"clean={clean_latency:.1f}s cap={PENDING_DRAIN_CLEAN_S:.0f}s "
                f"{inflight_detail[:100]}"
            ),
        )
        # P2: the survivor is not starved — it keeps serving fresh traffic
        # after the scale-in.
        report.invariant(
            "P2",
            recovery_rate >= 0.95,
            context="survivor_service",
            detail=f"recovery {ok_n}/20",
        )
        # Topology convergence is the elastic family's plain boolean
        # contract (same assertions as elastic_remove_flow), folded into
        # the case verdict rather than a graded property.
        topo_ok = gone_from_snapshot and gone_from_file and alive_1

        survivor_done = sum(1 for o in survivor_out if o[2] == "completed")
        return (
            report.passed and topo_ok,
            f"victim={victim}(stranded={stranded}, routed={victim_routed}, "
            f"wave={wave}, rejects={schedule_rejects}), "
            f"at_removal=(running={running_at_removal}, "
            f"waiting={waiting_at_removal}), "
            f"shapes={json.dumps(shape_counts)}, "
            f"terminal_latency=(med={lat_med:.1f}s, max={lat_max:.1f}s, "
            f"cap={PENDING_DRAIN_TERMINAL_S:.0f}s), "
            f"survivor_fired_completed={survivor_done}/{len(survivor_out)}, "
            f"master_cleanup={clean_latency:.1f}s("
            f"ok={inflight_ok}), "
            f"recovery={ok_n}/20({recovery_rate:.0%}), "
            f"topology=(snap={gone_from_snapshot}, "
            f"file={gone_from_file}, alive1={alive_1}), "
            f"grades: {report.summary()}",
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        # Restore perf (survivor; the victim is gone and set_perf on a
        # removed engine harmlessly 404s), consume every fired request to
        # a terminal state (wait_end + cancel fallback — a parked
        # FetchResponse whose stream never ends would leak master-side
        # inflight/ledger entries into later cases), then the usual
        # dynamic-engine hygiene.
        try:
            ops.set_perf(survivor, prefill_fixed_ms=100.0)
        except Exception:
            pass
        for rid, resp, handle, _route in fired:
            try:
                if not handle.snap.terminated:
                    handle.wait_end(20.0)
                if not handle.snap.completed and not handle.snap.error:
                    ops.cancel(rid, resp)
            except Exception:
                try:
                    ops.cancel(rid, resp)
                except Exception:
                    pass
        try:
            _cleanup_dynamic(ops, env)
        except Exception:
            pass
        try:
            AssertUtils.inflight_clean(_master_http(ops), 30.0)
        except Exception:
            pass


# ===========================================================================
# Scale-out traffic-preference shape (user-named coverage gap, 2026-09):
# does traffic favour the freshly added, queue-empty engine?
# ===========================================================================
#
# Routing scores prefill candidates by projected TTFT, which folds in the
# queue/ledger depth: a freshly added engine starts with an empty queue and
# an empty ledger, so it scores LOWEST and is necessarily preferred at
# first — the cost-aware-routing design intent (the newcomer absorbs the
# excess and helps rebalance).  The CORRECT shape is: transient burst
# allowed, sustained exclusivity NOT allowed, old engines never starved,
# self-converging — the newcomer's queue fills, its projected TTFT rises,
# routing falls back into the RANDOM_WITHIN_TOLERANCE parity window and
# the distribution returns to uniform.  elastic_rebalance pins only the
# post-scale-out steady share (<60%); elastic_add_preference pins the
# add_flow SHAPE around it: transient burst + steady re-flattening +
# oscillation probe, with the same accepted-counter delta caliber.

# Measurement windows, in seconds, timed from the instant the master view
# converges (discovered==alive==3): every window below is an
# accepted-counter DELTA between snapshot samples taken at/after that
# instant, so discovery-convergence counts cannot pollute any window.
ADD_PREF_BASELINE_S = 15.0  # pre-add steady split on the 2 old prefills (obs)
ADD_PREF_TOTAL_S = 45.0  # post-add measurement window
ADD_PREF_TRANSIENT_S = 10.0  # leading transient sub-window
ADD_PREF_TRANSIENT_PEAK_S = 5.0  # transient sub-slices for the peak capture
# The steady window is the remaining 35s split into 5 equal 7s sub-windows
# for the oscillation observation (sample offsets 10, 17, 24, 31, 38, 45).
ADD_PREF_STEADY_SUBWINDOWS = 5
# CONTRACT bands for the steady-window newcomer share (P1, case override —
# the same override mechanism as balance_overload_avoid_decode's P5 delta
# caliber): normal/loose = 60%, the elastic_rebalance parity band, so
# sustained exclusivity (>60% of the steady traffic on the newcomer)
# breaks the contract at EVERY grade; strict = 50%, a quality bar for
# near-uniform convergence.  CALIBRATION PLAN (task #61 discipline):
# first runs record the observed steady share; if it lands far below 60%,
# tighten the normal/loose tiers accordingly.
ADD_PREF_SHARE_BANDS = {"strict": 0.50, "normal": 0.60, "loose": 0.60}
# Old-engine starvation floor (P2 hard invariant): each of the two
# pre-existing prefills keeps >= 10% of the steady-window traffic.
ADD_PREF_OLD_FLOOR = 0.10


def _accepted_timeline(ops, engine_names, offsets_s):
    """Sample per-engine accepted counters at *offsets_s* seconds from NOW.

    While waiting for the next offset the master's HTTP health is probed
    at most once per second (any non-200 flips the returned flag).  Every
    downstream window is a DELTA between samples, so counts that piled up
    before the caller starts the timeline cannot leak into a window.
    Returns (samples, master_200) with samples aligned to *offsets_s*
    (the first offset should be 0.0 so the baseline sample is immediate).
    """
    samples = []
    master_200 = True
    last_probe = 0.0
    t0 = time.monotonic()
    idx = 0
    while idx < len(offsets_s):
        if time.monotonic() - t0 >= offsets_s[idx]:
            snap = ops.snapshot_by_name()
            samples.append(
                (
                    offsets_s[idx],
                    {n: int(snap.get(n, {}).get("accepted", 0)) for n in engine_names},
                )
            )
            idx += 1
            continue
        if time.monotonic() - last_probe >= 1.0:
            code = http_get_status(
                f"{_master_http(ops)}/rtp_llm/inflight_status", timeout=5
            )
            if code != 200:
                master_200 = False
            last_probe = time.monotonic()
        time.sleep(min(1.0, max(0.05, offsets_s[idx] - (time.monotonic() - t0))))
    return samples, master_200


@case(
    "elastic_add_preference",
    profiles=["batch-window"],  # elastic family: BATCH dispatcher + fault axes
    source=(
        "user-named gap: post-scale-out traffic preference shape "
        "(queue-empty newcomer)"
    ),
)
def elastic_add_preference(ctx: CaseContext):
    """Post-scale-out traffic-preference SHAPE (user-named coverage gap:
    "流量会不会偏好没排队的新引擎" — does traffic favour the freshly added,
    queue-empty engine?).

    Scenario: the SHARED elastic env (2P+4D, dynamic file discovery —
    reusing elastic_spec keeps the fingerprint compatible, so no extra
    environment rebuild; the prefill axis 2→3 is the measured dimension).
    A steady background flow of UNIQUE-KEY requests (block key derived
    from a fresh rid every time — the aff family's free-unique-key
    construction) runs to prefill so no prefix affinity can pin traffic:
    the measurement sees the pure queue/ledger routing dimension.  After
    the master view converges (discovered==alive==2) a ~15s baseline
    records the pre-add split; add_engine brings a 3rd prefill; after
    discovered==alive==3 a 45s measurement window runs — leading 10s
    transient, remaining 35s steady (5 x 7s sub-windows) — then the
    dynamic engine is removed in the finally hygiene.

    Behaviour: routing scores prefill candidates by projected TTFT
    (queue/ledger depth included).  The newcomer starts empty-queue and
    empty-ledger, so it scores LOWEST and is necessarily preferred at
    first — by design: cost-aware routing lets the newcomer absorb the
    excess and helps rebalance.  As its queue fills, its projected TTFT
    rises and routing falls back into the RANDOM_WITHIN_TOLERANCE parity
    window: the distribution self-converges to near-uniform.

    Expected (CONTRACT — the behaviour the system SHOULD have, not
    necessarily what it has today):
      1. The scale-out takes effect: the newcomer receives traffic
         (post-add window delta > 0).
      2. (observation, no hard band) transient burst ALLOWED: the first
         10s newcomer share may sit well above the uniform 1/3 — the
         recorded 5s-slice peak calibrates a future band.
      3. (invariant, P1 override) steady share bounded: over the last
         35s the newcomer's share stays under the 60% elastic_rebalance
         parity band (sustained exclusivity breaks the contract at every
         grade; strict tier 50%).  CALIBRATION PLAN: first runs record
         the observed value — if it lands far below 60%, tighten.
      4. (invariant, P2) old engines not starved: each of the two
         pre-existing prefills keeps >= 10% of the steady-window traffic.
      5. (observation, first round) no oscillation: the newcomer's share
         across the five 7s steady sub-windows is recorded; a repeated
         boom-bust pattern is a finding candidate, not yet a hard band.
      6. Master HTTP 200 across both measurement windows; background
         flow success >= 90%; topology converges (discovery file entry +
         master discovered==alive==3) before the window opens.

    Prediction (current master, from the mechanism): the empty-ledger
    newcomer necessarily scores lowest at first, so the transient share
    will sit clearly above the uniform 1/3; as its ledger fills the
    score flattens and the steady share should return into the [1/3,
    60%) band well within the 35s window.  A steady share above 60%
    (sustained exclusivity) or an old engine below the 10% floor is a
    finding.
    """
    env, ops = _elastic_env(ctx)
    base = rid_base(ctx, "elastic")
    report = GradeReport(run_grade=ctx.grade)
    flow: Optional[_BackgroundFlow] = None
    new_name = ""
    new_port = 0
    try:
        # Shared-env hygiene + initial-topology convergence: routable-but-
        # dead leftovers from earlier cases would poison the baseline (see
        # elastic_rebalance's baseline comment for the eviction timing).
        _cleanup_dynamic(ops, env)
        converged2 = _wait_master_topology(
            ops, "PREFILL", env.spec.n_prefill, MASTER_EVICT_S
        )
        if not converged2:
            info = ops.master_info() or {}
            entry = (info.get("worker_summary", {}) or {}).get("PREFILL") or {}
            return False, (
                f"initial prefill topology did not converge: "
                f"discovered={entry.get('discovered', '?')} "
                f"alive={entry.get('alive', '?')} "
                f"(need discovered==alive=={env.spec.n_prefill})"
            )

        flow = _BackgroundFlow(ops, base, interval_s=0.2)
        flow.start()
        time.sleep(1.0)  # ramp up before the baseline window

        olds = [f"prefill-{i}" for i in range(env.spec.n_prefill)]
        base_samples, base_m200 = _accepted_timeline(
            ops, olds, [0.0, ADD_PREF_BASELINE_S]
        )
        base_split = {
            n: base_samples[1][1].get(n, 0) - base_samples[0][1].get(n, 0) for n in olds
        }
        base_total = sum(base_split.values())
        base_share = {
            n: (v / base_total if base_total else 0.0) for n, v in base_split.items()
        }

        # -- THE SCALE-OUT EVENT: a 3rd prefill via the production chain
        #    (/add_engine -> discovery-file rewrite -> master sync).
        status, body = ops.add_engine("prefill")
        if status != 200:
            return False, f"add_engine failed: {status} {body}"
        new_name = body["engine"]
        new_port = body["port"]
        in_file = wait_for(
            lambda: _discovery_has_http_port(env, new_port - 1),
            ADD_CONVERGENCE_S,
            0.1,
        )
        topo3 = _wait_master_topology(ops, "PREFILL", 3, MASTER_EVICT_S)
        if not (in_file and topo3):
            return False, (
                f"{new_name} never converged: file={in_file}, "
                f"alive={ops.master_alive_count('PREFILL')} "
                f"(need discovered==alive==3)"
            )

        # -- post-add measurement window, offset from the CONVERGENCE
        #    instant (the discovery convergence period cannot pollute it).
        names = olds + [new_name]
        steady_len = ADD_PREF_TOTAL_S - ADD_PREF_TRANSIENT_S
        step = steady_len / ADD_PREF_STEADY_SUBWINDOWS
        offsets = [0.0, ADD_PREF_TRANSIENT_PEAK_S, ADD_PREF_TRANSIENT_S]
        offsets += [
            ADD_PREF_TRANSIENT_S + step * (k + 1)
            for k in range(ADD_PREF_STEADY_SUBWINDOWS)
        ]
        samples, m200 = _accepted_timeline(ops, names, offsets)

        total, ok = flow.stop()
        rate = ok / total if total else 0.0

        def delta(i_lo: int, i_hi: int, name: str) -> int:
            return samples[i_hi][1].get(name, 0) - samples[i_lo][1].get(name, 0)

        def window(i_lo: int, i_hi: int) -> dict:
            return {n: delta(i_lo, i_hi, n) for n in names}

        def share_of(counts: dict, name: str) -> float:
            tot = sum(counts.values())
            return (counts[name] / tot) if tot else 0.0

        last = len(samples) - 1
        transient = window(0, 2)  # [0, 10) after convergence
        steady = window(2, last)  # [10, 45)
        steady_subs = [window(2 + k, 3 + k) for k in range(ADD_PREF_STEADY_SUBWINDOWS)]
        steady_total = sum(steady.values())
        if steady_total <= 0:
            return False, (
                f"measurement void: no traffic landed in the steady window "
                f"(flow={ok}/{total})"
            )

        trans_new_share = share_of(transient, new_name)
        trans_peak = max(
            share_of(window(0, 1), new_name), share_of(window(1, 2), new_name)
        )
        steady_new_share = share_of(steady, new_name)
        old_shares = {n: share_of(steady, n) for n in olds}
        sub_shares = [share_of(s, new_name) for s in steady_subs]
        swing = (max(sub_shares) - min(sub_shares)) if sub_shares else 0.0

        # -- CONTRACT ASSERTIONS -------------------------------------------
        # #1 the scale-out took effect: the newcomer received traffic.
        new_got_traffic = (transient[new_name] + steady[new_name]) > 0
        # #3 steady share bounded (P1 override; ADD_PREF_SHARE_BANDS holds
        #    the calibration plan).
        report.check(
            "P1",
            steady_new_share,
            bands=ADD_PREF_SHARE_BANDS,
            context="new_engine_steady_share",
            detail=(
                f"{new_name} took {steady[new_name]}/{steady_total} "
                f"in steady window"
            ),
        )
        # #4 old engines not starved (P2 hard invariant).
        report.invariant(
            "P2",
            all(v >= ADD_PREF_OLD_FLOOR for v in old_shares.values()),
            context="old_engine_steady_floor",
            detail=(
                f"{olds[0]}={old_shares[olds[0]]:.1%}, "
                f"{olds[1]}={old_shares[olds[1]]:.1%}, "
                f"floor={ADD_PREF_OLD_FLOOR:.0%}"
            ),
        )
        # #2/#5 stay observation-only (transient peak, sub-window swing);
        # #6 availability folds into the case verdict below.

        base_share_str = "/".join(f"{base_share[n]:.0%}" for n in olds)
        sub_str = ",".join(f"{v:.0%}" for v in sub_shares)
        ok_verdict, detail, _rep = report.finish(
            f"new_engine={new_name}(grpc={new_port}), "
            f"baseline[0-{ADD_PREF_BASELINE_S:.0f}s]={base_split[olds[0]]}"
            f"+{base_split[olds[1]]}({base_share_str}, obs), "
            f"transient[0-{ADD_PREF_TRANSIENT_S:.0f}s]=+{transient[olds[0]]}"
            f"+{transient[olds[1]]}+{transient[new_name]}, "
            f"new_share={trans_new_share:.0%}"
            f"(peak5s={trans_peak:.0%}, obs-only), "
            f"steady[{ADD_PREF_TRANSIENT_S:.0f}-{ADD_PREF_TOTAL_S:.0f}s]="
            f"+{steady[olds[0]]}+{steady[olds[1]]}+{steady[new_name]}, "
            f"new_share={steady_new_share:.0%}(cap 60%), "
            f"old_floor=(p0 {old_shares[olds[0]]:.0%}, "
            f"p1 {old_shares[olds[1]]:.0%}, floor 10%), "
            f"sub_shares=[{sub_str}](swing={swing:.0%}, obs-only), "
            f"new_got_traffic={new_got_traffic}, "
            f"flow={ok}/{total}({rate:.1%}, >=90% required), "
            f"master_200=(baseline={base_m200}, window={m200}), "
            f"topology=(file={in_file}, discovered==alive==3={topo3}), "
            f"grades: {report.summary()}"
        )
        return (
            ok_verdict and new_got_traffic and rate >= 0.90 and m200 and base_m200,
            detail,
            _rep,
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if flow is not None:
            flow.stop()
        try:
            _cleanup_dynamic(ops, env)
        except Exception:
            pass
