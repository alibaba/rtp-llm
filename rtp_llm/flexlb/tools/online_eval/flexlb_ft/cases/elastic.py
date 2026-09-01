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
from ..harness import (
    AssertUtils,
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
        # (relaxed to 90s — covers the 30s stale-inflight TTL).
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 90.0
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
