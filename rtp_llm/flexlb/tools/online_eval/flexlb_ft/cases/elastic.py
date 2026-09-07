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

User ruling 2026-09 tightens the scale-in contract: a PLANNED engine
removal under load must not lose or fail any request ("不能丢，不能有
失败的请求") — every background request reaches a terminal state with
zero errors and zero hangs.  The mock's /remove_engine therefore defaults
to GRACEFUL drain (strip discovery first, wait bounded for in-flight
work, then tear down — production rolling scale-in order); the
zero-failure assertion below pins that contract, and the mock's
``mode="abrupt"`` parameter keeps the legacy hard teardown available
for chaos-style cases.  elastic_concurrent_ops stays exempt (see its
docstring: concurrent add/remove crossfire is a robustness extreme,
chaos-adjacent, not a planned scale-in).
"""

from __future__ import annotations

import json
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ..context import CaseContext, CaseDef, rid_base
from ..grade import GradeReport
from ..harness import (
    OMIT,
    TTL_DRAIN_TIMEOUT_S,
    AssertUtils,
    BalanceSampler,
    ConfigOverride,
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
    fault_env_perf,
    http_get_status,
    wait_for,
)
from .kv import _fam_keys

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
# Elastic cases (migrated from the legacy elastic group, category
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
        # Graceful scale-in (mock default): the call strips the discovery
        # entry first, waits bounded for the in-flight set to finish, and
        # only then tears the engine down — the production rolling order.
        status, rm_body = ops.remove_engine(engine_name=new_name)
        if status != 200:
            flow.stop()
            return False, f"remove_engine failed: {status} {rm_body}"
        rm_body = rm_body or {}

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
        # shared env).
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), TTL_DRAIN_TIMEOUT_S
        )
        # CONTRACT (user ruling 2026-09): a planned scale-in under load is
        # ZERO-FAILURE — no lost request (every fired request returned), no
        # failed request (no stream error), no hang (the flow consumes each
        # stream to its terminal state with a 10s cap; a hang surfaces as an
        # error).  The legacy >=90% tolerance described the mock's old hard
        # teardown (shutdownNow cutting in-flight streams); with the graceful
        # drain in place the correct behaviour is the one asserted here.
        zero_fail = total > 0 and ok == total
        passed = zero_fail and gone_from_snapshot and gone_from_file and inflight_ok
        return passed, (
            f"removed={new_name}(grpc={new_port}, accepted_at_removal={accepted_at_removal}, "
            f"drained={rm_body.get('drained')}, drain_ms={rm_body.get('drain_ms')}), "
            f"flow_success={ok}/{total}({rate:.1%}, zero-failure contract), "
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
    source="elastic acceptance: 3x add→verify→remove under load→verify cycle",
)
def elastic_add_remove_cycle(ctx: CaseContext):
    """Planned add/remove cycle under a background flow — every round's
    removal must be ZERO-FAILURE (user ruling 2026-09: "负载流运行中移除
    一个引擎，要确保请求不能丢，不能有失败的请求").

    Each round runs a background flow across the removal (the mock's
    graceful drain strips discovery first and waits out the in-flight
    set), then asserts the flow returned every request successfully —
    no error, no hang — on top of the original file/topology/traffic
    checks.
    """
    env, ops = _elastic_env(ctx)
    base = rid_base(ctx, "elastic")
    flow: Optional[_BackgroundFlow] = None
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

            # Background flow across the removal — the planned-scale-in
            # zero-failure contract (same caliber as elastic_remove_flow).
            flow = _BackgroundFlow(ops, base, interval_s=0.2)
            flow.start()
            time.sleep(0.5)  # some in-flight traffic before the mutation
            status_rm, _ = ops.remove_engine(engine_name=name)
            f_total, f_ok = flow.stop()
            flow = None  # stopped; nothing for the finally to reap
            flow_zero_fail = status_rm == 200 and f_total > 0 and f_ok == f_total

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
                and flow_zero_fail
                and file_rm_ok
                and parsable
            )
            all_ok = all_ok and round_ok
            round_details.append(
                f"r{round_no}[{name}]: file={file_ok} alive={alive_ok} "
                f"traffic={traffic_ok} rm={status_rm} "
                f"flow={f_ok}/{f_total}(zero-fail={f_ok == f_total and f_total > 0}) "
                f"file_rm={file_rm_ok} parsable={parsable}"
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
        if flow is not None:
            flow.stop()
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
    """Concurrent add/remove storm — robustness extreme, deliberately EXEMPT
    from the zero-failure contract.

    Unlike elastic_remove_flow / elastic_add_remove_cycle (a PLANNED
    scale-in under a steady flow, where the user ruling 2026-09 demands
    zero failures), this crossfire is chaos-adjacent: four threads
    add/remove engines every few hundred ms while the victim set keeps
    changing, discovery and routing race each other, and an engine may be
    removed while it is the only healthy candidate — availability below
    100% is a legitimate outcome of that stress shape.  The case keeps the
    hard assertions on what must NEVER break (master HTTP 200, discovery
    file parses and equals the services map, no residue) plus a
    conservative >=50% health floor so a total blackout still fails.
    """
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
        config_overrides=ConfigOverride(
            ordering="priority",
            decision="fixed_window",
            dispatcher="batch",
            queue_timeout_ms=OMIT,
            max_inflight_batches=2,
        ),
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

    Prediction (current master + graceful mock remove, from the code
    walk): /remove_engine (graceful default since 2026-09) strips the
    discovery entry first, then WAITS for the engine's in-flight set —
    the two 8s batch leases run to completion (~8s) before the teardown.
    The master, however, drops the endpoint's route as soon as the file
    changes (~1s): PrefillEndpoint.closeEndpoint runs WorkerBatcher.
    stopAndDrain which fail-closes every MASTER-side queued item to
    BATCH_DISPATCH_FAILED within seconds (the stranded set — fast visible
    terminal).  The already-dispatched batches finish on the engine, but
    after the route drop no master status poll ever pulls their
    completion records back — the client stream then parks until the
    stale-inflight TTL (30s) fails it (slow but bounded, <=40s cap).
    Expected observed shape: stranded = fast fail (~1-3s), dispatched =
    slow fail near the 30s TTL, master accounting clean ~30-45s — the
    case passes in that mixed shape.  If the retirement chain fails to
    drain the queue, the stranded items wait out queueTimeout (1h) ->
    inflight_clean(50s) FAILS -> finding.  (The dispatched-request
    slow-fail leg is itself the master-side scale-in drain gap — no
    completion-pull protocol survives route removal; related to the F7
    finding family.)
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
# near-uniform convergence.  CALIBRATION PLAN (calibration discipline):
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


# ===========================================================================
# Balance-metrics v2 elastic cases (design doc
# flexlb-balance-metrics-v2-design.md, part 2): KV-full shrink (two drain
# variants in ONE case), KV-skew shrink hot/cold (paired contrast),
# transient imbalance bound, steady-state recovery.
#
# Shared skeleton (design part-2 general conventions): a BalanceSampler
# polls the mock /metrics?per_engine=true and the master
# /rtp_llm/inflight_status once per second for the whole run; every case
# milestone is mark()'d (sampler-relative seconds); windows are cut
# relative to the event marks — W_base = 20s pre-event (>= 20 samples),
# W_tr = 2 x statusStaleAfterMs post-event, W_ss = >= 60s from t_settle
# (= max(remove_engine return, inflight-clean pass, master alive
# convergence)) in 3s x 20 sub-windows, recovery predicates reading the
# LAST THIRD (design §1.4); the sampler store is dumped as
# balance_sampler.json.gz into the case run directory and event timestamps
# + window bounds go into the case detail.
#
# Assertion policy (STRICT — the round's core discipline): a threshold
# gates the verdict (report.check / report.invariant) ONLY when it derives
# from a mechanism constant / explicit config / the case's own construction
# (drain 60s, stale 10s, TTL 30s, reserveRatio 0.05, explicit queue
# capacities, K_reject).  Every dimension the design flags "first-run
# calibration" is an OBSERVATION — the value plus its band key and design
# formula go into the case detail, never into report.check: all TPS dims
# (PT/PT2), exec CV (PL), hit steady-state recovery (PC >= baseline-0.15
# and >= 50% rebound), swing tolerance, skew recovery-duration contrast.
# A FAIL is evidence, never silenced by re-tuning the threshold.
# ===========================================================================

# ---- mechanism constants (Python-side named constants with provenance
# comments — NO Java imports, mirroring the PENDING_DRAIN_* precedent) ----
# Mock graceful-drain default cap: MockControlServer.DEFAULT_DRAIN_TIMEOUT_MS
# = 60_000 (flexlb-mock-engine MockControlServer.java — the drain field).
BAL_GRACEFUL_DRAIN_DEFAULT_MS = 60_000
# Master endpoint staleness: WorkerRegistryConfig.statusStaleAfterMs
# default = 10_000 (flexlb-common WorkerRegistryConfig.java);
# build_flexlb_config emits statusStaleAfterMs = max(10_000,
# status_rpc_ms * 2) and the fault axes run status_rpc_ms = 1_000, so the
# effective value IS 10_000 here.
BAL_STATUS_STALE_AFTER_MS = 10_000
# KV LRU reserve: MockLruBlockCache.DEFAULT_RESERVE_RATIO = 0.05 — the pool
# never admits past 95% occupancy; the design's PK upper bound.
BAL_KV_RESERVE_RATIO = 0.05
BAL_OCCUPANCY_CAP = 1.0 - BAL_KV_RESERVE_RATIO  # 0.95
# Java mock decode concurrency default (NOT overridden by these cases —
# the decode-side PQ capacity bound): JavaMockEngineCluster
# DEFAULT_DECODE_MAX_CONCURRENCY = 128.
BAL_JAVA_DECODE_MAX_CONCURRENCY = 128
# Inflight-stale TTL (build_flexlb_config stale_inflight_ms fault default).
BAL_STALE_INFLIGHT_MS = 30_000
# W_tr = 2 x statusStaleAfterMs (design §1.4 derivation: removal
# visibility = discovery strip ~1s + stale window 10s; doubled to ride out
# slow CI machines).
BAL_TRANSIENT_S = 2 * (BAL_STATUS_STALE_AFTER_MS / 1000.0)  # 20.0s
# W_base: pre-event steady baseline (>= 20 samples at 1s sampling).
BAL_BASELINE_S = 20.0
# W_ss: post-settle pure observation window, 3s x 20 sub-windows; recovery
# predicates read the last third (design §1.4).
BAL_STEADY_S = 60.0
BAL_STEADY_SUBWINDOW_S = 3.0
# Steady-state recovery tolerances (design §1.2/§1.4 rows): KV occupancy
# spread vs baseline, request-share ceiling/floor, queue depth, oscillation
# fingerprint (two consecutive same-direction sub-window departures).
BAL_KV_SPREAD_TOL = 0.05
BAL_SHARE_TOL = 0.10
BAL_SHARE_MIN_FLOOR = 0.10
BAL_QUEUE_DEPTH_MAX = 2.0


def _mechanism_bands(bound: float) -> dict:
    """Three-tier case-override bands from a mechanism-derived bound.

        All three tiers take the SAME value: the bound is mechanical (a Java
    # constant / explicit config value), not a graded quality level — the
        same convention as the GRADE_BANDS PK note and the ADD_PREF_SHARE_BANDS
        case-override precedent (case-level bands gate the verdict; the
        GRADE_BANDS default never does for these cases).
    """
    return {"strict": bound, "normal": bound, "loose": bound}


def _bal_occupancy(entry: Optional[dict]) -> Optional[float]:
    """(cache_blocks - available_blocks) / cache_blocks; None when the
    pool size is 0/absent (the caller fails loud on None)."""
    if not entry:
        return None
    total = entry.get("cache_blocks") or 0
    if total <= 0:
        return None
    return (total - int(entry.get("available_blocks") or 0)) / total


def _bal_rejects(entry: Optional[dict]) -> int:
    """kv_admission_fails + lack_mem_rejects (cumulative counters)."""
    if not entry:
        return 0
    return int(entry.get("kv_admission_fails", 0)) + int(
        entry.get("lack_mem_rejects", 0)
    )


def _bal_k_reject(victim_demand_blocks: float, survivor_free_blocks: float) -> int:
    """K_reject = ceil(max(0, victim demand - survivor free)) (design
    §1.3-4 / task-brief caliber).

    Every REJECTED request needs at least ONE block, so the block
    shortfall is an UPPER BOUND on the reject count: the transient-window
    survivor delta (lack_mem_rejects + kv_admission_fails) must stay
    <= K_reject.  The victim demand is measured as the victim's OCCUPIED
    blocks at removal (running leases + parked LRU keys): the LRU share
    does not itself produce admission rejects, so this caliber leans to
    the LOOSER (bound-raising) direction — recorded here rather than
    silently tightened; a first-run breach is arbitration evidence.
    """
    return math.ceil(max(0.0, victim_demand_blocks - survivor_free_blocks))


def _bal_occupancy_series(
    sampler: BalanceSampler, engine: str, t_lo: float, t_hi: float
) -> list:
    """Per-sample occupancy [(t, occ)] over [t_lo, t_hi] from the
    cache/available gauge pair (same-tick alignment — both gauges land in
    one /metrics scrape; samples missing either side are skipped)."""
    total_pts = sampler.window_series("mock_engine_cache_blocks", t_lo, t_hi).get(
        engine, []
    )
    avail_pts = sampler.window_series("mock_engine_available_blocks", t_lo, t_hi).get(
        engine, []
    )
    avail_by_t = {t: v for t, v in avail_pts}
    out: list = []
    for t, total in total_pts:
        avail = avail_by_t.get(t)
        if avail is None or total <= 0:
            continue
        out.append((t, (total - avail) / total))
    return out


def _bal_hit_rate(
    sampler: BalanceSampler, t_lo: float, t_hi: float, engines=None
) -> Optional[float]:
    """Cluster key-hit rate over [t_lo, t_hi]: sum of per-engine counter
    deltas (hits / requested); None when no in-window traffic (fail-loud
    at the caller — a missing rate is never a passing one)."""
    hits = sampler.window_series(
        "mock_engine_cache_key_hits_total", t_lo, t_hi, mode="delta"
    )
    reqs = sampler.window_series(
        "mock_engine_cache_keys_requested_total", t_lo, t_hi, mode="delta"
    )
    keys = set(hits) | set(reqs)
    if engines is not None:
        keys &= set(engines)
    tot_h = sum(hits.get(k, 0.0) for k in keys)
    tot_r = sum(reqs.get(k, 0.0) for k in keys)
    if tot_r <= 0:
        return None
    return tot_h / tot_r


def _bal_cluster_tps(
    sampler: BalanceSampler, t_lo: float, t_hi: float
) -> Optional[float]:
    """Cluster TPS over [t_lo, t_hi]: sum of per-engine window means of
    rtp_llm_context_tps (the production-caliber window gauge; the sampler
    drains one window per 1s scrape).  None without samples."""
    series = sampler.window_series("rtp_llm_context_tps", t_lo, t_hi)
    per = []
    for pts in series.values():
        if pts:
            per.append(sum(v for _, v in pts) / len(pts))
    return sum(per) if per else None


def _bal_engine_mean(series_pts: list) -> Optional[float]:
    """Mean of a [(t, v)] window slice (None when empty)."""
    if not series_pts:
        return None
    return sum(v for _, v in series_pts) / len(series_pts)


def _obs_note(key: str, formula: str, value) -> str:
    """Format one OBSERVATION item (value + band key + design formula;
    observation-only, never gates the verdict)."""
    v = "n/a" if value is None else f"{value:.3f}"
    return f"OBS[{key}] {formula}={v}"


def _bal_snap_series_mean(
    sampler: BalanceSampler, metric: str, t_lo: float, t_hi: float
) -> dict:
    """{engine: window mean} of a gauge metric (engines with no in-window
    samples are absent)."""
    out: dict = {}
    for eng, pts in sampler.window_series(metric, t_lo, t_hi).items():
        if pts:
            out[eng] = sum(v for _, v in pts) / len(pts)
    return out


# ===========================================================================
# elastic_kv_full_shrink — KV-full scale-in, two drain variants
# (design §2.1)
# ===========================================================================

# Private decode pool: 24 blocks (EnvSpec.decode_cache_blocks is a
# first-class field forwarded to the mock as --decode-kv-pool-blocks).
# Decode block demand per request = ceil(inputLen / block_size)
# (JavaMockEngineCluster.decodeDemandBlocks; input_len 2048 / 1024 = 2
# blocks), so ~12 concurrent in-flight decode requests saturate a 24-block
# pool (24 x (1 - reserveRatio 0.05) = 22.8 usable — the 12th 2-block
# request crosses the reserve line and RETRY-fails).
FULL_SHRINK_DECODE_CACHE_BLOCKS = 24
FULL_SHRINK_INPUT_LEN = 2048
FULL_SHRINK_FILL_BATCH = 8  # fill-wave fire granularity per poll round
FULL_SHRINK_FILL_TIMEOUT_S = 60.0
FULL_SHRINK_AVAIL_TOL = 1  # pre-assertion "full": available_blocks <= 1
# Decode-tail construction — DESIGN CONFLICT, recorded per the brief: the
# design text says "set_perf decode_step_ms"; the mock's /set_perf
# implements decode_scale (a multiplier on stepMs — MockPerformanceModel
# decodeMs = steps x stepMs x scale; handleSetPerf accepts only
# prefill_fixed_ms / decode_scale / max_prefill_concurrency /
# max_waiting_batches / decode_retry_*).  Implemented with decode_scale,
# the sanctioned slow-decode channel.  stepMs ~= 19.5 + 0.175 x running ms
# (production DSv4 fit code default, tokens_per_step 2.6 MTP fold ->
# output_len 13 = 5 steps):
#   drain_ok tail <= 8s:   5 x ~21ms x 60  ~= 6.3s (drain waits it out)
#   drain_timeout tail > 5s: 5 x ~21ms x 100 ~= 10.5s (crosses the 5s cap)
FULL_SHRINK_OUTPUT_LEN = 13
FULL_SHRINK_DRAIN_OK_SCALE = 60.0
FULL_SHRINK_DRAIN_TIMEOUT_SCALE = 100.0
FULL_SHRINK_DRAIN_TIMEOUT_MS = 5_000
# drain_ms ~= timeout PROVES the fallback branch: [5.0, 10.0]s — the lower
# bound IS the configured timeout (the drain loop cannot return early
# while work is pending), the upper adds a 5s management margin for the
# drain loop's check granularity + teardown.
FULL_SHRINK_DRAIN_MS_LO = 5.0
FULL_SHRINK_DRAIN_MS_HI = 10.0
# Terminal-state deadline: the SAME derivation as PENDING_DRAIN_TERMINAL_S
# (stale-inflight TTL 30s + settlement + margin).
FULL_SHRINK_TERMINAL_S = 40.0
FULL_SHRINK_CLEAN_S = 50.0
# 8211 — the synchronous decode-side KV-allocation failure the master
# surfaces when the D pool cannot serve the net demand after the retry
# window (JavaMockEngineCluster.reserveDecodeLease -> master-surface
# ack).  Fill-phase 8211 rejections are the "full" COUNTER evidence and
# sit OUTSIDE the zero-failure contract (they never entered the victim's
# in-flight set — admission refused them).
FULL_SHRINK_DECODE_KV_FAIL_CODE = 8211


def _full_shrink_spec(ctx: CaseContext) -> EnvSpec:
    """Private env for elastic_kv_full_shrink: 2P+2D, 24-block decode
    pools, dynamic file discovery, legacy fault axes.

    The decode_cache_blocks fingerprint differs from every other spec
    (elastic_spec 3000 / quota / pending-drain), so the one-shot
    initial-engine victims (decode-0 and decode-1, permanently removed —
    a removed initial engine never comes back on its port/name) never
    poison a shared env: this spec owns a private one (same reason as
    _pending_drain_spec).
    """
    return EnvSpec(
        label=f"fault_full_shrink_{ctx.profile}",
        n_prefill=2,
        n_decode=2,
        perf=fault_env_perf(),
        master_profile=ctx.profile,
        discovery="discovery_file",
        decode_cache_blocks=FULL_SHRINK_DECODE_CACHE_BLOCKS,
        config_overrides=ConfigOverride(
            ordering="priority",
            decision="fixed_window",
            dispatcher="batch",
            queue_timeout_ms=OMIT,
        ),
    )


@case(
    "elastic_kv_full_shrink",
    profiles=["batch-window"],  # elastic family: BATCH dispatcher + fault axes
    source=(
        "balance-metrics v2 design §2.1 "
        "(flexlb-balance-metrics-v2-design.md): KV-full scale-in, "
        "drain_ok + drain_timeout variants"
    ),
)
def elastic_kv_full_shrink(ctx: CaseContext):
    """KV-full scale-in: graceful drain under a saturated decode pool
    (design §2.1), both drain branches in ONE case — 2P+2D with 24-block
    decode pools, so the two decodes each carry one variant.

    Scenario: a long decode tail (set_perf decode_scale — see the
    FULL_SHRINK_DRAIN_* conflict note: the mock has no decode_step_ms
    field, decode_scale is the sanctioned channel) makes concurrent
    decode requests hold their blocks long enough to fill the pool with
    real in-flight work (no set_kv_pressure shortcut — the design
    explicitly wants the long-tail construction).  A pre-assertion proves
    "full" on BOTH engines: available_blocks <= 1 AND the counter delta
    (kv_admission_fails + lack_mem_rejects) > 0 (counter evidence, not
    eyeballing).

    Background-flow structure (conflict handling, see the assertion
    list): the flow runs through W_base and W_ss but is PAUSED around the
    remove+clean event windows — inflight_clean is a hard ≤50s assertion
    and a never-stopping flow keeps master inflight permanently non-zero,
    which would make the deadline unmeasurable.  The design's "background
    traffic" survives in the baseline/steady windows; the trade-off is
    recorded here.

    Expected (CONTRACT — thresholds are mechanism-derived unless marked
    OBS):
      Variant 1 — drain_ok (victim decode-0, default 60s cap):
      1. rm_body.drained is True — the mechanism's own proof that the
         victim's engine-side in-flight set ran to completion.
      2. Zero-failure contract (user ruling 2026-09): every fired request
         ends completed or as a fill-phase 8211 admission refusal (the
         "full" evidence, pre-event by construction — a mid-drain kill
         would surface as 8510 and break this); no rpc_error / hang /
         empty close; terminal within FULL_SHRINK_TERMINAL_S = 40s of the
         removal (stale-TTL derivation, same as PENDING_DRAIN_TERMINAL_S).
      3. AssertUtils.inflight_clean within 50s (PENDING_DRAIN_CLEAN_S
         caliber).
      Variant 2 — drain_timeout (victim decode-1, drain_timeout_ms=5000,
      decode tail > 5s so the drain necessarily times out):
      4. rm_body.drained is False and drain_ms in [5.0, 10.0]s — the
         timeout branch (drain_ms ~= timeout; bounds derivation on the
         constant).
      5. Victim decode in-flight ends with the DECODE_GENERATION_RETIRED
         terminal — client-visible shape: stream error code 8510
         (StrategyErrorType.BATCH_DISPATCH_FAILED — the code the master
         maps decode-settled terminals to, RequestRegistry.
         applyDecodeSettledTerminalLocked) and message containing
         "Decode endpoint generation retired" (EndpointEventProjector's
         detail string).  An error-code assertion, not just "has error".
      6. Terminal within 40s; no other failure family (fill 8211 still
         allowed, pre-event by construction).
      Both variants — transient KV bounds (PK, mechanism bands):
      7. Survivor occupancy peak over W_tr <= 0.95 (= 1 - reserveRatio
         0.05, MockLruBlockCache.DEFAULT_RESERVE_RATIO).
      8. Survivor Δ(lack_mem + admission_fails) over W_tr <= K_reject =
         ceil(max(0, victim occupied blocks - survivor free blocks))
         (caliber note on _bal_k_reject: occupied includes parked LRU
         keys, the looser direction).
      Steady-state recovery (after variant 1, on the 2-decode cluster
      {decode-1, newcomer}; last third of W_ss):
      9. KV occupancy spread <= baseline + 0.05 (PK spread).
      10. Mock waiting depth peak <= 2 (PQ steady row).
      11. Request share max <= max(baseline + 0.10, 1/n + 0.15) and
          min >= 0.10 (P1 steady + P2 floor, case-level bands).
      12. OBS[PT]: steady cluster TPS (design §1.2 PT row — first-run
          calibration, never gates).
    """
    env = ctx.env_manager.ensure(_full_shrink_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "elastic")
    flows: list = []
    sampler: Optional[BalanceSampler] = None
    fired: list = []  # every (rid, resp, handle) ever fired — finally drain
    obs: list = []
    try:
        snap = ops.snapshot_by_name()
        for name in ("decode-0", "decode-1"):
            if name not in snap:
                return False, (
                    f"{name} missing from the private env (one-shot "
                    f"victims: a rerun needs a fresh process); "
                    f"engines={sorted(snap)}"
                )
        _cleanup_dynamic(ops, env)
        if not _wait_master_topology(ops, "DECODE", 2, MASTER_EVICT_S):
            return False, "decode topology did not converge to 2"

        sampler = BalanceSampler(ops.mock_http_port, ops.master_http_port)
        sampler.start()

        # ---- W_base: pre-event baseline (background traffic only) ----
        flow_base = _BackgroundFlow(ops, base, interval_s=0.5)
        flows.append(flow_base)
        flow_base.start()
        time.sleep(1.0)  # ramp
        sampler.mark("baseline_start")
        time.sleep(BAL_BASELINE_S)
        t_base = sampler.mark("baseline_end")
        flow_base.stop()

        base_occ = {}
        for eng in ("decode-0", "decode-1"):
            base_occ[eng] = _bal_engine_mean(
                _bal_occupancy_series(sampler, eng, 0.0, t_base)
            )
        base_occ_vals = [v for v in base_occ.values() if v is not None]
        base_occ_spread = (
            max(base_occ_vals) - min(base_occ_vals) if len(base_occ_vals) == 2 else 0.0
        )
        base_share_deltas = sampler.window_series(
            "mock_engine_accepted_total", 0.0, t_base, mode="delta"
        )
        base_share_vals = [
            base_share_deltas.get(e, 0.0) for e in ("decode-0", "decode-1")
        ]
        base_share_tot = sum(base_share_vals)
        base_share_max = (
            max(base_share_vals) / base_share_tot if base_share_tot > 0 else None
        )
        base_tps = _bal_cluster_tps(sampler, 0.0, t_base)

        # ---- fill helper: fire long-decode waves until BOTH pools full ----
        def _both_full(names) -> bool:
            s = ops.snapshot_by_name()
            return all(
                int(s.get(n, {}).get("available_blocks", 10**9))
                <= FULL_SHRINK_AVAIL_TOL
                for n in names
            )

        def _fill(names: list, handles: list) -> tuple:
            deadline = time.monotonic() + FULL_SHRINK_FILL_TIMEOUT_S
            while time.monotonic() < deadline:
                for _ in range(FULL_SHRINK_FILL_BATCH):
                    rid = ops.next_request_id(base)
                    try:
                        resp = ops.schedule(
                            rid,
                            input_len=FULL_SHRINK_INPUT_LEN,
                            output_len=FULL_SHRINK_OUTPUT_LEN,
                            block_keys=[rid * 100 + j for j in range(3)],
                        )
                    except Exception:
                        continue
                    if resp.code != 200 or not resp.success:
                        continue
                    handle = ops.start_stream(resp, rid)
                    handles.append((rid, resp, handle))
                    fired.append((rid, resp, handle))
                if _both_full(names):
                    return True, ""
                time.sleep(0.5)
            return (
                _both_full(names),
                f"fill timeout after {FULL_SHRINK_FILL_TIMEOUT_S:.0f}s",
            )

        def _classify(handles: list) -> list:
            """[(kind, code, message)] for every handle, terminal or not."""
            out = []
            for rid, resp, handle in handles:
                s = handle.snap
                if s.completed and not s.error:
                    out.append(("completed", None, None, rid))
                elif s.stream_error_code is not None:
                    out.append(
                        ("error", s.stream_error_code, s.stream_error_message, rid)
                    )
                elif s.error:
                    out.append(("rpc_error", None, str(s.error)[:60], rid))
                elif s.terminated:
                    out.append(("empty", None, None, rid))
                else:
                    out.append(("hang", None, None, rid))
            return out

        # ---- variant 1: drain_ok (victim = decode-0) ----
        for name in ("decode-0", "decode-1"):
            ops.set_perf(name, decode_scale=FULL_SHRINK_DRAIN_OK_SCALE)
        time.sleep(1.0)  # perf sync

        v1_handles: list = []
        v1_full, v1_fill_detail = _fill(("decode-0", "decode-1"), v1_handles)
        if not v1_full:
            return False, f"variant-1 construction failed: {v1_fill_detail}"

        pre1 = ops.snapshot_by_name()
        v1_avail_ok = all(
            int(pre1.get(n, {}).get("available_blocks", 10**9)) <= FULL_SHRINK_AVAIL_TOL
            for n in ("decode-0", "decode-1")
        )
        v1_rejects = sum(_bal_rejects(pre1.get(n)) for n in ("decode-0", "decode-1"))
        if not v1_avail_ok or v1_rejects <= 0:
            return False, (
                f"variant-1 pre-assertion failed: pools full={v1_avail_ok}, "
                f"kv rejects={v1_rejects} (need >0 — counter evidence)"
            )
        # K_reject inputs (victim demand = occupied blocks at removal;
        # survivor free = available blocks, same snapshot).
        v1_demand = (
            pre1["decode-0"]["cache_blocks"] - pre1["decode-0"]["available_blocks"]
        )
        v1_free = pre1["decode-1"]["available_blocks"]
        k_reject_1 = _bal_k_reject(v1_demand, v1_free)

        t_v1_remove = sampler.mark("v1_remove")
        status, rm_body = ops.remove_engine(engine_name="decode-0")
        if status != 200:
            return False, f"remove_engine(decode-0) failed: {status} {rm_body}"
        rm1 = rm_body or {}
        v1_drained = bool(rm1.get("drained"))
        v1_drain_ms = rm1.get("drain_ms")

        # terminal collection for the fill wave (40s cap + margin)
        for rid, resp, handle in v1_handles:
            handle.wait_end(FULL_SHRINK_TERMINAL_S + 10.0)
        t_v1_terminal = time.monotonic()
        v1_out = _classify(v1_handles)
        v1_kinds = {}
        for kind, code, _msg, rid in v1_out:
            v1_kinds.setdefault((kind, code), []).append(rid)
        v1_bad = [
            (k, c, r)
            for (k, c), rids in v1_kinds.items()
            if (k, c) not in (("completed", None), ("error", 8211))
            for r in rids[:3]
        ]

        t_clean0 = time.monotonic()
        v1_clean_ok, v1_clean_detail = AssertUtils.inflight_clean(
            _master_http(ops), FULL_SHRINK_CLEAN_S
        )
        v1_clean_s = time.monotonic() - t_clean0
        t_v1_clean = sampler.mark("v1_clean")

        # restore the 2-decode cluster: add a dynamic decode engine
        status, body = ops.add_engine("decode")
        if status != 200:
            return False, f"add_engine(decode) failed: {status} {body}"
        v2_survivor = body["engine"]
        v2_topo_ok = _wait_master_topology(ops, "DECODE", 2, MASTER_EVICT_S)
        t_v1_alive = sampler.mark("v1_alive")
        t_v1_settle = max(t_v1_remove, t_v1_clean, t_v1_alive)

        # ---- W_ss (variant 1): steady-state recovery on {decode-1, new} ----
        flow_ss = _BackgroundFlow(ops, base, interval_s=0.5)
        flows.append(flow_ss)
        flow_ss.start()
        time.sleep(1.0)  # ramp
        time.sleep(BAL_STEADY_S)
        flow_ss.stop()
        t_ss1_end = sampler.mark("v1_steady_end")
        ss1_engines = ("decode-1", v2_survivor)
        ss1_tail_lo = t_v1_settle + BAL_STEADY_S * 2.0 / 3.0

        # 9. KV occupancy spread (last third)
        ss1_occ = {}
        for eng in ss1_engines:
            ss1_occ[eng] = _bal_engine_mean(
                _bal_occupancy_series(sampler, eng, ss1_tail_lo, t_ss1_end)
            )
        ss1_occ_vals = [v for v in ss1_occ.values() if v is not None]
        AssertUtils.balanced(
            report,
            "PK",
            {
                e: _bal_occupancy_series(sampler, e, ss1_tail_lo, t_ss1_end)
                for e in ss1_engines
            },
            (ss1_tail_lo, t_ss1_end),
            "spread",
            _mechanism_bands(base_occ_spread + BAL_KV_SPREAD_TOL),
            context="kv_full_shrink_v1_steady_spread",
        )

        # 10. waiting depth peak (last third)
        AssertUtils.balanced(
            report,
            "PQ",
            {
                e: sampler.window_series(
                    "mock_engine_waiting", ss1_tail_lo, t_ss1_end
                ).get(e, [])
                for e in ss1_engines
            },
            (ss1_tail_lo, t_ss1_end),
            "peak",
            _mechanism_bands(BAL_QUEUE_DEPTH_MAX),
            context="kv_full_shrink_v1_steady_depth",
        )

        # 11. request share (counter-delta caliber — computed directly,
        # balanced() aggregates value series, not counter deltas)
        ss1_share_deltas = sampler.window_series(
            "mock_engine_accepted_total", t_v1_settle, t_ss1_end, mode="delta"
        )
        ss1_share_vals = [ss1_share_deltas.get(e, 0.0) for e in ss1_engines]
        ss1_share_tot = sum(ss1_share_vals)
        if ss1_share_tot <= 0:
            return False, "variant-1 steady window void: no decode traffic"
        ss1_share_max = max(ss1_share_vals) / ss1_share_tot
        ss1_share_min = min(ss1_share_vals) / ss1_share_tot
        ss1_share_cap = max(
            (base_share_max or 0.0) + BAL_SHARE_TOL,
            1.0 / len(ss1_engines) + 0.15,
        )
        report.check(
            "P1",
            ss1_share_max,
            bands=_mechanism_bands(ss1_share_cap),
            context="kv_full_shrink_v1_steady_share_max",
            detail=(
                f"deltas={dict(zip(ss1_engines, ss1_share_vals))}, "
                f"cap=max(base+0.10, 1/n+0.15)={ss1_share_cap:.3f}"
            ),
        )
        report.invariant(
            "P2",
            ss1_share_min >= BAL_SHARE_MIN_FLOOR,
            context="kv_full_shrink_v1_steady_share_min",
            detail=f"min_share={ss1_share_min:.3f}, floor={BAL_SHARE_MIN_FLOOR}",
        )

        # 12. OBS[PT] steady cluster TPS
        obs.append(
            _obs_note(
                "PT",
                "steady_cluster_tps(design §1.2 PT row; first-run calibration)",
                _bal_cluster_tps(sampler, ss1_tail_lo, t_ss1_end),
            )
        )

        # 7/8. PK transient bounds — survivor occupancy peak + K_reject
        # over W_tr (the sampler kept sampling through the drain).
        occ1_series = _bal_occupancy_series(
            sampler, "decode-1", t_v1_remove, t_v1_remove + BAL_TRANSIENT_S
        )
        AssertUtils.balanced(
            report,
            "PK",
            {"decode-1": occ1_series},
            (t_v1_remove, t_v1_remove + BAL_TRANSIENT_S),
            "peak",
            _mechanism_bands(BAL_OCCUPANCY_CAP),
            context="kv_full_shrink_v1_transient_occupancy",
        )
        rej1_deltas = sampler.window_series(
            "mock_engine_kv_admission_fails_total",
            t_v1_remove,
            t_v1_remove + BAL_TRANSIENT_S,
            mode="delta",
        ).get("decode-1", 0.0) + sampler.window_series(
            "mock_engine_lack_mem_rejects_total",
            t_v1_remove,
            t_v1_remove + BAL_TRANSIENT_S,
            mode="delta",
        ).get(
            "decode-1", 0.0
        )
        report.check(
            "PK",
            rej1_deltas,
            bands=_mechanism_bands(k_reject_1),
            context="kv_full_shrink_v1_transient_rejects",
            detail=(
                f"K_reject=ceil(max(0, victim_occupied({v1_demand}) "
                f"- survivor_free({v1_free})))={k_reject_1}"
            ),
        )

        # ---- variant 2: drain_timeout (victim = decode-1) ----
        for name in ("decode-1", v2_survivor):
            ops.set_perf(name, decode_scale=FULL_SHRINK_DRAIN_TIMEOUT_SCALE)
        time.sleep(1.0)  # perf sync

        v2_handles: list = []
        v2_full, v2_fill_detail = _fill(("decode-1", v2_survivor), v2_handles)
        if not v2_full:
            return False, f"variant-2 construction failed: {v2_fill_detail}"

        pre2 = ops.snapshot_by_name()
        v2_avail_ok = all(
            int(pre2.get(n, {}).get("available_blocks", 10**9)) <= FULL_SHRINK_AVAIL_TOL
            for n in ("decode-1", v2_survivor)
        )
        v2_rejects = sum(_bal_rejects(pre2.get(n)) for n in ("decode-1", v2_survivor))
        if not v2_avail_ok or v2_rejects <= 0:
            return False, (
                f"variant-2 pre-assertion failed: pools full={v2_avail_ok}, "
                f"kv rejects={v2_rejects} (need >0)"
            )
        v2_demand = (
            pre2["decode-1"]["cache_blocks"] - pre2["decode-1"]["available_blocks"]
        )
        v2_free = pre2[v2_survivor]["available_blocks"]
        k_reject_2 = _bal_k_reject(v2_demand, v2_free)

        t_v2_remove = sampler.mark("v2_remove")
        status, rm_body = ops.remove_engine(
            engine_name="decode-1", drain_timeout_ms=FULL_SHRINK_DRAIN_TIMEOUT_MS
        )
        if status != 200:
            return False, f"remove_engine(decode-1) failed: {status} {rm_body}"
        rm2 = rm_body or {}
        v2_drained = bool(rm2.get("drained"))
        v2_drain_ms = rm2.get("drain_ms")
        v2_running_at_removal = rm2.get("running_at_removal")

        for rid, resp, handle in v2_handles:
            handle.wait_end(FULL_SHRINK_TERMINAL_S + 10.0)
        v2_out = _classify(v2_handles)
        v2_retired = [
            (code, msg, rid)
            for kind, code, msg, rid in v2_out
            if kind == "error" and code == 8510
        ]
        v2_retired_ok = all(
            msg and "Decode endpoint generation retired" in msg
            for _c, msg, _r in v2_retired
        )
        v2_bad = [
            (k, c, r)
            for k, c, _m, r in v2_out
            if (k, c) not in (("completed", None), ("error", 8211), ("error", 8510))
        ]

        # 7/8 (variant 2). PK transient bounds on the survivor.
        occ2_series = _bal_occupancy_series(
            sampler, v2_survivor, t_v2_remove, t_v2_remove + BAL_TRANSIENT_S
        )
        AssertUtils.balanced(
            report,
            "PK",
            {v2_survivor: occ2_series},
            (t_v2_remove, t_v2_remove + BAL_TRANSIENT_S),
            "peak",
            _mechanism_bands(BAL_OCCUPANCY_CAP),
            context="kv_full_shrink_v2_transient_occupancy",
        )
        rej2_deltas = sampler.window_series(
            "mock_engine_kv_admission_fails_total",
            t_v2_remove,
            t_v2_remove + BAL_TRANSIENT_S,
            mode="delta",
        ).get(v2_survivor, 0.0) + sampler.window_series(
            "mock_engine_lack_mem_rejects_total",
            t_v2_remove,
            t_v2_remove + BAL_TRANSIENT_S,
            mode="delta",
        ).get(
            v2_survivor, 0.0
        )
        report.check(
            "PK",
            rej2_deltas,
            bands=_mechanism_bands(k_reject_2),
            context="kv_full_shrink_v2_transient_rejects",
            detail=(
                f"K_reject=ceil(max(0, victim_occupied({v2_demand}) "
                f"- survivor_free({v2_free})))={k_reject_2}"
            ),
        )

        # ---- CONTRACT ASSERTIONS -------------------------------------------
        # Variant 1: drained=True (the drain ran the victim's in-flight
        # set to completion — the mechanism's own zero-loss proof).
        report.invariant(
            "P6",
            v1_drained,
            context="kv_full_shrink_v1_drained",
            detail=f"drained={v1_drained}, drain_ms={v1_drain_ms}",
        )
        # Variant 1: zero-failure contract — every fired request is
        # completed or a fill-phase 8211 admission refusal; a mid-drain
        # kill would surface as 8510 and break this at every grade.
        report.invariant(
            "P6",
            not v1_bad,
            context="kv_full_shrink_v1_zero_failure",
            detail=f"violations={v1_bad[:3]}, kinds={ {str(k): len(v) for k, v in v1_kinds.items()} }",
        )
        # Variant 1: master accounting clean within 50s.
        report.invariant(
            "P6",
            v1_clean_ok,
            context="kv_full_shrink_v1_inflight_clean",
            detail=(
                f"clean={v1_clean_s:.1f}s cap={FULL_SHRINK_CLEAN_S:.0f}s "
                f"{v1_clean_detail[:80]}"
            ),
        )
        # Variant 2: the timeout branch — drained=False and drain_ms in
        # [5.0, 10.0]s (drain_ms ~= the configured timeout).
        drain_ms_v = float(v2_drain_ms) if v2_drain_ms is not None else -1.0
        report.invariant(
            "P6",
            (not v2_drained)
            and FULL_SHRINK_DRAIN_MS_LO <= drain_ms_v <= FULL_SHRINK_DRAIN_MS_HI,
            context="kv_full_shrink_v2_drain_timeout_branch",
            detail=(
                f"drained={v2_drained}, drain_ms={v2_drain_ms} "
                f"(expect False and [{FULL_SHRINK_DRAIN_MS_LO}, "
                f"{FULL_SHRINK_DRAIN_MS_HI}]s = timeout 5s + management "
                f"margin)"
            ),
        )
        # Variant 2: DECODE_GENERATION_RETIRED terminal — 8510 + the
        # retirement message (error-code assertion, not just "has error");
        # at least one victim in-flight request existed (running_at_removal
        # consistency recorded in the detail).
        report.invariant(
            "P6",
            bool(v2_retired) and v2_retired_ok and not v2_bad,
            context="kv_full_shrink_v2_decode_retired",
            detail=(
                f"retired={len(v2_retired)} (code 8510, msg ok={v2_retired_ok}), "
                f"running_at_removal={v2_running_at_removal}, "
                f"other_failures={v2_bad[:3]}"
            ),
        )

        # ---- survivor service recovery + final accounting ----
        try:
            ops.set_perf(v2_survivor, decode_scale=1.0)
        except Exception:
            pass
        ok_n, _err_n, _ = _run_batch(ops, base, 20)
        recovery_rate = ok_n / 20.0
        report.invariant(
            "P2",
            recovery_rate >= 0.95,
            context="kv_full_shrink_survivor_service",
            detail=f"recovery {ok_n}/20",
        )

        events_summary = json.dumps(dict(sampler._events)) if sampler else "{}"
        ok_verdict, detail, _rep = report.finish(
            f"v1(drain_ok): drained={v1_drained} "
            f"drain_ms={v1_drain_ms}, rejects_pre={v1_rejects}, "
            f"kinds={ {str(k): len(v) for k, v in v1_kinds.items()} }, "
            f"clean={v1_clean_s:.1f}s(ok={v1_clean_ok}), "
            f"k_reject={k_reject_1}(rej_delta={rej1_deltas:.0f}), "
            f"steady(share_max={ss1_share_max:.2f}/cap {ss1_share_cap:.2f}, "
            f"share_min={ss1_share_min:.2f}), "
            f"v2(drain_timeout): drained={v2_drained} "
            f"drain_ms={v2_drain_ms}, retired_8510={len(v2_retired)}"
            f"(running_at_removal={v2_running_at_removal}), "
            f"k_reject={k_reject_2}(rej_delta={rej2_deltas:.0f}), "
            f"recovery={ok_n}/20, "
            f"base(occ_spread={base_occ_spread:.3f}, "
            f"share_max={base_share_max}, tps={base_tps}), "
            f"windows=(W_base[0,{t_base:.0f}], "
            f"W_tr1[{t_v1_remove:.0f},{t_v1_remove + BAL_TRANSIENT_S:.0f}], "
            f"W_ss1[{t_v1_settle:.0f},{t_ss1_end:.0f}], "
            f"W_tr2[{t_v2_remove:.0f},{t_v2_remove + BAL_TRANSIENT_S:.0f}]), "
            f"events={events_summary}, "
            f"{'; '.join(obs)}, "
            f"grades: {report.summary()}"
        )
        return ok_verdict, detail, _rep
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        for f in flows:
            try:
                f.stop()
            except Exception:
                pass
        for rid, resp, handle in fired:
            try:
                if not handle.snap.terminated:
                    handle.wait_end(10.0)
                if not handle.snap.completed and not handle.snap.error:
                    ops.cancel(rid, resp)
            except Exception:
                try:
                    ops.cancel(rid, resp)
                except Exception:
                    pass
        if sampler is not None:
            try:
                sampler.stop()
                sampler.dump(
                    ctx.case_dir("elastic_kv_full_shrink") / "balance_sampler.json.gz"
                )
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
# elastic_kv_skew_shrink_hot / _cold — KV-skew scale-in contrast
# (design §2.2): shrink the family-heavy engine vs the family-light one
# under sustained affinity traffic, hit-drop model + recovery contrast.
# ===========================================================================

# Asymmetric prefill: hot fast (50ms) / cold slow (500ms) — the cost
# router (ESTIMATED_TTFT) concentrates both traffic and family seeds on
# the hot engine (the balance_overload_avoid family proved the mechanism
# controllable; same caliber as its perf axes).
SKEW_PREFILL_FAST_MS = 50.0
SKEW_PREFILL_SLOW_MS = 500.0
SKEW_PREFILL_SYMMETRIC_MS = 100.0  # post-seed restore (fault_env_perf caliber)
# Prefix families (kv.py _fam_keys: 10 blocks x 1024 tokens per family,
# 1000-key stride).  8 families x 3 rounds = 24 seed requests.
SKEW_FAMILIES = 8
SKEW_FAM_ROUNDS = 3
SKEW_FAM_INPUT_LEN = 2048
# Pre-assertion "skew holds": hot's cache_key count >= R x cold's (the
# design's "R 由构造流量配比导出，目标 >= 3").
SKEW_MIN_RATIO = 3.0
# PC transient bounds (design §2.2 assertion 1 — construction-derived,
# NOT eyeballed): shrink hot — measured cluster hit drop <= expected
# drop + 0.10, expected drop = hot-held family traffic share x family
# hit rate (both construction-known: the share from the seeded routing
# split, the hit rate from the measured baseline window); shrink cold —
# drop <= 0.10 absolute (cold holds a minor family share).  Evaluated in
# the LOWER form tr_hit >= base_hit - cap so the PC key keeps its GRADE_
# BANDS lower-kind semantics.
SKEW_DROP_TOLERANCE = 0.10
SKEW_COLD_DROP_ABS = 0.10
# Family-affinity pump cadence through the measurement windows (the
# sustained affinity traffic the PC recovery slope needs).
SKEW_PUMP_INTERVAL_S = 0.15
# PC steady OBSERVATION (design §2.2 assertion 2 — first-run calibration,
# never gates): last-third hit >= baseline - 0.15 and >= trough + 50% of
# (baseline - trough) rebound; shrink cold has almost no recovery need.
SKEW_PC_STEADY_BASE_TOL = 0.15
SKEW_PC_REBOUND = 0.5


def _skew_spec(ctx: CaseContext, variant: str) -> EnvSpec:
    """Private env for one KV-skew case: 2P+2D, dynamic file discovery,
    fault axes.

    DESIGN DEVIATION, recorded per the brief: the design says "2P，无
    decode"; a PD-split cluster with n_decode=0 exposes no decode
    endpoint, so requests can never complete — the minimal WORKING shape
    (2P+2D) is used instead.  The dimension under test is prefill-side
    family placement; the decode axis is not measured.  Each variant owns
    a distinct label fingerprint so the one-shot initial-engine victim
    (prefill-0 for hot / prefill-1 for cold) never poisons the other
    variant's env (same reason as _pending_drain_spec).
    """
    return EnvSpec(
        label=f"fault_kv_skew_{variant}_{ctx.profile}",
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


def _run_kv_skew_shrink(ctx: CaseContext, shrink_hot: bool):
    """Shared body of the KV-skew pair (design §2.2).

    Scenario: asymmetric prefill perf routes family seeds onto the hot
    engine; a pre-assertion proves the skew (hot cache_key count >= 3x
    cold's, construction-derived R); perf is restored to symmetric; a
    family-affinity pump keeps hitting the families through every window
    (W_base 20s / W_tr 2x staleAfter 20s / W_ss 60s from t_settle, last
    third = recovery view).  The shrink then removes hot (elastic_kv_
    skew_shrink_hot) or cold (elastic_kv_skew_shrink_cold) gracefully.

    Expected (CONTRACT — thresholds are construction/mechanism-derived
    unless marked OBS):
      1. PC transient (HARD, construction-derived): shrink hot — the
         W_tr cluster hit rate stays >= baseline - (hot family-traffic
         share x baseline hit rate + 0.10): the expected-miss injection is
         exactly the retired family share, anything beyond +0.10 is
         routing chaos / stale cache accounting (a mechanism-bug signal);
         shrink cold — hit >= baseline - 0.10 absolute.
      2. PC steady (OBS, first-run calibration): last-third hit >=
         baseline - 0.15 AND >= trough + 50% of (baseline - trough).
      3. Steady balance (HARD): PQ — survivor waiting depth peak (last
         third) <= 2; PK — survivor occupancy peak <= 0.95.  The P1 share
         and PK spread rows are DEGENERATE post-shrink (2P -> 1P leaves a
         single prefill engine — share/spread are undefined at n=1) and
         are recorded rather than asserted (design-conflict note, see the
         final report).
      4. Recovery duration (OBS, first round): seconds from the removal
         to the first 10s window back within baseline - 0.15 — the hot
         vs cold contrast value itself.
      5. Survivor keeps serving (P2): post-window recovery batch >= 95%.

    t_settle caliber note: the inflight-clean component of the design's
    t_settle triple is NOT applicable here (the never-stopping affinity
    pump keeps master inflight non-zero by design); t_settle =
    max(remove return, master alive convergence) — recorded conflict
    handling, not a silent relaxation of any gated assertion.
    """
    variant = "hot" if shrink_hot else "cold"
    env = ctx.env_manager.ensure(_skew_spec(ctx, variant))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "elastic")
    sampler: Optional[BalanceSampler] = None
    pump_stop = threading.Event()
    pump_thread: Optional[threading.Thread] = None
    obs: list = []
    hot = "prefill-0"
    cold = "prefill-1"
    victim = hot if shrink_hot else cold
    survivor = cold if shrink_hot else hot
    try:
        snap = ops.snapshot_by_name()
        if hot not in snap or cold not in snap:
            return False, (
                f"{hot}/{cold} missing from the private env (one-shot "
                f"victim: a rerun needs a fresh process); engines={sorted(snap)}"
            )
        _cleanup_dynamic(ops, env)
        if not _wait_master_topology(ops, "PREFILL", 2, MASTER_EVICT_S):
            return False, "prefill topology did not converge to 2"

        # ---- asymmetric seed: concentrate family seeds on the hot engine
        ops.set_perf(hot, prefill_fixed_ms=SKEW_PREFILL_FAST_MS)
        ops.set_perf(cold, prefill_fixed_ms=SKEW_PREFILL_SLOW_MS)
        time.sleep(1.0)  # perf sync

        addr_map = ops.addr_to_name()
        fams = [_fam_keys(base + 10_000 + i * 100_000, 0) for i in range(SKEW_FAMILIES)]
        hot_routes = 0
        total_routes = 0
        for _round in range(SKEW_FAM_ROUNDS):
            for keys in fams:
                rid = ops.next_request_id(base)
                addr, err = ops.run_one_request(
                    rid,
                    input_len=SKEW_FAM_INPUT_LEN,
                    output_len=2,
                    block_keys=keys,
                    stream_timeout_s=30.0,
                )
                if err is not None:
                    continue
                total_routes += 1
                if addr_map.get(addr) == hot:
                    hot_routes += 1
        hot_share_constructed = hot_routes / total_routes if total_routes else None

        seed_snap = ops.snapshot_by_name()
        hot_keys = int(seed_snap.get(hot, {}).get("cache_keys", 0))
        cold_keys = int(seed_snap.get(cold, {}).get("cache_keys", 0))
        if hot_keys < SKEW_MIN_RATIO * max(cold_keys, 1) or hot_keys <= 0:
            return False, (
                f"skew construction failed: hot_keys={hot_keys}, "
                f"cold_keys={cold_keys} "
                f"(need hot >= {SKEW_MIN_RATIO:.0f}x cold), "
                f"routes={hot_routes}/{total_routes}"
            )

        # restore symmetric perf (design: "再 set_perf 恢复对称")
        ops.set_perf(hot, prefill_fixed_ms=SKEW_PREFILL_SYMMETRIC_MS)
        ops.set_perf(cold, prefill_fixed_ms=SKEW_PREFILL_SYMMETRIC_MS)
        time.sleep(1.0)  # perf sync

        sampler = BalanceSampler(ops.mock_http_port, ops.master_http_port)
        sampler.start()

        def _pump_loop() -> None:
            while not pump_stop.is_set():
                for keys in fams:
                    if pump_stop.is_set():
                        break
                    rid = ops.next_request_id(base)
                    ops.run_one_request(
                        rid,
                        input_len=SKEW_FAM_INPUT_LEN,
                        output_len=2,
                        block_keys=keys,
                        stream_timeout_s=30.0,
                    )
                    pump_stop.wait(SKEW_PUMP_INTERVAL_S)

        pump_thread = threading.Thread(
            target=_pump_loop, name=f"skew-fam-pump-{variant}", daemon=True
        )
        pump_thread.start()

        # ---- W_base ----
        sampler.mark("baseline_start")
        time.sleep(BAL_BASELINE_S)
        t_base = sampler.mark("baseline_end")
        base_hit = _bal_hit_rate(sampler, 0.0, t_base)
        if base_hit is None:
            return False, "baseline window void: no cache-key traffic"
        base_tps = _bal_cluster_tps(sampler, 0.0, t_base)

        # ---- the shrink event (graceful) ----
        t_remove = sampler.mark("remove")
        status, rm_body = ops.remove_engine(engine_name=victim)
        if status != 200:
            return False, f"remove_engine({victim}) failed: {status} {rm_body}"
        rm = rm_body or {}

        # ---- W_tr ----
        time.sleep(BAL_TRANSIENT_S)
        t_tr_end = sampler.mark("transient_end")
        tr_hit = _bal_hit_rate(sampler, t_remove, t_tr_end)
        if tr_hit is None:
            return False, "transient window void: no cache-key traffic"
        drop = base_hit - tr_hit

        alive_ok = _wait_master_alive(ops, "PREFILL", 1, MASTER_EVICT_S)
        t_alive = sampler.mark("alive")
        t_settle = max(t_remove, t_alive)

        # ---- W_ss ----
        time.sleep(BAL_STEADY_S)
        t_ss_end = sampler.mark("steady_end")
        ss_tail_lo = t_settle + BAL_STEADY_S * 2.0 / 3.0
        ss_hit = _bal_hit_rate(sampler, ss_tail_lo, t_ss_end)

        # ---- 1. PC transient (HARD, construction-derived; LOWER form so
        #      the PC key keeps its lower-kind semantics) ----
        if shrink_hot:
            expected_drop = (hot_share_constructed or 0.0) * base_hit
            cap = expected_drop + SKEW_DROP_TOLERANCE
            floor = base_hit - cap
            report.check(
                "PC",
                tr_hit,
                bands=_mechanism_bands(floor),
                context=f"kv_skew_{variant}_transient_hit",
                detail=(
                    f"drop={drop:.3f} <= expected({expected_drop:.3f})"
                    f"+{SKEW_DROP_TOLERANCE}: expected = hot_share"
                    f"({hot_share_constructed:.3f}) x base_hit({base_hit:.3f})"
                ),
            )
        else:
            floor = base_hit - SKEW_COLD_DROP_ABS
            report.check(
                "PC",
                tr_hit,
                bands=_mechanism_bands(floor),
                context=f"kv_skew_{variant}_transient_hit",
                detail=(
                    f"drop={drop:.3f} <= {SKEW_COLD_DROP_ABS:.2f} absolute "
                    f"(cold holds a minor family share)"
                ),
            )

        # ---- 3. steady balance (HARD): PQ depth + PK occupancy on the
        #      survivor (P1 share / PK spread degenerate at n=1 — recorded)
        AssertUtils.balanced(
            report,
            "PQ",
            {
                survivor: sampler.window_series(
                    "mock_engine_waiting", ss_tail_lo, t_ss_end
                ).get(survivor, [])
            },
            (ss_tail_lo, t_ss_end),
            "peak",
            _mechanism_bands(BAL_QUEUE_DEPTH_MAX),
            context=f"kv_skew_{variant}_steady_depth",
        )
        AssertUtils.balanced(
            report,
            "PK",
            {survivor: _bal_occupancy_series(sampler, survivor, ss_tail_lo, t_ss_end)},
            (ss_tail_lo, t_ss_end),
            "peak",
            _mechanism_bands(BAL_OCCUPANCY_CAP),
            context=f"kv_skew_{variant}_steady_occupancy",
        )

        # ---- 2. PC steady (OBS) + 4. recovery duration (OBS) ----
        rebound_cap = (
            tr_hit + SKEW_PC_REBOUND * (base_hit - tr_hit)
            if base_hit > tr_hit
            else None
        )
        obs.append(
            _obs_note(
                "PC",
                "steady_hit>=base-0.15 and >=trough+50%rebound "
                "(design §2.2 #2, first-run calibration)",
                ss_hit,
            )
        )
        obs.append(
            _obs_note(
                "PC",
                "rebound_cap(trough+50%(base-trough))",
                rebound_cap,
            )
        )
        rec_s = None
        win = 10.0
        t = t_remove
        while t + win <= t_ss_end:
            r = _bal_hit_rate(sampler, t, t + win)
            if r is not None and r >= base_hit - SKEW_PC_STEADY_BASE_TOL:
                rec_s = (t + win / 2.0) - t_remove
                break
            t += win
        obs.append(
            _obs_note(
                "PC",
                "recovery_duration_s(first 10s window back within base-0.15; "
                "hot-vs-cold contrast value, first round)",
                rec_s,
            )
        )
        obs.append(
            _obs_note(
                "PT",
                "steady_cluster_tps(design §1.2 PT row; first-run calibration)",
                _bal_cluster_tps(sampler, ss_tail_lo, t_ss_end),
            )
        )

        # ---- 5. survivor keeps serving ----
        ok_n, _err_n, _ = _run_batch(ops, base, 20)
        recovery_rate = ok_n / 20.0
        report.invariant(
            "P2",
            recovery_rate >= 0.95,
            context=f"kv_skew_{variant}_survivor_service",
            detail=f"recovery {ok_n}/20",
        )

        events_summary = json.dumps(dict(sampler._events)) if sampler else "{}"
        ok_verdict, detail, _rep = report.finish(
            f"variant={variant}, victim={victim}(survivor={survivor}), "
            f"skew=(hot_keys={hot_keys}, cold_keys={cold_keys}, "
            f"ratio={hot_keys / max(cold_keys, 1):.1f}x), "
            f"hot_share_constructed={hot_share_constructed}, "
            f"hit=(base={base_hit:.3f}, tr={tr_hit:.3f}, "
            f"drop={drop:.3f}, steady={ss_hit}), "
            f"rm=(drained={rm.get('drained')}, "
            f"drain_ms={rm.get('drain_ms')}), "
            f"alive_ok={alive_ok}, recovery={ok_n}/20, "
            f"base_tps={base_tps}, "
            f"windows=(W_base[0,{t_base:.0f}], "
            f"W_tr[{t_remove:.0f},{t_tr_end:.0f}], "
            f"W_ss[{t_settle:.0f},{t_ss_end:.0f}]), "
            f"events={events_summary}, "
            f"share/spread degenerate at n=1 (recorded), "
            f"{'; '.join(obs)}, "
            f"grades: {report.summary()}"
        )
        return ok_verdict, detail, _rep
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        pump_stop.set()
        if pump_thread is not None:
            pump_thread.join(10.0)
        if sampler is not None:
            try:
                sampler.stop()
                sampler.dump(
                    ctx.case_dir(f"elastic_kv_skew_shrink_{variant}")
                    / "balance_sampler.json.gz"
                )
            except Exception:
                pass
        for name in (hot, cold):
            try:
                ops.set_perf(name, prefill_fixed_ms=100.0)
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


@case(
    "elastic_kv_skew_shrink_hot",
    profiles=["batch-window"],  # elastic family: BATCH dispatcher + fault axes
    source=(
        "balance-metrics v2 design §2.2 "
        "(flexlb-balance-metrics-v2-design.md): KV-skew shrink, hot variant"
    ),
)
def elastic_kv_skew_shrink_hot(ctx: CaseContext):
    """KV-skew scale-in, hot variant — see _run_kv_skew_shrink (the
    paired contrast removes the family-HEAVY engine; the cold variant
    removes the family-light one)."""
    return _run_kv_skew_shrink(ctx, shrink_hot=True)


@case(
    "elastic_kv_skew_shrink_cold",
    profiles=["batch-window"],  # elastic family: BATCH dispatcher + fault axes
    source=(
        "balance-metrics v2 design §2.2 "
        "(flexlb-balance-metrics-v2-design.md): KV-skew shrink, cold variant"
    ),
)
def elastic_kv_skew_shrink_cold(ctx: CaseContext):
    """KV-skew scale-in, cold variant — see _run_kv_skew_shrink (the
    paired contrast removes the family-LIGHT engine; the hot variant
    removes the family-heavy one)."""
    return _run_kv_skew_shrink(ctx, shrink_hot=False)


# ===========================================================================
# elastic_transient_imbalance_bound — abrupt scale-in under sustained
# load, every transient dimension bounded by an EXPLICIT capacity
# (design §2.3)
# ===========================================================================

# Capacity axes — ALL explicit so the bound is KNOWN by construction
# (design §2.3: the thresholds are config-derived, never guessed):
#   * prefill-side engine queue cap: performance JSON
#     "prefill.max_waiting_batches" (MockPerformanceModel reads it at
#     start; /set_perf max_waiting_batches overrides the same value at
#     runtime) — the JSON channel is used so the INITIAL engines are
#     born capped.
#   * master batcher waiting capacity: FLEXLB_CONFIG scheduler.capacity
#     maxWaitingRequestsPerPrefillWorker (flexlb_cfg.ConfigOverride's
#     parameterised channel; Java default 1024 when omitted — an explicit
#     value here pins the bound).
#   * decode-side engine concurrency: NOT overridden — Java default
#     JavaMockEngineCluster.DEFAULT_DECODE_MAX_CONCURRENCY = 128
#     (BAL_JAVA_DECODE_MAX_CONCURRENCY above).
# Surpassing the engine-side cap rejects at the engine
# ("prefill waiting queue full (backpressure)", JavaMockEngineCluster
# ~L1290) — the load shape below is sized to stay under it, so any
# failure stays attributable to the victim, not to a capacity reject.
TRANSIENT_MAX_WAITING_BATCHES = 16
TRANSIENT_MAX_WAITING_REQUESTS_PER_WORKER = 64
# Load shape: a dense background pump plus one crossing burst fired
# just before the abrupt removal (the worst transient: in-flight work
# on the victim + a queue shock on the survivors).
TRANSIENT_PUMP_INTERVAL_S = 0.1
TRANSIENT_BURST_N = 30
TRANSIENT_BURST_CONCURRENCY = 15
# Burst streams must outlive the whole W_tr (fail-close settlement
# rides out the 10s stale window).
TRANSIENT_BURST_STREAM_TIMEOUT_S = 45.0


def _transient_spec(ctx: CaseContext) -> EnvSpec:
    """Private env for elastic_transient_imbalance_bound: 3P+2D, all
    capacity axes explicit, dynamic file discovery, fault/admission axes.

    Topology note (construction choice, not a threshold change): the
    design does not fix the role count; 2P would leave ONE prefill
    survivor (request-share degenerates at n=1, the skew-pair lesson),
    so one extra prefill keeps the post-event share row measurable.
    The n_prefill=3 fingerprint is unique across the elastic specs, so
    the one-shot initial-engine victim (prefill-1) never poisons a
    shared env (same reason as _pending_drain_spec).
    """
    perf = fault_env_perf()
    perf["prefill"]["max_waiting_batches"] = TRANSIENT_MAX_WAITING_BATCHES
    return EnvSpec(
        label=f"fault_transient_bound_{ctx.profile}",
        n_prefill=3,
        n_decode=2,
        perf=perf,
        master_profile=ctx.profile,
        discovery="discovery_file",
        config_overrides=ConfigOverride(
            ordering="priority",
            decision="fixed_window",
            dispatcher="batch",
            queue_timeout_ms=60_000,
            max_waiting_requests_per_prefill_worker=(
                TRANSIENT_MAX_WAITING_REQUESTS_PER_WORKER
            ),
        ),
    )


@case(
    "elastic_transient_imbalance_bound",
    profiles=["batch-window"],  # elastic family: BATCH dispatcher + fault axes
    source=(
        "balance-metrics v2 design §2.3 "
        "(flexlb-balance-metrics-v2-design.md): abrupt scale-in under "
        "sustained load, transient bounds from explicit capacities"
    ),
)
def elastic_transient_imbalance_bound(ctx: CaseContext):
    """Abrupt scale-in under sustained load (design §2.3): the worst
    transient — a hard teardown while the master still dispatches to
    the dead address inside the 10s stale window (W_tr = 2 x stale =
    20s, the BAL_TRANSIENT_S derivation), riding on a dense background
    pump plus a one-shot crossing burst.  Every gated threshold is an
    EXPLICIT capacity (a config value / a Java mechanism constant / the
    case's own construction); the TPS-trough row is an OBSERVATION
    (design flags it first-run calibration).

    Capacity map (all bounds known by construction):
      * prefill engine queue: perf JSON prefill.max_waiting_batches = 16.
        UNIT NOTE (recorded per the brief): the sampler's
        mock_engine_waiting series counts REQUESTS for prefill engines
        (waitingPrefillRequests) while the cap counts BATCHES; the
        assertion follows the task brief literally (waiting peak <=
        cap) — the request-level equivalent cap is batchCap x
        directBatchSizeMax (JavaMockEngineCluster.directWaitingRequestCap),
        noted here for review rather than silently widened.
      * master batcher: FLEXLB_CONFIG scheduler.capacity
        maxWaitingRequestsPerPrefillWorker = 64.  FIELD-MAP NOTE
        (recorded per the brief): the master /rtp_llm/inflight_status
        prefill plane carries NO waiting/queued field (ip_port /
        inflight_batches / inflight_requests / inflight_route_requests
        only); the closest available proxy is inflight_requests (the
        master's per-endpoint in-flight REQUEST count — same unit as
        the capacity).  Asserted as the master-side decision view; the
        mapping is recorded in the case detail, no data source is
        invented.  in-flight subsumes waiting (it includes
        dispatched-but-running work), so the bound leans STRICT.
      * decode engine concurrency: Java default 128
        (JavaMockEngineCluster.DEFAULT_DECODE_MAX_CONCURRENCY, not
        overridden) — decode-side waiting (decodePendingQueueSize)
        peak stays <= 128.

    Expected (CONTRACT — thresholds are capacity/mechanism-derived
    unless marked OBS):
      1. PQ transient (HARD): surviving prefill engine waiting peak
         over W_tr <= 16 (explicit perf value); decode waiting peak
         <= 128 (Java default); master-side per-survivor
         inflight_requests peak <= 64 (explicit FLEXLB_CONFIG value,
         field map above).
      2. PK transient (HARD): survivor occupancy peak over W_tr <=
         0.95 (= 1 - reserveRatio 0.05, MockLruBlockCache.
         DEFAULT_RESERVE_RATIO); survivor Δ(lack_mem+admission_fails)
         <= K_reject = ceil(max(0, victim occupied blocks - summed
         survivor free blocks)).  Caliber note: ALL four survivors'
         free blocks enter the sum and ALL four survivors' reject
         deltas enter the measured side — the victim's re-routed
         in-flight demand lands on prefill cache blocks AND decode
         leases, so the all-survivor reading is the formula's natural
         unit (recorded for review).
      3. PT transient (OBS, first-run calibration): W_tr cluster TPS
         mean >= baseline x (n_after/n_before) x 0.85 = baseline x
         4/5 x 0.85 (0.85 is a design value; a breach is arbitration
         evidence, never silenced).
      4. Fail-close locality (HARD): every failed request in the whole
         run (dense pump + crossing burst) is either victim-routed
         (its prefill address == the removed engine's address) or was
         never routed (master admission refusal, addr is None —
         counted as an observation, never hidden); requests routed to
         SURVIVORS end with ZERO failures (errors do not spill over).
      5. Steady-state recovery (HARD, last third of W_ss from
         t_settle): P1 — prefill-plane survivor share max <=
         max(baseline+0.10, 1/n+0.15) and min >= 0.10 (the event
         plane; the decode plane's share is recorded); PQ — all
         survivor waiting peaks <= 2; PK — decode-plane AND
         prefill-plane occupancy spread <= baseline + 0.05.
      6. Survivor keeps serving (P2): post-window recovery batch
         >= 95%.

    t_settle caliber note: the inflight-clean component of the design's
    t_settle triple is NOT applicable here (the never-stopping pump
    keeps master inflight non-zero by design); t_settle = max(remove
    return, master alive convergence) — same recorded conflict
    handling as the skew pair, not a silent relaxation of any gated
    assertion.
    """
    env = ctx.env_manager.ensure(_transient_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "elastic")
    sampler: Optional[BalanceSampler] = None
    pump_stop = threading.Event()
    pump_thread: Optional[threading.Thread] = None
    obs: list = []
    victim = "prefill-1"
    pre_survivors = ("prefill-0", "prefill-2")
    dec_survivors = ("decode-0", "decode-1")
    all_survivors = pre_survivors + dec_survivors
    try:
        snap = ops.snapshot_by_name()
        for name in (victim,) + all_survivors:
            if name not in snap:
                return False, (
                    f"{name} missing from the private env (one-shot "
                    f"victim: a rerun needs a fresh process); "
                    f"engines={sorted(snap)}"
                )
        _cleanup_dynamic(ops, env)
        if not _wait_master_topology(ops, "PREFILL", 3, MASTER_EVICT_S):
            return False, "prefill topology did not converge to 3"

        addr_map = ops.addr_to_name()
        victim_addr = next((a for a, n in addr_map.items() if n == victim), None)
        if not victim_addr:
            return False, "victim address lookup failed"

        sampler = BalanceSampler(ops.mock_http_port, ops.master_http_port)
        sampler.start()

        # dense background pump — records (addr, err) per request for
        # the fail-close locality verdict
        pump_records: list = []
        pump_lock = threading.Lock()

        def _pump_loop() -> None:
            while not pump_stop.is_set():
                rid = ops.next_request_id(base)
                try:
                    addr, err = ops.run_one_request(
                        rid,
                        output_len=2,
                        block_keys=[rid * 100 + 1],
                        stream_timeout_s=30.0,
                    )
                except Exception as exc:  # defensive: never kill the pump
                    addr, err = None, repr(exc)
                with pump_lock:
                    pump_records.append((addr, err))
                pump_stop.wait(TRANSIENT_PUMP_INTERVAL_S)

        pump_thread = threading.Thread(
            target=_pump_loop, name="transient-pump", daemon=True
        )
        pump_thread.start()

        # ---- W_base: pre-event baseline (pump running) ----
        sampler.mark("baseline_start")
        time.sleep(BAL_BASELINE_S)
        t_base = sampler.mark("baseline_end")

        base_p_share_deltas = sampler.window_series(
            "mock_engine_accepted_total", 0.0, t_base, mode="delta"
        )
        pre_names = (victim,) + pre_survivors
        p_vals = [base_p_share_deltas.get(e, 0.0) for e in pre_names]
        p_tot = sum(p_vals)
        base_p_share_max = max(p_vals) / p_tot if p_tot > 0 else None
        base_d_occ = {}
        for eng in dec_survivors:
            base_d_occ[eng] = _bal_engine_mean(
                _bal_occupancy_series(sampler, eng, 0.0, t_base)
            )
        base_d_vals = [v for v in base_d_occ.values() if v is not None]
        base_d_spread = (
            max(base_d_vals) - min(base_d_vals)
            if len(base_d_vals) == len(dec_survivors)
            else 0.0
        )
        base_p_occ = {}
        for eng in pre_names:
            base_p_occ[eng] = _bal_engine_mean(
                _bal_occupancy_series(sampler, eng, 0.0, t_base)
            )
        base_p_vals = [v for v in base_p_occ.values() if v is not None]
        base_p_spread = (
            max(base_p_vals) - min(base_p_vals)
            if len(base_p_vals) == len(pre_names)
            else 0.0
        )
        base_tps = _bal_cluster_tps(sampler, 0.0, t_base)

        # K_reject inputs (pre-event snapshot: victim occupied vs the
        # summed free blocks of ALL survivors — see the docstring note).
        pre = ops.snapshot_by_name()
        victim_demand = pre[victim]["cache_blocks"] - pre[victim]["available_blocks"]
        survivor_free = sum(int(pre[n]["available_blocks"]) for n in all_survivors)
        k_reject = _bal_k_reject(victim_demand, survivor_free)

        # ---- crossing burst, then the abrupt removal (worst transient) ----
        def _burst_one(rid: int) -> tuple:
            try:
                return ops.run_one_request(
                    rid,
                    output_len=2,
                    block_keys=[rid * 100 + 1],
                    stream_timeout_s=TRANSIENT_BURST_STREAM_TIMEOUT_S,
                )
            except Exception as exc:
                return None, repr(exc)

        burst_pool = ThreadPoolExecutor(max_workers=TRANSIENT_BURST_CONCURRENCY)
        burst_futures = [
            burst_pool.submit(_burst_one, ops.next_request_id(base))
            for _ in range(TRANSIENT_BURST_N)
        ]

        t_remove = sampler.mark("remove")
        status, rm_body = ops.remove_engine(engine_name=victim, mode="abrupt")
        if status != 200:
            for f in burst_futures:
                f.cancel()
            burst_pool.shutdown(wait=False)
            return False, (
                f"remove_engine({victim}, abrupt) failed: {status} {rm_body}"
            )

        burst_records = [f.result() for f in burst_futures]
        burst_pool.shutdown()
        sampler.mark("burst_settled")

        # ---- W_tr: the stale window rides out (pump keeps running) ----
        time.sleep(BAL_TRANSIENT_S)
        t_tr_end = sampler.mark("transient_end")

        alive_ok = _wait_master_topology(ops, "PREFILL", 2, MASTER_EVICT_S)
        t_alive = sampler.mark("alive_converged")
        t_settle = max(t_remove, t_alive)

        # ---- W_ss: steady observation (pump keeps running) ----
        time.sleep(BAL_STEADY_S)
        t_ss_end = sampler.mark("steady_end")
        ss_tail_lo = t_settle + BAL_STEADY_S * 2.0 / 3.0

        # ---- 1. PQ transient: engine side (explicit capacities) ----
        AssertUtils.balanced(
            report,
            "PQ",
            {
                e: sampler.window_series("mock_engine_waiting", t_remove, t_tr_end).get(
                    e, []
                )
                for e in pre_survivors
            },
            (t_remove, t_tr_end),
            "peak",
            _mechanism_bands(TRANSIENT_MAX_WAITING_BATCHES),
            context="transient_prefill_waiting_peak",
        )
        AssertUtils.balanced(
            report,
            "PQ",
            {
                e: sampler.window_series("mock_engine_waiting", t_remove, t_tr_end).get(
                    e, []
                )
                for e in dec_survivors
            },
            (t_remove, t_tr_end),
            "peak",
            _mechanism_bands(BAL_JAVA_DECODE_MAX_CONCURRENCY),
            context="transient_decode_waiting_peak",
        )

        # ---- 1b. PQ transient: master-side decision view ----
        # inflight_status prefill plane has NO waiting/queued field; the
        # closest same-unit proxy is inflight_requests (field map in the
        # docstring).  The victim's own dead-address backlog (until the
        # stale cleanup zeroes it) is EXCLUDED from the survivor bound
        # and recorded as an observation instead.
        infl = sampler.window_series("inflight_requests", t_remove, t_tr_end)
        victim_master_key = f"master:prefill:{victim_addr}"
        master_survivor_series = {
            k: pts
            for k, pts in infl.items()
            if k.startswith("master:prefill:") and k != victim_master_key
        }
        AssertUtils.balanced(
            report,
            "PQ",
            master_survivor_series,
            (t_remove, t_tr_end),
            "peak",
            _mechanism_bands(TRANSIENT_MAX_WAITING_REQUESTS_PER_WORKER),
            context="transient_master_inflight_peak",
        )
        victim_master_pts = infl.get(victim_master_key, [])
        victim_master_peak = (
            max(v for _, v in victim_master_pts) if victim_master_pts else None
        )

        # ---- 2. PK transient: survivor occupancy + K_reject ----
        AssertUtils.balanced(
            report,
            "PK",
            {
                e: _bal_occupancy_series(sampler, e, t_remove, t_tr_end)
                for e in all_survivors
            },
            (t_remove, t_tr_end),
            "peak",
            _mechanism_bands(BAL_OCCUPANCY_CAP),
            context="transient_survivor_occupancy_peak",
        )
        rej_delta = 0.0
        for e in all_survivors:
            rej_delta += sampler.window_series(
                "mock_engine_kv_admission_fails_total",
                t_remove,
                t_tr_end,
                mode="delta",
            ).get(e, 0.0)
            rej_delta += sampler.window_series(
                "mock_engine_lack_mem_rejects_total",
                t_remove,
                t_tr_end,
                mode="delta",
            ).get(e, 0.0)
        report.check(
            "PK",
            rej_delta,
            bands=_mechanism_bands(k_reject),
            context="transient_survivor_rejects",
            detail=(
                f"K_reject=ceil(max(0, victim_occupied({victim_demand}) "
                f"- survivor_free_sum({survivor_free})))={k_reject} "
                f"(all-survivor caliber, see docstring)"
            ),
        )

        # ---- 3. PT transient (OBS) ----
        tr_tps = _bal_cluster_tps(sampler, t_remove, t_tr_end)
        pt_floor = (
            base_tps * (4.0 / 5.0) * 0.85
            if base_tps is not None and tr_tps is not None
            else None
        )
        obs.append(
            _obs_note(
                "PT",
                "tr_cluster_tps(design §2.3 #3; first-run calibration)",
                tr_tps,
            )
        )
        obs.append(_obs_note("PT", "tr_tps_floor(base x 4/5 x 0.85)", pt_floor))

        # ---- 4. fail-close locality (whole run: pump + burst) ----
        with pump_lock:
            all_records = list(pump_records)
        all_records += burst_records
        victim_fail = [
            r for r in all_records if r[1] is not None and r[0] == victim_addr
        ]
        survivor_fail = [
            r
            for r in all_records
            if r[1] is not None and r[0] is not None and r[0] != victim_addr
        ]
        unrouted_fail = [r for r in all_records if r[1] is not None and r[0] is None]
        victim_fail_kinds = sorted({str(r[1])[:60] for r in victim_fail})[:3]
        report.invariant(
            "P6",
            not survivor_fail,
            context="transient_fail_close_locality",
            detail=(
                f"victim_routed_failures={len(victim_fail)} (allowed), "
                f"survivor_routed_failures={len(survivor_fail)} "
                f"(must be 0), "
                f"unrouted_failures={len(unrouted_fail)} (master "
                f"admission, observed), victim_addr={victim_addr}, "
                f"victim_fail_kinds={victim_fail_kinds}"
            ),
        )
        obs.append(
            _obs_note(
                "P6",
                "unrouted_failures(master admission refusals, whole run)",
                len(unrouted_fail),
            )
        )
        obs.append(
            _obs_note(
                "P6",
                "victim_master_inflight_peak(dead-address backlog until "
                "stale cleanup)",
                victim_master_peak,
            )
        )

        # ---- 5. steady-state recovery (last third of W_ss) ----
        # P1 — prefill-plane survivor share (the event plane)
        ss_share = sampler.window_series(
            "mock_engine_accepted_total", t_settle, t_ss_end, mode="delta"
        )
        p_ss_vals = [ss_share.get(e, 0.0) for e in pre_survivors]
        p_ss_tot = sum(p_ss_vals)
        if p_ss_tot <= 0:
            return False, "steady window void: no prefill traffic"
        p_ss_max = max(p_ss_vals) / p_ss_tot
        p_ss_min = min(p_ss_vals) / p_ss_tot
        p_share_cap = max(
            (base_p_share_max or 0.0) + BAL_SHARE_TOL,
            1.0 / len(pre_survivors) + 0.15,
        )
        report.check(
            "P1",
            p_ss_max,
            bands=_mechanism_bands(p_share_cap),
            context="transient_steady_prefill_share_max",
            detail=(
                f"deltas={dict(zip(pre_survivors, p_ss_vals))}, "
                f"cap=max(base+0.10, 1/n+0.15)={p_share_cap:.3f}"
            ),
        )
        report.invariant(
            "P2",
            p_ss_min >= BAL_SHARE_MIN_FLOOR,
            context="transient_steady_prefill_share_min",
            detail=f"min_share={p_ss_min:.3f}, floor={BAL_SHARE_MIN_FLOOR}",
        )
        # decode-plane share: recorded, not gated (event plane is prefill)
        d_ss_vals = [ss_share.get(e, 0.0) for e in dec_survivors]
        d_ss_tot = sum(d_ss_vals)
        d_ss_max = max(d_ss_vals) / d_ss_tot if d_ss_tot > 0 else None
        obs.append(_obs_note("P1", "steady_decode_share_max(recorded)", d_ss_max))

        # PQ — waiting depth peak (all survivors, last third)
        AssertUtils.balanced(
            report,
            "PQ",
            {
                e: sampler.window_series(
                    "mock_engine_waiting", ss_tail_lo, t_ss_end
                ).get(e, [])
                for e in all_survivors
            },
            (ss_tail_lo, t_ss_end),
            "peak",
            _mechanism_bands(BAL_QUEUE_DEPTH_MAX),
            context="transient_steady_waiting_depth",
        )

        # PK — occupancy spread (decode plane AND prefill plane)
        AssertUtils.balanced(
            report,
            "PK",
            {
                e: _bal_occupancy_series(sampler, e, ss_tail_lo, t_ss_end)
                for e in dec_survivors
            },
            (ss_tail_lo, t_ss_end),
            "spread",
            _mechanism_bands(base_d_spread + BAL_KV_SPREAD_TOL),
            context="transient_steady_decode_occ_spread",
        )
        AssertUtils.balanced(
            report,
            "PK",
            {
                e: _bal_occupancy_series(sampler, e, ss_tail_lo, t_ss_end)
                for e in pre_survivors
            },
            (ss_tail_lo, t_ss_end),
            "spread",
            _mechanism_bands(base_p_spread + BAL_KV_SPREAD_TOL),
            context="transient_steady_prefill_occ_spread",
        )

        # ---- 6. survivor keeps serving ----
        ok_n, _err_n, _ = _run_batch(ops, base, 20)
        recovery_rate = ok_n / 20.0
        report.invariant(
            "P2",
            recovery_rate >= 0.95,
            context="transient_survivor_service",
            detail=f"recovery {ok_n}/20",
        )

        events_summary = json.dumps(dict(sampler._events)) if sampler else "{}"
        ok_verdict, detail, _rep = report.finish(
            f"victim={victim}(abrupt), survivors={all_survivors}, "
            f"caps=(batches={TRANSIENT_MAX_WAITING_BATCHES}, "
            f"master_waiting={TRANSIENT_MAX_WAITING_REQUESTS_PER_WORKER}, "
            f"decode_conc={BAL_JAVA_DECODE_MAX_CONCURRENCY}), "
            f"fail_close=(victim_routed={len(victim_fail)}, "
            f"survivor_routed={len(survivor_fail)}, "
            f"unrouted={len(unrouted_fail)}), "
            f"k_reject={k_reject}(rej_delta={rej_delta:.0f}), "
            f"base=(p_share_max={base_p_share_max}, "
            f"d_occ_spread={base_d_spread:.3f}, "
            f"p_occ_spread={base_p_spread:.3f}, tps={base_tps}), "
            f"alive_ok={alive_ok}, recovery={ok_n}/20, "
            f"windows=(W_base[0,{t_base:.0f}], "
            f"W_tr[{t_remove:.0f},{t_tr_end:.0f}], "
            f"W_ss[{t_settle:.0f},{t_ss_end:.0f}]), "
            f"events={events_summary}, "
            f"{'; '.join(obs)}, "
            f"grades: {report.summary()}"
        )
        return ok_verdict, detail, _rep
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        pump_stop.set()
        if pump_thread is not None:
            pump_thread.join(10.0)
        if sampler is not None:
            try:
                sampler.stop()
                sampler.dump(
                    ctx.case_dir("elastic_transient_imbalance_bound")
                    / "balance_sampler.json.gz"
                )
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
# elastic_steady_state_recovery — graceful decode scale-in under a
# never-stopping background flow, a pure 60s steady observation window
# (design §2.4)
# ===========================================================================

# Sustained background-flow cadence — dense enough to keep every
# per-engine counter moving through all windows, never stopped (the
# design's explicit construction: "持续背景流不断").
STEADY_RECOVERY_PUMP_INTERVAL_S = 0.2


def _steady_recovery_spec(ctx: CaseContext) -> EnvSpec:
    """Private env for elastic_steady_state_recovery: 2P+4D (the heavier
    role is decode — four pools carry the KV load — so the victim is a
    decode engine, leaving THREE decode survivors so the share /
    spread rows stay non-degenerate), dynamic file discovery, fault
    axes.

    The label fingerprint differs from every other elastic spec, so the
    one-shot initial-engine victim (decode-0, permanently removed) never
    poisons a shared env (same reason as _pending_drain_spec).
    """
    return EnvSpec(
        label=f"fault_steady_recovery_{ctx.profile}",
        n_prefill=2,
        n_decode=4,
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


def _bal_exec_cv(
    sampler: BalanceSampler, metric: str, t_lo: float, t_hi: float, engines
) -> Optional[float]:
    """Cross-engine CV of per-engine window means (sample std / mean).

    Same caliber as AssertUtils.balanced's cv stat (sample standard
    deviation, ddof=1); None when fewer than 2 engines carry samples or
    the mean is <= 0 — the caller fails loud / records n/a, never a
    fabricated zero.
    """
    means = []
    for e in engines:
        pts = sampler.window_series(metric, t_lo, t_hi).get(e, [])
        if pts:
            means.append(sum(v for _, v in pts) / len(pts))
    if len(means) < 2:
        return None
    m = sum(means) / len(means)
    if m <= 0:
        return None
    var = sum((v - m) ** 2 for v in means) / (len(means) - 1)
    return (var**0.5) / m


@case(
    "elastic_steady_state_recovery",
    profiles=["batch-window"],  # elastic family: BATCH dispatcher + fault axes
    source=(
        "balance-metrics v2 design §2.4 "
        "(flexlb-balance-metrics-v2-design.md): graceful scale-in, "
        "steady-state recovery over a pure 60s observation window"
    ),
)
def elastic_steady_state_recovery(ctx: CaseContext):
    """Graceful decode scale-in under a never-stopping background flow
    (design §2.4): victim decode-0 drains gracefully (default 60s cap,
    BAL_GRACEFUL_DRAIN_DEFAULT_MS), the pump never stops, and W_ss is a
    pure 60s observation window from t_settle in 3s x 20 sub-windows
    (BAL_STEADY_SUBWINDOW_S); recovery predicates read the LAST THIRD
    (design §1.4).

    Expected (CONTRACT — thresholds are mechanism/tolerance-derived
    unless marked OBS):
      1. Steady recovery, last third of W_ss (HARD):
         P1 — survivor share max <= max(baseline + 0.10, 1/n + 0.15)
         and min >= 0.10 (P2 floor);
         PQ — all survivor waiting peaks <= 2;
         PK — decode-plane occupancy spread <= baseline + 0.05 AND
         per-survivor occupancy peak <= 0.95.
      2. Oscillation fingerprint (HARD): two CONSECUTIVE sub-windows
         departing the share target in the SAME direction beyond ±0.10
         -> fail (a persistent one-sided drift, not noise).  REFERENCE
         NOTE: the per-engine pre-event baseline share (1/4) is no longer
         the recovery target after the shrink — the natural rebalance
         is 1/n_after, the same n-rescaling the cap formula's 1/n + 0.15
         term encodes; the fingerprint measures oscillation AROUND the
         settled point, and the reference choice is recorded here for
         review.
      3. OBS (first-run calibration, never gating): swing — max
         |sub-window share - 1/n| vs the band baseline_swing + 0.10
         (design: the add_preference swing observation promoted to a
         band pending calibration); PL — steady cross-engine exec-ms CV
         vs max(baseline_cv + 0.10, 0.25); PC — steady cluster hit rate
         vs baseline - 0.15; PT — steady per-engine TPS max/min vs 1.5.
      4. Survivor keeps serving (P2): post-window recovery batch
         >= 95%; the pump's own whole-run success rate is recorded.

    t_settle caliber note: the inflight-clean component of the design's
    t_settle triple is NOT applicable (the never-stopping pump keeps
    master inflight non-zero by design); t_settle = max(remove return,
    master alive convergence) — same recorded conflict handling as the
    skew pair and the transient case, not a silent relaxation.
    """
    env = ctx.env_manager.ensure(_steady_recovery_spec(ctx))
    ops = ctx.engine_ops(env)
    report = GradeReport(run_grade=ctx.grade)
    base = rid_base(ctx, "elastic")
    sampler: Optional[BalanceSampler] = None
    pump_stop = threading.Event()
    pump_thread: Optional[threading.Thread] = None
    obs: list = []
    victim = "decode-0"
    dec_all = ("decode-0", "decode-1", "decode-2", "decode-3")
    dec_survivors = ("decode-1", "decode-2", "decode-3")
    try:
        snap = ops.snapshot_by_name()
        for name in dec_all:
            if name not in snap:
                return False, (
                    f"{name} missing from the private env (one-shot "
                    f"victim: a rerun needs a fresh process); "
                    f"engines={sorted(snap)}"
                )
        _cleanup_dynamic(ops, env)
        if not _wait_master_topology(ops, "DECODE", 4, MASTER_EVICT_S):
            return False, "decode topology did not converge to 4"

        sampler = BalanceSampler(ops.mock_http_port, ops.master_http_port)
        sampler.start()

        # never-stopping background flow (records its own success rate)
        pump_total = [0]
        pump_ok = [0]
        pump_lock = threading.Lock()

        def _pump_loop() -> None:
            while not pump_stop.is_set():
                rid = ops.next_request_id(base)
                try:
                    _addr, err = ops.run_one_request(
                        rid,
                        output_len=2,
                        block_keys=[rid * 100 + 1],
                        stream_timeout_s=30.0,
                    )
                    ok = err is None
                except Exception:
                    ok = False
                with pump_lock:
                    pump_total[0] += 1
                    if ok:
                        pump_ok[0] += 1
                pump_stop.wait(STEADY_RECOVERY_PUMP_INTERVAL_S)

        pump_thread = threading.Thread(
            target=_pump_loop, name="steady-recovery-pump", daemon=True
        )
        pump_thread.start()

        # ---- W_base: pre-event baseline (pump running) ----
        sampler.mark("baseline_start")
        time.sleep(BAL_BASELINE_S)
        t_base = sampler.mark("baseline_end")

        base_share_deltas = sampler.window_series(
            "mock_engine_accepted_total", 0.0, t_base, mode="delta"
        )
        d_base_vals = [base_share_deltas.get(e, 0.0) for e in dec_all]
        d_base_tot = sum(d_base_vals)
        if d_base_tot <= 0:
            return False, "baseline window void: no decode traffic"
        base_share_max = max(d_base_vals) / d_base_tot
        base_occ = {}
        for eng in dec_all:
            base_occ[eng] = _bal_engine_mean(
                _bal_occupancy_series(sampler, eng, 0.0, t_base)
            )
        base_occ_vals = [v for v in base_occ.values() if v is not None]
        base_occ_spread = (
            max(base_occ_vals) - min(base_occ_vals)
            if len(base_occ_vals) == len(dec_all)
            else 0.0
        )
        base_hit = _bal_hit_rate(sampler, 0.0, t_base)
        base_cv = _bal_exec_cv(
            sampler, "mock_engine_decode_ms_avg", 0.0, t_base, dec_all
        )

        # baseline swing: same 3s sub-window caliber as the steady side,
        # against the pre-event 1/4 target (n = 4 here)
        base_target = 1.0 / len(dec_all)
        base_devs = []
        n_base_sub = int(BAL_BASELINE_S / BAL_STEADY_SUBWINDOW_S)
        for i in range(n_base_sub):
            lo = i * BAL_STEADY_SUBWINDOW_S
            hi = lo + BAL_STEADY_SUBWINDOW_S
            d = sampler.window_series(
                "mock_engine_accepted_total", lo, hi, mode="delta"
            )
            vals = [d.get(e, 0.0) for e in dec_all]
            tot = sum(vals)
            if tot <= 0:
                continue
            base_devs.extend(abs(v / tot - base_target) for v in vals)
        base_swing = max(base_devs) if base_devs else None

        # ---- the shrink event (graceful, default 60s cap) ----
        t_remove = sampler.mark("remove")
        status, rm_body = ops.remove_engine(engine_name=victim)
        if status != 200:
            return False, f"remove_engine({victim}) failed: {status} {rm_body}"
        rm = rm_body or {}

        # ---- W_tr (pump keeps running) ----
        time.sleep(BAL_TRANSIENT_S)
        t_tr_end = sampler.mark("transient_end")

        alive_ok = _wait_master_topology(ops, "DECODE", 3, MASTER_EVICT_S)
        t_alive = sampler.mark("alive_converged")
        t_settle = max(t_remove, t_alive)

        # ---- W_ss: pure observation, 3s x 20 sub-windows ----
        time.sleep(BAL_STEADY_S)
        t_ss_end = sampler.mark("steady_end")
        ss_tail_lo = t_settle + BAL_STEADY_S * 2.0 / 3.0

        # ---- 1. steady recovery, last third (HARD) ----
        # P1 — survivor share (counter-delta caliber over the last third)
        ss_share = sampler.window_series(
            "mock_engine_accepted_total", ss_tail_lo, t_ss_end, mode="delta"
        )
        d_ss_vals = [ss_share.get(e, 0.0) for e in dec_survivors]
        d_ss_tot = sum(d_ss_vals)
        if d_ss_tot <= 0:
            return False, "steady window void: no decode traffic"
        d_ss_max = max(d_ss_vals) / d_ss_tot
        d_ss_min = min(d_ss_vals) / d_ss_tot
        d_share_cap = max(
            base_share_max + BAL_SHARE_TOL,
            1.0 / len(dec_survivors) + 0.15,
        )
        report.check(
            "P1",
            d_ss_max,
            bands=_mechanism_bands(d_share_cap),
            context="steady_share_max",
            detail=(
                f"deltas={dict(zip(dec_survivors, d_ss_vals))}, "
                f"cap=max(base+0.10, 1/n+0.15)={d_share_cap:.3f}"
            ),
        )
        report.invariant(
            "P2",
            d_ss_min >= BAL_SHARE_MIN_FLOOR,
            context="steady_share_min",
            detail=f"min_share={d_ss_min:.3f}, floor={BAL_SHARE_MIN_FLOOR}",
        )

        # PQ — waiting depth peak (all survivors, last third)
        AssertUtils.balanced(
            report,
            "PQ",
            {
                e: sampler.window_series(
                    "mock_engine_waiting", ss_tail_lo, t_ss_end
                ).get(e, [])
                for e in dec_survivors
            },
            (ss_tail_lo, t_ss_end),
            "peak",
            _mechanism_bands(BAL_QUEUE_DEPTH_MAX),
            context="steady_waiting_depth",
        )

        # PK — occupancy spread (survivors, last third)
        AssertUtils.balanced(
            report,
            "PK",
            {
                e: _bal_occupancy_series(sampler, e, ss_tail_lo, t_ss_end)
                for e in dec_survivors
            },
            (ss_tail_lo, t_ss_end),
            "spread",
            _mechanism_bands(base_occ_spread + BAL_KV_SPREAD_TOL),
            context="steady_occ_spread",
        )
        # PK — per-survivor occupancy peak <= 0.95 (reserveRatio bound)
        AssertUtils.balanced(
            report,
            "PK",
            {
                e: _bal_occupancy_series(sampler, e, ss_tail_lo, t_ss_end)
                for e in dec_survivors
            },
            (ss_tail_lo, t_ss_end),
            "peak",
            _mechanism_bands(BAL_OCCUPANCY_CAP),
            context="steady_occ_peak",
        )

        # ---- 2. oscillation fingerprint (HARD): 3s sub-window share
        # series vs the n-rescaled target — two consecutive
        # same-direction departures beyond ±0.10 fail ----
        n_sub = int(BAL_STEADY_S / BAL_STEADY_SUBWINDOW_S)
        target = 1.0 / len(dec_survivors)
        sub_shares = {e: [] for e in dec_survivors}
        for i in range(n_sub):
            lo = t_settle + i * BAL_STEADY_SUBWINDOW_S
            hi = lo + BAL_STEADY_SUBWINDOW_S
            d = sampler.window_series(
                "mock_engine_accepted_total", lo, hi, mode="delta"
            )
            vals = [d.get(e, 0.0) for e in dec_survivors]
            tot = sum(vals)
            if tot <= 0:
                continue  # traffic-less sub-window: no departure signal
            for e, v in zip(dec_survivors, vals):
                sub_shares[e].append(v / tot)
        osc_bad = []
        for e, seq in sub_shares.items():
            for i in range(len(seq) - 1):
                d1 = seq[i] - target
                d2 = seq[i + 1] - target
                if abs(d1) > BAL_SHARE_TOL and abs(d2) > BAL_SHARE_TOL and d1 * d2 > 0:
                    osc_bad.append((e, i, round(d1, 3), round(d2, 3)))
        report.check(
            "P1",
            len(osc_bad),
            bands=_mechanism_bands(0),
            context="steady_oscillation_fingerprint",
            detail=(
                f"two consecutive same-direction sub-window departures "
                f"beyond ±{BAL_SHARE_TOL} from the n-rescaled target "
                f"{target:.3f}: {osc_bad[:3]}"
            ),
        )

        # ---- 3. observations (first-run calibration, never gating) ----
        ss_dev = [abs(s - target) for seq in sub_shares.values() for s in seq]
        ss_swing = max(ss_dev) if ss_dev else None
        obs.append(
            _obs_note(
                "P1",
                "subwindow_share_swing(max|share-1/n|; band: "
                "<=base_swing+0.10, design §2.4, first-run)",
                ss_swing,
            )
        )
        obs.append(
            _obs_note(
                "P1",
                "swing_band(base_swing+0.10)",
                (base_swing + 0.10) if base_swing is not None else None,
            )
        )
        ss_cv = _bal_exec_cv(
            sampler,
            "mock_engine_decode_ms_avg",
            ss_tail_lo,
            t_ss_end,
            dec_survivors,
        )
        pl_cap = max((base_cv or 0.0) + 0.10, 0.25)
        obs.append(
            _obs_note(
                "PL",
                "steady_exec_cv(design §1.2 PL row; first-run calibration)",
                ss_cv,
            )
        )
        obs.append(_obs_note("PL", "pl_band(max(base_cv+0.10, 0.25))", pl_cap))
        ss_hit = _bal_hit_rate(sampler, ss_tail_lo, t_ss_end)
        obs.append(
            _obs_note(
                "PC",
                "steady_hit>=base-0.15 (design §2.4/§1.2; first-run)",
                ss_hit,
            )
        )
        obs.append(
            _obs_note(
                "PC",
                "pc_floor(base_hit-0.15)",
                (base_hit - 0.15) if base_hit is not None else None,
            )
        )
        tps_means = []
        for e in dec_survivors:
            pts = sampler.window_series(
                "rtp_llm_context_tps", ss_tail_lo, t_ss_end
            ).get(e, [])
            if pts:
                tps_means.append(sum(v for _, v in pts) / len(pts))
        pt_ratio = (
            max(tps_means) / min(tps_means)
            if len(tps_means) >= 2 and min(tps_means) > 0
            else None
        )
        obs.append(
            _obs_note(
                "PT",
                "steady_tps_max/min<=1.5 (design §1.2; first-run)",
                pt_ratio,
            )
        )

        # ---- 4. survivor keeps serving ----
        ok_n, _err_n, _ = _run_batch(ops, base, 20)
        recovery_rate = ok_n / 20.0
        report.invariant(
            "P2",
            recovery_rate >= 0.95,
            context="steady_survivor_service",
            detail=f"recovery {ok_n}/20",
        )
        with pump_lock:
            pump_rate = pump_ok[0] / pump_total[0] if pump_total[0] else None

        events_summary = json.dumps(dict(sampler._events)) if sampler else "{}"
        ok_verdict, detail, _rep = report.finish(
            f"victim={victim}(graceful), "
            f"rm=(drained={rm.get('drained')}, "
            f"drain_ms={rm.get('drain_ms')}), "
            f"steady=(share_max={d_ss_max:.3f}/cap {d_share_cap:.3f}, "
            f"share_min={d_ss_min:.3f}, osc_violations={len(osc_bad)}), "
            f"base=(share_max={base_share_max:.3f}, "
            f"occ_spread={base_occ_spread:.3f}, swing={base_swing}, "
            f"hit={base_hit}, exec_cv={base_cv}), "
            f"alive_ok={alive_ok}, recovery={ok_n}/20, "
            f"pump=({pump_ok[0]}/{pump_total[0]}), "
            f"windows=(W_base[0,{t_base:.0f}], "
            f"W_tr[{t_remove:.0f},{t_tr_end:.0f}], "
            f"W_ss[{t_settle:.0f},{t_ss_end:.0f}], "
            f"tail[{ss_tail_lo:.0f},{t_ss_end:.0f}]), "
            f"events={events_summary}, "
            f"{'; '.join(obs)}, "
            f"grades: {report.summary()}"
        )
        return ok_verdict, detail, _rep
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        pump_stop.set()
        if pump_thread is not None:
            pump_thread.join(10.0)
        if sampler is not None:
            try:
                sampler.stop()
                sampler.dump(
                    ctx.case_dir("elastic_steady_state_recovery")
                    / "balance_sampler.json.gz"
                )
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
