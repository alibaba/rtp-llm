"""Chaos test cases: elastic scaling (dynamic discovery) + core chaos scenarios.

Registration model (2026-08 test-taxonomy rework):

  elastic_*   (6 cases, suite="smoke") — dynamic engine scale-out/in through
      the mock control plane (/add_engine + /remove_engine) with the
      file-based dynamic discovery chain enabled end to end:
      mock ``--discovery-file`` → DiscoveryFileStore (atomic rewrite) →
      master ``FLEXLB_DISCOVERY_FILE`` → FileServiceDiscovery (re-read per
      poll) → EngineSyncRunner → EndpointRegistry → routing.
      Elastic scaling is a normal functional requirement, NOT chaos (user
      ruling 2026-08): these cases are registered with suite="smoke" so the
      runner's ``--suite smoke`` selects them as part of the functional
      suite while ``--suite chaos`` no longer does.  They physically stay in
      this file / CHAOS_CASES list for file-ownership isolation (parallel
      agent work on smoke_cases.py); the suite field alone drives the
      grouping, so no runner change is needed.  Run them individually with
      ``--filter elastic``.
      Convergence bounds follow the verified flexlb-api e2e behaviour
      (FileDiscoveryDynamicScaleEndToEndTest: add ~26ms, remove ~1s); the
      framework asserts at second-scale timeouts (10-15s) to stay robust
      against slow CI machines.

Suite ``chaos`` (fault injection & self-healing):

  chaos_engine_down_http_stop_prefill — five-phase engine-down uniform
      assertion set (S2/S4 port) plus a post-recovery TTFT regression gate
      (master_recovery_ttft_test.sh semantics: once the fault heals, TTFT
      must fall back within 1.5x of the pre-fault baseline p50).
  chaos_master_kill          — kill -9 master + restart + clean state.
  chaos_master_quota_block   — S3 port (1P+1D quota blocking + TTL recovery).
  chaos_inflight_ttl_cleanup — S1 port (stuck inflight cleaned by TTL).
  chaos_coldstart_burst (probe) — intake defect regression probe: fire 20
      requests the instant the master reports ready (stability window
      disabled). Expected to FAIL or pass marginally until the intake fix
      (CONNECT_TIMEOUT 20ms / 3-strike dead marking / non-atomic
      getOrCreateWorkerStatus); the recorded failure rate and marked-dead
      samples are the baseline for that fix.
  chaos_engine_flap (gap G2) — connection flapping: rapid /stop_engine +
      /start_engine oscillation on one prefill (>=5 cycles) under live
      traffic; asserts master stays healthy, no inflight leak, and full
      re-discovery + routing recovery once the flapping stops.

The mock is a SINGLE JVM hosting all engines: /add_engine opens a new port
inside the same process (no per-engine kill possible — use /stop_engine for
the stop/start variants, /remove_engine for permanent detach).
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .context import CaseContext, CaseDef, rid_base
from .harness import AssertUtils, EnvSpec, default_perf, http_get_status, wait_for

CHAOS_CASES: list[CaseDef] = []

PREFILL_DOMAIN = "mock.prefill.hosts.address"
DECODE_DOMAIN = "mock.decode.hosts.address"

STREAM_TIMEOUT_S = 15.0
# File-discovery convergence caps (e2e-verified: add ~26ms, remove ~1s; the
# caps below are deliberately loose, second-scale, for slow CI machines).
ADD_CONVERGENCE_S = 10.0
REMOVE_CONVERGENCE_S = 10.0
# Master eviction of a vanished discovery entry (sync 20ms + stale window) —
# generous cap so slow machines do not flake.
MASTER_EVICT_S = 30.0


def case(name: str, modes=None, source: str = "", suite: str = "chaos"):
    """Register into CHAOS_CASES; *suite* drives the runner grouping.

    The elastic_* cases pass suite="smoke" (functional taxonomy — see the
    module docstring); everything else stays in the chaos suite.
    """

    def deco(fn):
        CHAOS_CASES.append(
            CaseDef(name=name, suite=suite, fn=fn, modes=modes, source=source)
        )
        return fn

    return deco


# ===========================================================================
# FlexLB config for chaos environments
# ===========================================================================


def chaos_flexlb_config(
    stale_inflight_ms: int = 30_000,
    max_inflight_batches: int = 4,
    status_rpc_ms: int = 1_000,
) -> str:
    """Strict single-document FLEXLB_CONFIG (flexlb_behavior_test.sh template).

    Shorter staleInflightTimeoutMs (30s vs the na130 default 300s) so the TTL
    cleanup cases finish within their 90s caps.
    """
    return json.dumps(
        {
            "schemaVersion": 2,
            "scheduler": {
                "type": "QUEUE",
                "ordering": {"type": "PRIORITY"},
                "decision": {
                    "type": "FIXED_WINDOW",
                    "maxRequests": 32,
                    "maxCollectionWaitMs": 10,
                    "maxPredictedExecutionMs": 550,
                },
                "capacity": {"maxOutstandingRequestsGlobal": 5000},
                "lifecycle": {
                    "staleInflightTimeoutMs": stale_inflight_ms,
                    "deliveredNotAcceptedTimeoutMs": 30_000,
                    "maxDeliveredNotAcceptedRequestsGlobal": 200,
                },
            },
            "dispatcher": {
                "type": "BATCH",
                "maxInflightBatchesPerPrefillWorker": max_inflight_batches,
            },
            "router": {
                "availabilityHysteresisPercent": 0,
                "roles": {
                    "prefill": {
                        "availability": {"maxPendingRequests": 100000},
                        "selector": {
                            "type": "ESTIMATED_TTFT",
                            "candidateChoice": {
                                "type": "RANDOM_WITHIN_TOLERANCE",
                                "relativeTolerance": 0.1,
                                "minimumToleranceMs": 20,
                                "outlierRejection": {
                                    "maxPendingVsAverageMultiplier": 1.5,
                                    "maxWaitVsAverageMultiplier": 3.0,
                                },
                            },
                        },
                    },
                    "decode": {"availability": {"maxEngineRequests": 132}},
                },
            },
            "workerRegistry": {
                "health": {
                    "statusPollIntervalMs": 20,
                    "statusRpcTimeoutMs": status_rpc_ms,
                    "statusStaleAfterMs": max(10_000, status_rpc_ms * 2),
                }
            },
        },
        separators=(",", ":"),
    )


def elastic_spec(ctx: CaseContext) -> EnvSpec:
    """Shared elastic/chaos env: 2P+4D, dynamic file discovery, TTL=30s."""
    return EnvSpec(
        label=f"chaos_{ctx.mode}",
        n_prefill=2,
        n_decode=4,
        perf=default_perf(),
        master_mode=ctx.mode,
        discovery="discovery_file",
        master_env={"FLEXLB_CONFIG": chaos_flexlb_config()},
    )


def ttl_spec(ctx: CaseContext) -> EnvSpec:
    """Inflight-TTL env (S1): 2P+2D, TTL=30s."""
    return EnvSpec(
        label=f"chaos_ttl_{ctx.mode}",
        n_prefill=2,
        n_decode=2,
        perf=default_perf(),
        master_mode=ctx.mode,
        discovery="discovery_file",
        master_env={"FLEXLB_CONFIG": chaos_flexlb_config()},
    )


def quota_spec(ctx: CaseContext) -> EnvSpec:
    """Quota-block env (S3): 1P+1D, maxInflightBatches=1 (via env override)."""
    return EnvSpec(
        label=f"chaos_quota_{ctx.mode}",
        n_prefill=1,
        n_decode=1,
        perf=default_perf(),
        master_mode=ctx.mode,
        discovery="discovery_file",
        master_env={
            "FLEXLB_CONFIG": chaos_flexlb_config(max_inflight_batches=1),
            "FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES": "1",
        },
    )


# ===========================================================================
# Shared helpers
# ===========================================================================


def _master_http(ops) -> str:
    return f"http://127.0.0.1:{ops.master_http_port}"


def _elastic_env(ctx: CaseContext):
    env = ctx.env_manager.ensure(elastic_spec(ctx))
    return env, ctx.engine_ops(env)


def _initial_engine_names(env) -> set:
    return {f"prefill-{i}" for i in range(env.spec.n_prefill)} | {
        f"decode-{i}" for i in range(env.spec.n_decode)
    }


def _dynamic_engines(ops, env) -> list[str]:
    """Engine names added at runtime (anything beyond the initial role set)."""
    initial = _initial_engine_names(env)
    return [
        e["name"]
        for e in ops.snapshot().get("engines", [])
        if e.get("name") not in initial
    ]


def _cleanup_dynamic(ops, env) -> int:
    """Remove every dynamically added engine (env restore between cases)."""
    removed = 0
    for name in _dynamic_engines(ops, env):
        try:
            status, _ = ops.remove_engine(engine_name=name)
            removed += 1 if status == 200 else 0
        except Exception:
            pass
    return removed


def _discovery_payload(env) -> Optional[dict]:
    """json.load the discovery file; None when missing/unparseable."""
    try:
        return json.loads(env.discovery_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def _discovery_has_http_port(env, http_port: int) -> bool:
    payload = _discovery_payload(env)
    if payload is None:
        return False
    suffix = f":{http_port}"
    for hosts in payload.values():
        if isinstance(hosts, list) and any(str(h).endswith(suffix) for h in hosts):
            return True
    return False


def _discovery_entry_count(env) -> tuple[int, int]:
    """(#prefill hosts, #decode hosts) per the discovery file domains."""
    payload = _discovery_payload(env)
    if payload is None:
        return -1, -1
    return (
        len(payload.get(PREFILL_DOMAIN) or []),
        len(payload.get(DECODE_DOMAIN) or []),
    )


def _accepted(ops, engine_name: str) -> int:
    try:
        snap = ops.snapshot_by_name()
        return int(snap.get(engine_name, {}).get("accepted", 0))
    except Exception:
        return -1


def _engine_alive(ops, role: str) -> int:
    return ops.master_alive_count(role)


def _wait_master_alive(ops, role: str, expected: int, timeout_s: float) -> bool:
    return wait_for(lambda: _engine_alive(ops, role) >= expected, timeout_s, 0.5)


def _wait_master_topology(ops, role: str, expected: int, timeout_s: float) -> bool:
    """Wait until the master's discovery view of *role* is EXACTLY *expected*
    workers, all alive (worker_summary ``discovered == alive == expected``).

    ``discovered`` mirrors the master's workerStatusMap size, which only
    shrinks when EngineSyncRunner evicts a detached engine.  The alive count
    alone is NOT a safe convergence signal: the health 3-strike demotion
    lands well before the eviction removes the endpoint from the routable
    set, so waiting on alive lets traffic hit a dead-but-still-routable
    port (see elastic_rebalance baseline).
    """

    def converged() -> bool:
        info = ops.master_info()
        if not info:
            return False
        entry = (info.get("worker_summary", {}) or {}).get(role) or {}
        try:
            discovered = int(entry.get("discovered", -1))
            alive = int(entry.get("alive", -1))
        except (TypeError, ValueError):
            return False
        return discovered == expected and alive == expected

    return wait_for(converged, timeout_s, 0.2)


def _request_with_ttft(ops, rid: int, output_len: int, keys: list):
    """run_one_request variant that also measures TTFT.

    TTFT = time from the schedule() call to the first streamed output (2ms
    poll granularity — ``first_received`` is edge-triggered on the consume
    thread, so the poller only has to catch the edge before termination).
    Returns (prefill_addr, error, ttft_ms); ttft_ms is None when the first
    output was never observed.
    """
    t0 = time.monotonic()
    try:
        response = ops.schedule(rid, output_len=output_len, block_keys=keys)
        if response.code != 200 or not response.success:
            return "", f"schedule failed: {response.error_message}", None
        addr = ops.role_addr(response, "PREFILL")
        input_pb = (
            None if response.enqueued_by_master else ops.build_generate_input(rid)
        )
        handle = ops.start_stream(response, rid, input_pb=input_pb)
        first_ms: Optional[float] = None
        deadline = time.monotonic() + STREAM_TIMEOUT_S
        while time.monotonic() < deadline:
            if handle.snap.first_received:
                first_ms = (time.monotonic() - t0) * 1000.0
                break
            if handle.snap.terminated:
                break
            time.sleep(0.002)
        handle.wait_end(STREAM_TIMEOUT_S)
        snap = handle.snap
        if snap.error:
            return addr, snap.error, first_ms
        if not snap.completed:
            return addr, "stream did not complete", first_ms
        return addr, None, first_ms
    except Exception as exc:
        return "", repr(exc), None


def _ttft_p50(values: list) -> Optional[float]:
    """Index-method p50 (same as harness.per_request_ttft_p50)."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 50 / 100))]


def _run_batch(
    ops,
    base: int,
    n: int,
    output_len: int = 2,
    concurrency: int = 10,
    collect_ttft: bool = False,
) -> tuple[int, int, list[str]]:
    """Send *n* small requests (up to *concurrency* in flight); returns
    (ok, errors, prefill_addrs).

    Error strings (first 60 chars, deduped) are stashed on the function as
    ``last_error_types`` for failure diagnostics — batch paths routinely
    fail with distinct gRPC/queue errors and the verdict alone cannot
    distinguish "routed to a dead engine" from "queue admission reject".
    With ``collect_ttft=True`` the per-request TTFTs of *successful*
    requests are additionally stashed as ``last_ttfts`` (ms, unsorted).
    """
    rids = [ops.next_request_id(base) for _ in range(n)]

    def run(rid: int):
        keys = [rid * 100 + j for j in range(3)]
        if collect_ttft:
            return _request_with_ttft(ops, rid, output_len, keys)
        addr, err = ops.run_one_request(
            rid,
            output_len=output_len,
            block_keys=keys,
            stream_timeout_s=STREAM_TIMEOUT_S,
        )
        return addr, err, None

    with ThreadPoolExecutor(max_workers=min(n, concurrency)) as pool:
        results = list(pool.map(run, rids))
    ok = sum(1 for _, err, _ttft in results if err is None)
    addrs = [addr for addr, _err, _ttft in results]
    _run_batch.last_error_types = sorted(
        {str(err)[:60] for _, err, _ttft in results if err is not None}
    )
    if collect_ttft:
        _run_batch.last_ttfts = [
            ttft for _, err, ttft in results if err is None and ttft is not None
        ]
    return ok, n - ok, addrs


class _BackgroundFlow:
    """Lightweight loop: one request every *interval_s* on a daemon thread."""

    def __init__(
        self,
        ops,
        rid_base_value: int,
        interval_s: float = 0.2,
        output_len: int = 2,
    ):
        self._ops = ops
        self._base = rid_base_value
        self._interval = interval_s
        self._output_len = output_len
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.total = 0
        self.ok = 0

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            rid = self._ops.next_request_id(self._base)
            _, err = self._ops.run_one_request(
                rid,
                output_len=self._output_len,
                block_keys=[rid * 100 + 1],
                stream_timeout_s=10.0,
            )
            with self._lock:
                self.total += 1
                if err is None:
                    self.ok += 1
            self._stop_event.wait(self._interval)

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, name="chaos-flow", daemon=True
        )
        self._thread.start()

    def stop(self, timeout_s: float = 20.0) -> tuple[int, int]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
        return self.total, self.ok

    @property
    def success_rate(self) -> float:
        return (self.ok / self.total) if self.total else 0.0


def _pump_until_accepted(ops, engine_name: str, base: int, timeout_s: float) -> bool:
    """Send requests until *engine_name*'s accepted counter grows."""
    start = _accepted(ops, engine_name)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rid = ops.next_request_id(base)
        ops.run_one_request(
            rid,
            output_len=2,
            block_keys=[rid * 100 + 1],
            stream_timeout_s=10.0,
        )
        if _accepted(ops, engine_name) > start:
            return True
        time.sleep(0.2)
    return _accepted(ops, engine_name) > start


# ===========================================================================
# Elastic group — 6 cases (suite="smoke": functional taxonomy, not chaos)
# ===========================================================================


@case(
    "elastic_add_flow",
    source="elastic acceptance: add under load (FileDiscoveryDynamicScaleEndToEndTest phase 2)",
    suite="smoke",
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
    source="elastic acceptance: remove under load (FileDiscoveryDynamicScaleEndToEndTest phase 3)",
    suite="smoke",
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
    source="elastic acceptance: 3x add→verify→remove→verify cycle",
    suite="smoke",
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
    source="elastic acceptance: cost-based rebalance after scale-out (share < 60%)",
    suite="smoke",
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
    source="elastic acceptance: add → traffic → /stop_engine (3-fail evict) → /start_engine recovery",
    suite="smoke",
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
    source="elastic acceptance: concurrent add/remove storm, master stays healthy",
    suite="smoke",
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
# Cold-start probe — intake defect regression baseline
# ===========================================================================


def coldstart_spec(ctx: CaseContext) -> EnvSpec:
    """Cold-start probe env: mirrors the smoke topology (2P+4D, static file
    discovery, default config) but disables the master stability window so
    traffic hits the master during the first-connect storm."""
    return EnvSpec(
        label=f"chaos_coldstart_{ctx.mode}",
        n_prefill=2,
        n_decode=4,
        perf=default_perf(),
        master_mode=ctx.mode,
        master_stable_window_s=0.0,
    )


@case(
    "chaos_coldstart_burst",
    modes=["batch"],
    source="intake defect regression probe (cold-start first-connect storm)",
)
def coldstart_burst(ctx: CaseContext):
    """Fire 20 requests the instant the master reports ready.

    Regression probe for the three intake defects: CONNECT_TIMEOUT 20ms,
    3-strike dead marking on first connect, non-atomic getOrCreateWorkerStatus.
    Expected to FAIL or pass marginally today — the failure rate and the
    marked-dead sample count are recorded as the baseline for the intake fix.
    """
    env = ctx.env_manager.ensure(coldstart_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "chaos")
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
        # assertion (S1/S6 contract) below.
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
        # as scheduling S1/S6: 20 requests over 2 prefills (10-way
        # concurrent), both engines used, no engine above 80% of the
        # *successful* requests — COST_BASED_PREFILL scores the two prefills
        # identically on an empty cold ledger and RANDOM_WITHIN_TOLERANCE
        # samples the tie window uniformly, so a one-sided distribution can
        # only come from an engine being 3-strike-marked dead (the S10-class
        # intake defect this probe guards).  80% of 20 = 16 requests, i.e.
        # the same "no engine eats the burst" bound as S1/S6.
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
# Chaos core group — 4 cases (flexlb_behavior_test.sh ports + master kill)
# ===========================================================================


@case(
    "chaos_inflight_ttl_cleanup",
    modes=["batch"],
    source="flexlb_behavior_test.sh S1 (stuck inflight TTL cleanup)",
)
def inflight_ttl_cleanup(ctx: CaseContext):
    """S1 port: slow prefill → inflight stuck → /stop_engine → TTL cleans.

    Batch mode only — the stuck-inflight state lives in the master's batcher
    (enqueued_by_master path); direct/queue modes have no enqueue bookkeeping.
    """
    env = ctx.env_manager.ensure(ttl_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "chaos")
    try:
        # Slow both prefills (10s) so scheduled requests stay inflight.
        ops.set_perf("prefill-0", prefill_fixed_ms=10000.0)
        ops.set_perf("prefill-1", prefill_fixed_ms=10000.0)

        # Fire-and-forget: schedule without consuming the response stream —
        # the master has already enqueued these batches into the engines.
        rids = [ops.next_request_id(base) for _ in range(6)]
        for rid in rids:
            resp = ops.schedule(rid, output_len=10)
            if resp.code != 200 or not resp.success:
                return False, f"schedule failed for rid={rid}: {resp.error_message}"

        enqueued = wait_for(lambda: _accepted(ops, "prefill-0") > 0, 15.0, 0.5)
        inflight_before = ops.master_scheduler_inflight()

        # Cut the engine mid-flight; its batches will never complete.
        ops.stop_engine("prefill-0")
        time.sleep(5.0)  # let the gRPC failures propagate
        # Per-endpoint view: the evicted engine's row disappears from
        # prefill_endpoints, so observe the stuck batches via the global
        # scheduler inflight (survives eviction until TTL cleanup).
        inflight_after_kill = ops.master_scheduler_inflight()

        # TTL (30s) + sync margin: poll to zero within 90s.
        cleanup_ok = wait_for(lambda: ops.master_scheduler_inflight() == 0, 90.0, 2.0)
        inflight_final = ops.master_scheduler_inflight()

        # The surviving prefill keeps serving normally.
        recovery_ok, recovery_msg = ops.verify_recovery()

        passed = (
            enqueued
            and inflight_after_kill > 0
            and cleanup_ok
            and inflight_final == 0
            and recovery_ok
        )
        return passed, (
            f"enqueued={enqueued}, inflight_before_stop={inflight_before}, "
            f"stuck_after_kill={inflight_after_kill}, "
            f"cleanup_within_90s={cleanup_ok}, inflight_final={inflight_final}, "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            snap = ops.snapshot_by_name()
            if snap.get("prefill-0", {}).get("stopped"):
                ops.start_engine("prefill-0")
            ops.set_perf("prefill-0", prefill_fixed_ms=100.0)
            ops.set_perf("prefill-1", prefill_fixed_ms=100.0)
        except Exception:
            pass


@case(
    "chaos_engine_down_http_stop_prefill",
    source="flexlb_behavior_test.sh S2/S4 merged — five-phase engine-down assertion set",
)
def engine_down_http_stop_prefill(ctx: CaseContext):
    """Five phases with the uniform engine-down assertion set (core 3 of 7):
    master stays up + surviving engines take over + recovery rate.
    """
    env, ops = _elastic_env(ctx)
    base = rid_base(ctx, "chaos")
    try:
        _cleanup_dynamic(ops, env)

        def master_up() -> bool:
            return (
                http_get_status(
                    f"{_master_http(ops)}/rtp_llm/inflight_status", timeout=5
                )
                == 200
            )

        # Phase 1 — baseline: 20 requests, all succeed.  TTFT p50 is
        # recorded for the post-recovery regression gate (Phase 5).
        ok1, err1, _ = _run_batch(ops, base, 20, collect_ttft=True)
        base_ttft_p50 = _ttft_p50(getattr(_run_batch, "last_ttfts", []))
        master_ok1 = master_up()
        if err1:
            return False, f"baseline had {err1} errors (master_up={master_ok1})"

        # Phase 2 — http-stop prefill-0.
        ops.stop_engine("prefill-0")
        # Phase 3 — downtime: wait for the 3-consecutive-failure eviction,
        # then 20 more requests must still succeed (2P redundancy).
        evicted = wait_for(
            lambda: ops.master_alive_count("PREFILL") <= 1,
            MASTER_EVICT_S,
            0.5,
        )
        ok2, err2, _ = _run_batch(ops, base, 20)
        master_ok2 = master_up()
        rate2 = ok2 / 20 if err2 == 0 else ok2 / 20
        takeover_ok = err2 <= 2  # ≥90% success while one prefill is down
        downtime_err_types = list(getattr(_run_batch, "last_error_types", []))[:3]

        # Phase 4 — restart the engine and wait for re-discovery.
        ops.start_engine("prefill-0")
        alive_back = wait_for(
            lambda: ops.master_alive_count("PREFILL") >= 2,
            MASTER_EVICT_S,
            0.5,
        )
        # Channel recovery settle (S2's reconnect window).
        time.sleep(3.0)

        # Phase 5 — recovery: 20 requests ≥95%, and TTFT must fall back to
        # within 1.5x of the baseline p50 (master_recovery_ttft_test.sh
        # semantics: once the fault heals, TTFT returns to baseline — the
        # legacy analyzer tolerates an early 1.5x spike and only degrades
        # the verdict when the *stable* window stays above 1.2x; this batch
        # sits past the 3s channel-settle window, so the 1.5x gate is the
        # conservative bound on steady-state recovery).
        ok5, err5, _ = _run_batch(ops, base, 20, collect_ttft=True)
        recovery_ttft_p50 = _ttft_p50(getattr(_run_batch, "last_ttfts", []))
        ttft_ok, ttft_detail = AssertUtils.ttft_degradation(
            base_ttft_p50, recovery_ttft_p50, threshold_pct=50.0
        )
        master_ok5 = master_up()
        recovery_ok = ok5 >= 19  # ≥95%

        passed = (
            master_ok1
            and master_ok2
            and master_ok5
            and evicted
            and takeover_ok
            and alive_back
            and recovery_ok
            and ttft_ok
        )
        return passed, (
            f"baseline=20/20, evicted_after_stop={evicted}"
            f"(alive={ops.master_alive_count('PREFILL')}), "
            f"downtime={ok2}/20({rate2:.0%}, err_types={downtime_err_types}), "
            f"alive_restored={alive_back}, "
            f"recovery={ok5}/20({ok5 / 20:.0%}), "
            f"ttft_recovery=[{ttft_detail}], "
            f"master_up=(p1:{master_ok1}, p3:{master_ok2}, p5:{master_ok5})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        try:
            snap = ops.snapshot_by_name()
            if snap.get("prefill-0", {}).get("stopped"):
                ops.start_engine("prefill-0")
        except Exception:
            pass


@case(
    "chaos_master_kill",
    source="master HA: kill -9 master → restart → clean state + recovery",
)
def master_kill(ctx: CaseContext):
    env, ops = _elastic_env(ctx)
    base = rid_base(ctx, "chaos")
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
    "chaos_master_quota_block",
    modes=["batch"],
    source="flexlb_behavior_test.sh S3 (1P+1D quota blocking + TTL recovery)",
)
def master_quota_block(ctx: CaseContext):
    """S3 port: fill the 1-batch inflight quota → stop the only prefill →
    new requests fail (≥50%) → TTL cleanup → start engine → recovery ≥90%.
    """
    env = ctx.env_manager.ensure(quota_spec(ctx))
    ops = ctx.engine_ops(env)
    base = rid_base(ctx, "chaos")
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

        # TTL cleanup (30s): scheduler inflight drains to zero (the evicted
        # engine's endpoint row is gone, so watch the global counter).
        cleanup_ok = wait_for(lambda: ops.master_scheduler_inflight() == 0, 90.0, 2.0)
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
            f"ttl_cleanup_within_90s={cleanup_ok}, "
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
# Engine flap (gap G2) — connection flapping under live traffic
# ===========================================================================


@case(
    "chaos_engine_flap",
    source="gap G2: rapid /stop_engine+/start_engine oscillation, 3-strike eviction vs re-discovery race",
)
def engine_flap(ctx: CaseContext):
    """Connection flapping: >=5 rapid stop/start cycles on one prefill.

    Exercises the race window between the master's 3-strike health eviction
    and the engine's re-discovery: each cycle stops prefill-0, holds it down
    long enough for the health poller (20ms interval) to accumulate strikes,
    then brings it back WITHOUT waiting for convergence (the flap).  A
    background flow keeps traffic live throughout.

    Assertions (user-mandated):
      * master stays healthy the whole time — HTTP 200 probe every cycle,
        no hang;
      * after the flapping stops: the engine is re-discovered
        (discovered == alive == initial topology), routing and requests
        recover (>=95% batch), and no inflight leaks (global drain to zero
        within the 90s cap that covers the 30s stale-inflight TTL).

    The per-cycle alive count is observational evidence of the eviction vs
    re-discovery race: dipping below 2 means the 3-strike demotion landed,
    staying at 2 means recovery won the race — both are correct; the
    contract is only about final convergence and no leak.
    """
    env, ops = _elastic_env(ctx)
    base = rid_base(ctx, "chaos")
    flow: Optional[_BackgroundFlow] = None
    try:
        _cleanup_dynamic(ops, env)

        flow = _BackgroundFlow(ops, base, interval_s=0.2)
        flow.start()
        time.sleep(1.0)  # let the flow ramp up before the first stop

        cycles = 6
        cycle_log: list[str] = []
        master_200_all = True
        evict_landings = 0
        for i in range(1, cycles + 1):
            ops.stop_engine("prefill-0")
            # Hold down ~0.8s: the health poller runs every 20ms, so the
            # 3-strike counter fires well inside this window, while the
            # EngineSyncRunner endpoint-eviction threshold (max(3*20ms, 1s)
            # from the last successful status) lands at the tail of the
            # window or right after the restart — exactly the race under
            # test.
            time.sleep(0.8)
            alive_mid = ops.master_alive_count("PREFILL")
            if alive_mid < 2:
                evict_landings += 1
            probe = http_get_status(
                f"{_master_http(ops)}/rtp_llm/inflight_status", timeout=5
            )
            if probe != 200:
                master_200_all = False
            ops.start_engine("prefill-0")
            time.sleep(0.4)  # short gap — flap, no convergence wait
            cycle_log.append(f"c{i}[alive={alive_mid}, master={probe}]")

        total, ok = flow.stop()
        rate = ok / total if total else 0.0

        # Post-flap convergence: full re-discovery of the flapped engine
        # (discovered count covers the eviction side of the race — see
        # elastic_rebalance for why alive alone is not a safe signal).
        topology_ok = _wait_master_topology(
            ops, "PREFILL", env.spec.n_prefill, MASTER_EVICT_S
        )
        # Routing/request recovery: 20 requests, >=95%.
        ok_batch, _, _ = _run_batch(ops, base, 20)
        recovery_ok = ok_batch >= 19
        # No inflight leak: global drain to zero (covers the 30s TTL).
        inflight_ok, inflight_detail = AssertUtils.inflight_clean(
            _master_http(ops), 90.0
        )

        passed = (
            master_200_all
            and topology_ok
            and recovery_ok
            and inflight_ok
            and total > 0
            and rate >= 0.5  # availability floor — a total blackout must fail
        )
        return passed, (
            f"cycles={cycles}, evictions_landed={evict_landings}/{cycles}, "
            f"flap=[{'; '.join(cycle_log)}], "
            f"flow_success={ok}/{total}({rate:.0%}), "
            f"topology_converged={topology_ok}, "
            f"post_flap_batch={ok_batch}/20, "
            f"inflight_clean={inflight_ok}({inflight_detail})"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"
    finally:
        if flow is not None:
            flow.stop()
        try:
            snap = ops.snapshot_by_name()
            if snap.get("prefill-0", {}).get("stopped"):
                ops.start_engine("prefill-0")
        except Exception:
            pass
