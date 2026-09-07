"""Shared engine_fault scenario components. Resource and timing semantics live here."""

from __future__ import annotations

import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from ..context import CaseContext, CaseDef, rid_base
from ..engine_ops import (
    _fence_residue_stable,
    clear_type_all,
    engine_inflight_clean,
    inject_type,
    inject_type_all,
)
from ..harness import (
    OMIT,
    TTL_DRAIN_TIMEOUT_S,
    AssertUtils,
    ConfigOverride,
    EnvSpec,
    _BackgroundFlow,
    _cleanup_dynamic,
    _elastic_env,
    _pump_until_accepted,
    _run_batch,
    _ttft_p50,
    _wait_master_alive,
    _wait_master_topology,
    default_perf,
    fault_env_perf,
    http_get_status,
    http_post_json,
    wait_for,
)

STREAM_TIMEOUT_S = 15.0
# 3-strike health demotion + eviction window (fault-family precedent).
MASTER_EVICT_S = 30.0
# Engine restart channel-reconnect settle window.
ENGINE_RECOVERY_WAIT_S = 3.0
# Anomaly-family timing knobs (anomaly_smoke.py E2/E3 calibration).
TIMEOUT_WAIT_S = 5.0
ANOMALY_STREAM_TIMEOUT_S = 10.0
WORKER_RECOVERY_WAIT_S = 3.0


def _master_http(ops) -> str:
    return f"http://127.0.0.1:{ops.master_http_port}"


def _prefill_names(ops) -> list[str]:
    snap = ops.snapshot()
    return [e["name"] for e in snap.get("engines", []) if e.get("role") == "prefill"]


def _measure_ttft(
    ops, rid: int, timeout_s: float = 12.0
) -> tuple[float, Optional[str], bool]:
    """schedule + stream one request, measuring first-output latency.

    Returns (ttft, error, enqueued_by_master) — the flag lets callers
    branch master-inflight checks on the actual delivery mode instead of
    inferring it from the profile."""
    response = ops.schedule(rid)
    if response.code != 200 or not response.success:
        return -1.0, f"schedule failed: {response.error_message}", False
    input_pb = None if response.enqueued_by_master else ops.build_generate_input(rid)
    handle = ops.start_stream(response, rid, input_pb=input_pb)
    t0 = time.monotonic()
    got_first = handle.wait_first_output(timeout_s)
    ttft = time.monotonic() - t0
    if not got_first:
        handle.cancel()
        return -1.0, "no first output", response.enqueued_by_master
    ended = handle.wait_end(timeout_s)
    if not ended or handle.snap.error:
        handle.cancel()
        return (
            ttft,
            f"stream error after first output: {handle.snap.error}",
            (response.enqueued_by_master),
        )
    return ttft, None, response.enqueued_by_master


def _inject_all_prefill(ops, config: dict) -> list[str]:
    snap = ops.snapshot()
    prefill_names = [
        e["name"] for e in snap.get("engines", []) if e.get("role") == "prefill"
    ]
    for name in prefill_names:
        ops.inject(name, config)
    return prefill_names


def _clear_all_prefill_inject(ops, names: list[str]) -> None:
    for name in names:
        try:
            ops.clear_inject(name)
        except Exception:
            pass


def _anomaly_error_case(
    ctx: CaseContext, inject_config: dict, wait_s: float, require_error_detail: bool
) -> tuple[bool, str]:
    """Shared body of the /inject error family (no_respond, enqueue_error).

    Verdict: the injected fault must surface on the request, the env
    recovers, and — under BATCH delivery, where the explicit cleanup
    Cancel can safely release the master ledger — the inflight ledger
    drains to zero (asserted since the 2026-09 eval batch A; NON_BATCH
    keeps the residue contract, see cancel_anomaly_path).
    """
    ops = ctx.ops()
    rid = ops.next_request_id(rid_base(ctx, "engine_fault"))
    error_observed = False
    error_detail = "no error observed"
    injected_names: list[str] = []
    response = None
    try:
        injected_names = _inject_all_prefill(ops, inject_config)
        try:
            response = ops.schedule(rid)
            if response.code != 200 or not response.success:
                error_observed = True
                error_detail = f"schedule error: {response.error_message}"
            else:
                input_pb = (
                    None
                    if response.enqueued_by_master
                    else ops.build_generate_input(rid)
                )
                handle = ops.start_stream(response, rid, input_pb=input_pb)
                ended = handle.wait_end(wait_s)
                if not ended:
                    error_observed = True
                    error_detail = f"stream timed out (no response within {wait_s}s)"
                if handle.snap.error:
                    error_observed = True
                    error_detail = f"stream error: {handle.snap.error}"
                elif require_error_detail and not handle.snap.completed:
                    error_observed = True
                    error_detail = "stream did not complete"
        except Exception as exc:
            error_observed = True
            error_detail = f"exception: {exc!r}"
        finally:
            _clear_all_prefill_inject(ops, injected_names)

        # Explicitly cancel the failed request to clean up server-side
        # inflight (scheduler keeps the entry until TTL eviction otherwise).
        if response is not None and response.success:
            try:
                ops.cancel(rid, response)
            except Exception:
                pass

        time.sleep(WORKER_RECOVERY_WAIT_S)
        recovery_ok, recovery_msg = ops.verify_recovery()
        batch_delivered = (
            response is not None and response.success and response.enqueued_by_master
        )
        if batch_delivered:
            # Window-insufficient instability fix (no_respond family): the
            # failed request's ledger settle after the explicit cancel can
            # ride the stale-TTL + ExpirationTimer drain (worst ~90s) —
            # the 10s window let a normal slow drain read as a FAIL.
            # Aligned to the TTL_DRAIN_TIMEOUT_S standard; the all-zero
            # assertion itself is unchanged (a true leak still fails).
            inflight_ok, inflight_detail = AssertUtils.inflight_clean(
                _master_http(ops), TTL_DRAIN_TIMEOUT_S
            )
        else:
            # NON_BATCH: see cancel_anomaly_path — client Cancel cannot
            # safely release a delivered ledger entry, so immediate-zero is
            # not asserted.
            inflight_ok, inflight_detail = True, "N/A (NON_BATCH residue contract)"
        # 修复（eval batch A）：BATCH 交付的失败请求在显式 cancel 清理后
        # master 账本必须排空——inflight_ok 升格进 passed（no_respond /
        # enqueue_error 受益）；NON_BATCH 维持 residue contract 不变。
        passed = error_observed and recovery_ok and (not batch_delivered or inflight_ok)
        return passed, (
            f"error_observed={error_observed} ({error_detail}), "
            f"inflight_clean={inflight_ok}({inflight_detail}), "
            f"recovery={recovery_msg}"
        )
    except Exception as exc:
        return False, f"exception: {exc!r}"


# ===========================================================================
# Process-level engine faults
# ===========================================================================


def _fault_spec(ctx: CaseContext) -> EnvSpec:
    """Env for fault cases whose requests die mid-flight (fetch_error,
    crash_after): short staleInflightTimeoutMs (30s vs the na130 default
    300s) so the TTL-cleanup contracts finish within their caps — the
    master ledger entry of an accepted-but-abandoned request is settled
    by the stale-inflight TTL, not by an immediate terminal (formerly
    harness._fault_spec; queueTimeoutMs stays at the functional-profile
    60s; decision/dispatcher axes are the ctx profile's own since the
    tier2 spec unpick).
    """
    return EnvSpec(
        label=f"inject_fault_{ctx.profile}",
        n_prefill=2,
        n_decode=2,
        perf=default_perf(),
        master_profile=ctx.profile,
        config_overrides=ConfigOverride(
            ordering="priority",
            queue_timeout_ms=60_000,
            stale_inflight_ms=30_000,
        ),
    )


# ===========================================================================
# Injected engine faults: error / stall shapes (anomaly E2/E3 + delays)
# ===========================================================================


# ===========================================================================
# Elastic/fault recovery contracts (E1-E6) — expected-behavior assertions
# ===========================================================================
#
# Assertion policy (task E1-E6 mandate): every assertion below states the
# CORRECT contract for engine recovery, never the current behaviour.  A
# failing case is a FINDING on the master (or, where the observation is the
# mock's own self-report, on the mock's restart fidelity) and is recorded,
# not worked around.  Observation surfaces used:
#
#   * master log lifecycle lines — "Created WorkerStatus generation {} for
#     worker: {ipPort}" (INFO, EngineSyncRunner), "worker {ipPort} marked
#     dead after 3 consecutive gRPC failures" (ERROR, GrpcWorkerStatusRunner
#     — the transport-failure retire path), "[remove]/[replace] retiring ..."
#     (INFO — discovery-driven retires);
#   * /rtp_llm/inflight_status per-endpoint ledger (inflight_requests /
#     inflight_batches per prefill ip_port);
#   * /rtp_llm/master/info worker_summary (discovered / alive per role);
#   * routing landing points + mock /snapshot (cache_key_set,
#     kv_tokens_used) for the KV-view contracts.
#
# The six cases share a DEDICATED env (recovery_{profile}, 2P+2D, file
# discovery, fault axes, 30s stale-inflight TTL) so the stop/start/
# injection cycles cannot leak state into — or inherit residue from — the
# shared fault_/kv_ family envs (the task-#87 family-env leakage lesson).


# Retire wait cap: connection-refused failures accumulate one per status
# poll tick (20ms) and the transport retire fires at 3 consecutive
# failures, so the bounded stop in these cases retires within ~1s; the cap
# stays at the fault-family precedent for slow CI machines.
RECOVERY_EVICT_S = 30.0
# Engine restart channel-reconnect settle (engine_down Phase-4 precedent).
RECOVERY_SETTLE_S = 3.0
# E3 crash_after arming window: the crash only fires when a fresh
# EnqueueBatch lands on the armed engine, so trigger requests are fired
# until every target reports stopped.  The master may route several
# triggers at a live/already-crashed peer before one lands on each armed
# target (post-crash dispatch failures also re-route), so the window
# covers a handful of 0.2s trigger rounds.
CRASH_TRIGGER_WINDOW_S = 15.0
# Status-gap shapes: E4 is a 2-tick (2 x 20ms poll) transient gap; E5 must
# exceed the 3-consecutive-failure retire threshold with no_respond's
# per-RPC 1s deadline (fault_env_config statusRpcTimeoutMs=1000) — 5s gives
# >= 4 timed-out polls, comfortably past 3.
E4_GAP_S = 0.045
E5_GAP_S = 5.0
# Post-recovery cache-status poll convergence.  The master's prefill cache
# poll is dynamically intervalled (DynamicCacheIntervalService, default
# 50ms..3000ms) and gated on status ticks, so this window must comfortably
# exceed the 3s ceiling to guarantee the poller has pulled the post-change
# key set (kv.py KV_SYNC_CONVERGENCE_S caliber, widened for the ceiling).
RECOVERY_KV_SYNC_S = 4.5
# The master routes its sync loggers (EngineSyncRunner / GrpcWorkerStatusRunner
# log through the logback "syncLogger") to <flexlb.log.path>/sync.log — by
# default the SHARED ~/ai-whale/logs/sync.log, NOT the per-env stdout capture
# in flexlb_master.log (that one only holds the Spring banner).  The recovery
# env pins the path to a per-env directory so generation/retire observations
# cannot be polluted by sibling runs; every read is an incremental scan from
# a byte offset snapshotted at case start (late async flushes of earlier
# cases then land before the offset only).
SYNC_LOG_ROOT = Path(tempfile.gettempdir()) / "flexlb_ft_sync"


def _recovery_spec(ctx: CaseContext, suffix: str = "") -> EnvSpec:
    """Dedicated E1-E6 env: 2P+2D, dynamic file discovery, PRIORITY
    ordering over the profile's own decision/dispatcher axes
    (profile-aware since the tier2 spec unpick), 30s
    TTL — deliberately a separate label from the shared fault_/kv_ envs.
    A non-empty *suffix* gives a case its OWN env: E2's routing-shape
    assertions (regime A stick / regime B spread) are perturbed by a
    shared env's accumulated soft state (an earlier case's retire storm
    leaves the stormed engine with a routing penalty that skews the
    no-affinity distribution toward the other engine — observed as a
    stable 4/5 bias in the shared-env runs).
    """
    label = f"recovery{suffix}_{ctx.profile}"
    return EnvSpec(
        label=label,
        n_prefill=2,
        n_decode=2,
        perf=fault_env_perf(),
        master_profile=ctx.profile,
        discovery="discovery_file",
        config_overrides=ConfigOverride(
            ordering="priority",
            queue_timeout_ms=OMIT,
        ),
        # Route ALL master logback output (application/sync/flexlb/pv) into
        # a per-env directory — the generation/retire observations below read
        # <dir>/sync.log instead of the shared ~/ai-whale/logs.
        master_extra_args=[f"--flexlb.log.path={SYNC_LOG_ROOT / label}"],
    )


def _recovery_env(ctx: CaseContext, suffix: str = ""):
    env = ctx.env_manager.ensure(_recovery_spec(ctx, suffix))
    return env, ctx.engine_ops(env)


def _engine_ip_port(ops, engine_name: str) -> str:
    """Master-facing address of *engine_name* — the ipPort the master logs
    and keys workerStatus entries by.

    With --unique-engine-ips (harness default on Linux) every engine
    advertises a derived 127.x.y.z loopback host, NOT 127.0.0.1, and the
    master keys/logs by that advertised pair — a hardcoded 127.0.0.1
    needle matches nothing in the sync log or the inflight ledger.  The
    mock /snapshot exposes the real pair as "http_addr" (advertised host
    + http port = grpc port - 1); only when that field is absent (older
    mock build) fall back to the legacy localhost form."""
    snap = ops.snapshot_by_name().get(engine_name, {})
    addr = str(snap.get("http_addr") or "").strip()
    if addr:
        return addr
    grpc_port = int(snap.get("port", 0))
    if grpc_port <= 0:
        raise RuntimeError(f"no grpc port for engine {engine_name}: {snap!r}")
    return f"127.0.0.1:{grpc_port - 1}"


def _sync_log_path(env) -> Path:
    """Per-env sync log (logback syncLogger → <flexlb.log.path>/sync.log)."""
    return SYNC_LOG_ROOT / env.spec.label / "sync.log"


def _master_log_offset(env) -> int:
    """Byte offset of the sync log at case start — every count below scans
    incrementally from here, so residue from earlier cases (and sibling runs)
    cannot leak into the observation."""
    try:
        return _sync_log_path(env).stat().st_size
    except OSError:
        return 0


def _master_log_count(env, needle: str, offset: int = 0) -> int:
    """Count sync-log lines containing *needle* (generation lifecycle
    observations), scanning from *offset* onward.  Returns 0 when the log
    is unavailable."""
    path = _sync_log_path(env)
    if not path.is_file():
        return 0
    count = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            if offset > 0:
                fh.seek(offset)
            for line in fh:
                if needle in line:
                    count += 1
    except OSError:
        return 0
    return count


def _created_generation_count(env, ip_port: str, offset: int = 0) -> int:
    """How many WorkerStatus generations the master has CREATED for *ip_port*
    since *offset* — "Created WorkerStatus generation {} for worker: {ip}"."""
    return _master_log_count(env, f"for worker: {ip_port}", offset)


def _retire_count(env, ip_port: str, offset: int = 0) -> int:
    """Transport-failure retirements logged for *ip_port* since *offset* —
    "worker {ip} marked dead after {} consecutive gRPC failures"."""
    return _master_log_count(env, f"worker {ip_port} marked dead", offset)


def _prefill_endpoint_ledger(ops, ip_port: str) -> Optional[dict]:
    """The /rtp_llm/inflight_status prefill entry for *ip_port*, or None."""
    data = ops.master_inflight()
    if not data:
        return None
    for ep in data.get("prefill_endpoints", []):
        if ep.get("ip_port") == ip_port:
            return ep
    return None


def _recovery_cache_keys(ops, engine_name: str) -> set:
    """Per-engine LRU key set from mock /snapshot (same shape as kv.py's
    _engine_cache_keys — kept local so category modules stay independent)."""
    entry = ops.snapshot_by_name().get(engine_name, {})
    if "cache_key_set" not in entry:
        raise RuntimeError(f"no 'cache_key_set' for engine {engine_name} in /snapshot")
    return set(int(k) for k in entry["cache_key_set"])


def _recovery_cache_evict(ops, engine_name: str, keys) -> None:
    """POST /cache_evict (same endpoint as kv.py — bumps cacheVersion so the
    master's next cache poll re-pulls the key set)."""
    status, body = http_post_json(
        f"http://127.0.0.1:{ops.mock_http_port}/cache_evict",
        {"engine": engine_name, "keys": [int(k) for k in keys]},
    )
    if status != 200:
        raise RuntimeError(
            f"cache_evict({engine_name}, {len(keys)} keys) failed: {status} {body}"
        )


def _fire_inflight(ops, base: int, n: int, **kwargs) -> list:
    """Fire *n* requests without consuming their streams — the master
    ledger entries stay live until consumed/cancelled (S4 drainage lesson:
    unconsumed entries linger and poison later phases).  Returns
    [(rid, response, engine_name)]."""
    fired = []
    for _ in range(n):
        rid = ops.next_request_id(base)
        try:
            resp = ops.schedule(rid, **kwargs)
        except Exception:
            continue
        if resp.code != 200 or not resp.success:
            continue
        addr = ops.role_addr(resp, "PREFILL")
        fired.append((rid, resp, ops.addr_to_name().get(addr, addr)))
    return fired


def _consume_fired(ops, fired: list, wait_s: float = STREAM_TIMEOUT_S) -> list:
    """Consume every fired request to a terminal state (cancel fallback).
    Returns [(rid, engine_name, completed)] for resurrection bookkeeping."""
    outcomes = []
    for rid, resp, name in fired:
        completed = False
        try:
            handle = ops.start_stream(resp, rid)
            ended = handle.wait_end(wait_s)
            completed = bool(ended and handle.snap.completed and not handle.snap.error)
        except Exception:
            completed = False
        if not completed:
            try:
                ops.cancel(rid, resp)
            except Exception:
                pass
        outcomes.append((rid, name, completed))
    return outcomes


def _ensure_started(ops, names) -> None:
    """Restore any stopped engine (env hygiene for the shared recovery env)."""
    try:
        snap = ops.snapshot_by_name()
        for n in names:
            if snap.get(n, {}).get("stopped"):
                ops.start_engine(n)
    except Exception:
        pass
