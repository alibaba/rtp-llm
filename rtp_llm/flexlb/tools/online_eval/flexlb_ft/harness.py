"""FlexLB mock-engine case-test harness.

Terminology (unified 2026-09, suite-reorg task #85): the mock engine
CASE test (场景测试) is this framework — flexlb_functional_tests.py plus
flexlb_ft/ — while the mock engine STRESS test (压测) is the separate
online_eval load pipeline.  The legacy "e2e test" / "chaos test" suite
wording is retired; fault injection is a mechanism inside case tests,
not a suite name.

Single-process test harness that replaces the legacy shell/python smoke
and fault scripts.  Provides:

  * EnvManager   — start/stop the Java mock engine cluster, the FlexLB master
                   (flexlb-api) and standalone victim JVMs, with health waits,
                   port planning, per-spec environment reuse and teardown.
  * ProcessOps   — managed subprocess handles (kill -9, restart, pgrep sweep).
  * ClientOps    — JavaLoadClient driver (all 35 env vars explicit) plus
                   client_events.jsonl parsing.
  * EngineOps    — mock HTTP control-plane + gRPC schedule/cancel/stream
                   (see engine_ops.py).
  * AssertUtils  — wait_for / inflight-clean / TTFT helpers.

Only the Python standard library is used apart from ``grpc`` /
``grpc_tools`` (already required by the legacy smoke tests).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TOOL_DIR = Path(__file__).resolve().parents[1]  # rtp_llm/flexlb/tools/online_eval
FLEXLB_DIR = Path(__file__).resolve().parents[3]  # rtp_llm/flexlb (maven root)
REPO_ROOT = Path(__file__).resolve().parents[5]  # repo root
PROTO_DIR = REPO_ROOT / "rtp_llm" / "cpp" / "model_rpc" / "proto"
MOCK_JAR = (
    FLEXLB_DIR
    / "flexlb-mock-engine"
    / "target"
    / "flexlb-mock-engine-1.0.0-SNAPSHOT-all.jar"
)
API_JAR = FLEXLB_DIR / "flexlb-api" / "target" / "flexlb-api-1.0.0-SNAPSHOT.jar"
MASTER_CONFIG = TOOL_DIR / "data" / "config" / "master_fixed_window.json"
TRACE_FILE = TOOL_DIR / "data" / "online_logs" / "trace_30min.jsonl"

MAVEN_PROFILES = "opensource,!internal"
JAVA_MODULE_OPTS = [
    "--add-modules",
    "ALL-SYSTEM",
    "--add-opens",
    "java.base/java.lang=ALL-UNNAMED",
    "--add-opens",
    "java.base/java.lang.invoke=ALL-UNNAMED",
    "--add-opens",
    "java.base/java.util=ALL-UNNAMED",
    "--add-opens",
    "java.base/java.util.concurrent=ALL-UNNAMED",
    "--add-opens=java.base/jdk.internal.misc=ALL-UNNAMED",
    "--add-opens",
    "java.base/java.nio=ALL-UNNAMED",
    "--add-opens",
    "java.base/sun.nio.ch=ALL-UNNAMED",
    "--add-opens",
    "java.instrument/sun.instrument=ALL-UNNAMED",
]

DEFAULT_MOCK_HEAP = "2g"
DEFAULT_MOCK_EVENT_LOOP_THREADS = 8
DEFAULT_MOCK_COMPLETION_THREADS = 4
DEFAULT_PREFILL_CACHE_BLOCKS = 6000
DEFAULT_DECODE_CACHE_BLOCKS = 3000
# Master ports are env-overridable so sibling framework instances can run
# concurrently without squatting on each other's fixed 18080/18081 (plus the
# gRPC port at http+2). Management defaults to http+1.
DEFAULT_MASTER_HTTP_PORT = int(os.environ.get("FLEXLB_FT_MASTER_HTTP_PORT", "18080"))
DEFAULT_MASTER_MANAGEMENT_PORT = int(
    os.environ.get(
        "FLEXLB_FT_MASTER_MANAGEMENT_PORT", str(DEFAULT_MASTER_HTTP_PORT + 1)
    )
)

# Every env var read by JavaLoadClient.Config.fromEnv().  Exported explicitly
# (unset ones become empty string — JavaLoadClient treats empty as unset) so
# no ambient environment can leak in.  Mirrors lib_load_client.sh.
LOAD_CLIENT_ENV_VARS = [
    "TRACE_FILE",
    "TARGET_ADDR",
    "GRPC_TARGET",
    "DURATION_S",
    "MAX_CONCURRENCY",
    "REPLAY_SPEED",
    "LOAD_CLIENT_WORKERS",
    "OUTPUT_DIR",
    "NUM_SHARDS",
    "SHARD_INDEX",
    "LIMIT",
    "TIMEOUT_MS",
    "SLA_TTFT_MS",
    "ZERO_OUTPUT_POLICY",
    "FETCH_OUTPUT_STREAM",
    "LOOP",
    "N_CHANNELS",
    "EVENT_LOOP_THREADS",
    "START_AT_EPOCH_MS",
    "RESPONSE_TIMEOUT",
    "SKIP_SERVER_LATENCY",
    "MODEL",
    "API_KEY",
    "GRADIENT",
    "GRADIENT_START_SPEED",
    "GRADIENT_MAX_SPEED",
    "MAX_INPUT_LEN",
    "MAX_OUTPUT_LEN",
    "PUSHGATEWAY_URL",
    "ENABLE_FALLBACK",
    "ENDPOINTS_FILE",
    "DRY_RUN",
    "SEND_MODE",
    "SEND_MODE_QPS",
]

# ---------------------------------------------------------------------------
# Perf configs (translated from legacy scripts)
# ---------------------------------------------------------------------------


def default_perf() -> dict:
    """Standard smoke perf config (run_matrix_smoke.sh / run_cancel_smoke.sh).

    Prefill duration is deliberately NOT configured here: the mock engine
    resolves it from the master-config FORMULA expression (or the
    production-fit code default when the estimator is omitted), so mock
    execution time and master routing predictions always share one formula.
    The legacy silent ``prefill.fixed_ms`` fallback was removed.

    Decode timing is likewise NOT configured (task #69): the mock prices
    decode per STEP with the production DSv4 fit — step_ms = 19.5 +
    0.175 x running, 2.6 tokens/step (MTP acceptance fold) — as the code
    default, aligning throughput/queueing economics with production
    (low-batch ~515 tok/s at running=4, full-batch ~7900 tok/s at 128).
    The former explicit ``step_ms_by_batch`` curve approximated the same
    step latencies but without the MTP fold, overstating decode duration
    ~2.6x; it was removed so all flexlb_ft cases run on the production
    caliber. Suites that need custom step pricing still declare
    ``step_ms_by_batch`` / ``step_base_ms`` explicitly.
    """
    return {
        "block_size": 1024,
        "sleep_scale": 1.0,
        "decode": {
            "scale": 1.0,
        },
    }


# ---------------------------------------------------------------------------
# Java resolution
# ---------------------------------------------------------------------------

_java21_cache: Optional[str] = None


def _java_major(java_bin: str) -> int:
    try:
        out = subprocess.run(
            [java_bin, "-version"], capture_output=True, text=True, timeout=30
        ).stderr
    except Exception:
        return 0
    import re as _re

    m = _re.search(r'version "(\d+)', out)
    return int(m.group(1)) if m else 0


def resolve_java21() -> str:
    """Resolve a JDK >= 21 binary path (JAVA_HOME → JAVA21_HOME → ~/java21 → homebrew)."""
    global _java21_cache
    if _java21_cache:
        return _java21_cache
    candidates = []
    for var in ("JAVA_HOME", "JAVA21_HOME"):
        home = os.environ.get(var)
        if home:
            candidates.append(f"{home}/bin/java")
    candidates.append(f"{Path.home()}/java21/bin/java")
    candidates.append("/opt/homebrew/opt/openjdk@21/bin/java")
    for cand in candidates:
        if (
            os.path.isfile(cand)
            and os.access(cand, os.X_OK)
            and _java_major(cand) >= 21
        ):
            _java21_cache = cand
            return cand
    raise RuntimeError(
        "Java 21+ is required (set JAVA_HOME/JAVA21_HOME). Tried: "
        + ", ".join(candidates)
    )


# ---------------------------------------------------------------------------
# HTTP helpers (urllib — no third-party dependency)
# ---------------------------------------------------------------------------


def http_get_json(url: str, timeout: float = 10.0) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def http_get_status(url: str, timeout: float = 10.0) -> int:
    """Return HTTP status code, or 0 on connection failure (master down etc.)."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception:
        return 0


def http_post_json(
    url: str, body: dict, timeout: float = 10.0
) -> tuple[int, Optional[dict]]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(payload)
            except (ValueError, json.JSONDecodeError):
                return resp.status, None
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        try:
            return exc.code, json.loads(payload)
        except (ValueError, json.JSONDecodeError):
            return exc.code, None
    except Exception as exc:
        return 0, {"error": repr(exc)}


# ---------------------------------------------------------------------------
# Wait helpers
# ---------------------------------------------------------------------------


def wait_for_port(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    last_error: Optional[Exception] = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError as exc:
            last_error = exc
            time.sleep(0.5)
    print(f"[harness] timeout waiting for {host}:{port}: {last_error}", file=sys.stderr)
    return False


def wait_for(
    predicate: Callable[[], bool],
    timeout_s: float,
    interval_s: float = 0.5,
    desc: str = "condition",
) -> bool:
    """Poll *predicate* until it returns True or *timeout_s* elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(interval_s)
    return False


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return False
        except OSError:
            return True


# ---------------------------------------------------------------------------
# ProcessOps
# ---------------------------------------------------------------------------


class ManagedProcess:
    """A subprocess started by the harness, restartable from its original argv."""

    def __init__(
        self, proc: subprocess.Popen, argv: list[str], env: dict, log_file: Path
    ):
        self.proc = proc
        self.argv = argv
        self.env = env
        self.log_file = log_file
        self.start_epoch = int(time.time())

    @property
    def pid(self) -> int:
        return self.proc.pid

    def alive(self) -> bool:
        return self.proc.poll() is None

    def wait(self, timeout_s: float = 15.0) -> bool:
        try:
            self.proc.wait(timeout=timeout_s)
            return True
        except subprocess.TimeoutExpired:
            return False

    def terminate(self, timeout_s: float = 10.0) -> None:
        """SIGTERM → wait → SIGKILL fallback."""
        if not self.alive():
            return
        try:
            self.proc.terminate()
        except OSError:
            pass
        if not self.wait(timeout_s):
            self.kill9()
        # Drain zombies.
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass

    def kill9(self) -> None:
        try:
            os.kill(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass

    def tail_log(self, lines: int = 40) -> str:
        try:
            text = self.log_file.read_text(encoding="utf-8", errors="replace")
            return "\n".join(text.splitlines()[-lines:])
        except Exception:
            return "<no log>"


class ProcessOps:
    """Static process utilities (kill by pid / pgrep pattern sweep)."""

    @staticmethod
    def start(
        argv: list[str], env: dict, log_file: Path, cwd: Optional[Path] = None
    ) -> ManagedProcess:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        out = open(log_file, "wb")
        try:
            proc = subprocess.Popen(
                argv,
                stdout=out,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(cwd) if cwd else None,
            )
        finally:
            out.close()
        return ManagedProcess(proc, argv, env, log_file)

    @staticmethod
    def kill9(pid: int) -> None:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    @staticmethod
    def restart(
        mp: ManagedProcess, new_log: Optional[Path] = None, timeout_s: float = 0.0
    ) -> ManagedProcess:
        """Start a fresh process with the same argv/env (old one must be dead)."""
        log = new_log or mp.log_file
        return ProcessOps.start(mp.argv, mp.env, log)


# ---------------------------------------------------------------------------
# EnvManager
# ---------------------------------------------------------------------------


@dataclass
class EnvSpec:
    """Declarative description of a full mock + master environment."""

    label: str = "env"
    n_prefill: int = 2
    n_decode: int = 4
    mock_heap: str = DEFAULT_MOCK_HEAP
    perf: dict = field(default_factory=default_perf)
    # Built-in scheduling profile (PROFILES) or "none" (master not started);
    # the FLEXLB_CONFIG document is generated from the profile axes unless
    # master_env overrides it (chaos/gate suites bring their own config).
    master_profile: str = "batch-window"
    master_env: dict = field(default_factory=dict)  # extra/override env vars
    spring_profile: str = "default"
    master_debug_log: bool = False
    # file (static endpoints.json → env vars, NoOpServiceDiscovery)
    # | domain (explicit DOMAIN_ADDRESS)
    # | discovery_file (dynamic: mock --discovery-file, master FLEXLB_DISCOVERY_FILE
    #   → FileServiceDiscovery; /add_engine + /remove_engine keep it in sync)
    discovery: str = "file"
    domain_addrs: dict = field(default_factory=dict)  # {prefill: "a,b", decode: "a,b"}
    prefill_cache_blocks: int = DEFAULT_PREFILL_CACHE_BLOCKS
    decode_cache_blocks: int = DEFAULT_DECODE_CACHE_BLOCKS
    master_extra_args: list = field(default_factory=list)
    event_loop_threads: int = DEFAULT_MOCK_EVENT_LOOP_THREADS
    completion_threads: int = DEFAULT_MOCK_COMPLETION_THREADS
    # Seconds the freshly started master must hold "alive == discovered" for
    # every role before start_master() returns (0 disables). Skips the
    # cold-start first-connect storm during which healthy engines can be
    # 3-strike-marked dead (CONNECT_TIMEOUT 20ms intake defect).
    master_stable_window_s: float = 3.0

    def fingerprint(self) -> str:
        return json.dumps(
            {
                "n_prefill": self.n_prefill,
                "n_decode": self.n_decode,
                "perf": self.perf,
                "master_profile": self.master_profile,
                "master_env": self.master_env,
                "discovery": self.discovery,
                "domain_addrs": self.domain_addrs,
                "prefill_cache_blocks": self.prefill_cache_blocks,
                "decode_cache_blocks": self.decode_cache_blocks,
                "spring_profile": self.spring_profile,
                "master_stable_window_s": self.master_stable_window_s,
            },
            sort_keys=True,
        )


# ---------------------------------------------------------------------------
# Scheduling profiles (schema-v2 FLEXLB_CONFIG axes)
# ---------------------------------------------------------------------------
#
# v2 exposes four behaviour axes through the single strict FLEXLB_CONFIG
# document: scheduler.type / scheduler.ordering.type /
# scheduler.decision.type / dispatcher.type (see
# rtp_llm/flexlb/docs/priority-scheduler-delivery-modes.md).  The legacy
# v1 env vars (DEFAULT_SCHEDULE_MODE / LOAD_BALANCE_STRATEGY / FLEXLB_BATCH_*)
# have zero consumers in the v2 Java code and are gone; a "profile" is now
# a named axis combination.
#
# Phase-1 profile set (user ruling 2026-08): all QUEUE + FIFO ordering.
# PRIORITY ordering / DIRECT / preemption / selector variants are left for a
# dedicated later phase.

PROFILES = (
    "batch-window",
    "single-nonbatch",
    "single-batch",
    "window-nonbatch",
)

# decision × dispatcher axes per profile (scheduler is QUEUE, ordering FIFO).
PROFILE_SPECS = {
    "batch-window": {"decision": "fixed_window", "dispatcher": "batch"},
    "single-nonbatch": {"decision": "single", "dispatcher": "non_batch"},
    "single-batch": {"decision": "single", "dispatcher": "batch"},
    "window-nonbatch": {"decision": "fixed_window", "dispatcher": "non_batch"},
}

# Semantic capabilities per profile, used by CaseDef.requires filtering
# (e.g. requires=["enqueue_batch"] keeps a case to BATCH-dispatch profiles).
# Capability vocabulary (stable identifiers, extended in later phases):
#   queue / fifo / fixed_window / single
#   batch_dispatch / enqueue_batch / fetch_response   — BATCH dispatcher
#   non_batch_dispatch / frontend_send / generate_stream — NON_BATCH dispatcher
PROFILE_CAPS = {
    "batch-window": {
        "queue",
        "fifo",
        "fixed_window",
        "batch_dispatch",
        "enqueue_batch",
        "fetch_response",
    },
    "single-nonbatch": {
        "queue",
        "fifo",
        "single",
        "non_batch_dispatch",
        "frontend_send",
        "generate_stream",
    },
    "single-batch": {
        "queue",
        "fifo",
        "single",
        "batch_dispatch",
        "enqueue_batch",
        "fetch_response",
    },
    "window-nonbatch": {
        "queue",
        "fifo",
        "fixed_window",
        "non_batch_dispatch",
        "frontend_send",
        "generate_stream",
    },
}


def profile_dispatches_batch(profile: str) -> bool:
    """True when *profile*'s dispatcher axis is BATCH (master sends via
    EnqueueBatch; clients consume FetchResponse)."""
    return PROFILE_SPECS[profile]["dispatcher"] == "batch"


# Production DSv4 prefill execution-time fit, verbatim from the master-side
# RoutingConfig.FormulaEstimatorConfig.DEFAULT_EXPRESSION constant (the
# intake3 test-line default). The harness injects it EXPLICITLY into every
# generated FLEXLB_CONFIG instead of relying on the Java code default: the
# production default is the upstream legacy "1 ms/token" sum, which
# overpredicts a 32k all-miss prefill by ~96x (32.8 s vs the fitted ~342 ms)
# and would poison every ledger-driven routing decision in these suites.
DSV4_PREFILL_EXPRESSION = (
    "max(196, -68.612174288157 + 0.993068319341 * (max(0, 287.3980926717 + 2.30134977837751 *"
    " batchSize + 0.158123254797307 * sum(hitCacheTokens / 1024.) + 0.575522710053703 *"
    " sum(computeTokens / 1024.) + 0.0517623430739831 * sum(computeTokens / 1024. * computeTokens /"
    " 1024.) + 0.0395308136993267 * sum(hitCacheTokens / 1024. * computeTokens / 1024.) +"
    " 0.0104363634681015 * sum(hitCacheTokens / 1024. * hitCacheTokens / 1024.) + 0.575522710053703 *"
    " max(sum(computeTokens / 1024.) - 16, 0) + 2.82077211814514 * max(sum(computeTokens / 1024.) -"
    " 32, 0) - 0.0254671429192862 * max(sum(computeTokens / 1024.) - 64, 0) + 2.15779213792494 *"
    " max(sum(computeTokens / 1024.) - 96, 0) + 0.247806025472364 * max(sum(hitCacheTokens / 1024.) -"
    " 32, 0) - 0.444522654549492 * max(sum(hitCacheTokens / 1024.) - 64, 0) - 0.427317020061895 *"
    " max(sum(hitCacheTokens / 1024.) - 128, 0) + 0.347029077528455 * max(sum(hitCacheTokens / 1024.)"
    " - 256, 0) - 0.298742307762735 * max(sum(hitCacheTokens / 1024.) - 384, 0) + 2.30134977837751 *"
    " max(batchSize - 8, 0) - 3.54884859699154 * max(batchSize - 16, 0) - 11.3438560779984 *"
    " max(batchSize - 24, 0) + 0.879751992138183 * sum(max(computeTokens / 1024. - 2, 0)) +"
    " 0.636364578079591 * sum(max(computeTokens / 1024. - 4, 0)) - 0.0513345988517118 *"
    " sum(max(computeTokens / 1024. - 8, 0)) - 0.332584389129357 * sum(max(hitCacheTokens / 1024. -"
    " 2, 0)) + 0.305819761192588 * sum(max(hitCacheTokens / 1024. - 4, 0)) - 0.287610979974721 *"
    " sum(max(hitCacheTokens / 1024. - 8, 0)) + 0.191310200712013 * sum(max(hitCacheTokens / 1024. -"
    " 12, 0)) + 0.0130251644478961 * max(batchSize - 8, 0) * sum(hitCacheTokens / 1024.) +"
    " 0.00981382840761646 * max(batchSize - 16, 0) * sum(hitCacheTokens / 1024.) - 0.0299132587297009"
    " * max(batchSize - 24, 0) * sum(hitCacheTokens / 1024.) + 0.0447455122487382 * max(batchSize -"
    " 8, 0) * sum(computeTokens / 1024.) + 0.0104635312001851 * max(batchSize - 16, 0) *"
    " sum(computeTokens / 1024.) + 0.0542737877321807 * max(batchSize - 24, 0) * sum(computeTokens /"
    " 1024.))))"
)


def build_flexlb_config(
    *,
    ordering: str = "fifo",  # fifo | priority
    decision: str = "fixed_window",  # fixed_window | single
    dispatcher: str = "batch",  # batch | non_batch
    # scheduler.decision (FIXED_WINDOW only; ignored for SINGLE)
    max_requests: int = 32,
    max_collection_wait_ms: int = 10,
    max_predicted_execution_ms: int = 550,
    # scheduler knobs (None → omit the key, keep the Java default)
    queue_timeout_ms: Optional[int] = None,
    max_outstanding: int = 5_000,
    stale_inflight_ms: int = 30_000,
    delivered_not_accepted_timeout_ms: int = 30_000,
    max_delivered_not_accepted: int = 200,
    # dispatcher knobs
    max_inflight_batches: int = 4,  # BATCH
    enqueue_rpc_timeout_ms: Optional[int] = None,  # BATCH; None → Java default 5000
    max_inflight_requests_per_worker: Optional[
        int
    ] = None,  # NON_BATCH; None → unlimited
    # workerRegistry.health
    status_rpc_ms: int = 1_000,
) -> str:
    """Unified strict schema-v2 FLEXLB_CONFIG generator.

    One template for every environment the framework boots: the four
    built-in profiles (via :func:`flexlb_config_for_profile`), the fault
    families (harness.fault_env_config) and the admission-gate cases
    (harness.admission_config) all delegate here.  The router gets
    the FORMULA execution-time estimator with the production DSv4 fit
    injected EXPLICITLY (:data:`DSV4_PREFILL_EXPRESSION`): the online
    LEARNING estimator only trains from completed EnqueueBatch groups, so a
    stable prediction cap for FIXED_WINDOW + NON_BATCH requires FORMULA —
    and the test line must not depend on the Java code default, which is the
    upstream legacy 1 ms/token expression.
    """
    if decision == "single":
        decision_cfg: dict = {"type": "SINGLE"}
    else:
        decision_cfg = {
            "type": "FIXED_WINDOW",
            "maxRequests": max_requests,
            "maxCollectionWaitMs": max_collection_wait_ms,
            "maxPredictedExecutionMs": max_predicted_execution_ms,
        }
    if dispatcher == "batch":
        dispatcher_cfg: dict = {
            "type": "BATCH",
            "maxInflightBatchesPerPrefillWorker": max_inflight_batches,
        }
        if enqueue_rpc_timeout_ms is not None:
            dispatcher_cfg["enqueueRpcTimeoutMs"] = enqueue_rpc_timeout_ms
    else:
        dispatcher_cfg = {"type": "NON_BATCH"}
        if max_inflight_requests_per_worker is not None:
            dispatcher_cfg["maxInflightRequestsPerPrefillWorker"] = (
                max_inflight_requests_per_worker
            )
    scheduler_cfg: dict = {
        "type": "QUEUE",
        "ordering": {"type": ordering.upper()},
        "decision": decision_cfg,
        "capacity": {"maxOutstandingRequestsGlobal": max_outstanding},
        "lifecycle": {
            "staleInflightTimeoutMs": stale_inflight_ms,
            "deliveredNotAcceptedTimeoutMs": delivered_not_accepted_timeout_ms,
            "maxDeliveredNotAcceptedRequestsGlobal": max_delivered_not_accepted,
        },
    }
    if queue_timeout_ms is not None:
        scheduler_cfg["queueTimeoutMs"] = queue_timeout_ms
    return json.dumps(
        {
            "schemaVersion": 2,
            "scheduler": scheduler_cfg,
            "dispatcher": dispatcher_cfg,
            "router": {
                "availabilityHysteresisPercent": 0,
                "roles": {
                    "prefill": {
                        "availability": {"maxPendingRequests": 100000},
                        # Production DSv4 prefill fit injected explicitly
                        # (see DSV4_PREFILL_EXPRESSION above): the test line
                        # stays on the production-fit caliber regardless of
                        # the Java code default.
                        "executionTimeEstimator": {
                            "type": "FORMULA",
                            "expression": DSV4_PREFILL_EXPRESSION,
                        },
                        # Bounded cache affinity, aligned with the production
                        # master template (data/config/master_fixed_window.json):
                        # a cache leader is preferred while its projected TTFT
                        # stays within maxExtraTtftMs of the best candidate and
                        # its reusable prefix covers >= minPrefixHitPercent.
                        # Without this key the affinity gate is disabled and
                        # prefix reuse degrades to tie-window randomness.
                        "cacheAffinity": {
                            "maxExtraTtftMs": 20,
                            "minPrefixHitPercent": 20,
                        },
                        "selector": {
                            "type": "ESTIMATED_TTFT",
                            "candidateChoice": {
                                "type": "RANDOM_WITHIN_TOLERANCE",
                                "relativeTolerance": 0.1,
                                "minimumToleranceMs": 20,
                                "outlierRejection": {
                                    "maxPendingVsAverageMultiplier": 1.5,
                                    "maxProjectedDrainVsAverageMultiplier": 3.0,
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


def flexlb_config_for_profile(profile: str, **overrides) -> str:
    """FLEXLB_CONFIG for a built-in profile; *overrides* forward to
    :func:`build_flexlb_config` (e.g. window size / TTL tuning)."""
    axes = PROFILE_SPECS[profile]
    kwargs = {
        "ordering": "fifo",
        "decision": axes["decision"],
        "dispatcher": axes["dispatcher"],
        # Queue deadline for functional profiles: tight enough that the
        # queue-timeout gate cases can observe expiry without waiting for
        # the Java default (1h).
        "queue_timeout_ms": 60_000,
    }
    kwargs.update(overrides)
    return build_flexlb_config(**kwargs)


# Master env that is actually consumed by the v2 code:
#   FLEXLB_CONFIG          — set per spec from the profile generator below
#   HIPPO_ROLE             — flexlb-sync (zookeeper elect / LB status)
#   OTEL_TRACE_SKIP_PATTERN — flexlb-api application.yml (spring tracing)
#   OTEL_EXPORTER_OTLP_ENDPOINT — OpenTelemetry SDK exporter ("none" disables)
# Every other legacy v1 var previously exported here had zero consumers in
# the v2 Java code and was removed (task #54 dead-env sweep).
BASE_MASTER_ENV = {
    "OTEL_TRACE_SKIP_PATTERN": ".*",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "none",
}


class FlexEnv:
    """A live environment: one mock cluster + one master (+ optional victims)."""

    def __init__(self, spec: EnvSpec, run_dir: Path, base_grpc_port: int):
        self.spec = spec
        self.run_dir = run_dir
        self.base_grpc_port = base_grpc_port
        self.mock_http_port = base_grpc_port - 1
        self.master_http_port = DEFAULT_MASTER_HTTP_PORT
        self.master_management_port = DEFAULT_MASTER_MANAGEMENT_PORT
        self.endpoint_file = run_dir / "endpoints.json"
        self.discovery_file = run_dir / "discovery.json"  # dynamic file discovery
        self.perf_file = run_dir / "perf.json"
        self.mock: Optional[ManagedProcess] = None
        self.master: Optional[ManagedProcess] = None
        self.victims: dict[str, ManagedProcess] = {}  # name -> process
        self.load_clients: list[ManagedProcess] = []
        self.master_start_count = 0

    # -- addresses ---------------------------------------------------------

    def master_http(self, path: str) -> str:
        return f"http://127.0.0.1:{self.master_http_port}{path}"


class EnvManager:
    """Owns the lifecycle of mock/master/victim JVMs; reuses env per spec."""

    def __init__(self, run_root: Path, keep: bool = False, verbose: bool = True):
        self.run_root = run_root
        self.keep = keep
        self.verbose = verbose
        self.current: Optional[FlexEnv] = None
        self._env_seq = 0

    # -- logging -----------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[env] {msg}", flush=True)

    # -- public API --------------------------------------------------------

    def ensure(self, spec: EnvSpec) -> FlexEnv:
        """Return a live env for *spec*; rebuild only when the spec changed."""
        if (
            self.current is not None
            and self.current.spec.fingerprint() == spec.fingerprint()
        ):
            return self.current
        if self.current is not None:
            self.teardown()
        return self._build(spec)

    def teardown(self) -> None:
        """Stop master → victims → load clients → mock cluster."""
        env = self.current
        if env is None:
            return
        self._log(f"tearing down env '{env.spec.label}'")
        self._stop_env_processes(env)
        if not self.keep:
            pass  # keep run dirs on disk (logs), like legacy scripts
        self.current = None

    def _stop_env_processes(self, env: FlexEnv) -> None:
        """Stop every process owned by *env* (idempotent, env refs cleared).

        Used by teardown() and by the _build() failure path: when a build
        dies mid-way self.current is still None, so the regular teardown()
        would see nothing and leak the JVMs this build already started.
        """
        for mp in env.load_clients:
            mp.terminate()
        env.load_clients.clear()
        for vic in env.victims.values():
            vic.terminate()
        env.victims.clear()
        if env.master is not None:
            env.master.terminate()
            env.master = None
            time.sleep(2)  # mirror stop_master() settle wait
        if env.mock is not None:
            env.mock.terminate()
            env.mock = None
        time.sleep(1)

    # -- env construction --------------------------------------------------

    def _pick_base_grpc_port(self, n_prefill: int, n_decode: int) -> int:
        forced = os.environ.get("FLEXLB_FT_MOCK_BASE_GRPC_PORT")
        if forced:
            return int(forced)
        base = 55151
        for _ in range(12):
            # mock http = base-1; engines base .. base+n-1; victim zone base+149..base+151
            needed = [base - 1] + list(range(base, base + n_prefill + n_decode))
            needed += [base + 149, base + 150, base + 151]
            if not any(port_in_use(p) for p in needed):
                return base
            base += 100
        raise RuntimeError("no free mock port range found")

    def _build(self, spec: EnvSpec) -> FlexEnv:
        self._env_seq += 1
        run_dir = self.run_root / f"env{self._env_seq}_{spec.label}"
        run_dir.mkdir(parents=True, exist_ok=True)
        base = self._pick_base_grpc_port(spec.n_prefill, spec.n_decode)
        env = FlexEnv(spec, run_dir, base)
        self._log(
            f"building env '{spec.label}' ({spec.n_prefill}P+{spec.n_decode}D, "
            f"mock_base={base}, dir={run_dir.name})"
        )

        # perf config
        env.perf_file.write_text(json.dumps(spec.perf, indent=2))

        try:
            # mock cluster
            self._start_mock(env)
            # master
            if spec.master_profile != "none":
                self.start_master(env)
        except Exception:
            # Build failed before self.current was assigned: teardown()
            # would see None and leak the mock/master JVMs already started.
            # Stop the partial env, then re-raise for the caller.
            self._log(f"env '{spec.label}' build failed — stopping partial env")
            self._stop_env_processes(env)
            raise
        self.current = env
        return env

    def _start_mock(self, env: FlexEnv) -> None:
        spec = env.spec
        if not MOCK_JAR.is_file():
            raise RuntimeError(
                f"mock engine jar not found: {MOCK_JAR} (build it first)"
            )
        java = resolve_java21()
        argv = [
            java,
            f"-Xms{spec.mock_heap}",
            f"-Xmx{spec.mock_heap}",
            "-XX:+ExitOnOutOfMemoryError",
            f"-Xlog:gc*,safepoint:{env.run_dir / 'mock_engine_gc.log'}:time,uptime,level,tags:filecount=3,filesize=20m",
            "-jar",
            str(MOCK_JAR),
            "--n-prefill",
            str(spec.n_prefill),
            "--n-decode",
            str(spec.n_decode),
            "--base-grpc-port",
            str(env.base_grpc_port),
            # macOS lo0 only has 127.0.0.1 (no whole 127/8 routing like Linux),
            # so the unique-IP advertisement (127.1.0.x) is unreachable there.
            "--unique-engine-ips",
            "false" if sys.platform == "darwin" else "true",
            "--event-loop-threads",
            str(spec.event_loop_threads),
            "--completion-threads",
            str(spec.completion_threads),
            "--performance",
            str(env.perf_file),
            "--master-config",
            str(MASTER_CONFIG),
            "--prefill-cache-blocks",
            str(spec.prefill_cache_blocks),
            "--decode-cache-blocks",
            str(spec.decode_cache_blocks),
            "--endpoint-file",
            str(env.endpoint_file),
            "--env-file",
            str(env.run_dir / "flexlb_env.txt"),
        ]
        if spec.discovery == "discovery_file":
            # Dynamic file discovery: the mock maintains the domain→hosts
            # mapping (kept in sync by /add_engine + /remove_engine) and the
            # master re-reads it via FLEXLB_DISCOVERY_FILE → FileServiceDiscovery
            # (mirrors run_online_eval.sh FLEXLB_DISCOVERY_FILE=auto wiring).
            argv += ["--discovery-file", str(env.discovery_file)]
        proc = ProcessOps.start(argv, dict(os.environ), env.run_dir / "mock_engine.log")
        env.mock = proc
        if not wait_for_port("127.0.0.1", env.mock_http_port, 60):
            raise RuntimeError(f"mock cluster failed to start:\n{proc.tail_log()}")
        # Wait for the discovery file (max 10s).
        if not wait_for(
            lambda: env.endpoint_file.exists() and env.endpoint_file.stat().st_size > 0,
            10,
            0.1,
        ):
            if not proc.alive():
                raise RuntimeError(f"mock engine exited:\n{proc.tail_log()}")
            raise RuntimeError(
                f"mock engine did not write endpoint file: {env.endpoint_file}"
            )
        if spec.discovery == "discovery_file" and not wait_for(
            lambda: env.discovery_file.exists()
            and env.discovery_file.stat().st_size > 0,
            10,
            0.1,
        ):
            raise RuntimeError(
                f"mock engine did not write discovery file: {env.discovery_file}"
            )
        self._log(f"mock cluster up (pid={proc.pid}, http={env.mock_http_port})")

    # -- master ------------------------------------------------------------

    def _master_env(self, env: FlexEnv) -> dict:
        spec = env.spec
        menv = dict(BASE_MASTER_ENV)
        if spec.master_profile != "none":
            menv["FLEXLB_CONFIG"] = flexlb_config_for_profile(spec.master_profile)
        menv["HIPPO_ROLE"] = f"flexlb_ft_{spec.label}"
        if spec.discovery == "file":
            payload = json.loads(env.endpoint_file.read_text(encoding="utf-8"))
            for key, value in payload.get("env", {}).items():
                menv[key] = str(value)
        elif spec.discovery == "discovery_file":
            # Dynamic file discovery: MODEL_SERVICE_CONFIG (domain endpoints)
            # comes from the mock's endpoint-file env section; host resolution
            # itself goes through FileServiceDiscovery reading the discovery
            # file the mock keeps in sync (DOMAIN_ADDRESS env vars unused).
            payload = json.loads(env.endpoint_file.read_text(encoding="utf-8"))
            service_config = payload.get("env", {}).get("MODEL_SERVICE_CONFIG")
            if service_config:
                menv["MODEL_SERVICE_CONFIG"] = str(service_config)
            menv["FLEXLB_DISCOVERY_FILE"] = str(env.discovery_file)
        elif spec.discovery == "domain":
            menv["MODEL_SERVICE_CONFIG"] = json.dumps(
                {
                    "service_id": "aigc.text-generation.generation.engine_service",
                    "load_balance": True,
                    "role_endpoints": [
                        {
                            "group": "mock",
                            "prefill_endpoint": {
                                "address": "mock.prefill.hosts.address",
                                "protocol": "http",
                                "path": "/",
                            },
                            "decode_endpoint": {
                                "address": "mock.decode.hosts.address",
                                "protocol": "http",
                                "path": "/",
                            },
                        }
                    ],
                },
                separators=(",", ":"),
            )
            menv["DOMAIN_ADDRESS:mock.prefill.hosts.address"] = spec.domain_addrs[
                "prefill"
            ]
            menv["DOMAIN_ADDRESS:mock.decode.hosts.address"] = spec.domain_addrs[
                "decode"
            ]
        menv.update(spec.master_env)  # spec overrides come last
        return menv

    def _master_ports_in_use(self, env: FlexEnv) -> list[int]:
        """Master's fixed ports: HTTP / management / gRPC (= http + 2)."""
        ports = [
            env.master_http_port,
            env.master_management_port,
            env.master_http_port + 2,
        ]
        return [p for p in ports if port_in_use(p)]

    def start_master(
        self, env: FlexEnv, log_name: Optional[str] = None
    ) -> ManagedProcess:
        spec = env.spec
        if not API_JAR.is_file():
            raise RuntimeError(f"flexlb-api jar not found: {API_JAR} (build it first)")
        if env.master is not None:
            raise RuntimeError("master already running; stop it first")
        # Pre-flight: a concurrently running master (sibling framework
        # instance) holds the fixed 18080/18081/18082 ports — wait for release
        # instead of dying on BindException while the readiness probe hits the
        # *foreign* master (mis-detected as "up").
        port_wait_s = float(os.environ.get("FLEXLB_FT_MASTER_PORT_WAIT_S", "120"))
        port_deadline = time.monotonic() + port_wait_s
        while True:
            busy = self._master_ports_in_use(env)
            if not busy:
                break
            if time.monotonic() >= port_deadline:
                raise RuntimeError(
                    f"master ports still busy after {port_wait_s:.0f}s (another "
                    f"master running?): {busy}"
                )
            self._log(f"master ports {busy} busy; waiting for release ...")
            time.sleep(5.0)
        java = resolve_java21()
        env.master_start_count += 1
        log_name = log_name or (
            "flexlb_master.log"
            if env.master_start_count == 1
            else f"flexlb_master_restart{env.master_start_count}.log"
        )
        argv = [
            java,
            *JAVA_MODULE_OPTS,
            "-jar",
            str(API_JAR),
            f"--server.port={env.master_http_port}",
            f"--management.server.port={env.master_management_port}",
            f"--spring.profiles.active={spec.spring_profile}",
        ]
        if spec.master_debug_log:
            argv.append("--logging.level.org.flexlb=DEBUG")
        argv.extend(spec.master_extra_args)
        proc = ProcessOps.start(argv, self._master_env(env), env.run_dir / log_name)
        env.master = proc
        if not wait_for_port("127.0.0.1", env.master_http_port, 90):
            raise RuntimeError(f"master failed to start:\n{proc.tail_log()}")
        # Guard against a foreign master squatting on the HTTP port: if our own
        # JVM died on BindException, the port probe above may still succeed
        # against the foreign process. Re-check our own pid.
        if not proc.alive():
            raise RuntimeError(
                f"master process exited during startup (port conflict?):\n"
                f"{proc.tail_log()}"
            )

        def _master_info() -> Optional[dict]:
            # /rtp_llm/master/info is a POST endpoint (GET returns 405);
            # http_post_json returns (status, payload) tuple.
            status, data = http_post_json(
                f"http://127.0.0.1:{env.master_http_port}/rtp_llm/master/info",
                {},
            )
            return data if status == 200 else None

        if not wait_for(
            lambda: (lambda d: bool(d and d.get("ready")))(_master_info()),
            timeout_s=30,
            interval_s=0.5,
        ):
            raise RuntimeError(
                "master HTTP up but engine sync not ready after 30s "
                "(check ~/ai-whale/logs/flexlb.log)"
            )

        # Stability window: hold "alive == discovered == spec topology" for
        # master_stable_window_s (default 3s) before returning. Skips the
        # cold-start first-connect storm; 0 disables (cold-start probe).
        window_s = spec.master_stable_window_s
        if window_s > 0:

            def _engines_stable() -> bool:
                data = _master_info()
                if not data or not data.get("ready"):
                    return False
                summary = data.get("worker_summary", {}) or {}
                for role, expected in (
                    ("PREFILL", spec.n_prefill),
                    ("DECODE", spec.n_decode),
                ):
                    if expected <= 0:
                        continue
                    entry = summary.get(role) or {}
                    try:
                        discovered = int(entry.get("discovered", -1))
                        alive = int(entry.get("alive", -1))
                    except (TypeError, ValueError):
                        return False
                    if discovered != expected or alive != discovered:
                        return False
                return True

            needed = max(1, int(round(window_s / 0.5)))
            stable_ticks = 0
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if _engines_stable():
                    stable_ticks += 1
                    if stable_ticks >= needed:
                        break
                else:
                    stable_ticks = 0
                time.sleep(0.5)
            if stable_ticks < needed:
                if not proc.alive():
                    raise RuntimeError(f"master exited:\n{proc.tail_log()}")
                raise RuntimeError(
                    f"master engines not stable (alive == discovered for "
                    f"{window_s:.0f}s) within 90s — check "
                    f"~/ai-whale/logs/flexlb.log"
                )
        self._log(
            f"master up (pid={proc.pid}, profile={spec.master_profile}, log={log_name})"
        )
        return proc

    def stop_master(self, env: FlexEnv, settle_s: float = 2.0) -> None:
        if env.master is not None:
            self._log(f"stopping master (pid={env.master.pid})")
            env.master.terminate()
            env.master = None
            time.sleep(settle_s)

    def kill_master9(self, env: FlexEnv) -> None:
        if env.master is not None:
            self._log(f"kill -9 master (pid={env.master.pid})")
            env.master.kill9()
            env.master = None

    # -- victims (engine-kill chaos) --------------------------------------

    def start_victim(
        self,
        env: FlexEnv,
        role: str,
        perf_file: Optional[Path] = None,
        heap: str = "1g",
    ) -> ManagedProcess:
        """Start a standalone single-engine JVM (role: prefill|decode) at base+150."""
        grpc_port = env.base_grpc_port + 150
        http_port = grpc_port - 1
        # Pre-flight: wait briefly for the ports to be released after a kill -9
        # (mirrors the legacy engine-kill script's stale-port check).
        for _ in range(10):
            if not port_in_use(grpc_port) and not port_in_use(http_port):
                break
            time.sleep(0.5)
        endpoint_file = env.run_dir / f"victim_endpoints_{role}.json"
        argv = [
            resolve_java21(),
            f"-Xms{heap}",
            f"-Xmx{heap}",
            "-XX:+ExitOnOutOfMemoryError",
            "-jar",
            str(MOCK_JAR),
            "--n-prefill",
            "1" if role == "prefill" else "0",
            "--n-decode",
            "1" if role == "decode" else "0",
            "--base-grpc-port",
            str(grpc_port),
            "--performance",
            str(perf_file or env.perf_file),
            "--master-config",
            str(MASTER_CONFIG),
            "--prefill-cache-blocks",
            str(env.spec.prefill_cache_blocks if role == "prefill" else 0),
            "--decode-cache-blocks",
            str(env.spec.decode_cache_blocks if role == "decode" else 0),
            "--endpoint-file",
            str(endpoint_file),
        ]
        proc = ProcessOps.start(
            argv, dict(os.environ), env.run_dir / f"victim_{role}.log"
        )
        env.victims[f"victim-{role}"] = proc
        if not wait_for_port("127.0.0.1", http_port, 30):
            raise RuntimeError(f"victim {role} failed to start:\n{proc.tail_log()}")
        if not wait_for(
            lambda: endpoint_file.exists() and endpoint_file.stat().st_size > 0, 10, 0.1
        ):
            raise RuntimeError(f"victim {role} did not write endpoint file")
        self._log(f"victim {role} up (pid={proc.pid}, grpc={grpc_port})")
        return proc

    # -- load client registration (for teardown) ---------------------------


# ---------------------------------------------------------------------------
# ClientOps — JavaLoadClient driver + result parsing
# ---------------------------------------------------------------------------


class LoadClientResult:
    """Raw client output handle.

    Phase B removed summary.json (the client records raw rows only), so the
    derived total/ok/errors summary fields are gone with it — per_request()
    rows are the sole client-side source (no-backward-compat). The underlying
    file is client_events.jsonl (renamed from per_request.jsonl together with
    the multi-component JSONL event streams).
    """

    def __init__(self, output_dir: Path, returncode: int):
        self.output_dir = output_dir
        self.returncode = returncode

    def per_request(self) -> list[dict]:
        path = self.output_dir / "client_events.jsonl"
        rows = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        return rows


def per_request_ttft_p50(rows: list[dict]) -> Optional[float]:
    """Median TTFT over successful rows (index method, as legacy scripts)."""
    values = []
    for row in rows:
        if row.get("ok") is False or row.get("status") == "error":
            continue
        ttft = row.get("ttft_ms")
        if ttft is None:
            continue
        try:
            values.append(float(ttft))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    values.sort()
    return values[int(len(values) * 50 / 100)]


class ClientOps:
    """Drives JavaLoadClient as a subprocess with fully explicit env."""

    def __init__(
        self, env_manager: EnvManager, jvm_xms: str = "4g", jvm_xmx: str = "4g"
    ):
        self.env_manager = env_manager
        self.jvm_xms = jvm_xms
        self.jvm_xmx = jvm_xmx

    def _base_env(self, overrides: dict) -> dict:
        menv = dict(os.environ)
        for var in LOAD_CLIENT_ENV_VARS:
            menv[var] = ""
        for key, value in overrides.items():
            menv[key] = str(value)
        return menv

    def _argv(self) -> list[str]:
        return [
            resolve_java21(),
            f"-Xms{self.jvm_xms}",
            f"-Xmx{self.jvm_xmx}",
            "-cp",
            str(MOCK_JAR),
            "org.flexlb.mockengine.JavaLoadClient",
        ]

    def run(
        self,
        overrides: dict,
        output_dir: Path,
        log_file: Path,
        timeout_s: Optional[float] = None,
        label: str = "load_client",
    ) -> LoadClientResult:
        """Run JavaLoadClient synchronously until it exits."""
        output_dir.mkdir(parents=True, exist_ok=True)
        argv = self._argv()
        menv = self._base_env({**overrides, "OUTPUT_DIR": str(output_dir)})
        proc = ProcessOps.start(argv, menv, log_file)
        try:
            proc.proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill9()
        rc = proc.proc.returncode if proc.proc.returncode is not None else -1
        return LoadClientResult(output_dir, rc)


# ---------------------------------------------------------------------------
# TTL-settle drain window (shared by every case whose leaked inflight slots
# settle via the master's stale-inflight TTL path)
# ---------------------------------------------------------------------------

# Worst-case master-side TTL settle: staleInflightTimeoutMs (30s) + the
# ExpirationTimer @Scheduled(60s) sweep + 5s margin.  A leaked slot's TTL
# expires at t+30s, but the sweeper only visits on its 60s period, so with
# worst-phase alignment the ledger entry survives until ~t+90s after its
# last touch.  Drain windows shorter than this (the legacy TTL+margin=60s
# or the 90s caps) let the residue bleed into the NEXT case on the shared
# env — the integration-round cascade (2026-09-01 task #87: 16 false
# FAILs with scheduler=4/8/5 constant residue and an all-zero engine
# side; every affected case was solo-PASS on a clean env).  Waiting longer
# is "wait for the settle", NOT a weaker assertion: the target stays
# all-zero and a true leak (a slot that is never released) still times
# out and fails the caller.
TTL_DRAIN_TIMEOUT_S = 95.0


# ---------------------------------------------------------------------------
# AssertUtils
# ---------------------------------------------------------------------------


class AssertUtils:
    """Shared assertion helpers used by both suites."""

    @staticmethod
    def inflight_clean(master_http: str, timeout_s: float = 10.0) -> tuple[bool, str]:
        """All-zero master inflight (scheduler + every endpoint)."""
        detail = "no response yet"
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            data = http_get_json(f"{master_http}/rtp_llm/inflight_status", timeout=5)
            if data is not None:
                sched = data.get("scheduler_inflight", 0)
                prefill_eps = data.get("prefill_endpoints", []) or []
                decode_eps = data.get("decode_endpoints", []) or []
                prefill_clean = all(
                    ep.get("inflight_batches", 0) == 0 for ep in prefill_eps
                )

                # decode_endpoints no longer emit the legacy inflight_requests
                # key (see HttpLoadBalanceServer.inflightStatus): fall back to
                # total_load (= confirmedEngineOwnedCount + inflight) so the
                # decode leg still observes a live field instead of a missing
                # key that always reads 0.
                def _decode_inflight(ep: dict) -> int:
                    return ep.get("inflight_requests", 0) or ep.get("total_load", 0)

                decode_clean = all(_decode_inflight(ep) == 0 for ep in decode_eps)
                if sched == 0 and prefill_clean and decode_clean:
                    return True, "all inflight zero"
                detail = (
                    f"scheduler={sched}, "
                    f"prefill={[(ep.get('ip_port'), ep.get('inflight_batches', 0)) for ep in prefill_eps]}, "
                    f"decode={[(ep.get('ip_port'), _decode_inflight(ep)) for ep in decode_eps]}"
                )
            time.sleep(0.5)
        return False, f"timeout waiting for inflight clean: {detail}"

    @staticmethod
    def ttft_degradation(
        base_p50: Optional[float], new_p50: Optional[float], threshold_pct: float = 50.0
    ) -> tuple[bool, str]:
        """new_p50 must not exceed base_p50 by more than threshold_pct %."""
        if base_p50 is None or new_p50 is None:
            return True, "ttft baseline unavailable — skipped comparison"
        degradation = (new_p50 - base_p50) / base_p50 * 100 if base_p50 > 0 else 0.0
        ok = degradation <= threshold_pct
        return ok, (
            f"ttft p50 {base_p50:.1f} → {new_p50:.1f} ms "
            f"({degradation:+.1f}%, limit {threshold_pct:.0f}%)"
        )


# ---------------------------------------------------------------------------
# Proto bootstrap (translated from online_eval/proto_utils.py)
# ---------------------------------------------------------------------------

_PROTO_CACHE: dict = {}


def _proto_out_dir() -> Path:
    out = os.environ.get("FLEXLB_EVAL_PROTO_OUT")
    if out:
        return Path(out)
    tmp = os.environ.get("TMPDIR") or tempfile.gettempdir()
    return Path(tmp) / "flexlb_eval_proto"


def _generate_proto(proto_name: str) -> tuple:
    out_dir = _proto_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    src = PROTO_DIR / proto_name
    if not src.is_file():
        raise FileNotFoundError(f"proto source not found: {src}")
    py_name = proto_name.replace(".proto", "_pb2.py")
    grpc_name = proto_name.replace(".proto", "_pb2_grpc.py")
    need_regen = True
    if (out_dir / py_name).is_file() and (out_dir / grpc_name).is_file():
        need_regen = src.stat().st_mtime > (out_dir / py_name).stat().st_mtime
    if need_regen:
        cmd = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={out_dir}",
            f"--grpc_python_out={out_dir}",
            proto_name,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    import importlib

    base = proto_name.replace(".proto", "")
    pb2 = importlib.import_module(f"{base}_pb2")
    pb2_grpc = importlib.import_module(f"{base}_pb2_grpc")
    return pb2, pb2_grpc


def ensure_proto_modules() -> tuple:
    """Engine-side protos (rpc_service.proto)."""
    if "rpc" not in _PROTO_CACHE:
        _PROTO_CACHE["rpc"] = _generate_proto("model_rpc_service.proto")
    return _PROTO_CACHE["rpc"]


def ensure_schedule_proto_modules() -> tuple:
    """Master-side protos (flexlb_service.proto)."""
    if "schedule" not in _PROTO_CACHE:
        _PROTO_CACHE["schedule"] = _generate_proto("flexlb_schedule_service.proto")
    return _PROTO_CACHE["schedule"]


def encode_unique_key(meta: dict) -> str:
    return "flexlb_eval:" + json.dumps(meta, separators=(",", ":"))


# ===========================================================================
# Shared case-level helpers (suite-reorg task #85)
#
# Environment constructors, traffic pumps and topology observers that used
# to live in chaos_cases.py / injection_gate_cases.py and are shared by
# several flexlb_ft/cases/ categories (elastic, master, engine_fault,
# status, admission).  They depend only on this module (EnvSpec /
# build_flexlb_config / default_perf / wait_for) plus duck-typed CaseContext
# / EngineOps arguments (annotated as strings to avoid an import cycle:
# context.py imports harness.py), so harness.py is their shared home.
# ===========================================================================

# Shared stream timeout for the fault-family helpers below (the per-category
# case modules declare their own same-valued constant for case bodies).
STREAM_TIMEOUT_S = 15.0

PREFILL_DOMAIN = "mock.prefill.hosts.address"
DECODE_DOMAIN = "mock.decode.hosts.address"


def fault_env_config(
    stale_inflight_ms: int = 30_000,
    max_inflight_batches: int = 4,
    status_rpc_ms: int = 1_000,
) -> str:
    """Fault-family FLEXLB_CONFIG (QUEUE + PRIORITY + FIXED_WINDOW + BATCH —
    the legacy fault axes), generated through the unified
    harness.build_flexlb_config template.

    Shorter staleInflightTimeoutMs (30s vs the na130 default 300s) so the TTL
    cleanup cases finish within their 90s caps.
    """
    return build_flexlb_config(
        ordering="priority",
        decision="fixed_window",
        dispatcher="batch",
        stale_inflight_ms=stale_inflight_ms,
        max_inflight_batches=max_inflight_batches,
        status_rpc_ms=status_rpc_ms,
    )


def fault_env_perf() -> dict:
    """Fault/elastic perf: an EXPLICIT flat prefill (performance JSON
    "prefill.fixed_ms" — the sanctioned explicit channel, cluster-wide so
    dynamically added engines inherit it too).

    The elastic cases measure discovery/removal/inflight semantics, not
    ledger-driven routing. The formula-driven default (~220ms at 2048
    tokens) doubles the in-flight window at removal time and amplifies a
    pre-existing intermittent drain stall (requests orphaned by the removed
    engine never complete decode) — the flat 100ms declaration restores the
    historical timing envelope this family was calibrated on.
    """
    perf = default_perf()
    perf["prefill"] = {"fixed_ms": 100.0, "scale": 1.0}
    return perf


def admission_config(
    queue_timeout_ms: int = 60_000,
    max_outstanding: int = 5_000,
    stale_inflight_ms: int = 30_000,
) -> str:
    """FLEXLB_CONFIG for the admission-gate cases: the legacy fault axes
    (QUEUE + PRIORITY + FIXED_WINDOW + BATCH) via the unified
    harness.build_flexlb_config template, with the admission knobs
    parameterised."""
    return build_flexlb_config(
        ordering="priority",
        decision="fixed_window",
        dispatcher="batch",
        queue_timeout_ms=queue_timeout_ms,
        max_outstanding=max_outstanding,
        stale_inflight_ms=stale_inflight_ms,
    )


def elastic_spec(ctx: "CaseContext") -> EnvSpec:
    """Shared elastic/fault env: 2P+4D, dynamic file discovery, TTL=30s."""
    return EnvSpec(
        label=f"fault_{ctx.profile}",
        n_prefill=2,
        n_decode=4,
        perf=fault_env_perf(),
        master_profile=ctx.profile,
        discovery="discovery_file",
        master_env={"FLEXLB_CONFIG": fault_env_config()},
    )


def ttl_spec(ctx: "CaseContext") -> EnvSpec:
    """Inflight-TTL env (S1): 2P+2D, TTL=30s."""
    return EnvSpec(
        label=f"fault_ttl_{ctx.profile}",
        n_prefill=2,
        n_decode=2,
        perf=fault_env_perf(),
        master_profile=ctx.profile,
        discovery="discovery_file",
        master_env={"FLEXLB_CONFIG": fault_env_config()},
    )


def quota_spec(ctx: "CaseContext") -> EnvSpec:
    """Quota-block env (S3): 1P+1D, maxInflightBatches=1 via FLEXLB_CONFIG
    (dispatcher.maxInflightBatchesPerPrefillWorker — the v1 env var
    FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES has no v2 consumer)."""
    return EnvSpec(
        label=f"fault_quota_{ctx.profile}",
        n_prefill=1,
        n_decode=1,
        perf=fault_env_perf(),
        master_profile=ctx.profile,
        discovery="discovery_file",
        master_env={"FLEXLB_CONFIG": fault_env_config(max_inflight_batches=1)},
    )


def coldstart_spec(ctx: "CaseContext") -> EnvSpec:
    """Cold-start probe env: mirrors the default topology (2P+4D, static
    file discovery, default config) but disables the master stability
    window so traffic hits the master during the first-connect storm."""
    return EnvSpec(
        label=f"fault_coldstart_{ctx.profile}",
        n_prefill=2,
        n_decode=4,
        perf=default_perf(),
        master_profile=ctx.profile,
        master_stable_window_s=0.0,
    )


def _fault_spec(ctx: "CaseContext") -> EnvSpec:
    """Env for fault cases whose requests die mid-flight (fetch_error,
    crash_after): short staleInflightTimeoutMs (30s vs the na130 default
    300s) because the VERIFIED contract is that a request already
    accepted by an engine but whose client stream dies is cleaned by the
    stale-inflight TTL, not by an immediate terminal (engine-side inflight
    DOES drain immediately; the master ledger entry lingers).  With the
    default env's 300s TTL the case would have to wait 5 minutes."""
    return EnvSpec(
        label=f"inject_fault_{ctx.profile}",
        n_prefill=2,
        n_decode=2,
        perf=default_perf(),
        master_profile=ctx.profile,
        master_env={"FLEXLB_CONFIG": admission_config(stale_inflight_ms=30_000)},
    )


def _elastic_env(ctx: "CaseContext"):
    env = ctx.env_manager.ensure(elastic_spec(ctx))
    return env, ctx.engine_ops(env)


def _initial_engine_names(env) -> set:
    return {f"prefill-{i}" for i in range(env.spec.n_prefill)} | {
        f"decode-{i}" for i in range(env.spec.n_decode)
    }


def _dynamic_engines(ops, env) -> list:
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


def _discovery_entry_count(env) -> tuple:
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
) -> tuple:
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
            target=self._loop, name="fault-flow", daemon=True
        )
        self._thread.start()

    def stop(self, timeout_s: float = 20.0) -> tuple:
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
