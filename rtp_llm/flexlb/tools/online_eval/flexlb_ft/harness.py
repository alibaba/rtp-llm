"""FlexLB functional + chaos e2e harness.

Single-process test harness that replaces the legacy shell/python smoke and
chaos scripts.  Provides:

  * EnvManager   — start/stop the Java mock engine cluster, the FlexLB master
                   (flexlb-api) and standalone victim JVMs, with health waits,
                   port planning, per-spec environment reuse and teardown.
  * ProcessOps   — managed subprocess handles (kill -9, restart, pgrep sweep).
  * ClientOps    — JavaLoadClient driver (all 36 env vars explicit) plus
                   summary.json / per_request.jsonl parsing.
  * EngineOps    — mock HTTP control-plane + gRPC schedule/cancel/stream
                   (see engine_ops.py).
  * AssertUtils  — wait_for / inflight-clean / recovery-rate / TTFT helpers.

Only the Python standard library is used apart from ``grpc`` /
``grpc_tools`` (already required by the legacy smoke tests).
"""

from __future__ import annotations

import atexit
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
DEFAULT_MASTER_HTTP_PORT = 18080
DEFAULT_MASTER_MANAGEMENT_PORT = 18081

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
    "SCHEDULE_ONLY",
    "LOOP",
    "N_CHANNELS",
    "EVENT_LOOP_THREADS",
    "START_AT_EPOCH_MS",
    "RESPONSE_TIMEOUT",
    "SKIP_SERVER_LATENCY",
    "MODEL",
    "API_KEY",
    "FLEXLB_EXPECT_FETCH_RESPONSE",
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
    """Standard smoke perf config (run_matrix_smoke.sh / run_cancel_smoke.sh)."""
    return {
        "block_size": 1024,
        "sleep_scale": 1.0,
        "prefill": {"fixed_ms": 100.0, "scale": 1.0},
        "decode": {
            "scale": 1.0,
            "step_ms_by_batch": [
                [1, 20.0],
                [2, 22.0],
                [4, 25.0],
                [8, 28.0],
                [16, 30.0],
                [32, 35.0],
                [64, 40.0],
                [128, 45.0],
                [256, 50.0],
            ],
        },
    }


def flat_perf(prefill_ms: float = 1000.0, decode_ms: float = 100.0) -> dict:
    """Chaos perf config (engine_disconnect / master_recovery): flat latencies."""
    steps = [[n, decode_ms] for n in (1, 2, 4, 8, 16, 32, 64, 128, 256)]
    return {
        "block_size": 1024,
        "sleep_scale": 1.0,
        "prefill": {"fixed_ms": prefill_ms, "scale": 1.0},
        "decode": {"scale": 1.0, "step_ms_by_batch": steps},
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


def http_save(url: str, path: Path, timeout: float = 10.0) -> bool:
    """GET a URL and save the body to *path* (returns False on failure)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            path.write_bytes(resp.read())
        return True
    except Exception:
        return False


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
    def pids_by_pattern(pattern: str) -> list[int]:
        """pgrep -f pattern (own pid excluded)."""
        try:
            out = subprocess.run(
                ["pgrep", "-f", pattern], capture_output=True, text=True, timeout=10
            ).stdout
            return [int(p) for p in out.split() if p.strip().isdigit()]
        except Exception:
            return []

    @staticmethod
    def kill9_pattern(pattern: str) -> int:
        """Kill -9 every pid matching *pattern* (except ourselves). Returns count."""
        count = 0
        for pid in ProcessOps.pids_by_pattern(pattern):
            if pid == os.getpid():
                continue
            ProcessOps.kill9(pid)
            count += 1
        return count

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
    master_mode: str = "batch"  # batch | direct | queue | custom | none
    master_env: dict = field(default_factory=dict)  # extra/override env vars
    spring_profile: str = "default"
    master_debug_log: bool = False
    discovery: str = "file"  # file (endpoints.json) | domain (DOMAIN_ADDRESS)
    domain_addrs: dict = field(default_factory=dict)  # {prefill: "a,b", decode: "a,b"}
    prefill_cache_blocks: int = DEFAULT_PREFILL_CACHE_BLOCKS
    decode_cache_blocks: int = DEFAULT_DECODE_CACHE_BLOCKS
    master_extra_args: list = field(default_factory=list)
    event_loop_threads: int = DEFAULT_MOCK_EVENT_LOOP_THREADS
    completion_threads: int = DEFAULT_MOCK_COMPLETION_THREADS

    def fingerprint(self) -> str:
        return json.dumps(
            {
                "n_prefill": self.n_prefill,
                "n_decode": self.n_decode,
                "perf": self.perf,
                "master_mode": self.master_mode,
                "master_env": self.master_env,
                "discovery": self.discovery,
                "domain_addrs": self.domain_addrs,
                "prefill_cache_blocks": self.prefill_cache_blocks,
                "decode_cache_blocks": self.decode_cache_blocks,
                "spring_profile": self.spring_profile,
            },
            sort_keys=True,
        )


MODE_STRATEGY = {
    "batch": {
        "LOAD_BALANCE_STRATEGY": "COST_BASED_PREFILL",
        "DEFAULT_SCHEDULE_MODE": "BATCH",
    },
    "direct": {
        "LOAD_BALANCE_STRATEGY": "SHORTEST_TTFT",
        "DEFAULT_SCHEDULE_MODE": "DIRECT",
    },
    "queue": {
        "LOAD_BALANCE_STRATEGY": "SHORTEST_TTFT",
        "DEFAULT_SCHEDULE_MODE": "QUEUE",
    },
}

DEFAULT_FLEXLB_CONFIG = json.dumps(
    {
        "schemaVersion": 2,
        "scheduler": {
            "type": "QUEUE",
            "ordering": {"type": "PRIORITY", "defaultPriority": 50},
            "decision": {
                "type": "FIXED_WINDOW",
                "maxRequests": 32,
                "maxCollectionWaitMs": 10,
                "maxPredictedExecutionMs": 550,
            },
            "capacity": {
                "maxOutstandingRequestsGlobal": 1000000,
                "maxWaitingRequestsPerPrefillWorker": 1024,
            },
        },
        "dispatcher": {"type": "BATCH", "enqueueRpcTimeoutMs": 5000},
        "router": {
            "availabilityHysteresisPercent": 30,
            "roles": {
                "prefill": {
                    "availability": {"maxPendingRequests": 100000},
                    "executionTimeEstimator": {"type": "FORMULA"},
                    "selector": {
                        "type": "ESTIMATED_TTFT",
                        "candidateChoice": {"type": "RANDOM_WITHIN_TOLERANCE"},
                    },
                },
                "decode": {
                    "availability": {"maxKvUsagePercent": 90, "maxEngineRequests": 132},
                    "kvReservation": {"maxOutputTokensForEstimate": 1000},
                    "selector": {"type": "KV_USAGE_WEIGHTED_RANDOM"},
                },
                "vit": {"selector": {"type": "RANDOM"}},
            },
        },
        "observability": {
            "cacheHit": {
                "recentKeyWindow": {
                    "writeEnabled": True,
                    "durationMs": 1800000,
                    "maxKeyOccurrences": 10000000,
                },
                "metricsEnabled": True,
                "requestTraceLogEnabled": False,
            }
        },
    }
)

BASE_MASTER_ENV = {
    "FLEXLB_CONFIG": DEFAULT_FLEXLB_CONFIG,
    "DECODE_LOAD_BALANCE_STRATEGY": "COST_BASED_DECODE",
    "DECODE_CONCURRENCY_LIMIT": "132",
    "FLEXLB_BATCH_ALGORITHM": "fixed_window",
    "FLEXLB_BATCH_FIXED_WAIT_MS": "10",
    "FLEXLB_BATCH_PREDICT_THRESHOLD_MS": "550",
    "FLEXLB_BATCH_SIZE_MAX": "32",
    "FLEXLB_BATCH_MIN_SIZE": "1",
    "FLEXLB_BATCH_FIXED_MAX_INFLIGHT_BATCHES": "4",
    "HYSTERESIS_BIAS_PERCENT": "0",
    "MAX_QUEUE_SIZE": "5000",
    "PREFILL_QUEUE_SIZE_THRESHOLD": "100000",
    "COST_SLO_MS": "30000",
    "COST_HOTSPOT_MULTIPLIER": "1.5",
    "STRATEGY_CONFIGS": "{}",
    "OTEL_TRACE_SKIP_PATTERN": ".*",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "none",
    "FLEXLB_EXPECT_FETCH_RESPONSE": "true",
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
        self.perf_file = run_dir / "perf.json"
        self.mock: Optional[ManagedProcess] = None
        self.master: Optional[ManagedProcess] = None
        self.victims: dict[str, ManagedProcess] = {}  # name -> process
        self.load_clients: list[ManagedProcess] = []
        self.master_start_count = 0

    # -- addresses ---------------------------------------------------------

    def grpc_port_of(self, role: str, index: int) -> int:
        """Port layout mirrors JavaMockEngineCluster: prefill engines first."""
        if role == "prefill":
            return self.base_grpc_port + index
        return self.base_grpc_port + self.spec.n_prefill + index

    def mock_http(self, path: str) -> str:
        return f"http://127.0.0.1:{self.mock_http_port}{path}"

    def master_http(self, path: str) -> str:
        return f"http://127.0.0.1:{self.master_http_port}{path}"


class EnvManager:
    """Owns the lifecycle of mock/master/victim JVMs; reuses env per spec."""

    def __init__(self, run_root: Path, keep: bool = False, verbose: bool = True):
        self.run_root = run_root
        self.keep = keep
        self.verbose = verbose
        self.current: Optional[FlexEnv] = None
        self._registered = False
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
        if not self.keep:
            pass  # keep run dirs on disk (logs), like legacy scripts
        self.current = None

    def register_atexit(self) -> None:
        if not self._registered:
            atexit.register(self._atexit_teardown)
            self._registered = True

    def _atexit_teardown(self) -> None:
        try:
            self.teardown()
        except Exception:
            pass

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

        # mock cluster
        self._start_mock(env)
        # master
        if spec.master_mode != "none":
            self.start_master(env)
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
        self._log(f"mock cluster up (pid={proc.pid}, http={env.mock_http_port})")

    # -- master ------------------------------------------------------------

    def _master_env(self, env: FlexEnv) -> dict:
        spec = env.spec
        menv = dict(BASE_MASTER_ENV)
        if spec.master_mode in MODE_STRATEGY:
            menv.update(MODE_STRATEGY[spec.master_mode])
        menv["HIPPO_ROLE"] = f"flexlb_ft_{spec.label}"
        if spec.discovery == "file":
            payload = json.loads(env.endpoint_file.read_text(encoding="utf-8"))
            for key, value in payload.get("env", {}).items():
                menv[key] = str(value)
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

    def start_master(
        self, env: FlexEnv, log_name: Optional[str] = None
    ) -> ManagedProcess:
        spec = env.spec
        if not API_JAR.is_file():
            raise RuntimeError(f"flexlb-api jar not found: {API_JAR} (build it first)")
        if env.master is not None:
            raise RuntimeError("master already running; stop it first")
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
        self._log(
            f"master up (pid={proc.pid}, mode={spec.master_mode}, log={log_name})"
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

    def restart_victim(
        self, env: FlexEnv, role: str, perf_file: Optional[Path] = None
    ) -> ManagedProcess:
        key = f"victim-{role}"
        old = env.victims.pop(key, None)
        if old is not None:
            old.kill9() if not old.alive() else old.terminate()
        log = env.run_dir / f"victim_{role}_restart.log"
        proc = self.start_victim(env, role, perf_file=perf_file)
        # keep the same log naming scheme as start_victim for simplicity
        return proc

    # -- load client registration (for teardown) ---------------------------

    def track_load_client(self, env: FlexEnv, mp: ManagedProcess) -> None:
        env.load_clients.append(mp)


# ---------------------------------------------------------------------------
# ClientOps — JavaLoadClient driver + result parsing
# ---------------------------------------------------------------------------


class LoadClientResult:
    def __init__(self, output_dir: Path, returncode: int):
        self.output_dir = output_dir
        self.returncode = returncode
        self.summary: Optional[dict] = None
        summary_file = output_dir / "summary.json"
        if summary_file.is_file():
            try:
                self.summary = json.loads(summary_file.read_text(encoding="utf-8"))
            except Exception:
                self.summary = None

    @property
    def total(self) -> int:
        if not self.summary:
            return 0
        for key in ("total_requests", "total", "requests"):
            if key in self.summary:
                return int(self.summary[key])
        return 0

    @property
    def ok(self) -> int:
        if not self.summary:
            return 0
        for key in ("completed", "ok", "success"):
            if key in self.summary:
                return int(self.summary[key])
        return 0

    @property
    def errors(self) -> int:
        if not self.summary:
            return 0
        for key in ("errors", "error", "failed"):
            if key in self.summary:
                return int(self.summary[key])
        return 0

    @property
    def success_rate(self) -> float:
        return (self.ok / self.total) if self.total else 0.0

    def ttft_p50(self) -> Optional[float]:
        if not self.summary:
            return None
        latency = self.summary.get("latency") or {}
        ttft = latency.get("ttft_ms") or {}
        if "p50" in ttft:
            try:
                return float(ttft["p50"])
            except (TypeError, ValueError):
                return None
        return None

    def per_request(self) -> list[dict]:
        path = self.output_dir / "per_request.jsonl"
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

    def describe(self) -> str:
        if self.summary is None:
            return f"no summary (rc={self.returncode})"
        return (
            f"total={self.total} ok={self.ok} errors={self.errors} "
            f"ttft_p50={self.ttft_p50()}"
        )


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

    def start_background(
        self, overrides: dict, output_dir: Path, log_file: Path
    ) -> ManagedProcess:
        """Start JavaLoadClient in the background (caller stops it later)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        argv = self._argv()
        menv = self._base_env({**overrides, "OUTPUT_DIR": str(output_dir)})
        proc = ProcessOps.start(argv, menv, log_file)
        return proc


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
                decode_clean = all(
                    ep.get("inflight_requests", 0) == 0 for ep in decode_eps
                )
                if sched == 0 and prefill_clean and decode_clean:
                    return True, "all inflight zero"
                detail = (
                    f"scheduler={sched}, "
                    f"prefill={[(ep.get('ip_port'), ep.get('inflight_batches', 0)) for ep in prefill_eps]}, "
                    f"decode={[(ep.get('ip_port'), ep.get('inflight_requests', 0)) for ep in decode_eps]}"
                )
            time.sleep(0.5)
        return False, f"timeout waiting for inflight clean: {detail}"

    @staticmethod
    def recovery_rate(result: LoadClientResult) -> tuple[bool, str]:
        """>= 95% success rate with at least one request (legacy chaos rule)."""
        if result.total == 0:
            return False, "recovery verification sent 0 requests"
        rate = result.ok / result.total
        return rate >= 0.95, f"recovery {result.ok}/{result.total} ({rate:.1%})"

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


def filter_trace(
    src: Path,
    dst: Path,
    max_ol: int,
    max_lines: Optional[int] = None,
    tag: Optional[str] = None,
    tag_field: str = "tag",
) -> int:
    """Filter a trace file to rows with ol <= max_ol (optionally cap lines /
    annotate each row so request ids stay unique across reruns — the legacy
    engine-kill script injects a ``_rt`` field, smoke scripts use ``tag``)."""
    count = 0
    with open(src, "r", encoding="utf-8") as fin, open(
        dst, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if max_lines is not None and count >= max_lines:
                break
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("ol", 0) > max_ol:
                continue
            if tag is not None:
                rec[tag_field] = tag
                line = json.dumps(rec)
            fout.write(line if line.endswith("\n") else line + "\n")
            count += 1
    return count
