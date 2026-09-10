"""FlexLB mock-engine case-test harness.

EnvManager starts, reuses and tears down Java mock clusters and Masters.
ProcessOps owns subprocess handles; ClientOps drives JavaLoadClient and reads
its request events. EngineOps provides mock control-plane and gRPC operations
from engine_ops.py. Shared helpers provide bounded waits and observations.

Functional cases run through scenario_runner.py. Performance traffic and
metric collection live in the separate stress/ pipeline.

Uses the Python standard library plus grpc and grpc_tools.
"""

from __future__ import annotations

import gzip
import json
import os
import shlex
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
from dataclasses import dataclass, field, fields
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
TRACE_FILE = TOOL_DIR / "data" / "online_logs" / "trace_30min.jsonl"

# ---------------------------------------------------------------------------
# FLEXLB_CONFIG SSOT — flexlb_cfg (re-exported)
# ---------------------------------------------------------------------------
#
# flexlb_cfg.py (one directory up, in online_eval/) is the single source
# of truth for every FLEXLB_CONFIG document this repo produces: the four
# functional profile axes, the stress-na130 render profile (the retired
# data/config/master_fixed_window.json), the strict schema-v2 builders
# and the ConfigOverride layering.  It lives on sys.path in every
# supported entrypoint (scenario_runner.py, the tests/, the
# run_online_eval.sh generator call) — the insert below makes the bare
# import robust regardless of how flexlb_test_framework.harness itself was imported.
#
# Re-exported for the scenario runner, backend and action implementations.
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from flexlb_cfg import (  # noqa: E402
    OMIT,
    PROFILE_CAPS,
    PROFILE_SPECS,
    PROFILES,
    STRESS_PROFILE,
    ConfigOverride,
    build_flexlb_config,
    parse_overrides,
    profile_dispatches_batch,
    render_env,
    render_process_config,
)

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
# Auto port-hunt (master block): when FLEXLB_FT_MASTER_HTTP_PORT is NOT
# pinned, _pick_master_http_port() bind-probes the default block and shifts
# forward by this stride — keeping the http / mgmt=+1 / gRPC=+2 layout —
# until a free block is found.  A pinned env var keeps the legacy semantics
# (start_master's wait loop owns contention): parallel_runner lanes pin
# DISJOINT blocks per lane, and silent shifting there would make lanes
# chase each other's blocks.  50 shifts cover 18080..18580.
MASTER_PORT_STRIDE = 10
MASTER_PORT_MAX_SHIFTS = 50

# Shared with the shell stress launcher; case values still come from YAML.
from online_eval.load_client import LOAD_CLIENT_ENV_VARS
from online_eval.metrics import parse_prometheus_samples

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

    Decode timing is likewise NOT configured: the mock prices
    decode per STEP with the production DSv4 fit — step_ms = 19.5 +
    0.175 x running, 2.6 tokens/step (MTP acceptance fold) — as the code
    default, aligning throughput/queueing economics with production
    (low-batch ~515 tok/s at running=4, full-batch ~7900 tok/s at 128).
    The former explicit ``step_ms_by_batch`` curve approximated the same
    step latencies but without the MTP fold, overstating decode duration
    ~2.6x; it was removed so all flexlb_test_framework cases run on the production
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


# Bind-probe address for PRE-START port hunting: every JVM the harness
# launches (mock NettyServer, Spring master --server.port) binds the
# WILDCARD 0.0.0.0 — a probe on 127.0.0.1 only proves the loopback
# interface free, so a foreign process bound to another interface
# (observed: "Failed to bind to address 0.0.0.0/0.0.0.0:55252" while
# the 127.0.0.1 probe passed) slips through and the JVM dies at startup.
# Post-start RELEASE detection (_master_ports_in_use) keeps 127.0.0.1 —
# an actually-running JVM on 0.0.0.0 shows up on loopback too.
PROBE_BIND_HOST = "0.0.0.0"


def _pick_master_http_port() -> tuple[int, str]:
    """Bind-probe the master port block and auto-shift when busy.

    The block layout is http / management=+1 / gRPC=+2 (README "master 组");
    the probe uses the same SO_REUSEADDR bind as port_in_use so a port in
    TIME_WAIT still counts as FREE only when the real JVM bind would
    succeed — matching the actual start semantics.

    Modes:
      * FLEXLB_FT_MASTER_HTTP_PORT pinned → returned verbatim ("explicit")
        with contention left to start_master's wait loop — a pinned port is
        an orchestrator contract (parallel_runner lanes), never silently
        moved;
      * unpinned → probe from the default 18080 block, shifting forward by
        MASTER_PORT_STRIDE (layout preserved) until a free block is found;
        the note records the provenance ("auto 18080" or
        "auto 18080->18090 (default block busy)") for the env build log
        and the results JSON — a shifted run must be traceable when
        debugging "which master did this case actually talk to".

    Returns (http_port, note).
    """
    explicit = os.environ.get("FLEXLB_FT_MASTER_HTTP_PORT")
    if explicit:
        port = int(explicit)
        return port, f"explicit {port}"
    base = DEFAULT_MASTER_HTTP_PORT
    for shift in range(MASTER_PORT_MAX_SHIFTS):
        cand = base + MASTER_PORT_STRIDE * shift
        if not any(port_in_use(p, PROBE_BIND_HOST) for p in (cand, cand + 1, cand + 2)):
            if shift == 0:
                return cand, f"auto {cand}"
            return cand, f"auto {base}->{cand} (default block busy)"
    raise RuntimeError(
        f"no free master port block found in "
        f"{base}..{base + MASTER_PORT_STRIDE * (MASTER_PORT_MAX_SHIFTS - 1) + 2} "
        f"(stride {MASTER_PORT_STRIDE})"
    )


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

    # -- Mode 2 (freeze) primitives, per-master directed -------------------

    def freeze(self) -> None:
        """SIGSTOP this process (frozen: port up, state retained)."""
        ProcessOps.sigstop(self.proc.pid)

    def unfreeze(self) -> None:
        """SIGCONT this process (hot recovery: state intact)."""
        ProcessOps.sigcont(self.proc.pid)

    def tail_log(self, lines: int = 40) -> str:
        try:
            text = self.log_file.read_text(encoding="utf-8", errors="replace")
            return "\n".join(text.splitlines()[-lines:])
        except Exception:
            return "<no log>"


# ---------------------------------------------------------------------------
# HA dual-master port plan (ruling, verified against the v2 Java code — see
# the MasterSpec docstring below for the full evidence chain)
# ---------------------------------------------------------------------------

# Tier-1 dual standalone (needConsistency stays OFF): each master gets its
# OWN port group — A: HTTP 18080 / mgmt 18081 / gRPC 18082, B: HTTP 18083 /
# mgmt 18084 / gRPC 18085.  With consistency disabled the three same-host
# assumptions (ZK leader id, port stitching, SELF_TARGET) are all inert, so
# distinct ports are the zero-risk layout and no FLEXLB_ADVERTISED_IP is
# needed; the client separates the two masters by gRPC port.
HA_TIER1_MASTER_A_HTTP_PORT = int(
    os.environ.get("FLEXLB_FT_HA_MASTER_A_HTTP_PORT", "18080")
)
HA_TIER1_MASTER_B_HTTP_PORT = int(
    os.environ.get("FLEXLB_FT_HA_MASTER_B_HTTP_PORT", "18083")
)

# Tier-2/3 ZK-activated layout: the SAME port group on DIFFERENT loopback
# IPs (127.0.0.1:18080..18082 + 127.0.0.2:18080..18082 + FLEXLB_ADVERTISED_IP
# injected per master).  Required because (verified in the Java code):
#   * ZookeeperMasterElectService.initializeIpAndPort() sets the ZK
#     LeaderSelector id to the BARE local IP (no port) — same-IP dual
#     instances collide and mis-elect (isStillMaster compares bare IPs);
#   * LBStatusConsistencyService.getMasterHostIpPort() stitches the LOCAL
#     server.port onto the leader IP — a distinct-port layout would forward
#     to the wrong port (A_IP:B_port is unreachable);
#   * FlexlbGrpcForwarder.sameHost() compares bare IPs — same-IP instances
#     would self-block every forward as SELF_TARGET.
# Activation additionally needs the production-side prerequisites
# (an FLEXLB_ADVERTISED_IP consumer in flexlb-sync + a per-address gRPC
# bind instead of NettyServerBuilder.forPort) which are NOT yet in the
# code; the harness injects the env/ports per this contract so the layout
# lights up the moment those land.  Tier-2 forwarding itself is covered by
# the JUnit layer (master_forward_matrix), not by this harness.
#
# RULING (2026-09-02): the same-host distinct-IP layout is DEAD.  The
# election localIp comes ONLY from InetAddress.getLocalHost() hostname
# resolution (ZookeeperMasterElectService L106-111 + LBStatusConsistency-
# Service L52, two independent sites, no env override channel), the gRPC
# wildcard bind (forPort) cannot start a second same-port instance, and
# same-IP distinct-port makes SELF_TARGET permanently true, blocking all
# forwarding; the production-side prerequisites (FLEXLB_ADVERTISED_IP
# consumer / per-address bind) will NOT land.  Tier-3 moves to a
# dual-container topology (each container gets its own network stack +
# hostname -> naturally distinct IPs on the SAME port, faithfully
# replicating production's one-IP-per-pod model) — phase 2.  The existing
# 127.0.0.1/.2 wiring below is kept ONLY as the env-injection contract
# reference (supersedes the "lights up the moment those land" expectation
# above).
HA_TIER3_MASTER_HTTP_PORT = int(
    os.environ.get("FLEXLB_FT_HA_TIER3_MASTER_HTTP_PORT", "18080")
)
HA_TIER3_MASTER_B_BIND_IP = os.environ.get(
    "FLEXLB_FT_HA_TIER3_MASTER_B_BIND_IP", "127.0.0.2"
)


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

    # ------------------------------------------------------------------
    # Mode 2 fault primitives (HA case test, brief p3: "SIGSTOP / SIGCONT
    # 原语（ProcessOps 一行级扩展）").  Directed at a specific master PID
    # through the masters registry (ManagedProcess.freeze / unfreeze).
    # ------------------------------------------------------------------

    @staticmethod
    def sigstop(pid: int) -> None:
        """Mode 2 (freeze) injection: process frozen, port stays up,
        application stops responding, in-memory state retained."""
        os.kill(pid, signal.SIGSTOP)

    @staticmethod
    def sigcont(pid: int) -> None:
        """Mode 2 recovery: thaw a SIGSTOP-frozen process (hot recovery —
        memory state intact, as opposed to Mode 1 kill -9 cold start)."""
        os.kill(pid, signal.SIGCONT)

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
class MasterSpec:
    """One flexlb master entry in the (optional) dual-master registry.

    Port-plan ruling (verified against the v2 Java code, 2026-09 HA case
    test task):

    * Tier-1 dual standalone — needConsistency stays OFF (the mock-line
      default).  Each master gets its OWN port group (A: HTTP 18080 /
      mgmt 18081 / gRPC 18082, B: HTTP 18083 / mgmt 18084 / gRPC 18085).
      With consistency disabled the three same-host assumptions are inert:
      ZookeeperMasterElectService.init() returns before touching ZK,
      LBStatusConsistencyService.getMasterHostIpPort() returns null (no
      forwarding, LOCAL_STANDALONE routing) and
      FlexlbGrpcForwarder.sameHost(ip, null) is false (no SELF_TARGET), so
      distinct ports are the zero-risk layout.  No FLEXLB_ADVERTISED_IP.

    * Tier-2/3 ZK-activated — FLEXLB_SYNC_CONSISTENCY_CONFIG set by the
      harness (EnvSpec.zk_consistency).  The layout MUST switch to
      same-port / different-IP (bind_ip 127.0.0.1 vs 127.0.0.2 +
      FLEXLB_ADVERTISED_IP): the ZK LeaderSelector id is the BARE local IP
      (ZookeeperMasterElectService.initializeIpAndPort), the forwarded
      master address stitches the LOCAL server.port onto the leader IP
      (LBStatusConsistencyService.getMasterHostIpPort) and SELF_TARGET
      compares bare IPs (FlexlbGrpcForwarder.sameHost) — a distinct-port
      same-IP pair breaks on all three.  Both instances share ONE
      HIPPO_ROLE: the ZK lock path is /master_lb_leader/{HIPPO_ROLE}, so
      the same roleId is what makes them mutual master/follower.

    RULING (2026-09-02): the same-host distinct-IP Tier-3 layout is
    DEAD — the election localIp comes only from InetAddress.getLocalHost()
    hostname resolution (ZookeeperMasterElectService L106-111 +
    LBStatusConsistencyService L52, two independent sites, no env
    override channel), the gRPC wildcard bind (forPort) cannot start a
    second same-port instance, and same-IP distinct-port makes
    SELF_TARGET permanently true, blocking all forwarding; the
    production-side prerequisites (FLEXLB_ADVERTISED_IP consumer /
    per-address bind) will NOT land.  Tier-3 moves to a dual-container
    topology (one network stack + hostname per container -> naturally
    distinct IPs on the same port, replicating production's
    one-IP-per-pod) — phase 2.  The 127.0.0.1/.2 wiring is kept only as
    the env-injection contract reference.

    Tier-2 forwarding semantics (four-state matrix, 8511, ForwardGuard)
    are covered by the JUnit layer (master_forward_matrix) — this harness
    only orchestrates processes/env; Tier-3 is deferred to the phase-2
    dual-container topology per the RULING above.
    """

    name: str  # registry key ("A" / "B" — brief p5/p6 scenario notation)
    http_port: int
    management_port: Optional[int] = None  # default http+1
    # Spring --server.address; Tier-1 stays 127.0.0.1 (distinct ports),
    # Tier-2/3 uses 127.0.0.1 vs 127.0.0.2 (same ports, distinct IPs).
    bind_ip: str = "127.0.0.1"
    # FLEXLB_ADVERTISED_IP (Tier-2/3): overrides the ZK-advertised localIp.
    # Has NO consumer in the flexlb Java code and none will land (see the
    # RULING in the docstring above) — kept as the env-injection contract
    # reference for the phase-2 dual-container Tier-3.
    advertised_ip: Optional[str] = None
    # Default: BOTH instances share spec.label's role (mutual backup).
    hippo_role: Optional[str] = None
    log_dir_name: Optional[str] = None  # default logs_{name} under run_dir
    extra_env: dict = field(default_factory=dict)  # per-master overrides
    extra_args: list = field(default_factory=list)  # per-master CLI args

    def grpc_port(self) -> int:
        """gRPC port = HTTP + 2 (FlexlbGrpcServer.FLEXLB_GRPC_PORT_OFFSET)."""
        return self.http_port + 2

    def management(self) -> int:
        return (
            self.management_port
            if self.management_port is not None
            else self.http_port + 1
        )

    def fingerprint(self) -> dict:
        return {
            "name": self.name,
            "http_port": self.http_port,
            "management_port": self.management(),
            "bind_ip": self.bind_ip,
            "advertised_ip": self.advertised_ip,
            "hippo_role": self.hippo_role,
            "extra_env": self.extra_env,
            "extra_args": self.extra_args,
        }


# ---------------------------------------------------------------------------
# ZK helper (Tier-2/3) — cross-agent contract with flexlb-sync (Mark)
# ---------------------------------------------------------------------------

# Contract constants — the SINGLE definition point the harness and the
# flexlb-sync test launcher are aligned on (org.flexlb.consistency.
# ZkTestingServerLauncher on the flexlb-sync src/test classpath).
# Field-tested contract (flexlb-sync owner, delivered 2026-09-02):
#   * boot sequence (from the maven root):
#       ./mvnw -q -pl flexlb-sync -am test-compile
#       ./mvnw -q -pl flexlb-sync dependency:build-classpath \
#               -Dmdep.outputFile=<ABS>/test-classpath.txt \
#               -Dmdep.includeScope=test
#       java -cp "flexlb-sync/target/test-classes:<deps>" \
#            org.flexlb.consistency.ZkTestingServerLauncher --port 0
#     -Dmdep.outputFile MUST be an absolute path (a relative one resolves
#     against the module basedir and creates nested directories);
#     build-classpath must NOT carry -am (every reactor module would write
#     its own outputFile — the nested-directory trap).
#   * readiness handshake: ONE stdout line "ZK_READY <connectString>" —
#     do NOT probe with 4lw (ZK 3.6.3 whitelists only "srvr"; ruok is
#     refused by default).  --port 0 auto-allocates; the ZK_READY line
#     carries the actual port.
#   * exit paths: SIGTERM (used by the harness) and stdin EOF — both
#     print "ZK_STOPPED" and exit 0.
#   * macOS caveat: 127.0.0.2 silently drops (SYN retransits 20s+) on
#     macOS — the Tier-2/3 same-port/distinct-IP layout only works on
#     Linux loopback (full 127/8 routed); run the ZK-tier cases remotely.
ZK_LAUNCHER_CLASS = "org.flexlb.consistency.ZkTestingServerLauncher"
ZK_READY_PREFIX = "ZK_READY"
ZK_STOPPED_PREFIX = "ZK_STOPPED"
# Full-argv escape hatch: when set, the harness shlex-splits it verbatim
# and uses it as the ZK helper command line (the exact mvnw classpath
# assembly stays swappable until the flexlb-sync owner's final report).
ZK_LAUNCH_CMD_ENV = "FLEXLB_FT_ZK_LAUNCH_CMD"
# How long the harness waits for the ZK_READY line before failing closed.
ZK_READY_TIMEOUT_S = float(os.environ.get("FLEXLB_FT_ZK_READY_TIMEOUT_S", "120"))


class ZkHelperOps:
    """Lifecycle glue for the ZK helper JVM.

    Default launch: resolve the flexlb-sync TEST classpath once per run-dir
    via ``mvnw -pl flexlb-sync -am dependency:build-classpath`` (cached in
    <run_dir>/zk_classpath.txt), then
    ``java -cp test-classes:classes:<deps> org.flexlb.consistency.ZkTestingServerLauncher``.
    Override the whole command through FLEXLB_FT_ZK_LAUNCH_CMD if the
    flexlb-sync assembly differs.
    """

    @staticmethod
    def find_ready_connect_string(log_file: Path) -> Optional[str]:
        """Scan the redirected stdout for the 'ZK_READY <connectString>' line."""
        try:
            text = log_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        for line in text.splitlines():
            parts = line.split()
            if parts and parts[0] == ZK_READY_PREFIX and len(parts) >= 2:
                return parts[1]
        return None

    @staticmethod
    def _mvnw(args: list, env: dict, timeout_s: int = 900, what: str = "mvnw") -> None:
        mvnw = FLEXLB_DIR / "mvnw"
        if not mvnw.is_file():
            raise RuntimeError(f"mvnw not found: {mvnw}")
        cmd = [str(mvnw), "-q", *args]
        # First run may compile reactor modules — allow a long window.
        proc = subprocess.run(
            cmd,
            cwd=str(FLEXLB_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{what} failed (rc={proc.returncode}):\n"
                f"{(proc.stderr or proc.stdout)[-2000:]}"
            )

    @staticmethod
    def _ensure_test_classpath(java_bin: str, run_dir: Path) -> str:
        """flexlb-sync TEST-scope dependency classpath via mvnw (cached).

        Field-tested two-step boot (flexlb-sync owner contract):
          1. ``-pl flexlb-sync -am test-compile`` — compiles the launcher
             itself (src/test) plus reactor deps;
          2. ``-pl flexlb-sync dependency:build-classpath`` (NO -am:
             with it every reactor module writes its own outputFile)
             with an ABSOLUTE -Dmdep.outputFile and includeScope=test
             (curator-test is test-scoped in flexlb-sync/pom.xml).
        Cached per run-dir so repeated env builds reuse one resolution.
        """
        cp_file = run_dir / "zk_classpath.txt"
        if cp_file.is_file() and cp_file.stat().st_size > 0:
            return cp_file.read_text(encoding="utf-8").strip()
        env = dict(os.environ)
        # mvnw honours JAVA_HOME; derive it from the resolved java binary.
        java_home = Path(java_bin).resolve().parents[1]
        if (java_home / "bin" / "java").exists():
            env.setdefault("JAVA_HOME", str(java_home))
        ZkHelperOps._mvnw(
            ["-pl", "flexlb-sync", "-am", "test-compile"],
            env,
            timeout_s=900,
            what="mvnw test-compile (flexlb-sync)",
        )
        ZkHelperOps._mvnw(
            [
                "-pl",
                "flexlb-sync",
                "dependency:build-classpath",
                f"-Dmdep.outputFile={cp_file}",  # absolute (run_dir is abs)
                "-Dmdep.includeScope=test",
            ],
            env,
            timeout_s=600,
            what="mvnw dependency:build-classpath (flexlb-sync)",
        )
        if not cp_file.is_file() or cp_file.stat().st_size == 0:
            raise RuntimeError(
                f"mvnw build-classpath produced no classpath file: {cp_file}"
            )
        return cp_file.read_text(encoding="utf-8").strip()

    @staticmethod
    def default_launch_argv(java_bin: str, run_dir: Path) -> list[str]:
        deps = ZkHelperOps._ensure_test_classpath(java_bin, run_dir)
        sync_target = FLEXLB_DIR / "flexlb-sync" / "target"
        cp = os.pathsep.join(
            [
                str(sync_target / "test-classes"),
                str(sync_target / "classes"),
                deps,
            ]
        )
        # --port 0: ZK auto-allocates a free port; the ZK_READY line
        # advertises the actual connectString (no port-collision risk
        # against sibling harness processes).
        return [java_bin, "-cp", cp, ZK_LAUNCHER_CLASS, "--port", "0"]


def _override_fingerprint(overrides: Optional[ConfigOverride]) -> Optional[dict]:
    """Stable JSON-serializable snapshot of a ConfigOverride (the OMIT
    sentinel serializes as its own token) — EnvSpec.fingerprint input."""
    if overrides is None:
        return None
    snapshot: dict = {}
    for f in fields(overrides):
        value = getattr(overrides, f.name)
        snapshot[f.name] = "OMIT" if value is OMIT else value
    return snapshot


@dataclass
class EnvSpec:
    """Declarative description of a full mock + master environment."""

    label: str = "env"
    n_prefill: int = 2
    n_decode: int = 4
    mock_heap: str = DEFAULT_MOCK_HEAP
    perf: dict = field(default_factory=default_perf)
    # Built-in scheduling profile (PROFILES) or "none" (master not
    # started); the FLEXLB_CONFIG document is rendered by flexlb_cfg from
    # the profile axes + config_overrides (or passthrough raw_config).
    master_profile: str = "batch-window"
    # Non-config master env overrides ONLY (e.g. FLEXLB_MONITOR_METRIC_
    # WHITELIST, FLEXLB_ADVERTISED_IP, FLEXLB_SYNC_CONSISTENCY_CONFIG).
    # FLEXLB_CONFIG must NOT ride here — the narrowed channel is
    # config_overrides (generator layering) / raw_config (negative-test
    # bypass); _master_env rejects a stray FLEXLB_CONFIG key.
    master_env: dict = field(default_factory=dict)
    # FLEXLB_CONFIG layer: flexlb_cfg.ConfigOverride applied on top of the
    # master_profile base document (base < profile < override — see
    # flexlb_cfg.render_env).  None = the profile document as-is.
    config_overrides: Optional[ConfigOverride] = None
    # Raw FLEXLB_CONFIG string — the negative-test channel (atpm
    # strict-reject variants inject deliberately-illegal documents).
    # Bypasses the generator entirely; takes precedence over
    # config_overrides.
    raw_config: Optional[str] = None
    spring_profile: str = "default"
    master_debug_log: bool = False
    # domain uses MODEL_SERVICE_CONFIG.hosts; file/discovery_file use
    # the mock-maintained MODEL_SERVICE_CONFIG.discovery_file.
    discovery: str = "file"
    domain_addrs: dict = field(default_factory=dict)  # {prefill: "a,b", decode: "a,b"}
    # Per-role KV pool size in BLOCKS — forwarded to the Java mock as
    # --prefill-kv-pool-blocks / --decode-kv-pool-blocks (NOT a key-count
    # cap: since KV v2 this is the total block count of the pool).
    prefill_cache_blocks: int = DEFAULT_PREFILL_CACHE_BLOCKS
    decode_cache_blocks: int = DEFAULT_DECODE_CACHE_BLOCKS
    master_extra_args: list = field(default_factory=list)
    event_loop_threads: int = DEFAULT_MOCK_EVENT_LOOP_THREADS
    completion_threads: int = DEFAULT_MOCK_COMPLETION_THREADS
    mock_auto_fetch: bool = False
    mock_fetch_attach_timeout_ms: int = 600_000
    # Seconds the freshly started master must hold "alive == discovered" for
    # every role before start_master() returns (0 disables). Skips the
    # cold-start first-connect storm during which healthy engines can be
    # 3-strike-marked dead (CONNECT_TIMEOUT 20ms intake defect).
    master_stable_window_s: float = 3.0
    # ------------------------------------------------------------------
    # HA dual-master orchestration (HA case test).  EMPTY (default) keeps
    # the single-master legacy path byte-identical: every new behaviour in
    # EnvManager is gated on this list being non-empty, so the existing
    # stress line and every pre-HA flexlb_test_framework case are untouched.
    # ------------------------------------------------------------------
    # Per-master registry (MasterSpec).  Non-empty => EnvManager starts one
    # flexlb-api JVM per entry (start_master_instance) instead of the
    # single shared master; both masters share the SAME mock cluster and
    # discovery file and poll it independently (brief p1).
    masters: list = field(default_factory=list)  # list[MasterSpec]
    # Tier-2/3 only: non-None starts the ZK helper JVM (Mark's contract —
    # org.flexlb.consistency.ZkTestingServerLauncher, "ZK_READY
    # <connectString>" on stdout) BEFORE the masters and injects
    # FLEXLB_SYNC_CONSISTENCY_CONFIG (needConsistency=true, zkHost=<helper
    # connectString>, zkTimeoutMs from this dict) into every master env.
    # Tier-1 dual-standalone specs leave this None: no ZK, no election, no
    # forwarding (needConsistency=false → LOCAL_STANDALONE, the mock line's
    # existing state).
    zk_consistency: Optional[dict] = None  # e.g. {"zkTimeoutMs": 30000}

    def fingerprint(self) -> str:
        return json.dumps(
            {
                "n_prefill": self.n_prefill,
                "n_decode": self.n_decode,
                "perf": self.perf,
                "master_profile": self.master_profile,
                "master_env": self.master_env,
                # config axes: overrides serialized field-by-field (OMIT as
                # its own token) so distinct override specs never collide;
                # raw_config contributes its exact string.
                "config_overrides": _override_fingerprint(self.config_overrides),
                "raw_config": self.raw_config,
                "discovery": self.discovery,
                "domain_addrs": self.domain_addrs,
                "prefill_cache_blocks": self.prefill_cache_blocks,
                "decode_cache_blocks": self.decode_cache_blocks,
                "spring_profile": self.spring_profile,
                "master_stable_window_s": self.master_stable_window_s,
                # HA axes: absent-equivalent ([], None) for every legacy
                # spec — the fingerprint VALUE changes (new keys) but stays
                # stable within a process, so ensure() reuse semantics are
                # unchanged and the legacy path never takes the new branch.
                "masters": [m.fingerprint() for m in self.masters],
                "zk_consistency": self.zk_consistency,
            },
            sort_keys=True,
        )


# ---------------------------------------------------------------------------
# Scheduling profiles / FLEXLB_CONFIG — owned by flexlb_cfg (SSOT)
# ---------------------------------------------------------------------------
#
# The profile axes (PROFILES / PROFILE_SPECS / PROFILE_CAPS), the DSv4
# prefill fit (DSV4_PREFILL_EXPRESSION), the strict schema-v2 builders
# (_build_preemption_cfg / _build_ordering_cfg / build_flexlb_config) and
# the render layer (render_env / render_process_config / ConfigOverride)
# moved to flexlb_cfg.py — the single source of truth for every
# FLEXLB_CONFIG document this repo produces — and are re-exported at the
# top of this module for the runner and action implementations.
#
# Profile semantics in one line: four functional case-test profiles
# (decision x dispatcher axes, scheduler QUEUE + FIFO ordering; values
# unchanged) plus the stress-na130 render profile — the former
# data/config/master_fixed_window.json, byte-for-byte identical, now
# generated.  See the flexlb_cfg module docstring for the full contract
# (layering, OMIT, axis re-typing, envelope projection).
# Master env that is actually consumed by the v2 code:
#   FLEXLB_CONFIG          — set per spec from the profile generator below
#   HIPPO_ROLE             — flexlb-sync (zookeeper elect / LB status)
#   OTEL_TRACE_SKIP_PATTERN — flexlb-api application.yml (spring tracing)
#   OTEL_EXPORTER_OTLP_ENDPOINT — OpenTelemetry SDK exporter ("none" disables)
# Every other legacy v1 var previously exported here had zero consumers in
# the v2 Java code and was removed (dead-env sweep).
BASE_MASTER_ENV = {
    "OTEL_TRACE_SKIP_PATTERN": ".*",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "none",
}


def _write_master_config(env: "FlexEnv") -> Path:
    """Render the SSOT envelope into run_dir/master_config.json.

    Single render, two projections: this file feeds the mock / victim
    ``--master-config`` argument, while the master env's FLEXLB_CONFIG
    (see EnvManager._master_env) comes from the same flexlb_cfg render —
    the two sides can never drift apart.  Replaces the retired checked-in
    data/config/master_fixed_window.json, which pinned the mock side to
    the stress-line document regardless of the master's profile.
    """
    path = env.run_dir / "master_config.json"
    spec = env.spec
    path.write_text(
        render_process_config(
            spec.master_profile,
            spec.config_overrides,
            raw_config=spec.raw_config,
        ),
        encoding="utf-8",
    )
    return path


class FlexEnv:
    """A live environment: one mock cluster + one master (+ optional victims)."""

    def __init__(self, spec: EnvSpec, run_dir: Path, base_grpc_port: int):
        self.spec = spec
        self.run_dir = run_dir
        self.base_grpc_port = base_grpc_port
        self.mock_http_port = base_grpc_port - 1
        # Master ports: auto-hunt (bind probe + forward shift) unless the
        # env var pins them — see _pick_master_http_port.  An explicit
        # FLEXLB_FT_MASTER_MANAGEMENT_PORT still wins (partial pin keeps
        # its pinned axis; only the UNPINNED http block auto-shifts).
        self.master_http_port, self.master_port_note = _pick_master_http_port()
        explicit_mgmt = os.environ.get("FLEXLB_FT_MASTER_MANAGEMENT_PORT")
        self.master_management_port = (
            int(explicit_mgmt) if explicit_mgmt else self.master_http_port + 1
        )
        self.endpoint_file = run_dir / "endpoints.json"
        self.discovery_file = run_dir / "discovery.json"  # dynamic file discovery
        self.perf_file = run_dir / "perf.json"
        self.mock: Optional[ManagedProcess] = None
        self.master: Optional[ManagedProcess] = None
        self.victims: dict[str, ManagedProcess] = {}  # name -> process
        self.load_clients: list[ManagedProcess] = []
        self.master_start_count = 0
        self.master_incarnations = []
        # HA dual-master registry (empty on the legacy single-master path —
        # env.master stays the single source of truth there).  A value of
        # None means "slot exists, process dead" (post-kill, pre-restart).
        self.masters: dict[str, Optional[ManagedProcess]] = {}
        self.master_specs: dict[str, MasterSpec] = {}
        self.masters_start_count: dict[str, int] = {}
        # ZK helper (Tier-2/3): ManagedProcess + advertised connectString.
        self.zk_helper: Optional[ManagedProcess] = None
        self.zk_connect_string: Optional[str] = None

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
        self._zk_ops_instance: Optional["ZkHelperOps"] = None

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
        # HA dual-master instances: SIGCONT first so a SIGSTOP-frozen JVM
        # can drain on SIGTERM (terminate() would escalate to kill -9
        # anyway — harmless, just noisier); then the ZK helper.
        for name, mp in list(env.masters.items()):
            if mp is not None:
                try:
                    mp.unfreeze()
                except Exception:
                    pass
                mp.terminate()
            env.masters[name] = None
        self._stop_zk_helper(env)
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
            # wildcard probe (PROBE_BIND_HOST): the mock binds 0.0.0.0, a
            # loopback-only probe misses foreign binds on other interfaces
            if not any(port_in_use(p, PROBE_BIND_HOST) for p in needed):
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
            f"mock_base={base}, master_http={env.master_http_port} "
            f"({env.master_port_note}), dir={run_dir.name})"
        )

        # perf config
        env.perf_file.write_text(json.dumps(spec.perf, indent=2))

        try:
            # mock cluster
            self._start_mock(env)
            if spec.masters:
                # HA dual-master path (gated on the registry being
                # non-empty — the single-master legacy branch below is
                # untouched).  Tier-2/3 boots the ZK helper first so the
                # masters can grab the election lock at startup; Tier-1
                # (zk_consistency=None) skips it entirely.
                if spec.zk_consistency is not None:
                    self.start_zk_helper(env)
                for mspec in spec.masters:
                    self.start_master_instance(env, mspec)
            elif spec.master_profile != "none":
                # master
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
            "--auto-fetch",
            str(spec.mock_auto_fetch).lower(),
            "--fetch-attach-timeout-ms",
            str(spec.mock_fetch_attach_timeout_ms),
            "--performance",
            str(env.perf_file),
            "--master-config",
            str(_write_master_config(env)),
            # stdout telemetry (java_mock_stats line every statsIntervalMs):
            # lets the archived mock_engine.log answer "which RPC counters
            # kept moving / stalled" for hang forensics (3c-class issues)
            # without any behavioral change to the engine itself.
            "--stats-stdout",
            "true",
            "--events-file",
            str(env.run_dir / "engine_events.jsonl"),
            "--prefill-kv-pool-blocks",
            str(spec.prefill_cache_blocks),
            "--decode-kv-pool-blocks",
            str(spec.decode_cache_blocks),
            "--endpoint-file",
            str(env.endpoint_file),
            "--env-file",
            str(env.run_dir / "flexlb_env.txt"),
        ]
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
        if not wait_for(
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

    def _master_env(self, env: FlexEnv, mspec: Optional[MasterSpec] = None) -> dict:
        """Build master configuration documents and the existing HA deployment identity."""
        spec = env.spec
        menv = dict(BASE_MASTER_ENV)
        if "FLEXLB_CONFIG" in spec.master_env:
            # Narrowed channel (SSOT migration): master_env is non-config
            # env only — a stray FLEXLB_CONFIG here would silently fork
            # away from the flexlb_cfg render that feeds the mock-side
            # master_config.json envelope.
            raise ValueError(
                "EnvSpec.master_env must not carry FLEXLB_CONFIG — use "
                "config_overrides (flexlb_cfg generator layering) or "
                "raw_config (negative-test bypass)"
            )
        if spec.master_profile != "none":
            if spec.raw_config is not None:
                # Negative-test channel: raw passthrough, generator
                # bypassed (atpm strict-reject variants).
                menv["FLEXLB_CONFIG"] = spec.raw_config
            else:
                menv["FLEXLB_CONFIG"] = render_env(
                    spec.master_profile, spec.config_overrides
                )
        menv["HIPPO_ROLE"] = f"flexlb_ft_{spec.label}"
        if spec.discovery == "file":
            payload = json.loads(env.endpoint_file.read_text(encoding="utf-8"))
            menv["MODEL_SERVICE_CONFIG"] = payload["env"]["MODEL_SERVICE_CONFIG"]
        elif spec.discovery == "discovery_file":
            payload = json.loads(env.endpoint_file.read_text(encoding="utf-8"))
            service_config = json.loads(payload["env"]["MODEL_SERVICE_CONFIG"])
            service_config["discovery_file"] = str(env.discovery_file)
            menv["MODEL_SERVICE_CONFIG"] = json.dumps(service_config)
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
            service_config = json.loads(menv["MODEL_SERVICE_CONFIG"])
            service_config["hosts"] = {
                f"mock.{role}.hosts.address": [
                    host.strip()
                    for host in spec.domain_addrs[role].split(",")
                    if host.strip()
                ]
                for role in ("prefill", "decode")
            }
            menv["MODEL_SERVICE_CONFIG"] = json.dumps(service_config)
        menv.update(spec.master_env)  # spec overrides come last
        if mspec is not None:
            # Per-instance layer (HA dual-master path only — the legacy
            # single-master path keeps mspec None and never reaches here).
            if spec.zk_consistency is not None:
                menv["HIPPO_ROLE"] = mspec.hippo_role or f"flexlb_ft_{spec.label}"
                if not env.zk_connect_string:
                    # Fail-closed: a master must never boot with
                    # needConsistency=true against a missing/dead ZK —
                    # a dual-master pair without the election quorum
                    # would split-brain.
                    raise RuntimeError(
                        "zk_consistency spec requires a live ZK helper "
                        "(connectString missing) — fail-closed"
                    )
                menv["FLEXLB_SYNC_CONSISTENCY_CONFIG"] = json.dumps(
                    {
                        "needConsistency": True,
                        "zookeeperConfig": {
                            "zkHost": env.zk_connect_string,
                            # Default 10s: session expiry inside the ≤60s
                            # convergence window; widen via the
                            # zk_consistency dict for slow-CI layouts.
                            "zkTimeoutMs": int(
                                spec.zk_consistency.get("zkTimeoutMs", 10000)
                            ),
                        },
                    },
                    separators=(",", ":"),
                )
            menv.update(mspec.extra_env)  # per-instance overrides come last
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
            # flexlbLogger (org.flexlb.util.Logger's slf4j name —
            # logback-spring.xml pins it at INFO with the FLEXLB file
            # appender) carries the [priority-scheduler] DEBUG lines into
            # ~/ai-whale/logs/flexlb.log; the org.flexlb switch alone never
            # reaches it (logger-name mismatch, round-2 O1 finding).
            argv.append("--logging.level.flexlbLogger=DEBUG")
        argv.extend(spec.master_extra_args)
        # The JVM's stdout redirection (flexlb_master.log) captures only
        # the console appender's first buffered lines — implementation-
        # period finding: a config-rejected master leaves ~11 stdout
        # lines with NO strict-parser message.  The full Spring startup
        # and ConfigValidationException stacks land in the logback file
        # appender at ~/ai-whale/logs/application.log (shared across
        # every master start in the container), so capture its size now
        # and append the bytes written by THIS start to the failure
        # diagnostics.
        # Honor an explicitly isolated log directory for every diagnostic read.
        # Offsets on the shared default file do not establish process ownership.
        log_root = Path.home() / "ai-whale" / "logs"
        for arg in spec.master_extra_args:
            if arg.startswith("--flexlb.log.path="):
                log_root = Path(arg.split("=", 1)[1])
        app_log = log_root / "application.log"
        try:
            app_log_offset = app_log.stat().st_size
        except OSError:
            app_log_offset = 0
        # Same offset discipline for the flexlbLogger file appender
        # (~/ai-whale/logs/flexlb.log — shared across every master in the
        # container): cases read "the bytes THIS master wrote" via
        # env.flexlb_log_offset.
        flexlb_log = log_root / "flexlb.log"
        try:
            env.flexlb_log_offset = flexlb_log.stat().st_size
        except OSError:
            env.flexlb_log_offset = 0
        # A8 (Daniel P2-3): same offset discipline for the pv.log request
        # journal (~/ai-whale/logs/pv.log — shared across every master in
        # the container): cases read only THIS master's rows via
        # env.pv_log_offset, with an
        # additional per-case requestId filter on top.
        pv_log = log_root / "pv.log"
        try:
            env.pv_log_offset = pv_log.stat().st_size
        except OSError:
            env.pv_log_offset = 0
        proc = ProcessOps.start(argv, self._master_env(env), env.run_dir / log_name)
        env.master = proc
        env.master_incarnations.append(
            dict(
                name="single",
                target=f"127.0.0.1:{env.master_http_port + 2}",
                generation=env.master_start_count,
                pid=proc.pid,
                started_epoch_ms=time.time() * 1000,
            )
        )

        def _app_log_tail_this_start(lines: int = 60) -> str:
            try:
                with open(app_log, "rb") as fh:
                    fh.seek(app_log_offset)
                    chunk = fh.read().decode("utf-8", errors="replace")
                return "\n".join(chunk.splitlines()[-lines:])
            except OSError:
                return ""

        # B2 (Daniel P3-2): poll readiness AND liveness — the strict-
        # config startup-failure variants (atpm_config_strict_reject) die
        # within seconds of launch, so waiting the full 90s port window
        # there only delays the failure report; the early exit mirrors
        # the mock-start polling above.  Only the process-dead branch
        # short-circuits — a live master keeps the full window.
        master_up = False
        master_deadline = time.monotonic() + 90
        while time.monotonic() < master_deadline:
            if port_in_use(env.master_http_port):
                master_up = True
                break
            if not proc.alive():
                break
            time.sleep(1.0)
        if not master_up:
            app_tail = _app_log_tail_this_start()
            extra = f"\n--- {app_log} (this start) ---\n" + app_tail if app_tail else ""
            raise RuntimeError(f"master failed to start:\n{proc.tail_log()}{extra}")
        # Guard against a foreign master squatting on the HTTP port: if our own
        # JVM died on BindException, the port probe above may still succeed
        # against the foreign process. Re-check our own pid.
        if not proc.alive():
            app_tail = _app_log_tail_this_start()
            extra = f"\n--- {app_log} (this start) ---\n" + app_tail if app_tail else ""
            raise RuntimeError(
                f"master process exited during startup (port conflict?):\n"
                f"{proc.tail_log()}{extra}"
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

    # -- HA dual-master orchestration (gated on spec.masters) --------------

    def _instance_ports_in_use(self, mspec: MasterSpec) -> list[int]:
        """Instance's fixed ports (HTTP / management / gRPC = http+2),
        probed on the instance's OWN bind ip.

        On the Tier-2/3 same-port layout the probe against 127.0.0.2 must
        not be confused by a sibling instance bound to 127.0.0.1 — distinct
        addresses coexist, so only a wildcard squatter (e.g. the current
        NettyServerBuilder.forPort gRPC bind, until the production-side
        per-address prerequisite lands) reports the port busy on both.
        """
        ports = [mspec.http_port, mspec.management(), mspec.grpc_port()]
        return [p for p in ports if port_in_use(p, mspec.bind_ip)]

    def start_master_instance(
        self, env: FlexEnv, mspec: MasterSpec, log_name: Optional[str] = None
    ) -> ManagedProcess:
        """Start ONE flexlb-api JVM per MasterSpec (HA dual-master path).

        Mirrors start_master()'s readiness ladder (port → /master/info
        ready → engine stable window) with every probe pointed at the
        instance's own bind ip/port, plus the per-instance argv/env keys:
        --server.address, --management.server.address, --flexlb.log.path
        (per-instance log dir), FLEXLB_ADVERTISED_IP and
        FLEXLB_SYNC_CONSISTENCY_CONFIG via _master_env(env, mspec).
        """
        spec = env.spec
        if not API_JAR.is_file():
            raise RuntimeError(f"flexlb-api jar not found: {API_JAR} (build it first)")
        if env.masters.get(mspec.name) is not None:
            raise RuntimeError(
                f"master instance '{mspec.name}' already running; stop it first"
            )
        # Pre-flight on the instance's own addresses (mirrors the single
        # master's sibling-instance wait; see _instance_ports_in_use).
        port_wait_s = float(os.environ.get("FLEXLB_FT_MASTER_PORT_WAIT_S", "120"))
        port_deadline = time.monotonic() + port_wait_s
        while True:
            busy = self._instance_ports_in_use(mspec)
            if not busy:
                break
            if time.monotonic() >= port_deadline:
                raise RuntimeError(
                    f"master instance '{mspec.name}' ports still busy after "
                    f"{port_wait_s:.0f}s on {mspec.bind_ip} (another master "
                    f"running? the Tier-2/3 same-port layout additionally "
                    f"needs the production-side per-address gRPC bind "
                    f"prerequisite): {busy}"
                )
            self._log(
                f"master '{mspec.name}' ports {busy} busy on {mspec.bind_ip}; "
                f"waiting for release ..."
            )
            time.sleep(5.0)
        java = resolve_java21()
        env.masters_start_count[mspec.name] = (
            env.masters_start_count.get(mspec.name, 0) + 1
        )
        n = env.masters_start_count[mspec.name]
        log_name = log_name or (
            f"flexlb_master_{mspec.name}.log"
            if n == 1
            else f"flexlb_master_{mspec.name}_restart{n}.log"
        )
        # Per-instance log dir — logback-spring.xml's springProperty
        # flexlb.log.path: two instances must never interleave one file.
        log_dir = env.run_dir / (mspec.log_dir_name or f"logs_{mspec.name}")
        argv = [
            java,
            *JAVA_MODULE_OPTS,
            "-jar",
            str(API_JAR),
            f"--server.port={mspec.http_port}",
            f"--management.server.port={mspec.management()}",
            f"--server.address={mspec.bind_ip}",
            # Management port follows the main bind ip too: without it
            # Spring binds 0.0.0.0 and the Tier-2/3 same-port pair would
            # collide on the management port even though the main HTTP
            # ports coexist on distinct addresses.
            f"--management.server.address={mspec.bind_ip}",
            f"--flexlb.log.path={log_dir}",
            f"--spring.profiles.active={spec.spring_profile}",
        ]
        if spec.master_debug_log:
            argv.append("--logging.level.org.flexlb=DEBUG")
        argv.extend(spec.master_extra_args)
        argv.extend(mspec.extra_args)
        proc = ProcessOps.start(
            argv, self._master_env(env, mspec), env.run_dir / log_name
        )
        env.masters[mspec.name] = proc
        env.master_specs[mspec.name] = mspec
        env.master_incarnations.append(
            dict(
                name=mspec.name,
                target=f"{mspec.bind_ip}:{mspec.grpc_port()}",
                generation=n,
                pid=proc.pid,
                started_epoch_ms=time.time() * 1000,
            )
        )

        base_url = f"http://{mspec.bind_ip}:{mspec.http_port}"

        def _master_info() -> Optional[dict]:
            # /rtp_llm/master/info is a POST endpoint (GET returns 405).
            status, data = http_post_json(f"{base_url}/rtp_llm/master/info", {})
            return data if status == 200 else None

        if not wait_for_port(mspec.bind_ip, mspec.http_port, 90):
            raise RuntimeError(
                f"master instance '{mspec.name}' failed to start:\n"
                f"{proc.tail_log()}"
            )
        # Foreign-squatter guard (same rationale as start_master).
        if not proc.alive():
            raise RuntimeError(
                f"master instance '{mspec.name}' exited during startup "
                f"(port conflict?):\n{proc.tail_log()}"
            )
        if not wait_for(
            lambda: (lambda d: bool(d and d.get("ready")))(_master_info()),
            timeout_s=30,
            interval_s=0.5,
        ):
            raise RuntimeError(
                f"master instance '{mspec.name}' HTTP up but engine sync not "
                f"ready after 30s (check {log_dir})"
            )
        # Stability window (same semantics as start_master).
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
                    raise RuntimeError(
                        f"master instance '{mspec.name}' exited:\n" f"{proc.tail_log()}"
                    )
                raise RuntimeError(
                    f"master instance '{mspec.name}' engines not stable "
                    f"(alive == discovered for {window_s:.0f}s) within 90s "
                    f"— check {log_dir}"
                )
        self._log(
            f"master instance '{mspec.name}' up (pid={proc.pid}, "
            f"http={base_url}, grpc={mspec.bind_ip}:{mspec.grpc_port()})"
        )
        return proc

    def _live_instance(self, env: FlexEnv, name: str) -> ManagedProcess:
        mp = env.masters.get(name)
        if mp is None:
            raise RuntimeError(f"master instance '{name}' is not running")
        return mp

    def master_instance_http(self, env: FlexEnv, name: str) -> str:
        """HTTP base URL of one registered instance (probing + assertions)."""
        mspec = env.master_specs[name]
        return f"http://{mspec.bind_ip}:{mspec.http_port}"

    def master_instance_target(self, env: FlexEnv, name: str) -> str:
        """gRPC target (bind_ip:grpc_port) — the GRPC_TARGETS entry format."""
        mspec = env.master_specs[name]
        return f"{mspec.bind_ip}:{mspec.grpc_port()}"

    def kill_master9_instance(self, env: FlexEnv, name: str) -> None:
        """Mode 1 directed fault: kill -9 ONE instance (cold-restart semantics).

        The registry slot flips to None ("dead, restartable") — the other
        instance keeps running untouched.
        """
        mp = self._live_instance(env, name)
        self._log(f"kill -9 master instance '{name}' (pid={mp.pid})")
        mp.kill9()
        env.masters[name] = None

    def stop_master_instance(
        self, env: FlexEnv, name: str, settle_s: float = 2.0
    ) -> None:
        """Graceful SIGTERM stop of ONE instance (orderly, drains on SIGTERM)."""
        mp = self._live_instance(env, name)
        self._log(f"stopping master instance '{name}' (pid={mp.pid})")
        mp.terminate()
        env.masters[name] = None
        time.sleep(settle_s)

    def restart_master_instance(self, env: FlexEnv, name: str) -> ManagedProcess:
        """Mode 1 recovery: fresh JVM from the SAME MasterSpec (cold start —
        in-memory state zeroed, converges from the zero-point)."""
        mspec = env.master_specs[name]
        return self.start_master_instance(env, mspec)

    def freeze_master_instance(self, env: FlexEnv, name: str) -> None:
        """Mode 2 directed fault: SIGSTOP ONE instance.

        Port stays up, process is unresponsive, in-memory state retained
        (hot-recovery semantics on SIGCONT — no cold restart).
        """
        mp = self._live_instance(env, name)
        self._log(f"SIGSTOP master instance '{name}' (pid={mp.pid})")
        mp.freeze()

    def unfreeze_master_instance(self, env: FlexEnv, name: str) -> None:
        """Mode 2 recovery: SIGCONT the frozen instance (hot recovery)."""
        mp = self._live_instance(env, name)
        self._log(f"SIGCONT master instance '{name}' (pid={mp.pid})")
        mp.unfreeze()

    # -- ZK helper (Tier-2/3, gated on spec.zk_consistency) ----------------

    def start_zk_helper(self, env: FlexEnv) -> None:
        """Boot the ZK helper JVM and wait for 'ZK_READY <connectString>'.

        Fail-closed by design: ANY startup failure raises here, and the
        _build() failure path then stops the partial env — the masters must
        never start against a missing/quorum-less ZK (split-brain guard).
        """
        if env.zk_helper is not None:
            raise RuntimeError("ZK helper already running")
        java = resolve_java21()
        override = os.environ.get(ZK_LAUNCH_CMD_ENV)
        if override:
            argv = shlex.split(override)
        else:
            argv = ZkHelperOps.default_launch_argv(java, env.run_dir)
        log_file = env.run_dir / "zk_helper.log"
        proc = ProcessOps.start(argv, dict(os.environ), log_file, cwd=FLEXLB_DIR)
        env.zk_helper = proc
        connect: Optional[str] = None
        deadline = time.monotonic() + ZK_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if not proc.alive():
                raise RuntimeError(
                    f"ZK helper exited during startup (fail-closed):\n"
                    f"{proc.tail_log()}"
                )
            connect = ZkHelperOps.find_ready_connect_string(log_file)
            if connect:
                break
            time.sleep(0.2)
        if not connect:
            raise RuntimeError(
                f"ZK helper did not print '{ZK_READY_PREFIX} <connectString>' "
                f"within {ZK_READY_TIMEOUT_S:.0f}s (fail-closed):\n"
                f"{proc.tail_log()}"
            )
        env.zk_connect_string = connect
        self._log(f"ZK helper up (pid={proc.pid}, connectString={connect})")

    def _stop_zk_helper(self, env: FlexEnv) -> None:
        if env.zk_helper is not None:
            self._log(f"stopping ZK helper (pid={env.zk_helper.pid})")
            # Contract exit path: SIGTERM (launcher also exits on stdin EOF).
            env.zk_helper.terminate()
            env.zk_helper = None
        env.zk_connect_string = None

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
            str(_write_master_config(env)),
            "--prefill-kv-pool-blocks",
            str(env.spec.prefill_cache_blocks if role == "prefill" else 0),
            "--decode-kv-pool-blocks",
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

    def run_async(
        self,
        overrides: dict,
        output_dir: Path,
        log_file: Path,
        label: str = "load_client_async",
    ) -> tuple[ManagedProcess, Path]:
        """Start JavaLoadClient WITHOUT waiting for it (HA background flow).

        HA cases keep a steady client running ACROSS fault injections
        (that is the whole point — observe the in-flight behaviour).  The
        process is registered on env.load_clients so a mid-case failure
        still gets cleaned up by _stop_env_processes().
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        argv = self._argv()
        menv = self._base_env({**overrides, "OUTPUT_DIR": str(output_dir)})
        proc = ProcessOps.start(argv, menv, log_file)
        env = self.env_manager.current
        if env is not None:
            env.load_clients.append(proc)
        return proc, output_dir

    def stop_async(
        self,
        proc: ManagedProcess,
        output_dir: Path,
        timeout_s: float = 15.0,
    ) -> LoadClientResult:
        """Terminate a run_async client and return its result handle.

        SIGTERM lets the JVM run its shutdown path (jsonl writers flush);
        rows still buffered at the exact kill instant may be lost, which is
        why HA assertions compare pre/post-injection WINDOWS instead of
        exact request totals.
        """
        proc.terminate(timeout_s=timeout_s)
        rc = proc.proc.returncode if proc.proc.returncode is not None else -1
        return LoadClientResult(output_dir, rc)


# ---------------------------------------------------------------------------
# BalanceSampler (balance-metrics v2 — design doc
# flexlb-balance-metrics-v2-design.md §1.5①)
# ---------------------------------------------------------------------------

# Mock per-engine series the sampler records (Prometheus series names; the
# emitter is MockControlServer.appendPerEngineMetrics — 6 gauge / 7 counter
# / 6 exec_ms / 3 TPS families).  The whitelist is a SAMPLER-side concern:
# the /metrics endpoint exposes the full ~25-per-engine surface and the
# case-side store keeps only these.  Master inflight_status fields ride the
# same per-key store under "master:*" keys (see BalanceSampler docstring).
BALANCE_MOCK_SERIES = (
    # gauges
    "mock_engine_running",
    "mock_engine_waiting",
    "mock_engine_cache_blocks",
    "mock_engine_available_blocks",
    "mock_engine_held_blocks",
    "mock_engine_referenced_blocks",
    # cumulative counters
    "mock_engine_accepted_total",
    "mock_engine_completed_total",
    "mock_engine_cancelled_total",
    "mock_engine_kv_admission_fails_total",
    "mock_engine_lack_mem_rejects_total",
    "mock_engine_cache_key_hits_total",
    "mock_engine_cache_keys_requested_total",
    # exec_ms family (gauge-style per-window stats)
    "mock_engine_prefill_ms_avg",
    "mock_engine_prefill_ms_p99",
    "mock_engine_prefill_ms_count",
    "mock_engine_decode_ms_avg",
    "mock_engine_decode_ms_p99",
    "mock_engine_decode_ms_count",
    # production-caliber TPS trio (window = scrape interval)
    "rtp_llm_context_tps",
    "rtp_llm_context_tps_with_cache",
    "rtp_llm_generate_tps",
)


def _http_get_text(url: str, timeout: float = 5.0) -> Optional[str]:
    from online_eval.telemetry import http_text

    try:
        return http_text(url, timeout)
    except Exception:
        return None


def _parse_per_engine_lines(body: str) -> list:
    """Select balance series using the shared exposition parser."""
    return [
        sample
        for sample in parse_prometheus_samples(body, "")
        if sample[0] in BALANCE_MOCK_SERIES
    ]


class BalanceSampler:
    """Per-engine balance-metrics sampler: a daemon thread polling the mock
    /metrics?per_engine=true and the master /rtp_llm/inflight_status once
    per second (design flexlb-balance-metrics-v2-design.md §1.5①).

    Storage is keyed by ENGINE NAME (the per_engine label) for mock series
    and by "master:scheduler" / "master:prefill:<ip:port>" /
    "master:decode:<ip:port>" for the master inflight view — one flat
    ``{key: {metric: [(t_rel_s, v)]}}`` store, so :meth:`window_series`
    addresses both planes with a single metric name (mock series by their
    Prometheus name, master rows by their JSON field name:
    scheduler_inflight / inflight_batches / inflight_requests /
    inflight_route_requests / master_queued / total_load /
    confirmed_accepted / confirmed_running).

    Lifecycle: start() → mark(event) at every case milestone (timestamps are
    seconds relative to the sampling start) → stop() → dump(path) to persist
    the full store + events as json.gz into the case run directory.  Failed
    rounds are skipped, never fatal (best-effort sampling, same contract as
    the G1 poller); an engine that disappears simply stops appending.
    """

    POLL_INTERVAL_S = 1.0

    # Master inflight_status fields worth recording per endpoint row (the
    # prefill plane carries the inflight trio, the decode plane the queued /
    # load quartet — see HttpLoadBalanceServer.inflightStatus).
    _PREFILL_FIELDS = (
        "inflight_batches",
        "inflight_requests",
        "inflight_route_requests",
    )
    _DECODE_FIELDS = (
        "master_queued",
        "total_load",
        "confirmed_accepted",
        "confirmed_running",
    )

    def __init__(self, mock_http_port: int, master_http_port: int):
        self._mock_url = f"http://127.0.0.1:{mock_http_port}/metrics?per_engine=true"
        self._master_url = (
            f"http://127.0.0.1:{master_http_port}/rtp_llm/inflight_status"
        )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._series: dict = {}
        # {engine_name: role} from the per_engine "role" label — the
        # per-role TPS routing key (P -> context_tps, D -> generate_tps,
        # design §1.2 dim 5; the off-role series stay 0 on the Java side).
        self._engine_roles: dict = {}
        self._events: dict = {}
        self._t0: Optional[float] = None

    # -- polling -----------------------------------------------------------

    def _record(self, key: str, metric: str, t_rel: float, value: float) -> None:
        with self._lock:
            per_key = self._series.setdefault(key, {})
            per_key.setdefault(metric, []).append((t_rel, value))

    def _poll_mock(self, t_rel: float) -> None:
        from online_eval.telemetry import shared_samples_since

        samples = shared_samples_since(
            self._mock_url, getattr(self, "_mock_sequence", 0)
        )
        if samples is not None:
            for sample in samples:
                self._mock_sequence = sample["sequence"]
                if sample["error"] is None and sample["monotonic_s"] >= self._t0:
                    self._ingest_mock(sample["body"], sample["monotonic_s"] - self._t0)
            return
        body = _http_get_text(self._mock_url, timeout=2.0)
        if body is not None:
            self._ingest_mock(body, t_rel)

    def _ingest_mock(self, body, t_rel):
        for name, labels, value in _parse_per_engine_lines(body):
            engine = labels.get("engine_name")
            if not engine:
                continue
            role = labels.get("role")
            if role:
                with self._lock:
                    self._engine_roles[engine] = role
            self._record(engine, name, t_rel, value)

    def _poll_master(self, t_rel: float) -> None:
        data = http_get_json(self._master_url, timeout=2.0)
        if not isinstance(data, dict):
            return
        sched = data.get("scheduler_inflight")
        if isinstance(sched, (int, float)):
            self._record("master:scheduler", "scheduler_inflight", t_rel, float(sched))
        for ep in data.get("prefill_endpoints", []) or []:
            key = f"master:prefill:{ep.get('ip_port')}"
            for field in self._PREFILL_FIELDS:
                v = ep.get(field)
                if isinstance(v, list):
                    v = len(v)
                if isinstance(v, (int, float)):
                    self._record(key, field, t_rel, float(v))
        for ep in data.get("decode_endpoints", []) or []:
            key = f"master:decode:{ep.get('ip_port')}"
            for field in self._DECODE_FIELDS:
                v = ep.get(field)
                if isinstance(v, (int, float)):
                    self._record(key, field, t_rel, float(v))

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            t_rel = round(started - self._t0, 3)
            try:
                self._poll_mock(t_rel)
            except Exception:
                pass  # best-effort round; skip
            try:
                self._poll_master(t_rel)
            except Exception:
                pass
            remaining = self.POLL_INTERVAL_S - (time.monotonic() - started)
            if remaining > 0:
                self._stop_event.wait(remaining)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._thread = threading.Thread(
            target=self._loop, name="balance-sampler", daemon=True
        )
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout_s)

    def mark(self, event_name: str) -> float:
        """Record an event timestamp (seconds relative to the sampling start).

        Returns the recorded timestamp so callers can reuse it as the anchor
        of their baseline/transient/steady windows (design §1.4).
        """
        if self._t0 is None:
            raise RuntimeError("BalanceSampler.mark() before start()")
        t_rel = round(time.monotonic() - self._t0, 3)
        with self._lock:
            self._events[event_name] = t_rel
        return t_rel

    def event_time(self, event_name: str) -> Optional[float]:
        with self._lock:
            return self._events.get(event_name)

    def engine_roles(self) -> dict:
        """{engine_name: role} snapshot ("prefill" / "decode", from the
        per_engine role label) — the per-role series-routing key for
        consumers (context vs generate TPS)."""
        with self._lock:
            return dict(self._engine_roles)

    # -- window extraction ----------------------------------------------------

    def window_series(self, metric: str, t_lo: float, t_hi: float, mode: str = "value"):
        """Extract one metric over [t_lo, t_hi] (seconds, sampler axis).

        mode="value" (gauge) returns {key: [(t_rel_s, v), ...]} — the
        in-window sample series per engine, the exact shape
        AssertUtils.balanced() consumes.  mode="delta" (counter) returns
        {key: last − first} — the in-window cumulative delta per engine
        (share/reject accounting: one poll per side means the endpoints
        bracket the window; a single-sample window yields 0).
        """
        out: dict = {}
        with self._lock:
            snapshot = {key: dict(per) for key, per in self._series.items()}
        for key, per_metric in snapshot.items():
            pts = per_metric.get(metric)
            if not pts:
                continue
            in_win = [(t, v) for t, v in pts if t_lo <= t <= t_hi]
            if not in_win:
                continue
            if mode == "delta":
                out[key] = in_win[-1][1] - in_win[0][1]
            else:
                out[key] = in_win
        return out

    def dump(self, path) -> None:
        """Persist the full store + events + roles + meta as json.gz
        (case evidence).

        Reliability contract (code-review fix, three properties):
          1. the payload snapshot is taken under the SAME lock the
             poller thread takes, with per-series list copies — a daemon
             thread that outlives stop()'s join can never mutate the
             dicts mid-serialization;
          2. atomic publish — write a ``.tmp`` sibling then os.replace
             onto the final path, so an interrupted dump never leaves a
             TRUNCATED evidence file behind;
          3. engine_roles rides along so the per-role TPS routing is
             reproducible from the artifact alone.
        """
        with self._lock:
            payload = {
                "sample_s": self.POLL_INTERVAL_S,
                "events": dict(self._events),
                "engine_roles": dict(self._engine_roles),
                "series": {
                    key: {metric: list(pts) for metric, pts in per.items()}
                    for key, per in self._series.items()
                },
            }

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(out_path.name + ".tmp")
        with gzip.open(tmp_path, "wt") as f:
            json.dump(payload, f)
        os.replace(tmp_path, out_path)


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
                    # 修复（eval batch A）：两键齐缺时 fail loudly——静默回
                    # 退 0 会把上游字段改名伪装成"账本干净"。
                    if "inflight_requests" not in ep and "total_load" not in ep:
                        raise RuntimeError(
                            "decode endpoint exposes neither inflight_requests "
                            "nor total_load — inflight_status schema changed: "
                            f"keys={sorted(ep.keys())}"
                        )
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
    def inflight_batches_peak(
        master_http: str,
        engine_role: str,
        window_s: float,
        interval_s: float = 0.2,
    ) -> tuple[list, int]:
        """Sample the master inflight ledger through *window_s* and return
        (samples, peak_batches).

        Each sample is (t_offset_s, scheduler_inflight, role_batches,
        role_requests): *role_batches* / *role_requests* sum the endpoint
        rows for *engine_role* ("prefill": inflight_batches +
        inflight_requests — the master's EnqueueBatch bookkeeping; "decode":
        total_load fallback when the legacy inflight_requests key is
        absent).  Polling every *interval_s* (the master's WorkerStatus
        reconcile runs at a 20ms cadence, so 200ms sampling resolves every
        engine-state transition that outlasts a poll tick).

        This is a SAMPLER, not a verdict: callers assert on the returned
        series (peak, event-driven decay, in-window convergence).  An
        empty sample list means the endpoint never answered inside the
        window — callers treat that as a failure (fail loud).
        """
        key = "prefill_endpoints" if engine_role == "prefill" else "decode_endpoints"
        samples: list = []
        t0 = time.monotonic()
        deadline = t0 + window_s
        while time.monotonic() < deadline:
            data = http_get_json(f"{master_http}/rtp_llm/inflight_status", timeout=5)
            if data is not None:
                batches = 0
                requests = 0
                for ep in data.get(key, []) or []:
                    raw = ep.get("inflight_batches", 0)
                    batches += len(raw) if isinstance(raw, list) else int(raw or 0)
                    req = ep.get("inflight_requests", 0)
                    if req is None and engine_role != "prefill":
                        req = ep.get("total_load", 0)
                    requests += int(req or 0)
                samples.append(
                    (
                        round(time.monotonic() - t0, 3),
                        int(data.get("scheduler_inflight", 0) or 0),
                        batches,
                        requests,
                    )
                )
            time.sleep(interval_s)
        peak = max((s[2] for s in samples), default=-1)
        return samples, peak

    @staticmethod
    def ttft_degradation(
        base_p50: Optional[float], new_p50: Optional[float], threshold_pct: float = 50.0
    ) -> tuple[bool, str]:
        """new_p50 must not exceed base_p50 by more than threshold_pct %.

        A missing baseline/recovery reading FAILS the gate instead of
        silently passing it (fail-loud, the kv_capacity_conflict P7
        philosophy: a missing timing value is a failure state, never a
        skip).  Callers whose Phase-1 baseline fully failed return before
        reaching this gate, so this cannot amplify an existing failure.
        """
        if base_p50 is None or new_p50 is None:
            # 修复（eval batch A）：基线缺失静默直通会令 TTFT 回归门无声
            # 消失——值缺失即失败态（对齐 kv_capacity_conflict 的处理）。
            missing = "baseline" if base_p50 is None else "recovery"
            return False, f"ttft {missing} p50 unavailable — gate not skippable"
        degradation = (new_p50 - base_p50) / base_p50 * 100 if base_p50 > 0 else 0.0
        ok = degradation <= threshold_pct
        return ok, (
            f"ttft p50 {base_p50:.1f} → {new_p50:.1f} ms "
            f"({degradation:+.1f}%, limit {threshold_pct:.0f}%)"
        )

    @staticmethod
    def balanced(
        report,
        prop: str,
        series_by_engine: dict,
        window: tuple,
        stat: str,
        bands,
        context: str = "",
    ) -> tuple:
        """Window-aggregate per-engine series into one statistic, then
        ``report.check(prop, value, bands=bands, context=context)``
        (balance-metrics v2 design §1.5③).

        *series_by_engine*: ``{engine: [(t_rel_s, v)]}`` — the value-mode
        product of :meth:`BalanceSampler.window_series` (any isomorphic
        mapping works).  *window*: ``(t_lo, t_hi)`` relative to the case
        event moment; each engine is aggregated to its in-window MEAN
        first (1s same-tick sampling makes the mean the window
        representative), then the inter-engine statistic is computed:

          max_share — max_i(v_i) / Σv_i   (request/load share caliber, P1)
          min       — min_i(v_i)          (starvation floor)
          cv        — sample stdev / mean across engines (mean 0 → None,
                      noted in context — CV is undefined there)
          spread    — max_i(v_i) − min_i  (absolute dispersion, KV occupancy)
          mean      — mean over engines
          peak      — max over ALL in-window raw samples (queue depth and
                      other absolute peaks — taken from the samples, NOT the
                      per-engine means, so instantaneous spikes survive)
          cluster   — Σ_i(per-engine mean)  (cluster aggregate: TPS sums,
                      count-weighted totals; equals the time-mean of the
                      per-tick cross-engine sum under same-tick sampling)

        A None value (no in-window samples, CV zero-mean, max_share zero
        denominator) is checked as NaN — every band comparison is False, so
        the property records FAIL: missing data fails loud, it never passes
        silently.  Returns ``(passed, value)`` so callers can quote the
        statistic in their case detail.
        """
        t_lo, t_hi = window
        per_engine: dict = {}
        all_samples: list = []
        for engine, pts in series_by_engine.items():
            in_win = [v for t, v in pts if t_lo <= t <= t_hi]
            if not in_win:
                continue
            per_engine[engine] = sum(in_win) / len(in_win)
            all_samples.extend(in_win)
        reason = ""
        value = None
        if stat == "peak":
            value = max(all_samples) if all_samples else None
            if value is None:
                reason = "no in-window samples"
        elif not per_engine:
            reason = "no in-window samples"
        elif stat == "cluster":
            value = sum(per_engine.values())
        elif stat == "mean":
            value = sum(per_engine.values()) / len(per_engine)
        elif stat == "min":
            value = min(per_engine.values())
        elif stat == "max_share":
            total = sum(per_engine.values())
            if total > 0:
                value = max(per_engine.values()) / total
            else:
                reason = "zero total share denominator"
        elif stat == "spread":
            value = max(per_engine.values()) - min(per_engine.values())
        elif stat == "cv":
            vals = list(per_engine.values())
            mean_v = sum(vals) / len(vals)
            if mean_v == 0:
                reason = "zero mean — CV undefined"
            else:
                var = (
                    sum((v - mean_v) ** 2 for v in vals) / (len(vals) - 1)
                    if len(vals) > 1
                    else 0.0
                )
                value = (var**0.5) / mean_v
        else:
            raise ValueError(f"unknown stat: {stat}")
        note = context
        if value is None:
            note = (context + "; " if context else "") + (reason or "value unavailable")
        passed = report.check(
            prop,
            value if value is not None else float("nan"),
            bands=bands,
            context=note,
        )
        return passed, value


def master_prefill_batches_sum(ops) -> int:
    """Sum of master-side prefill inflight_batches across every endpoint
    (-1 when the inflight endpoint is unreachable).

    Shared caliber (status.py / admission.py): the batches a master still
    holds OPEN on its prefill endpoints — one EnqueueBatch's bookkeeping
    stays counted until every member settles (engine splits never multiply
    it; see PrefillState.stats().batchCount()).
    """
    data = ops.master_inflight()
    if data is None:
        return -1
    total = 0
    for ep in data.get("prefill_endpoints", []) or []:
        batches = ep.get("inflight_batches", 0)
        total += len(batches) if isinstance(batches, list) else int(batches)
    return total


def master_prefill_requests_sum(ops) -> int:
    """Sum of master-side prefill inflight_requests (locally-owned member
    accounting) across every endpoint — the member-count caliber behind
    the batch bookkeeping (inflight_batches alone cannot expose whether a
    failed ack member still occupies the batch ledger)."""
    data = ops.master_inflight()
    if data is None:
        return -1
    return sum(
        int(ep.get("inflight_requests", 0) or 0)
        for ep in data.get("prefill_endpoints", []) or []
    )


def master_decode_requests_sum(ops) -> int:
    """Sum of master-side decode inflight_requests across every endpoint
    (total_load caliber — decode rows expose layered admission fields,
    inflight_requests only on the legacy schema)."""
    data = ops.master_inflight()
    if data is None:
        return -1
    return sum(
        int(ep.get("inflight_requests", 0) or ep.get("total_load", 0) or 0)
        for ep in data.get("decode_endpoints", []) or []
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
# Shared environment, traffic and topology helpers
#
# Action implementations use these helpers with their environment and
# EngineOps resources. They share EnvSpec, ConfigOverride and wait_for here.
# ===========================================================================

# Shared stream timeout for the fault-family helpers below (the per-category
# case modules declare their own same-valued constant for case bodies).
STREAM_TIMEOUT_S = 15.0

PREFILL_DOMAIN = "mock.prefill.hosts.address"
DECODE_DOMAIN = "mock.decode.hosts.address"


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
    port.
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
