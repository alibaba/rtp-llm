"""Lifecycle ownership for mock clusters, Masters and victim engines."""

from __future__ import annotations

import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Optional

from flexlb_cfg import render_env, render_process_config
from mode_profiles import resolve_address_plan
from runtime.environment_config import (
    DEFAULT_MASTER_HTTP_PORT,
    MASTER_PORT_STRIDE,
    MASTER_PORT_MAX_SHIFTS,
    EnvSpec,
    MasterSpec,
)
from runtime.java_runtime import JAVA_MODULE_OPTS, resolve_java21
from runtime.master_artifact import configure_master
from runtime.network import (
    PROBE_BIND_HOST,
    http_post_json,
    port_accepting,
    port_in_use,
    wait_for,
    wait_for_port,
)
from runtime.paths import API_JAR, FLEXLB_DIR, MOCK_JAR
from runtime.process import ManagedProcess, ProcessOps
from runtime.resource_plan import VICTIM_OFFSETS
from runtime.zk_helper import (
    ZK_LAUNCH_CMD_ENV,
    ZK_READY_PREFIX,
    ZK_READY_TIMEOUT_S,
    ZkHelperOps,
)


BASE_MASTER_ENV = {
    "OTEL_TRACE_SKIP_PATTERN": ".*",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "none",
}


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


def _write_master_config(env: "FlexEnv") -> Path:
    """Render the SSOT envelope into run_dir/master_config.json.

    Single render, two projections: this file feeds the mock / victim
    ``--master-config`` argument, while the master env's FLEXLB_CONFIG
    (see EnvManager._master_env) comes from the same flexlb_cfg render —
    both projections use the selected profile and the same overrides.
    """
    path = env.run_dir / "master_config.json"
    spec = env.spec
    path.write_text(
        render_process_config(
            spec.master_profile,
            spec.config_overrides,
            jvm_heap=spec.master_jvm_heap,
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
            needed += [base + offset for offset in VICTIM_OFFSETS]
            # wildcard probe (PROBE_BIND_HOST): the mock binds 0.0.0.0, a
            # loopback-only probe misses foreign binds on other interfaces
            if not any(port_in_use(p, PROBE_BIND_HOST) for p in needed):
                return base
            base += 100
        raise RuntimeError("no free mock port range found")

    def _build(self, spec: EnvSpec) -> FlexEnv:
        self._env_seq += 1
        run_dir = spec.run_dir or self.run_root / f"env{self._env_seq}_{spec.label}"
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
                self.start_master(env, log_name=spec.master_log_name)
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
            str(resolve_address_plan(
                spec.runtime_mode, unique_loopback_supported=sys.platform != "darwin"
            )["unique_engine_ips"]).lower(),
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
            str(spec.diagnostic_events).lower(),
            "--events-file",
            str(env.run_dir / "engine_events.jsonl") if spec.diagnostic_events else "",
            "--prefill-kv-pool-blocks",
            str(spec.prefill_cache_blocks),
            "--decode-kv-pool-blocks",
            str(spec.decode_cache_blocks),
            "--endpoint-file",
            str(env.endpoint_file),
            "--env-file",
            str(env.run_dir / "flexlb_env.txt"),
        ]
        argv += spec.mock_extra_args
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
        if os.environ.get("FLEXLB_SYNC_DIAGNOSTICS") == "1":
            menv["FLEXLB_SYNC_DIAGNOSTICS"] = "1"
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
        return [p for p in ports if port_accepting(p)]

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
            *spec.master_jvm_args,
            *JAVA_MODULE_OPTS,
            "-jar",
            str(API_JAR),
            f"--server.port={env.master_http_port}",
            f"--management.server.port={env.master_management_port}",
            f"--spring.profiles.active={spec.spring_profile}",
        ]
        if not (spec.master_pv_log if spec.master_pv_log is not None else spec.diagnostic_events):
            argv.append("--logging.level.pvLogger=WARN")
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
        master_env = configure_master(env, self._master_env(env), API_JAR)
        proc = ProcessOps.start(argv, master_env, env.run_dir / log_name)
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
            if port_accepting(env.master_http_port):
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
        return [p for p in ports if port_accepting(p, mspec.bind_ip)]

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
        if not spec.diagnostic_events:
            argv.append("--logging.level.pvLogger=WARN")
        if spec.master_debug_log:
            argv.append("--logging.level.org.flexlb=DEBUG")
        argv.extend(spec.master_extra_args)
        argv.extend(mspec.extra_args)
        proc = ProcessOps.start(
            argv, configure_master(env, self._master_env(env, mspec), API_JAR), env.run_dir / log_name
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

    def restart_master_instance(self, env: FlexEnv, name: str) -> ManagedProcess:
        """Mode 1 recovery: fresh JVM from the SAME MasterSpec (cold start —
        in-memory state zeroed, converges from the zero-point)."""
        mspec = env.master_specs[name]
        return self.start_master_instance(env, mspec)

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
        grpc_port = env.base_grpc_port + VICTIM_OFFSETS[1]
        http_port = grpc_port - 1
        # Pre-flight: wait briefly for the ports to be released after a kill -9
        # (mirrors the legacy engine-kill script's stale-port check).
        for _ in range(10):
            if not port_accepting(grpc_port) and not port_accepting(http_port):
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
