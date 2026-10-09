"""Lifecycle ownership for mock clusters, Masters and victim engines."""

from __future__ import annotations

import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Optional

from flexlb_cfg import render_process_config
from mode_profiles import resolve_address_plan
from runtime.environment_config import (
    DEFAULT_MASTER_HTTP_PORT,
    MASTER_PORT_STRIDE,
    MASTER_PORT_MAX_SHIFTS,
    EnvSpec,
    MasterSpec,
)
from runtime.java_runtime import resolve_java21
from runtime.master_lifecycle import MasterLifecycle
from runtime.network import PROBE_BIND_HOST, port_accepting, port_in_use, wait_for, wait_for_port
from runtime.paths import FLEXLB_DIR, MOCK_JAR
from runtime.process import ManagedProcess, ProcessOps
from runtime.resource_plan import VICTIM_OFFSETS
from runtime.zk_helper import (
    ZK_LAUNCH_CMD_ENV,
    ZK_READY_PREFIX,
    ZK_READY_TIMEOUT_S,
    ZkHelperOps,
)


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
    (see MasterLifecycle._master_env) comes from the same flexlb_cfg render —
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
        self.master_lifecycle = MasterLifecycle(self._log)
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


    def start_master(self, env: FlexEnv, log_name: Optional[str]=None) -> ManagedProcess:
        return self.master_lifecycle.start_master(env, log_name)

    def stop_master(self, env: FlexEnv, settle_s: float=2.0) -> None:
        return self.master_lifecycle.stop_master(env, settle_s)

    def kill_master9(self, env: FlexEnv) -> None:
        return self.master_lifecycle.kill_master9(env)

    # -- HA dual-master orchestration (gated on spec.masters) --------------


    def start_master_instance(self, env: FlexEnv, mspec: MasterSpec, log_name: Optional[str]=None) -> ManagedProcess:
        return self.master_lifecycle.start_master_instance(env, mspec, log_name)


    def master_instance_target(self, env: FlexEnv, name: str) -> str:
        return self.master_lifecycle.master_instance_target(env, name)

    def kill_master9_instance(self, env: FlexEnv, name: str) -> None:
        return self.master_lifecycle.kill_master9_instance(env, name)

    def restart_master_instance(self, env: FlexEnv, name: str) -> ManagedProcess:
        return self.master_lifecycle.restart_master_instance(env, name)

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
