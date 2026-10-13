"""Master JVM configuration, readiness and single/dual process lifecycle.

The environment owner supplies its logger and an explicit environment. Process
references and incarnation history stay on that environment for cleanup.
"""

from __future__ import annotations
import json
import hashlib
import time
import urllib.request
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from flexlb_cfg import render_env
from adapters.environment_config import MasterSpec
from runtime.java_runtime import JAVA_MODULE_OPTS, resolve_java21
from adapters.master_artifact import configure_master
from adapters.master_control import master_url
from runtime.network import http_post_json, port_accepting
from runtime.startup import startup_deadline, wait_ready
from artifacts.startup import startup_phase
from runtime.paths import API_JAR
from runtime.process import ManagedProcess, ProcessOps

if TYPE_CHECKING:
    from adapters.environment import FlexEnv


BASE_MASTER_ENV = {
    "OTEL_TRACE_SKIP_PATTERN": ".*",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "none",
}


class MasterLifecycle:
    def __init__(self, log):
        self._log = log

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
            if mspec.advertised_ip is not None:
                menv["FLEXLB_ADVERTISED_IP"] = mspec.advertised_ip
            if spec.zk_consistency is not None:
                menv["HIPPO_ROLE"] = mspec.hippo_role or "flexlb_ft_" + hashlib.sha256(str(env.run_dir.resolve()).encode()).hexdigest()[:16]
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
                            "zkTimeoutMs": spec.zk_consistency.session_timeout_ms,
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
        self, env: FlexEnv, log_name: Optional[str] = None, *, deadline=None
    ) -> ManagedProcess:
        deadline = startup_deadline(deadline, 240)
        deadline.check()
        spec = env.spec
        if not API_JAR.is_file():
            raise RuntimeError(f"flexlb-api jar not found: {API_JAR} (build it first)")
        if env.master is not None:
            raise RuntimeError("master already running; stop it first")
        # Pre-flight: a concurrently running master (sibling framework
        # instance) holds the fixed 18080/18081/18082 ports — wait for release
        # instead of dying on BindException while the readiness probe hits the
        # *foreign* master (mis-detected as "up").
        wait_ready(deadline, 'master ports available',
                   lambda: not self._master_ports_in_use(env))
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

        try:
            self._wait_master(env, proc, 'single', '127.0.0.1', env.master_http_port, deadline)
        except Exception as exc:
            app_tail = _app_log_tail_this_start()
            extra = f"\n--- {app_log} (this start) ---\n" + app_tail if app_tail else ""
            raise type(exc)(str(exc) + extra) from exc
        self._log(
            f"master up (pid={proc.pid}, profile={spec.master_profile}, log={log_name})"
        )
        return proc

    def _wait_master(self, env, proc, name, host, port, deadline):
        deadline = startup_deadline(deadline, 240)
        with startup_phase(env.run_dir, f'master/{name}', deadline):
            wait_ready(deadline, f'master/{name}/http',
                       lambda: port_accepting(port, host), process=proc)
            if env.spec.zk_consistency is not None:
                # Production sidecars invoke this hook to join the election.
                # Opening the HTTP listener alone does not register a leader.
                deadline.check()
                with urllib.request.urlopen(f'http://{host}:{port}/hook/after_start',
                        timeout=min(10, deadline.remaining())) as response:
                    if response.status != 200 or response.read().decode().strip() != 'success':
                        raise RuntimeError('Master online hook failed')
            def info():
                status, data = http_post_json(
                    master_url(host, port, 'info'), {},
                    timeout=min(3, deadline.remaining()))
                return data if status == 200 and isinstance(data, dict) else None
            wait_ready(deadline, f'master/{name}/sync',
                       lambda: bool((info() or {}).get('ready')), process=proc)
            if env.spec.master_stable_window_s > 0:
                def topology():
                    data = info()
                    if not data or not data.get('ready'):
                        return False
                    summary = data.get('worker_summary', {})
                    for role, expected in [('PREFILL', env.spec.n_prefill), ('DECODE', env.spec.n_decode)]:
                        if expected <= 0:
                            continue
                        entry = summary.get(role, {})
                        try:
                            if int(entry.get('discovered', -1)) != expected or int(entry.get('alive', -1)) != expected:
                                return False
                        except (TypeError, ValueError):
                            return False
                    return True
                wait_ready(deadline, f'master/{name}/topology', topology,
                           process=proc, stable_s=env.spec.master_stable_window_s)

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

    def _instance_ports_in_use(self, mspec: MasterSpec) -> list[int]:
        """Instance's fixed ports (HTTP / management / gRPC = http+2),
        probed on the instance's OWN bind ip.

        On the Tier-2/3 same-port layout the probe against 127.0.0.2 must
        not be confused by a sibling bound to another IP at the same port.
        """
        ports = [mspec.http_port, mspec.management(), mspec.grpc_port()]
        return [p for p in ports if port_accepting(p, mspec.bind_ip)]

    def start_master_instance(
        self, env: FlexEnv, mspec: MasterSpec, log_name: Optional[str] = None, *, deadline=None
    ) -> ManagedProcess:
        """Start ONE flexlb-api JVM per MasterSpec (HA dual-master path).

        Mirrors start_master()'s readiness ladder (port → /master/info
        ready → engine stable window) with every probe pointed at the
        instance's own bind ip/port, plus the per-instance argv/env keys:
        --server.address, --management.server.address, --flexlb.log.path
        (per-instance log dir), FLEXLB_ADVERTISED_IP and
        FLEXLB_SYNC_CONSISTENCY_CONFIG via _master_env(env, mspec).
        """
        deadline = startup_deadline(deadline, 240)
        deadline.check()
        spec = env.spec
        if not API_JAR.is_file():
            raise RuntimeError(f"flexlb-api jar not found: {API_JAR} (build it first)")
        if env.masters.get(mspec.name) is not None:
            raise RuntimeError(
                f"master instance '{mspec.name}' already running; stop it first"
            )
        wait_ready(deadline, f'master/{mspec.name}/ports available',
                   lambda: not self._instance_ports_in_use(mspec))
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
        self._wait_master(env, proc, mspec.name, mspec.bind_ip, mspec.http_port, deadline)
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

    def restart_master_instance(self, env: FlexEnv, name: str, *, deadline=None) -> ManagedProcess:
        """Mode 1 recovery: fresh JVM from the SAME MasterSpec (cold start —
        in-memory state zeroed, converges from the zero-point)."""
        mspec = env.master_specs[name]
        return self.start_master_instance(env, mspec, deadline=deadline)
