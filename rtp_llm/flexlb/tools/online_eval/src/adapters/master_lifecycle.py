"""Master JVM configuration, readiness and single/dual process lifecycle.

The environment owner supplies its logger and an explicit environment. Process
references and incarnation history stay on that environment for cleanup.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from adapters.environment_config import MasterSpec
from adapters.master_artifact import configure_master
from adapters.master_control import master_url
from artifacts.startup import startup_phase
from file_io import read_json
from flexlb_cfg import render_env
from runtime.java_runtime import JAVA_MODULE_OPTS, resolve_java21
from runtime.network import http_post_json, port_accepting
from runtime.paths import API_JAR
from runtime.process import ManagedProcess, ProcessOps
from runtime.startup import startup_deadline, wait_ready

if TYPE_CHECKING:
    from adapters.environment import FlexEnv


BASE_MASTER_ENV = {"OTEL_TRACE_SKIP_PATTERN": ".*", "OTEL_EXPORTER_OTLP_ENDPOINT": "none"}


def master_jvm_options(spec):
    """Apply the declared heap to the actual JVM, including HA instances."""
    options = list(spec.master_jvm_args)
    if spec.master_jvm_heap is not None:
        options = [item for item in options if not item.startswith("-Xmx")]
        options.append("-Xmx" + spec.master_jvm_heap)
    return options


def master_arguments(env, mspec=None):
    """One JVM/logging policy for fixed and discovered Master instances."""
    spec = env.spec
    http = mspec.http_port if mspec else env.master_http_port
    management = mspec.management() if mspec else env.master_management_port
    argv = [
        resolve_java21(),
        *master_jvm_options(spec),
        *JAVA_MODULE_OPTS,
        "-jar",
        str(API_JAR),
        f"--server.port={http}",
        f"--management.server.port={management}",
        f"--spring.profiles.active={spec.spring_profile}",
    ]
    if mspec:
        log_dir = env.run_dir / (mspec.log_dir_name or f"logs_{mspec.name}")
        argv.extend((f"--server.address={mspec.bind_ip}", f"--management.server.address={mspec.bind_ip}", f"--flexlb.log.path={log_dir}"))
    if not (spec.master_pv_log if spec.master_pv_log is not None else spec.diagnostic_events):
        argv.append("--logging.level.pvLogger=WARN")
    if spec.master_debug_log:
        argv.extend(("--logging.level.org.flexlb=DEBUG", "--logging.level.flexlbLogger=DEBUG"))
    argv.extend(spec.master_extra_args)
    if mspec:
        argv.extend(mspec.extra_args)
    return argv


class MasterLifecycle:
    def __init__(self, log):
        self._log = log

    def master_environment(self, env: FlexEnv, mspec: Optional[MasterSpec] = None) -> dict:
        """Build master configuration documents and the existing HA deployment identity."""
        spec = env.spec
        menv = dict(BASE_MASTER_ENV)
        if "FLEXLB_CONFIG" in spec.master_env:
            # Keep generator/raw-config inputs and the archived launch configuration aligned.
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
                menv["FLEXLB_CONFIG"] = render_env(spec.master_profile, spec.config_overrides)
        menv["HIPPO_ROLE"] = f"flexlb_ft_{spec.label}"
        if spec.discovery == "file":
            payload = read_json(env.endpoint_file)
            menv["MODEL_SERVICE_CONFIG"] = payload["env"]["MODEL_SERVICE_CONFIG"]
        elif spec.discovery == "discovery_file":
            payload = read_json(env.endpoint_file)
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
                            "prefill_endpoint": {"address": "mock.prefill.hosts.address", "protocol": "http", "path": "/"},
                            "decode_endpoint": {"address": "mock.decode.hosts.address", "protocol": "http", "path": "/"},
                        }
                    ],
                },
                separators=(",", ":"),
            )
            service_config = json.loads(menv["MODEL_SERVICE_CONFIG"])
            service_config["hosts"] = {
                f"mock.{role}.hosts.address": [host.strip() for host in spec.domain_addrs[role].split(",") if host.strip()]
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
                    raise RuntimeError("zk_consistency spec requires a live ZK helper (connectString missing) — fail-closed")
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
        ports = [env.master_http_port, env.master_management_port, env.master_http_port + 2]
        return [p for p in ports if port_accepting(p)]

    def start_master(self, env: FlexEnv, log_name: Optional[str] = None, *, deadline=None) -> ManagedProcess:
        return self._start_master(env, None, log_name, deadline)

    def _start_master(self, env, mspec, log_name, deadline):
        """Launch, register and check one owned JVM through the same lifecycle."""
        deadline = startup_deadline(deadline, 240)
        deadline.check()
        name = mspec.name if mspec else "single"
        if not API_JAR.is_file():
            raise RuntimeError(f"flexlb-api jar not found: {API_JAR} (build it first)")
        current = env.masters.get(name) if mspec else env.master
        if current is not None:
            raise RuntimeError(f"master {name!r} already running; stop it first")
        ports = (lambda: self._instance_ports_in_use(mspec)) if mspec else (lambda: self._master_ports_in_use(env))
        wait_ready(deadline, f"master/{name}/ports available", lambda: not ports())
        if mspec:
            generation = env.masters_start_count.get(name, 0) + 1
            env.masters_start_count[name] = generation
        else:
            env.master_start_count += 1
            generation = env.master_start_count
        suffix = f"_{name}" if mspec else ""
        log_name = log_name or (f"flexlb_master{suffix}.log" if generation == 1 else f"flexlb_master{suffix}_restart{generation}.log")
        argv = master_arguments(env, mspec)
        log_root = Path.home() / "ai-whale" / "logs"
        for arg in argv:
            if arg.startswith("--flexlb.log.path="):
                log_root = Path(arg.split("=", 1)[1])
        offsets = {}
        for filename in ("application.log", "flexlb.log", "pv.log"):
            path = log_root / filename
            try:
                offsets[filename] = path.stat().st_size
            except OSError:
                offsets[filename] = 0
        if not mspec:
            env.flexlb_log_offset = offsets["flexlb.log"]
            env.pv_log_offset = offsets["pv.log"]
        proc = ProcessOps.start(argv, configure_master(env, self.master_environment(env, mspec), API_JAR), env.run_dir / log_name)
        host = mspec.bind_ip if mspec else "127.0.0.1"
        http = mspec.http_port if mspec else env.master_http_port
        grpc = mspec.grpc_port() if mspec else http + 2
        if mspec:
            env.masters[name] = proc
            env.master_specs[name] = mspec
        else:
            env.master = proc
        env.master_incarnations.append(
            dict(name=name, target=f"{host}:{grpc}", generation=generation, pid=proc.pid, started_epoch_ms=time.time() * 1000)
        )
        try:
            self._wait_master(env, proc, name, host, http, deadline)
        except Exception as exc:
            app_log = log_root / "application.log"
            try:
                with app_log.open("rb") as stream:
                    stream.seek(offsets["application.log"])
                    tail = "\n".join(stream.read().decode("utf-8", errors="replace").splitlines()[-60:])
            except OSError:
                tail = ""
            extra = f"\n--- {app_log} (this start) ---\n{tail}" if tail else ""
            raise type(exc)(str(exc) + extra) from exc
        self._log(f"master {name!r} up (pid={proc.pid}, http={host}:{http}, grpc={host}:{grpc}, log={log_name})")
        return proc

    def _wait_master(self, env, proc, name, host, port, deadline):
        deadline = startup_deadline(deadline, 240)
        with startup_phase(env.run_dir, f"master/{name}", deadline):
            wait_ready(deadline, f"master/{name}/http", lambda: port_accepting(port, host), process=proc)
            if env.spec.zk_consistency is not None:
                # Production sidecars invoke this hook to join the election.
                # Opening the HTTP listener alone does not register a leader.
                deadline.check()
                with urllib.request.urlopen(f"http://{host}:{port}/hook/after_start", timeout=min(10, deadline.remaining())) as response:
                    if response.status != 200 or response.read().decode().strip() != "success":
                        raise RuntimeError("Master online hook failed")

            def info():
                status, data = http_post_json(master_url(host, port, "info"), {}, timeout=min(3, deadline.remaining()))
                return data if status == 200 and isinstance(data, dict) else None

            wait_ready(deadline, f"master/{name}/sync", lambda: bool((info() or {}).get("ready")), process=proc)
            if env.spec.master_stable_window_s > 0:

                def topology():
                    data = info()
                    if not data or not data.get("ready"):
                        return False
                    summary = data.get("worker_summary", {})
                    for role, expected in [("PREFILL", env.spec.n_prefill), ("DECODE", env.spec.n_decode)]:
                        if expected <= 0:
                            continue
                        entry = summary.get(role, {})
                        try:
                            if int(entry.get("discovered", -1)) != expected or int(entry.get("alive", -1)) != expected:
                                return False
                        except (TypeError, ValueError):
                            return False
                    return True

                wait_ready(deadline, f"master/{name}/topology", topology, process=proc, stable_s=env.spec.master_stable_window_s)

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

    def start_master_instance(self, env: FlexEnv, mspec: MasterSpec, log_name: Optional[str] = None, *, deadline=None) -> ManagedProcess:
        return self._start_master(env, mspec, log_name, deadline)

    def _live_instance(self, env: FlexEnv, name: str) -> ManagedProcess:
        mp = env.masters.get(name)
        if mp is None:
            raise RuntimeError(f"master instance '{name}' is not running")
        return mp

    def master_instance_target(self, env: FlexEnv, name: str) -> str:
        """gRPC target (bind_ip:grpc_port)."""
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
