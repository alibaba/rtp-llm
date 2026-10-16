"""Java mock backend: bounded owned processes and finite, recorded RPC batches.

Environment lifecycle imports happen only on setup, after lease checks.
"""

import json
import os
import signal
import subprocess
import threading
from dataclasses import replace

from scenario.request_batch import RequestBatch
from scenario.backend_spec import make_env_spec, configure_master_sync_log
from runtime.deadline import StageTimeout


class BoundedOps:
    """A lane-scoped facade; ordinary operations preserve EngineOps semantics."""

    def __init__(self, ops, budget, lease):
        self._ops, self._budget, self._lease = ops, budget, lease
        self._additions = 0
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self._ops, name)

    def add_engine(self, role, port=None):
        if port is not None:
            raise ValueError("scenario adapters cannot allocate explicit worker ports")
        if role not in ("prefill", "decode"):
            raise ValueError("unknown engine role")
        with self._lock:
            if self._additions >= self._budget["max_dynamic_additions"]:
                raise ValueError("declared cumulative engine-add budget exhausted")
            # Count attempts conservatively: an HTTP failure can follow a server
            # side allocation, so its port must not silently become free budget.
            self._additions += 1
            status, body = self._ops.add_engine(role)
            if status == 200:
                allocated = body.get("port") if isinstance(body, dict) else None
                first = self._lease["mock_base"]
                last = (
                    first
                    + self._budget.get(
                        "max_environment_workers", self._budget["initial_workers"]
                    )
                    + self._budget["max_dynamic_additions"]
                    - 1
                )
                if type(allocated) is not int or not first <= allocated <= last:
                    raise RuntimeError(
                        "mock allocated a worker outside its declared lane budget"
                    )
            return status, body


class JavaMockBackend:
    def __init__(self, lease):
        self.lease = lease
        self.manager = self.raw_ops = None
        self.environments = []
        self.owned_processes = {}

    def retain_environment(self, env):
        if env not in self.environments:
            self.environments.append(env)

    def remember_processes(self, env):
        for mp in [
            env.mock,
            env.master,
            env.zk_helper,
            *env.masters.values(),
            *env.victims.values(),
            *env.load_clients,
        ]:
            if mp is not None:
                self.owned_processes[mp.pid] = mp

    def setup(self, ctx, plan, deadline):
        from artifacts.startup import startup_phase
        with startup_phase(ctx.artifact_dir, "environment", deadline):
            return self._setup(ctx, plan, deadline)

    def _setup(self, ctx, plan, deadline, raw_config=None):
        if ":" in str(ctx.artifact_dir.resolve()):
            raise ValueError(
                "Java mock artifact path contains the JVM -Xlog colon delimiter"
            )
        workers = plan["n_prefill"] + plan["n_decode"]
        budget = ctx.instance["resource_budget"]
        bound = budget.get("max_environment_workers", budget["initial_workers"])
        if (
            workers > bound
            or workers + budget["max_dynamic_additions"] > self.lease["worker_capacity"]
        ):
            raise ValueError(
                "environment topology exceeds compiled or leased worker capacity"
            )
        from adapters.engine_ops import EngineOps
        from adapters.environment import EnvManager

        spec = make_env_spec(plan, ctx.instance["profile"], self.lease)
        spec.diagnostic_events = ctx.instance.get("collection_profile", "diagnostic") == "diagnostic"
        # Keep the first-epoch artifact layout compatible with existing runs.
        # Later environments never overwrite its config, logs or cleanup proof.
        artifact_dir = (
            ctx.artifact_dir
            if ctx.env_epoch == 1
            else ctx.artifact_dir / f"environment-epoch-{ctx.env_epoch}"
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self.current_artifact_dir = artifact_dir
        if raw_config is not None:
            spec.raw_config = raw_config
        sync_log_path = (
            configure_master_sync_log(spec, artifact_dir, ctx.env_epoch)
            if plan.get("master_sync_log", False)
            else None
        )
        private_log = sync_log_path.parent if sync_log_path else None
        if not spec.masters and private_log is None:
            private_log = (artifact_dir / "master-logs").resolve()
            private_log.mkdir(exist_ok=False)
            spec.master_extra_args.append(f"--flexlb.log.path={private_log}")
        # Dual Master startup already assigns one private directory per owner.
        # A single Master always gets one too, not only negative-test probes.
        ctx.master_log_dir = private_log
        self.manager = EnvManager(
            artifact_dir / "environment",
            on_environment=self.retain_environment,
            on_processes=self.remember_processes,
            defer_cleanup=True,
            # Invalid configuration probes must reach Master parsing while
            # the mock starts from its valid baseline configuration.
            mock_spec=replace(spec, raw_config=None) if raw_config is not None else None,
        )
        (artifact_dir / "environment.json").write_text(
            json.dumps(
                dict(
                    fingerprint=spec.fingerprint(),
                    lease=self.lease,
                    resolved_config=plan["resolved_config"],
                    raw_config=raw_config,
                    raw_config_target="master" if raw_config is not None else None,
                    env_epoch=ctx.env_epoch,
                    master_log_dir=str(private_log) if private_log else None,
                    master_sync_log_path=str(sync_log_path) if sync_log_path else None,
                ),
                indent=2,
            )
            + "\n"
        )
        env = self.manager.ensure(spec, deadline=deadline)
        env.master_log_dir = private_log
        env.master_sync_log_path = sync_log_path
        ctx.master_sync_log_path = sync_log_path
        deadline.check()
        if (
            env.base_grpc_port != self.lease["mock_base"]
            or env.master_http_port != self.lease["master_base"]
            or env.master_management_port != self.lease["master_base"] + 1
        ):
            raise RuntimeError("environment manager changed a leased port")
        self.raw_ops = EngineOps("127.0.0.1", env.master_http_port, env.mock_http_port)
        ops = BoundedOps(self.raw_ops, ctx.instance["resource_budget"], self.lease)
        return env, ops

    def probe_startup(self, ctx, plan, raw_config, deadline):
        """Actually launch the raw config; parsing a Python mirror is no probe."""
        first = len(self.environments)
        error = None
        try:
            ctx.env, ctx.ops = self._setup(ctx, plan, deadline, raw_config=raw_config)
        except RuntimeError as exc:
            error = repr(exc)
        # Timeout/interrupt/IO failure is not an expected parser rejection.
        deadline.check()
        masters = [
            env.master for env in self.environments[first:] if env.master is not None
        ]
        if not masters:
            raise RuntimeError("startup probe has no owned Master process evidence")
        logs = []
        app_log = self.current_artifact_dir / "master-logs" / "application.log"
        for path in [app_log, *[mp.log_file for mp in masters]]:
            if path.exists():
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 262144))
                    logs.append(
                        dict(
                            path=str(path),
                            tail=stream.read(262144).decode("utf-8", errors="replace"),
                        )
                    )
        return dict(
            startup_error=error,
            started=error is None,
            master_pids=[mp.pid for mp in masters],
            master_returncodes=[mp.proc.poll() for mp in masters],
            logs=logs,
            current_absent_before_cleanup=self.manager.current is None,
        )

    def start_requests(self, ctx, params, deadline):
        from adapters.resource_evidence import request_evidence
        batch = RequestBatch(ctx, params)
        handle = ctx.register_resource(
            "requests", batch, batch.cleanup, historical=True, evidence=request_evidence,
            request_stop=lambda: batch.cancel("cleanup")
        )
        batch.submit(deadline)
        return handle

    def wait_requests(self, ctx, requests, deadline):
        return requests.wait(deadline)

    def cancel_requests(self, ctx, requests, deadline):
        deadline.check()
        return requests.cancel_server(deadline)

    def teardown(self, ctx, deadline):
        errors = []
        if self.raw_ops is not None:
            try:
                self.raw_ops.close()
            except Exception as exc:
                errors.append(("channels", exc))
        processes = list(self.owned_processes.values())
        for env in self.environments:
            processes.extend(env.load_clients)
            processes.extend(env.victims.values())
            processes.extend(env.masters.values())
            processes.extend([env.master, env.zk_helper, env.mock])
        unique = {mp.pid: mp for mp in processes if mp is not None}
        # Signal every owned PID before any wait. No process-name scans or kills.
        for mp in unique.values():
            if mp.alive():
                try:
                    os.kill(mp.pid, signal.SIGCONT)
                    mp.proc.terminate()
                except ProcessLookupError:
                    pass
                except Exception as exc:
                    errors.append((str(mp.pid) + " signal", exc))
        for mp in unique.values():
            if not mp.alive():
                continue
            remaining = max(0, deadline.expires_at - ctx.clock())
            try:
                mp.proc.wait(timeout=min(2, remaining / max(1, len(unique))))
            except subprocess.TimeoutExpired:
                try:
                    mp.proc.kill()
                except ProcessLookupError:
                    pass
                except Exception as exc:
                    errors.append((str(mp.pid) + " kill", exc))
            except Exception as exc:
                errors.append((str(mp.pid) + " wait", exc))
        remaining_pids = []
        for mp in unique.values():
            try:
                mp.proc.wait(timeout=max(0, deadline.expires_at - ctx.clock()))
            except Exception as exc:
                remaining_pids.append(mp.pid)
                errors.append((str(mp.pid) + " reap", exc))
        (ctx.artifact_dir / "process-cleanup.json").write_text(
            json.dumps(
                dict(owned_pids=list(unique), remaining_pids=remaining_pids), indent=2
            )
            + "\n"
        )
        current = getattr(self, "current_artifact_dir", ctx.artifact_dir)
        if current == ctx.artifact_dir:
            current = ctx.artifact_dir / f"environment-epoch-{ctx.env_epoch}"
        current.mkdir(parents=True, exist_ok=True)
        (current / "process-cleanup.json").write_text(
            (ctx.artifact_dir / "process-cleanup.json").read_text()
        )
        if remaining_pids:
            raise StageTimeout(f"owned processes not reaped: {remaining_pids}")
        if self.manager is not None:
            self.manager.current = None
        if errors:
            message = "; ".join(name + ": " + str(exc) for name, exc in errors)
            error_type = StageTimeout if any(isinstance(exc, (TimeoutError, subprocess.TimeoutExpired))
                                             for _, exc in errors) else RuntimeError
            raise error_type(message) from errors[0][1]
