"""Workload policy: retain independent observations and own continuous telemetry.

Functional execution remains fail-fast. Both policies reuse stage dispatch,
resource ownership and deadline cleanup; workloads never suppress ERROR/TIMEOUT.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from ..scenario.runtime import execute_instance

ROOT = Path(__file__).resolve().parents[2]


class WorkloadPolicy:
    def attach(self, ctx):
        self.options = ctx.instance["workload_runtime"]
        self.anchor = dict(monotonic_s=ctx.clock(), epoch_s=time.time())
        self.collectors = {}
        self.events = []
        self.expected_telemetry = []
        self.telemetry_windows = {}
        self.environments = {}
        self.environment_metadata = {}
        self.master_log_directories = {}
        self.master_incarnations = {}
        self.final_snapshots = set()
        self.snapshot_errors = []

    def continue_after_failure(self, spec):
        return spec.get("purpose") == "observation"

    def _snapshot_servers(self, ctx, env, epoch, remaining_s):
        if epoch in self.final_snapshots or remaining_s <= 0:
            return
        import urllib.request

        targets = {
            name: f"{spec.bind_ip}:{spec.http_port}"
            for name, spec in getattr(env, "master_specs", {}).items()
        }
        if not targets:
            targets = {"single": f"127.0.0.1:{env.master_http_port}"}
        directory = ctx.artifact_dir / "telemetry" / str(epoch)
        directory.mkdir(parents=True, exist_ok=True)
        expires = ctx.clock() + remaining_s
        for name, address in targets.items():
            try:
                timeout = min(
                    self.options["collector_shutdown_s"], expires - ctx.clock()
                )
                if timeout <= 0:
                    raise TimeoutError("final server snapshot budget expired")
                with urllib.request.urlopen(
                    f"http://{address}/rtp_llm/server_latency", timeout=timeout
                ) as response:
                    snapshot = json.load(response)
                if not isinstance(snapshot, dict) or "arrival_count" not in snapshot:
                    raise ValueError("server latency snapshot lacks arrival_count")
                (directory / f"server-latency-{name}.json").write_text(
                    json.dumps(snapshot)
                )
            except Exception as exc:
                self.snapshot_errors.append(
                    dict(env_epoch=epoch, master=name, error=str(exc))
                )
        self.final_snapshots.add(epoch)

    def before_stage(self, ctx, spec):
        if (
            spec["action"] == "teardown"
            and ctx.env is not None
            and self.options["capture_metrics"]
        ):
            self._snapshot_servers(
                ctx,
                ctx.env,
                ctx.env_epoch,
                max(0, ctx.instance_deadline_s - ctx.clock()),
            )
        self.events.append(
            dict(
                stage=spec["id"],
                action=spec["action"],
                event="start",
                monotonic_s=ctx.clock(),
                epoch_s=time.time(),
                env_epoch=ctx.env_epoch,
            )
        )

    def after_stage(self, ctx, spec, row):
        self.events.append(
            dict(
                stage=spec["id"],
                action=spec["action"],
                event="end",
                status=row["status"],
                master_generations=dict(getattr(ctx.env, "masters_start_count", {})),
                monotonic_s=ctx.clock(),
                epoch_s=time.time(),
                env_epoch=ctx.env_epoch,
            )
        )
        if (
            not self.options["capture_metrics"]
            or ctx.env is None
            or ctx.env_epoch in self.collectors
        ):
            return
        env = ctx.env
        self.environments[str(ctx.env_epoch)] = str(env.run_dir)
        self.environment_metadata[str(ctx.env_epoch)] = dict(
            n_prefill=env.spec.n_prefill,
            n_decode=env.spec.n_decode,
            send_mode="case program",
        )
        self.master_incarnations[str(ctx.env_epoch)] = getattr(
            env, "master_incarnations", []
        )
        self.collectors[ctx.env_epoch] = []
        specs = getattr(env, "master_specs", {})
        self.master_log_directories[str(ctx.env_epoch)] = {
            name: str(Path(env.run_dir) / (value.log_dir_name or f"logs_{name}"))
            for name, value in specs.items()
        }
        if not specs and getattr(env, "master_log_dir", None):
            self.master_log_directories[str(ctx.env_epoch)]["single"] = str(
                env.master_log_dir
            )
        masters = [(name, value.management()) for name, value in specs.items()] or [
            ("single", env.master_management_port)
        ]
        from online_eval.telemetry import SharedMetricSource

        source = SharedMetricSource(
            f"http://127.0.0.1:{env.mock_http_port}/metrics?per_engine=true",
            ctx.artifact_dir / "telemetry" / str(ctx.env_epoch),
            self.options["sample_interval_s"],
            history_limit=self.options["sample_history_limit"],
        )
        window = dict(started_epoch_s=time.time(), ended_epoch_s=None)
        self.telemetry_windows[f"{ctx.env_epoch}/mock"] = window
        source.start()
        self.expected_telemetry.append(f"{ctx.env_epoch}/mock")

        def stop_source(deadline):
            window["ended_epoch_s"] = time.time()
            source.stop(
                min(self.options["collector_shutdown_s"], max(0, deadline.remaining()))
            )

        ctx.add_cleanup(
            "shared-mock-telemetry-" + str(ctx.env_epoch),
            stop_source,
        )
        for name, port in masters:
            self._start_collector(ctx, name, port)
        epoch = ctx.env_epoch
        ctx.add_cleanup(
            "final-server-snapshot-" + str(epoch),
            lambda deadline: self._snapshot_servers(
                ctx, env, epoch, deadline.remaining()
            ),
        )

    def _start_collector(self, ctx, name, port):
        env = ctx.env
        directory = ctx.artifact_dir / "telemetry" / str(ctx.env_epoch)
        directory.mkdir(parents=True, exist_ok=True)
        self.expected_telemetry.append(f"{ctx.env_epoch}/master-{name}")
        window = dict(started_epoch_s=time.time(), ended_epoch_s=None)
        self.telemetry_windows[f"{ctx.env_epoch}/master-{name}"] = window
        log = (directory / ("collector-" + name + ".log")).open("w")
        try:
            child = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "stress/eval_collectors.py"),
                    "--secondary-interval",
                    str(self.options["sample_interval_s"]),
                    "--prometheus-port",
                    str(port),
                    "--prometheus-out",
                    str(directory / ("master-" + name + ".prom")),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        except BaseException:
            log.close()
            raise
        self.collectors[ctx.env_epoch].append(child)

        def stop(deadline):
            window["ended_epoch_s"] = time.time()
            try:
                exited_early = child.poll() is not None
                if not exited_early:
                    child.terminate()
                try:
                    child.wait(
                        timeout=min(
                            self.options["collector_shutdown_s"],
                            max(0, deadline.remaining()),
                        )
                    )
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                    raise RuntimeError(
                        "workload collector did not stop within cleanup budget"
                    )
                if exited_early or child.returncode not in (0, -15):
                    raise RuntimeError(
                        f"workload collector exited unexpectedly: {child.returncode}"
                    )
            finally:
                log.close()

        ctx.add_cleanup("workload-collector-" + str(ctx.env_epoch) + "-" + name, stop)

    def finalize(self, ctx, result):
        from .report import write_report

        evidence = ctx.artifact_dir / "workload-evidence.json"
        records = []
        incomplete = []
        expected_outages = []
        producers = {}
        for handle, value, historical in ctx._resources.values():
            if handle["kind"] == "master_fault" and getattr(value, "injected", False):
                expected_outages.append(
                    dict(
                        source=f"{handle['env_epoch']}/master-{value.target}",
                        started_epoch_s=value.started_epoch_s,
                        ended_epoch_s=value.restored_epoch_s or time.time(),
                        reason=value.mode,
                    )
                )
            if hasattr(value, "evidence_snapshot"):
                snapshot = value.evidence_snapshot()
                producers.setdefault(str(handle["env_epoch"]), set()).add(
                    "python" if snapshot.get("producer_kind") == "python" else id(value)
                )
                records.append(dict(resource=handle, **snapshot))
                if not snapshot["complete"]:
                    incomplete.append(dict(resource=handle, errors=snapshot["errors"]))
                continue
            if hasattr(value, "snapshot_records"):
                producers.setdefault(str(handle["env_epoch"]), set()).add("python")
                from online_eval.requests import completeness

                rows = value.snapshot_records()
                integrity = completeness(rows)
                if not integrity["complete"]:
                    incomplete.append(
                        dict(
                            resource=handle,
                            errors=["request consumers did not all finish"],
                            detail=integrity,
                        )
                    )
            elif handle["kind"] == "ha_rows" and isinstance(value, list):
                rows = value
            else:
                continue
            records.append(dict(resource=handle, records=rows))
        for epoch, values in producers.items():
            if epoch in self.environment_metadata:
                self.environment_metadata[epoch]["load_client_workers"] = len(values)
        payload = dict(
            schema_version=1,
            instance_id=result["id"],
            clock_anchor=self.anchor,
            phases=self.events,
            request_resources=records,
            incomplete_request_resources=incomplete,
            expected_telemetry=self.expected_telemetry,
            telemetry_windows=self.telemetry_windows,
            expected_outages=expected_outages,
            master_incarnations=self.master_incarnations,
            final_server_snapshot_errors=self.snapshot_errors,
        )
        from .evidence import join_evidence

        joined = join_evidence(payload, self.environments)
        (ctx.artifact_dir / "request-engine-evidence.json").write_text(
            json.dumps(joined, indent=2, allow_nan=False) + "\n"
        )
        payload["request_engine_join"] = str(
            ctx.artifact_dir / "request-engine-evidence.json"
        )
        evidence.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        result["workload"] = dict(
            evidence=str(evidence),
            capture_metrics=self.options["capture_metrics"],
            runtime_configuration=dict(self.options),
            runtime_validity=(
                "INVALID"
                if incomplete
                or joined["issues"]
                or result["status"] in ("ERROR", "TIMEOUT")
                or any(x["status"] != "PASS" for x in result["cleanup"])
                else "VALID"
            ),
            # A valid execution is not proof that a calibrated performance band passed.
            performance_verdict="NOT_EVALUATED",
        )
        from .aggregate import aggregate_workload

        result["workload"]["stress_aggregates"] = aggregate_workload(
            ctx.artifact_dir,
            joined,
            self.environments,
            self.anchor,
            ctx.instance["execution"]["cleanup_timeout_s"],
            master_log_directories=self.master_log_directories,
            environment_metadata=self.environment_metadata,
        )
        if any(
            row["status"] == "ERROR" for row in result["workload"]["stress_aggregates"]
        ):
            result["workload"]["runtime_validity"] = "INVALID"
        write_report(ctx.artifact_dir, result, payload)


def execute_workload(instance, backend, handlers=None, artifact_dir=".", **kwargs):
    if instance.get("test_kind") != "workload":
        raise ValueError("workload executor requires a workload plan")
    return execute_instance(
        instance, backend, handlers, artifact_dir, _policy=WorkloadPolicy(), **kwargs
    )
