"""Workload policy: retain independent observations and own continuous telemetry.

Functional execution remains fail-fast. Both policies reuse stage dispatch,
resource ownership and deadline cleanup; workloads never suppress ERROR/TIMEOUT.
"""

import json
import time
from pathlib import Path

from flexlb_eval.scenario.runtime import execute_instance

ROOT = Path(__file__).resolve().parents[3]


class WorkloadPolicy:
    def attach(self, ctx):
        self.options = dict(ctx.instance["workload_runtime"])
        self.profile = ctx.instance.get("collection_profile", "request")
        self.monitors = {}
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
        if (
            self.profile != "diagnostic"
            or epoch in self.final_snapshots
            or remaining_s <= 0
        ):
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
        if spec["action"] == "teardown" and ctx.env_epoch in self.monitors:
            # Stop the monitoring interval before intentional target teardown.
            self.monitors[ctx.env_epoch].stop(self.options["collector_shutdown_s"])
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
        masters = [
            (name, value.bind_ip, value.management()) for name, value in specs.items()
        ] or [("single", "127.0.0.1", env.master_management_port)]
        from flexlb_eval.monitoring.session import PrometheusSession

        targets = {
            "mock": f"http://127.0.0.1:{env.mock_http_port}/metrics?per_engine=true"
        }
        targets.update(
            {
                "master-" + name: f"http://{host}:{port}/prometheus"
                for name, host, port in masters
            }
        )
        source = PrometheusSession(
            ctx.artifact_dir / "telemetry" / str(ctx.env_epoch),
            targets,
            self.options["sample_interval_s"],
            self.options.get("max_sample_gap_s", 5),
        )
        window = dict(started_epoch_s=time.time(), ended_epoch_s=None)
        for name in targets:
            self.telemetry_windows[f"{ctx.env_epoch}/{name}"] = window
            self.expected_telemetry.append(f"{ctx.env_epoch}/{name}")

        # Register cleanup before startup so a failed start cannot leak a process.
        def stop_source(deadline):
            window["ended_epoch_s"] = time.time()
            source.stop(
                min(self.options["collector_shutdown_s"], max(0, deadline.remaining()))
            )

        ctx.add_cleanup("prometheus-" + str(ctx.env_epoch), stop_source)
        source.start()
        self.monitors[ctx.env_epoch] = source
        ctx.monitor = source
        epoch = ctx.env_epoch
        ctx.add_cleanup(
            "final-server-snapshot-" + str(epoch),
            lambda deadline: self._snapshot_servers(
                ctx, env, epoch, deadline.remaining()
            ),
        )

    def finalize(self, ctx, result):
        from flexlb_eval.workload.report import write_report
        from flexlb_eval.workload.evidence_analysis import analyze_report

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
                # Keep one canonical request journal, not a second full copy in workload JSON.
                manifest = {
                    k: v
                    for k, v in snapshot.items()
                    if k not in ("records", "issued", "unfinished")
                }
                if self.profile == "diagnostic":
                    manifest = snapshot
                elif hasattr(value, "directory"):
                    manifest["request_journal"] = str(
                        value.directory / "client_lifecycle.jsonl"
                    )
                records.append(dict(resource=handle, **manifest))
                if not snapshot["complete"]:
                    incomplete.append(dict(resource=handle, errors=snapshot["errors"]))
                continue
            if hasattr(value, "snapshot_records"):
                producers.setdefault(str(handle["env_epoch"]), set()).add("python")
                from flexlb_eval.runtime.requests import completeness

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
        joined = {"issues": [], "requests": []}
        # Request/engine correlation is a dedicated diagnostic, never a curve source.
        if self.profile == "diagnostic":
            from flexlb_eval.workload.evidence import join_evidence

            joined = join_evidence(payload, self.environments)
            joined_path = ctx.artifact_dir / "request-engine-evidence.json"
            joined_path.write_text(json.dumps(joined, allow_nan=False) + "\n")
            payload["request_engine_join"] = str(joined_path)
        payload["collection_profile"] = self.profile
        payload["monitor_backend"] = "prometheus"
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
        # Log-derived legacy aggregates are no longer automatic, nor report curves.
        result["workload"]["stress_aggregates"] = []
        result["workload"]["collection_profile"] = self.profile
        result["workload"]["monitor_backend"] = "prometheus"
        analysis = analyze_report(ctx.artifact_dir, result, payload)
        bundle = write_report(ctx.artifact_dir, analysis)
        result["workload"]["report"] = str(bundle / "report.html")


def execute_workload(instance, backend, handlers=None, artifact_dir=".", **kwargs):
    if instance.get("test_kind") != "workload":
        raise ValueError("workload executor requires a workload plan")
    return execute_instance(
        instance, backend, handlers, artifact_dir, _policy=WorkloadPolicy(), **kwargs
    )
