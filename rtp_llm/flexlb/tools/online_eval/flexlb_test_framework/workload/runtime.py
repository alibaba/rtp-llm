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

    def continue_after_failure(self, spec):
        return spec.get("purpose") == "observation"

    def before_stage(self, ctx, spec):
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
        self.collectors[ctx.env_epoch] = []
        specs = getattr(env, "master_specs", {})
        masters = [(name, value.management()) for name, value in specs.items()] or [
            ("single", env.master_management_port)
        ]
        for index, (name, port) in enumerate(masters):
            self._start_collector(ctx, name, port, include_mock=index == 0)

    def _start_collector(self, ctx, name, port, include_mock):
        env = ctx.env
        directory = ctx.artifact_dir / "telemetry" / str(ctx.env_epoch)
        directory.mkdir(parents=True, exist_ok=True)
        log = (directory / ("collector-" + name + ".log")).open("w")
        try:
            child = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "stress/eval_collectors.py"),
                    "--secondary-interval",
                    str(self.options["sample_interval_s"]),
                    *(
                        [
                            "--mock-port",
                            str(env.mock_http_port),
                            "--mock-out",
                            str(directory / "mock.prom"),
                        ]
                        if include_mock
                        else []
                    ),
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
        for handle, value, historical in ctx._resources.values():
            if hasattr(value, "snapshot_records"):
                rows = value.snapshot_records()
            elif handle["kind"] == "ha_rows" and isinstance(value, list):
                rows = value
            else:
                continue
            records.append(dict(resource=handle, records=rows))
        payload = dict(
            schema_version=1,
            instance_id=result["id"],
            clock_anchor=self.anchor,
            phases=self.events,
            request_resources=records,
        )
        evidence.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        result["workload"] = dict(
            evidence=str(evidence),
            capture_metrics=self.options["capture_metrics"],
            runtime_configuration=dict(self.options),
            runtime_validity=(
                "INVALID"
                if result["status"] in ("ERROR", "TIMEOUT")
                or any(x["status"] != "PASS" for x in result["cleanup"])
                else "VALID"
            ),
            # A valid execution is not proof that a calibrated performance band passed.
            performance_verdict="NOT_EVALUATED",
        )
        write_report(ctx.artifact_dir, result, payload)


def execute_workload(instance, backend, handlers=None, artifact_dir=".", **kwargs):
    if instance.get("test_kind") != "workload":
        raise ValueError("workload executor requires a workload plan")
    return execute_instance(
        instance, backend, handlers, artifact_dir, _policy=WorkloadPolicy(), **kwargs
    )
