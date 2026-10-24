"""Environment-scoped monitoring, event capture and final server snapshots."""

import json
import time

from monitoring.collection_plan import instance_plan
from monitoring.query_plan import DEFAULT_PLAN


class RunMonitor:
    def attach(self, ctx):
        self.options = dict(ctx.instance["workload_runtime"])
        self.profile = ctx.instance.get("collection_profile", "request")
        self.monitors = {}
        self.anchor = dict(monotonic_s=ctx.clock(), epoch_s=time.time())
        self.events = []
        self.expected_telemetry = []
        self.telemetry_windows = {}
        self.environments = {}
        self.environment_metadata = {}
        self.master_incarnations = {}
        self.final_snapshots = set()
        self.snapshot_errors = []

    def _snapshot_servers(self, ctx, env, epoch, remaining_s):
        if self.profile != "diagnostic" or epoch in self.final_snapshots or remaining_s <= 0:
            return
        import urllib.request

        targets = {name: f"{spec.bind_ip}:{spec.http_port}" for name, spec in getattr(env, "master_specs", {}).items()}
        if not targets:
            targets = {"single": f"127.0.0.1:{env.master_http_port}"}
        directory = ctx.artifact_dir / "telemetry" / str(epoch)
        directory.mkdir(parents=True, exist_ok=True)
        expires = ctx.clock() + remaining_s
        for name, address in targets.items():
            try:
                timeout = min(self.options["collector_shutdown_s"], expires - ctx.clock())
                if timeout <= 0:
                    raise TimeoutError("final server snapshot budget expired")
                with urllib.request.urlopen(f"http://{address}/rtp_llm/server_latency", timeout=timeout) as response:
                    snapshot = json.load(response)
                if not isinstance(snapshot, dict) or "arrival_count" not in snapshot:
                    raise ValueError("server latency snapshot lacks arrival_count")
                (directory / f"server-latency-{name}.json").write_text(json.dumps(snapshot))
            except Exception as exc:
                self.snapshot_errors.append(dict(env_epoch=epoch, master=name, error=str(exc)))
        self.final_snapshots.add(epoch)

    def _stop_monitor(self, ctx, source, deadline):
        shutdown = self.options["collector_shutdown_s"]
        remaining = max(0, deadline.expires_at - ctx.clock())
        source.stop(shutdown, export=remaining > 0, materialize=False, archive_timeout=max(0, remaining - shutdown), deadline=deadline)

    def before_stage(self, ctx, spec):
        if spec["action"] == "teardown" and ctx.env_epoch in self.monitors:
            # Stop the monitoring interval before intentional target teardown.
            self._stop_monitor(ctx, self.monitors[ctx.env_epoch], ctx.stage_deadline)
        if spec["action"] == "teardown" and ctx.env is not None and self.options["capture_metrics"]:
            self._snapshot_servers(ctx, ctx.env, ctx.env_epoch, max(0, ctx.instance_deadline_s - ctx.clock()))
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
        if not self.options["capture_metrics"] or ctx.env is None or ctx.env_epoch in self.monitors:
            return
        env = ctx.env
        self.environments[str(ctx.env_epoch)] = str(env.run_dir)
        self.environment_metadata[str(ctx.env_epoch)] = dict(
            n_prefill=env.spec.n_prefill, n_decode=env.spec.n_decode, send_mode="case program"
        )
        self.master_incarnations[str(ctx.env_epoch)] = getattr(env, "master_incarnations", [])
        specs = getattr(env, "master_specs", {})
        masters = [(name, value.bind_ip, value.management()) for name, value in specs.items()] or [
            ("single", "127.0.0.1", env.master_management_port)
        ]
        from monitoring.session import PrometheusSession

        targets = {"mock": f"http://127.0.0.1:{env.mock_http_port}/metrics?per_engine=true"}
        targets.update({"master-" + name: f"http://{host}:{port}/prometheus" for name, host, port in masters})
        source = PrometheusSession(
            ctx.artifact_dir / "telemetry" / str(ctx.env_epoch),
            targets,
            self.options["sample_interval_s"],
            self.options.get("max_sample_gap_s", 5),
            query_plan=ctx.instance["execution"]["monitoring"].get("query_plan", DEFAULT_PLAN),
            metric_plan=instance_plan(ctx.instance, ctx.case_sources),
            sources=ctx.case_sources,
            target_kinds={name: "mock" if name == "mock" else "master" for name in targets},
        )
        window = dict(started_epoch_s=time.time(), ended_epoch_s=None)
        for name in source.targets:
            self.telemetry_windows[f"{ctx.env_epoch}/{name}"] = window
            self.expected_telemetry.append(f"{ctx.env_epoch}/{name}")

        # Register cleanup before startup so a failed start cannot leak a process.
        def stop_source(deadline):
            window["ended_epoch_s"] = time.time()
            self._stop_monitor(ctx, source, deadline)

        ctx.add_cleanup("prometheus-" + str(ctx.env_epoch), stop_source)
        from artifacts.startup import startup_phase

        self.monitors[ctx.env_epoch] = source
        with startup_phase(ctx.artifact_dir, "monitoring", ctx.stage_deadline):
            source.start(deadline=ctx.stage_deadline)
        ctx.monitor = source
        epoch = ctx.env_epoch
        ctx.add_cleanup(
            "final-server-snapshot-" + str(epoch), lambda deadline: self._snapshot_servers(ctx, env, epoch, deadline.remaining())
        )
