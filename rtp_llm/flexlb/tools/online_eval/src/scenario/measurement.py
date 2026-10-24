"""Continuous-measurement hooks over the common scenario executor."""

from file_io import write_json
from monitoring.collection_plan import instance_plan
from monitoring.run_monitor import RunMonitor
from scenario.run_result import EvidenceResult, ReportResult, WorkloadResult
from scenario.runtime import execute_instance


class WorkloadPolicy:
    def attach(self, ctx):
        self.reports = ctx.instance.get("reports")
        self.monitoring = RunMonitor()
        self.monitoring.attach(ctx)

    def continue_after_failure(self, spec):
        return spec.get("purpose") == "observation"

    def before_stage(self, ctx, spec):
        self.monitoring.before_stage(ctx, spec)

    def after_stage(self, ctx, spec, row):
        self.monitoring.after_stage(ctx, spec, row)

    def finalize(self, ctx, result, deadline, *, workload=None):
        if workload is None:
            workload = WorkloadResult()
        monitor = self.monitoring
        evidence = ctx.artifact_dir / "workload-evidence.json"
        records = []
        incomplete = []
        expected_outages = []
        producers = {}
        for handle, snapshot in ctx.export_evidence(monitor.profile, deadline):
            expected_outages.extend(snapshot.outages)
            if snapshot.producer is not None:
                producers.setdefault(str(handle["env_epoch"]), set()).add(snapshot.producer)
            if snapshot.requests:
                records.append(dict(resource=handle, **snapshot.requests))
            if snapshot.errors:
                incomplete.append(dict(resource=handle, errors=list(snapshot.errors)))
        for epoch, values in producers.items():
            if epoch in monitor.environment_metadata:
                monitor.environment_metadata[epoch]["load_client_workers"] = len(values)
        from reporting.timeline import freeze, traffic_event

        events = list(ctx.report_events)
        onset = traffic_event(records)
        if onset is not None:
            if any(event["id"] == "traffic_started" for event in events):
                raise ValueError("traffic_started is reserved for issued-request evidence")
            events.append(onset)
        timeline = freeze(ctx.instance["reporting"], events=events, phases=monitor.events) if "reporting" in ctx.instance else None
        payload = dict(
            workload_evidence_schema_version=1,
            instance_id=result["id"],
            clock_anchor=monitor.anchor,
            phases=monitor.events,
            events=events,
            report_timeline=timeline,
            request_resources=records,
            incomplete_request_resources=incomplete,
            expected_telemetry=monitor.expected_telemetry,
            telemetry_windows=monitor.telemetry_windows,
            expected_outages=expected_outages,
            master_incarnations=monitor.master_incarnations,
            final_server_snapshot_errors=monitor.snapshot_errors,
        )
        joined = {"issues": [], "requests": []}
        # Request/engine correlation is a dedicated diagnostic, never a curve source.
        if monitor.profile == "diagnostic":
            from analysis.request_engine import join_evidence

            joined = join_evidence(payload, monitor.environments)
            joined_path = ctx.artifact_dir / "request-engine-evidence.json"
            write_json(joined_path, joined)
            payload["request_engine_join"] = str(joined_path)
        payload["collection_profile"] = monitor.profile
        payload["monitor_backend"] = "prometheus"
        from artifacts.provenance import collect

        payload["runtime_provenance"] = collect(ctx.artifact_dir, monitor.environments, monitor.environment_metadata)
        write_json(evidence, payload)
        workload.details.update(
            evidence=str(evidence), capture_metrics=monitor.options["capture_metrics"], runtime_configuration=dict(monitor.options)
        )
        workload.runtime_validity = (
            "INVALID"
            if incomplete
            or joined["issues"]
            or result["status"] in ("ERROR", "TIMEOUT")
            or any(x["status"] != "PASS" for x in result["cleanup"])
            else "VALID"
        )
        workload.details["collection_profile"] = monitor.profile
        workload.details["monitor_backend"] = "prometheus"
        from monitoring.archive import export_metrics

        metric_plan = instance_plan(ctx.instance, ctx.case_sources)
        metric_store = export_metrics(ctx.artifact_dir, metric_plan)
        metric_store.document["run"] = dict(
            id=result["id"],
            configuration_sha256=result.get("implementation", {}).get("configuration_sha256"),
            report_timeline=timeline,
            phases=monitor.events,
            events=events,
        )
        metric_store.save(ctx.artifact_dir)
        workload.details["metrics"] = str(ctx.artifact_dir / "metrics.json")
        from scenario.registry import registry

        selected = {spec["producer"] for spec in metric_store.document["definitions"].values() if "producer" in spec}
        case_name = result.get("implementation", {}).get("configuration", {}).get("case")
        metrics = registry().cases[case_name].definition.metrics if selected and case_name is not None else None
        if selected and (metrics is None or selected - metrics.CONTRACTS.keys()):
            raise ValueError("metric sources are not owned by the executing case")
        if metrics is not None and selected:
            deadline.check()
            metrics.project(ctx.artifact_dir, selected, deadline)
        finalized = analyze_evidence(ctx.artifact_dir, result, payload, workload)
        analysis = finalized.analysis
        analysis["metric_directory"] = str(ctx.artifact_dir)
        if metrics is not None and selected and hasattr(metrics, "finalize"):
            import json
            from copy import deepcopy
            from types import MappingProxyType

            metadata = metrics.finalize(ctx.artifact_dir, MappingProxyType(deepcopy(analysis)), selected, deadline)
            if not isinstance(metadata, dict) or metadata.keys() - selected:
                raise ValueError("case metric finalization must return selected source metadata")
            json.dumps(metadata, allow_nan=False)
            analysis["producer_results"] = metadata
            deadline.check()
            from monitoring.metric_store import MetricStore

            analysis["series"], analysis["statistic_sources"], _, _ = MetricStore.read(ctx.artifact_dir).series(monitor.anchor["epoch_s"])
        deadline.check()
        return finalized

    def render(self, ctx, analysis, deadline):
        from scenario.registry import write_views as write_views

        deadline.check()
        links = write_views(ctx.artifact_dir, analysis, self.reports, deadline=deadline)
        return ReportResult(links)


def execute_workload(instance, backend, handlers=None, artifact_dir=".", **kwargs):
    if instance.get("test_kind") != "workload":
        raise ValueError("workload executor requires a workload plan")
    return execute_instance(instance, backend, handlers, artifact_dir, _policy=WorkloadPolicy(), **kwargs)


def analyze_evidence(directory, result, evidence, workload):
    from monitoring.metric_store import MetricStore
    from monitoring.quality import assess_collection
    from reporting.workload import analysis_document
    from scenario.outcome import apply_outcome

    series, sources, gaps, errors = MetricStore.read(directory).series(evidence["clock_anchor"]["epoch_s"])
    workload.quality = assess_collection(
        series,
        gaps,
        errors,
        evidence,
        capture_metrics=workload.details.get("capture_metrics", False),
        max_gap_s=workload.details.get("runtime_configuration", {}).get("max_sample_gap_s", 0),
    )
    snapshot = dict(result, workload=workload.to_dict())
    apply_outcome(snapshot)
    return EvidenceResult(workload, analysis_document(directory, snapshot, evidence, series, sources))
