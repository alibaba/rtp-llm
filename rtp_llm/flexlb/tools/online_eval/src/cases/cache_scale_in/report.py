"""Present frozen cache-gate results and published metrics; never adjudicate."""

from runtime.observation import evidence_origin

import hashlib
import json
from pathlib import Path

from reporting.view_config import view
from reporting import run_meta, write_bundle
from reporting.run_context import KPI_LABELS, title
from reporting.curves import materialize, project_panels
from reporting.view_sections import view_details, view_table


def prepare_report(directory, evidence):
    from monitoring.session import archived_series
    from reporting.view_config import view
    from statistics import median

    rows = evidence["samples"]
    anchor = evidence_origin(evidence)
    series, sources, gaps, errors = archived_series(directory, anchor)
    archive_paths = sorted(Path(directory).glob("telemetry/*/queries.json"))
    presentation = view("cache_scale_in.yaml")
    from monitoring.metric_store import MetricStore
    from reporting.metric_binding import monitoring_audit
    inventory = monitoring_audit(MetricStore.read(directory), presentation)
    metric_defs = presentation["charts"]["curves"]
    diagnostic_only = set(presentation["metrics"]["diagnostic_only"])
    curves = []
    audit = []
    found = set()
    for key, points in series.items():
        if "promql" not in sources[key]:
            continue
        _, source, metric, label_json = key.split("/", 3)
        if metric == "up":
            continue
        labels = json.loads(label_json)
        identity = sources[key]["metric_id"]
        kind, metric = identity.split("/", 1)
        if labels.get("role") == "decode":
            continue
        from reporting.metric_binding import bindings
        selected_styles = bindings(presentation, identity, labels)
        if not selected_styles:
            if identity in diagnostic_only:
                audit.append([identity, "—", "DIAGNOSTIC_ONLY", sources[key]["promql"]])
            continue
        for curve_id, definition in selected_styles:
            name = definition["name"]
            qualifiers = [str(value) for label, value in sorted(labels.items()) if label not in {"role"}]
            if qualifiers:
                name += " · " + ", ".join(qualifiers)
            found.add((kind, metric))
            visible_end = max((r["t"] for r in rows), default=0)
            valid_points = sum(
                value is not None and 0 <= timestamp <= visible_end
                for timestamp, value in points
            )
            expected = max(1, round((visible_end + 1) / max(sources[key].get("step", 1), 0.001)))
            coverage = min(1, valid_points / expected)
            curves.append(materialize(
                curve_id, definition, points, name=name,
                description=sources[key]["promql"], provenance=sources[key],
            ))
            audit.append([name, f"{coverage:.0%}", "OK" if coverage >= 0.8 else "SPARSE", sources[key]["promql"]])
    for identity, definition in metric_defs.items():
        kind, metric = identity.split("/", 1)
        if (kind, metric) not in found and identity == "mock/context_completed_qps":
            audit.append([definition["name"], "0%", "MISSING", "本次归档没有该监控序列"])

    by_id = {curve["metric_id"]: curve for curve in curves}
    def values(metric_id, start=None, end=None):
        return [
            point["y"]
            for point in by_id.get(metric_id, {}).get("points", [])
            if point["y"] is not None
            and (start is None or point["x"] >= start)
            and (end is None or point["x"] <= end)
        ]
    baseline_start, baseline_end = evidence.get("baseline_start"), evidence.get("baseline_end")
    sent = values("client/actual_send_qps", baseline_start, baseline_end)
    success = values("client/success_qps", baseline_start, baseline_end)
    failures = values("client/error_qps", baseline_start, baseline_end)
    monitor_warnings = [str(error) for error in errors]
    if not archive_paths:
        monitor_warnings.append("缺少 Prometheus queries.json 归档；监控曲线不可用")
    if sent and success and median(sent) > 0 and median(success) / median(sent) < 0.9:
        monitor_warnings.append(
            f"缩容前客户端成功率偏低：success/send 中位数 {median(success) / median(sent):.1%}"
        )
    if sent and failures and median(sent) > 0 and median(failures) / median(sent) > 0.05:
        monitor_warnings.append(
            f"缩容前客户端错误率偏高：error/send 中位数 {median(failures) / median(sent):.1%}"
        )
    monitoring_status = "WARN" if monitor_warnings else "OK"
    return dict(curves=curves, audit=audit, sources=sources, gaps=gaps, errors=errors,
                metric_classification=inventory,
                monitoring_status=monitoring_status, monitor_warnings=monitor_warnings)


def build_spec(directory, evidence, result, prepared):
    presentation = view("cache_scale_in.yaml")
    rows = evidence["samples"]
    curves = list(prepared["curves"])
    from monitoring.metric_store import MetricStore
    survivor_style = presentation["charts"]["curves"]["derived/survivor_hit_ratio"]
    store = MetricStore.read(directory)
    for observation in store.document["metrics"].get("derived/survivor_hit_ratio", []):
        curves.append(materialize(
            "derived/survivor_hit_ratio", survivor_style, observation["points"],
            origin=evidence_origin(evidence), provenance=observation["provenance"],
            description="冻结门禁窗口：仅 survivors 的 hit/context counter delta，x 为窗口结束时刻",
        ))
    audit = prepared["audit"]
    sources, gaps, errors = (prepared[key] for key in ("sources", "gaps", "errors"))
    monitoring_status = prepared["monitoring_status"]
    monitor_warnings = prepared["monitor_warnings"]
    spec = dict(
        run_id=evidence["provenance"]["instance"],
        title=title(evidence["provenance"]["instance"]),
        subtitle=presentation["report"]["subtitle"].format(
            verdict=result["verdict"], monitoring_status=monitoring_status),
        timeOriginLabel="秒；t=0 为观测开始",
        kpis=[
            dict(label=KPI_LABELS["verdict"], value=result["verdict"]),
            dict(label=KPI_LABELS["monitoring"], value=monitoring_status,
                 tone="danger" if monitor_warnings else "success"),
        ],
        panels=project_panels(curves, presentation),
        sections=[
            view_table(presentation, "audit", audit),
            view_details(presentation, "monitoring", dict(status=monitoring_status, warnings=monitor_warnings)),
            view_details(presentation, "checks", result),
            view_details(presentation, "measurement", dict(scope=result["measurement_scope"],
                    client_attribution=evidence.get("client_attribution"),
                    removals=evidence.get("removals"),
                    reinterpretation=evidence.get("reinterpretation"))),
            view_details(presentation, "sources", dict(queries=sources, gaps=gaps, errors=errors,
                metric_classification=prepared["metric_classification"])),
        ],
        timeAxis=dict(min=0, max=max((r["t"] for r in rows), default=1)),
    )
    from reporting.events import attach_events
    origin = evidence_origin(evidence)
    return attach_events(spec, presentation, origin=origin, events=evidence.get("events", []))


def write_report(directory, evidence, result, prepared=None, *, run=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prepared = prepared if prepared is not None else prepare_report(directory, evidence)
    presentation = view("cache_scale_in.yaml")
    spec = build_spec(directory, evidence, result, prepared)
    provenance = evidence.get("provenance", {})
    meta = run_meta(
        dict(id=provenance["instance"]),
        implementation=dict(
            files=provenance.get("files"), master=provenance.get("master_artifact")
        ),
        workload=provenance.get("trace"),
        configuration={
            k: provenance.get(k)
            for k in ("topology", "capacity", "performance", "master_config", "actual_master_config")
        },
        environment=provenance.get("client_environment"),
        evidence=[
            dict(
                path="../../../cache-gate-evidence.json",
                sha256=hashlib.sha256(
                    (directory / "cache-gate-evidence.json").read_bytes()
                ).hexdigest(),
            )
        ],
    )
    if run is not None:
        from reporting.run_context import selected_spec
        from reporting.run_context import provenance_from
        spec = selected_spec(spec, run, presentation)
        meta = provenance_from(run, meta)
        result = dict(result, run=run)
    else:
        from reporting.timeline import archived
        from runtime.observation import verdict_status
        spec = archived(spec, directory, presentation, status=verdict_status(result['verdict']))
    return write_bundle(
        directory,
        "run",
        "cache-scale-in",
        result,
        spec,
        meta=meta,
        producer="cache-gate",
        role="gate",
    )


def validate_view(path, data, fail):
    from reporting.view_schema import validate_section_contract

    validate_section_contract(path, data, {
        "audit": 4, "monitoring": None, "checks": None,
        "measurement": None, "sources": None,
    }, fail)


def render_view(directory, run, presentation):
    from workload.gate_result import load_gate
    evidence, result = load_gate(directory, "cache")
    return write_report(directory, evidence, result, run=run) / "report.html"
