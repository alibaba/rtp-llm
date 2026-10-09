"""Present frozen cache-gate results and published metrics; never adjudicate."""

import hashlib
import json
from pathlib import Path

from reporting.view_config import view
from reporting import details, table, run_meta, write_bundle


def prepare_report(directory, evidence):
    from monitoring.session import archived_series
    from reporting.view_config import view
    from statistics import median

    rows = evidence["samples"]
    anchor = rows[0]["epoch_s"] - rows[0]["t"] if rows else 0
    series, sources, gaps, errors = archived_series(directory, anchor)
    archive_paths = sorted(Path(directory).glob("telemetry/*/queries.json"))
    presentation = view("cache_scale_in_overview.yaml")
    metric_defs = presentation["curves"]
    diagnostic_only = set(presentation["diagnostic_only"])
    curves = []
    audit = []
    found = set()
    for key, points in series.items():
        _, source, metric, label_json = key.split("/", 3)
        if metric == "up":
            continue
        labels = json.loads(label_json)
        kind = "mock" if source == "mock" else "client" if source.startswith("client-") else "master"
        identity = f"{kind}/{metric}"
        if labels.get("role") == "decode":
            continue
        from reporting.metric_binding import bindings
        selected_styles = bindings(presentation, identity, labels)
        if not selected_styles:
            if identity not in diagnostic_only:
                raise ValueError(f"unclassified monitoring metric {identity}")
            audit.append([identity, "—", "DIAGNOSTIC_ONLY", sources[key]["promql"]])
            continue
        for curve_id, definition in selected_styles:
            name, group, axis, unit, color, hidden = (
                definition[field] for field in ("name", "group", "axis", "unit", "color", "hidden")
            )
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
            curves.append(
                dict(
                    curve_id=curve_id, metric_id=definition["metric_id"],
                    name=name,
                    group=group,
                    axis=axis,
                    unit=unit,
                    color=color,
                    points=[dict(x=t, y=v) for t, v in points],
                    hidden=hidden,
                    description=sources[key]["promql"], provenance=sources[key],
                )
            )
            audit.append([name, f"{coverage:.0%}", "OK" if coverage >= 0.8 else "SPARSE", sources[key]["promql"]])
    for identity, definition in metric_defs.items():
        kind, metric = identity.split("/", 1)
        if (kind, metric) not in found and identity in {
            "mock/context_completed_qps",
            "master/schedule_responses_qps"
        }:
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
                monitoring_status=monitoring_status, monitor_warnings=monitor_warnings)


def report_panels(curves, presentation):
    """Project archived monitoring curves into independent presentation panels."""
    panels = []
    for descriptor in presentation["panels"]:
        selected = [dict(curve, hidden=False) for metric_id in descriptor["curve_ids"]
                    for curve in curves
                    if curve["curve_id"] == metric_id]
        missing = [presentation["curves"][metric_id]["name"]
                   for metric_id in descriptor["curve_ids"] if not any(
            curve["curve_id"] == metric_id
            for curve in selected)]
        caption = descriptor["caption"] if selected else descriptor["empty_caption"]
        if selected and missing:
            caption += " 缺少监控序列：" + "、".join(missing) + "。"
        panels.append(dict(id=descriptor["id"], title=descriptor["title"],
                           timeX=True, axes=descriptor["axes"],
                           series=selected, caption=caption))
    return panels


def build_spec(directory, evidence, result, prepared):
    presentation = view("cache_scale_in_overview.yaml")
    rows = evidence["samples"]
    curves = list(prepared["curves"])
    from monitoring.metric_store import MetricStore
    survivor_style = presentation["curves"]["derived/survivor_hit_ratio"]
    curves.append(dict(curve_id="derived/survivor_hit_ratio", metric_id="derived/survivor_hit_ratio",
                       **{field: survivor_style[field]
                          for field in ("name", "group", "axis", "unit", "color", "hidden")},
                       description="冻结门禁窗口：仅 survivors 的 hit/context counter delta，x 为窗口结束时刻",
                       points=[dict(x=t-(rows[0]["epoch_s"]-rows[0]["t"] if rows else 0), y=value)
                               for observation in MetricStore.read(directory).document["metrics"].get("derived/survivor_hit_ratio", [])
                               for t, value in observation["points"]]))
    audit = prepared["audit"]
    sources, gaps, errors = (prepared[key] for key in ("sources", "gaps", "errors"))
    monitoring_status = prepared["monitoring_status"]
    monitor_warnings = prepared["monitor_warnings"]
    return dict(
        run_id="cache-scale-in",
        title=presentation["title"],
        subtitle=presentation["subtitle"].format(
            verdict=result["verdict"], monitoring_status=monitoring_status),
        meta=dict(
            params=evidence["criteria"],
            sampling=presentation["meta"]["sampling"],
            sources=dict(
                aggregate=presentation["meta"]["aggregate"],
                engineDist=presentation["meta"]["engine_dist"],
                runDir=str(Path(directory).resolve()),
            ),
        ),
        timeOriginLabel=presentation["time_origin"],
        events=evidence.get("events", []),
        kpis=[
            dict(label=presentation["kpis"]["verdict"], value=result["verdict"]),
            dict(label=presentation["kpis"]["monitoring"], value=monitoring_status,
                 tone="danger" if monitor_warnings else "success"),
        ],
        panels=report_panels(curves, presentation),
        sections=[
            table(presentation["sections"]["audit"], presentation["audit_columns"], audit),
            details(presentation["sections"]["monitoring"], dict(status=monitoring_status, warnings=monitor_warnings)),
            details(presentation["sections"]["verdict"], result),
            details("测量口径与请求归属", dict(scope=result["measurement_scope"],
                    client_attribution=evidence.get("client_attribution"),
                    removals=evidence.get("removals"),
                    reinterpretation=evidence.get("reinterpretation"))),
            details(presentation["sections"]["sources"], dict(queries=sources, gaps=gaps, errors=errors)),
        ],
        timeAxis=dict(min=0, max=max((r["t"] for r in rows), default=1)),
    )


def write_report(directory, evidence, result, prepared=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prepared = prepared if prepared is not None else prepare_report(directory, evidence)
    spec = build_spec(directory, evidence, result, prepared)
    provenance = evidence.get("provenance", {})
    meta = run_meta(
        dict(id="cache-scale-in", instance=provenance.get("instance")),
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
    write_bundle(
        directory,
        "run",
        "cache-scale-in",
        result,
        spec,
        meta=meta,
        producer="cache-gate",
        role="gate",
    )
    return spec
