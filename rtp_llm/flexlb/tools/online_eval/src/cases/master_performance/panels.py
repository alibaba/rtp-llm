"""Selectable request and archived Prometheus views; never affect gate decisions."""


from monitoring.session import archived_series
from reporting.curves import materialize, project_panels, apply_panel_presets
from reporting.catalog import PALETTE


def prepare_curves(directory, evidence, presentation):
    import json

    lo = evidence.get("window", {}).get("start_epoch_ms", 0)
    duration = evidence.get("criteria", {}).get("measure_s", 1)
    curves, audit = [], []

    def add(curve_id, points, description, *, name=None, labels=None, provenance=None):
        style = presentation["charts"]["curves"][curve_id]
        curves.append(materialize(
            curve_id, style, points, name=name, labels=labels,
            unit=store.document["definitions"][style["metric_id"]]["unit"],
            description=description, provenance=provenance,
            color_index=list(presentation["charts"]["curves"]).index(curve_id),
        ))

    from monitoring.metric_store import MetricStore
    store = MetricStore.read(directory)
    from reporting.metric_binding import monitoring_audit
    inventory = monitoring_audit(store, presentation)
    from reporting.metric_binding import bindings
    for identity, rows in store.document["metrics"].items():
        if not identity.startswith("request/"):
            continue
        for row in rows:
            for curve_id, _ in bindings(presentation, identity, row["labels"]):
                add(curve_id, [(t-lo/1000, value) for t,value in row["points"]],
                    "逐请求证据；cohort 和完成窗口由指标生产器冻结，不替代整窗门禁 p99",
                    labels=row["labels"], provenance=row["provenance"])

    series, sources, gaps, errors = archived_series(directory, lo / 1000)
    for key, points in series.items():
        if "promql" not in sources[key]:
            continue
        epoch, source, metric, label_json = key.split("/", 3)
        if metric == "up":
            continue
        labels = json.loads(label_json)
        identity = sources[key]["metric_id"]
        selected = bindings(presentation, identity, labels)
        residual = {k:v for k,v in labels.items() if k != "role"}
        for curve_id, style in selected:
            name = style["name"]
            if residual:
                name += " · " + ", ".join(f"{k}={v}" for k, v in sorted(residual.items()))
            if any(c["name"] == name for c in curves):
                name += " · epoch " + epoch
            visible = [(t, v) for t, v in points if 0 <= t <= duration]
            add(curve_id, visible, sources[key]["promql"], name=name, labels=labels, provenance=sources[key])
            audit.append(dict(name=name, samples=sum(v is not None for _,v in visible), **sources[key]))
    return curves, dict(queries=audit, gaps=gaps, errors=errors, available=bool(series), metric_classification=inventory)


def report_panels(curves, criteria, presentation):
    """Show the four measured views; keep gate floors separate from measurements."""
    panels = project_panels(curves, presentation)
    duration = criteria.get("measure_s", 1)
    floor_metrics = {
        "mock/rtp_llm_context_tps_engine_mean/P": "mock/rtp_llm_context_tps",
        "mock/rtp_llm_context_tps_with_cache_engine_mean/P": "mock/rtp_llm_context_tps_with_cache",
        "mock/rtp_llm_generate_tps_engine_mean/D": "mock/rtp_llm_generate_tps",
    }
    for panel, descriptor in zip(panels, presentation["charts"]["panels"]):
        selected = panel["series"]
        if descriptor["id"] == "engine-tps":
            floors = criteria.get("engine_tps", {})
            for metric_id in descriptor["curve_ids"]:
                metric = floor_metrics[metric_id]
                if metric in floors:
                    source = next((curve for curve in selected if curve["curve_id"] == metric_id), None)
                    name = presentation["charts"]["curves"][metric_id]["name"]
                    selected.append(dict(
                        curve_id=metric_id + "/gate_floor", source_type="configuration",
                        name=name + " 门禁线", group="门禁", axis="forward",
                        unit="执行 tok/s", color=source["color"] if source else presentation["charts"]["curves"][metric_id].get("color") or PALETTE[len(selected) % len(PALETTE)],
                        dash=[6, 4], hidden=False,
                        points=[dict(x=t, y=floors[metric]) for t in (0, duration)],
                        description="场景配置中的绝对下界；不是实测值",
                    ))
        apply_panel_presets(panel, descriptor)
    return panels
