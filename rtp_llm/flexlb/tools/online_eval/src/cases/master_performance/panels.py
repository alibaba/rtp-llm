"""Selectable request and archived Prometheus views; never affect gate decisions."""


from monitoring.metric_store import MetricStore
from reporting.curves import materialize, project_panels


def prepare_curves(directory, evidence, presentation):
    import json

    lo = evidence.get("window", {}).get("start_epoch_ms", 0)
    curves, audit = [], []

    def add(curve_id, points, description, *, name=None, labels=None, provenance=None):
        style = presentation.charts.curves[curve_id]
        curves.append(materialize(
            curve_id, style, points, name=name, labels=labels,
            unit=store.document["definitions"][style.metric_id]["unit"],
            description=description, provenance=provenance,
            color_index=list(presentation.charts.curves).index(curve_id),
        ))

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

    series, sources, gaps, errors = MetricStore.read(directory).series(lo / 1000)
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
            name = style.name
            if residual:
                name += " · " + ", ".join(f"{k}={v}" for k, v in sorted(residual.items()))
            if any(c["name"] == name for c in curves):
                name += " · epoch " + epoch
            visible = points
            add(curve_id, visible, sources[key]["promql"], name=name, labels=labels, provenance=sources[key])
            audit.append(dict(name=name, samples=sum(v is not None for _,v in visible), **sources[key]))
    return curves, dict(queries=audit, gaps=gaps, errors=errors, available=bool(series), metric_classification=inventory)


def report_panels(curves, result, presentation, *, origin=0):
    from reporting.gate_lines import apply_gate_lines
    return apply_gate_lines(dict(panels=project_panels(curves, presentation), timeOriginEpochS=origin),
                            presentation, result.get('gate_references', []))['panels']
