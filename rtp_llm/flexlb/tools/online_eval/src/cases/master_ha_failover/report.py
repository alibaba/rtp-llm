"""Project HA request and Master-state evidence into the case-selected view."""

from reporting import write_bundle
from reporting.view_sections import view_details
from reporting.run_context import title, KPI_LABELS, canonical_spec, provenance_from


def build_spec(payload, presentation):
    from monitoring.metric_store import MetricStore
    store = MetricStore.read(payload["metric_directory"])
    anchor = payload["clock_anchor"]["epoch_s"]
    observations = store.document["metrics"]
    metadata = payload["ha_metric_metadata"]
    panels = []
    for descriptor in presentation["charts"]["panels"]:
        curves = []
        for curve_id in descriptor["curve_ids"]:
            style = presentation["charts"]["curves"][curve_id]
            identity = style["metric_id"]
            for row in observations.get(identity, []):
                labels = row["labels"]
                color = style["color"]
                if any(labels.get(k) != v for k,v in style["labels"].items()):
                    continue
                curves.append(dict(curve_id=curve_id, metric_id=identity,
                    name=style["name"].format(**labels), group=style["group"].format(**labels),
                    unit=store.document["definitions"][identity]["unit"],
                    axis=style["axis"], color=color,
                    points=[dict(x=t-anchor, y=value) for t, value in row["points"]],
                    provenance=row["provenance"]))
        panels.append(dict(
            id=descriptor["id"], title=descriptor["title"], caption=descriptor["caption"],
            timeX=True, axes=descriptor["axes"], series=curves,
        ))
    sections = [
        view_details(presentation, "sources", dict(metadata,
            metrics=payload["metric_directory"] + "/metrics.json",
            monitoring=payload["workload"].get("telemetry_completeness"),
            telemetry_errors=payload["workload"].get("telemetry_integrity_errors", []),
            telemetry_warnings=payload["workload"].get("telemetry_warnings", []))),
    ]
    maximum = max(
        [point["x"] for panel in panels for curve in panel["series"] for point in curve["points"]]
        + [1]
    )
    spec = dict(
        run_id=payload["id"], title=title(payload["id"]), subtitle=presentation["report"]["subtitle"],
        timeOriginLabel="秒；t=0 为 workload 运行开始", timeAxis=dict(min=0, max=maximum),
        kpis=[dict(label=KPI_LABELS["execution"], value=payload["status"]),
              dict(label=KPI_LABELS["validity"], value=payload["workload"]["runtime_validity"]),
              dict(label=KPI_LABELS["request_count"], value=metadata["request_count"])],
        panels=panels, sections=sections,
    )
    from reporting.events import attach_events
    attach_events(spec, presentation, origin=anchor,
                  phases=payload.get("phases", []), events=payload.get("events", []))
    spec["timeAxis"]["max"] = max(maximum, max((event["t"] for event in spec["events"]), default=1))
    return spec


def write_report(directory, payload, presentation):
    spec = canonical_spec(build_spec(payload, presentation), payload)
    bundle = write_bundle(
        directory, "run", payload["id"] + "-" + presentation["report"]["id"],
        dict(payload, report_view="master_ha_core.yaml"),
        spec, meta=provenance_from(payload), producer=presentation["report"]["producer"],
    )
    return bundle / "report.html"


def validate_view(path, data, fail):
    from reporting.view_schema import validate_section_contract
    validate_section_contract(path, data, {"sources": None}, fail)
