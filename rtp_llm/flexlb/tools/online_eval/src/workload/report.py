"""Presentation of already analyzed workload evidence; no status mutation."""

import copy
import os
from pathlib import Path
from workload.report_panels import build_panels
from reporting.view_config import custom_view
from reporting import (
    bundle_path,
    details,
    links,
    table,
    run_meta,
    write_bundle,
)


def build_spec(payload, directory, reports=None):
    series = payload["series"]
    panels = build_panels(series, payload.get("statistic_sources", {}),
                          reports["default"] if reports is not None else None)
    target = bundle_path(directory, "run", payload["id"]).resolve()
    items = []
    for aggregate in payload["workload"].get("stress_aggregates", []):
        if aggregate["status"] != "GENERATED":
            continue
        aggregate_links = [("Environment " + aggregate["env_epoch"], aggregate["report"])]
        aggregate_links += [
            (name + " metrics", entry["report"])
            for name, entry in aggregate.get("master_aggregates", {}).items()
        ]
        items += [
            dict(label=title, href=os.path.relpath(path, target))
            for title, path in aggregate_links
        ]
    gates = payload.get("gate_reports")
    if gates is None:
        gates = [payload["gate_report"]] if payload.get("gate_report") else []
    if reports is not None:
        selected = {custom_view(name)["report"] for name in reports["custom"]}
        gates = [gate for gate in gates if Path(gate).parent.name in selected]
        missing = selected - {Path(gate).parent.name for gate in gates}
        if missing:
            raise ValueError("declared report was not produced: " + ", ".join(sorted(missing)))
    for gate in gates:
        items.append(
            dict(
                label="Gate · " + Path(gate).parent.name,
                href=os.path.relpath(gate, target),
            )
        )
    return dict(
        run_id=payload["id"],
        title=payload["id"],
        timeOriginLabel="t=0 = workload 运行开始",
        subtitle="Independent checks and continuous telemetry",
        kpis=[
            dict(label="Execution", value=payload["status"]),
            dict(label="Validity", value=payload["workload"]["runtime_validity"]),
            dict(label="Performance", value="NOT_EVALUATED"),
        ],
        panels=panels,
        timeAxis=dict(
            min=0, max=max((p[0] for v in series.values() for p in v), default=1) or 1
        ),
        sections=[
            table(
                "Independent checks",
                ["Stage / check", "Status", "Evidence"],
                [
                    [r["stage"] + "/" + r["id"], r["status"], r]
                    for r in payload["checks"]
                ],
            ),
            details("Playback iterations", payload["iterations"]),
            details("Traffic semantics", payload["traffic_manifests"]),
            links("Statistical reports", items),
        ],
    )


def write_report(directory, analysis, *, reports=None):
    payload = copy.deepcopy(analysis)
    if reports is not None:
        payload["report_views"] = copy.deepcopy(reports)
    target = bundle_path(directory, "run", payload["id"]).resolve()
    for source in payload.get("request_sources", []):
        source["path"] = os.path.relpath(source["path"], target)
    meta = run_meta(
        dict(id=payload["id"], kind="run"),
        implementation=payload["implementation"],
        workload=payload["traffic_manifests"],
        configuration=dict(
            declared=payload["configuration"],
            sha256=payload["configuration_sha256"],
            runtime=payload["workload"].get("runtime_configuration"),
        ),
        clock=payload["clock_anchor"],
        evidence=payload["request_sources"],
    )
    return write_bundle(
        directory,
        "run",
        payload["id"],
        payload,
        build_spec(analysis, directory, reports=reports),
        meta=meta,
        producer="workload",
    )
