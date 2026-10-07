"""Assemble the default monitoring view and link case-selected produced views."""

import copy
import json
import os
from pathlib import Path

from reporting import (
    bundle_path, details, links, load_analysis, read_bundle, run_meta, table,
    write_bundle,
)
from reporting.view_config import DEFAULT_VIEW, view
from workload.report_panels import build_panels


def _gate_sources(payload):
    paths = payload.get("gate_reports")
    if paths is None:
        paths = [payload["gate_report"]] if payload.get("gate_report") else []
    sources = {}
    for entry in paths:
        directory = read_bundle(entry)
        sources[directory.name] = load_analysis(directory)
    return sources


def build_spec(payload, directory, view_links=None):
    presentation = view(DEFAULT_VIEW)
    series = payload["series"]
    target = bundle_path(directory, "run", payload["id"]).resolve()
    panels = build_panels(series, payload.get("statistic_sources", {}), presentation)
    sections = [
        table("Independent checks", ["Stage / check", "Status", "Evidence"], [
            [row["stage"] + "/" + row["id"], row["status"], row]
            for row in payload["checks"]
        ]),
        details("门禁详细结果", {
            "checks": payload["checks"],
            "analyzers": _gate_sources(payload),
        }),
        details("Playback iterations", payload["iterations"]),
        details("Traffic semantics", payload["traffic_manifests"]),
    ]
    if payload.get("unavailable_report_views"):
        sections.append(details("未生成的报告视角", payload["unavailable_report_views"]))
    items = [
        dict(label=Path(name).stem, href=os.path.relpath(Path(path).resolve(), target))
        for name, path in (view_links or {}).items() if name != DEFAULT_VIEW
    ]
    for aggregate in payload["workload"].get("stress_aggregates", []):
        if aggregate["status"] == "GENERATED":
            items.append(dict(label="Environment " + aggregate["env_epoch"],
                              href=os.path.relpath(aggregate["report"], target)))
    if items:
        sections.append(links("其他报告视角", items))
    return dict(
        run_id=payload["id"], title=presentation["title"],
        subtitle=presentation["subtitle"],
        timeOriginLabel="t=0 = workload 运行开始",
        kpis=[
            dict(label="Execution", value=payload["status"]),
            dict(label="Validity", value=payload["workload"]["runtime_validity"]),
        ],
        panels=panels,
        timeAxis=dict(min=0, max=max(
            (point[0] for points in series.values() for point in points), default=1
        ) or 1),
        sections=sections,
    )


def write_report(directory, analysis, *, view_links=None):
    payload = copy.deepcopy(analysis)
    payload["report_view"] = DEFAULT_VIEW
    target = bundle_path(directory, "run", payload["id"]).resolve()
    for source in payload.get("request_sources", []):
        source["path"] = os.path.relpath(source["path"], target)
    meta = run_meta(
        dict(id=payload["id"], kind="run", view=DEFAULT_VIEW),
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
        directory, "run", payload["id"], payload,
        build_spec(analysis, directory, view_links), meta=meta, producer="workload",
    )


def write_views(directory, analysis, names=None):
    """Verify produced views, then publish the default view with links to them."""
    analysis = copy.deepcopy(analysis)
    names = names or [DEFAULT_VIEW]
    paths = {DEFAULT_VIEW: bundle_path(directory, "run", analysis["id"]) / "report.html"}
    for name in names:
        if name == DEFAULT_VIEW:
            continue
        presentation = view(name)
        if presentation["kind"] == "ha":
            from workload.ha_view import write_report as write_ha_report

            paths[name] = write_ha_report(directory, analysis, presentation)
            continue
        expected = bundle_path(directory, "run", presentation["report"])
        if not expected.exists() and analysis["status"] in {"FAIL", "ERROR", "TIMEOUT", "BLOCKED"}:
            analysis.setdefault("unavailable_report_views", []).append(dict(
                view=name, producer=presentation["producer"], status="NOT_PRODUCED",
                reason="运行未成功完成，生产该视角的阶段可能未执行；未补算结论。",
            ))
            continue
        bundle = read_bundle(expected)
        manifest = json.loads((bundle / "manifest.json").read_text())
        if manifest.get("producer") != presentation["producer"]:
            raise ValueError("report producer mismatch for " + name)
        paths[name] = bundle / manifest["entrypoint"]
    write_report(directory, analysis, view_links=paths)
    return paths
