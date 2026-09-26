"""Build each case-selected monitoring view from archived analysis and validated evidence."""

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


def report_identity(run_id, view_name):
    return run_id if view_name == DEFAULT_VIEW else run_id + "--" + Path(view_name).stem


def _gate_sources(payload):
    paths = payload.get("gate_reports")
    if paths is None:
        paths = [payload["gate_report"]] if payload.get("gate_report") else []
    sources = {}
    for entry in paths:
        directory = read_bundle(entry)
        sources[directory.name] = dict(
            result=load_analysis(directory),
            spec=json.loads((directory / "report-spec.json").read_text()),
        )
    return sources


def _metrics_panels(presentation, source):
    original = source["spec"]
    panels, missing = [], []
    for index, selection in enumerate(presentation["panels"]):
        curves = []
        axes = {}
        for name in selection["metrics"]:
            found = [
                (panel, curve)
                for panel in original.get("panels", [])
                for curve in panel.get("series", [])
                if curve.get("name") == name or curve.get("name", "").startswith(name + " · ")
            ]
            if not found:
                missing.append(name)
            for panel, curve in found:
                curves.append(copy.deepcopy(curve))
                if curve.get("axis") in panel.get("axes", {}):
                    axes[curve["axis"]] = copy.deepcopy(panel["axes"][curve["axis"]])
        panels.append(dict(
            id="view-" + str(index), title=selection["title"], overlay=True,
            timeX=True, axes=axes, series=curves,
            presets={"全部": [curve["name"] for curve in curves]},
            caption="曲线取自已归档的专属分析；缺采保持空值。",
        ))
    return panels, missing


def build_spec(payload, directory, view_name=DEFAULT_VIEW, view_links=None):
    presentation = view(view_name)
    sources = _gate_sources(payload)
    if presentation["kind"] == "default":
        panels = build_panels(payload["series"], payload.get("statistic_sources", {}), presentation)
        time_axis = dict(min=0, max=max(
            (point[0] for points in payload["series"].values() for point in points),
            default=1,
        ) or 1)
        time_origin = "t=0 = workload 运行开始"
        events = []
        missing = []
    else:
        source_name = presentation["source_report"]
        if source_name not in sources:
            raise ValueError("view source report was not produced: " + source_name)
        source = sources[source_name]
        panels, missing = _metrics_panels(presentation, source)
        time_axis = source["spec"].get("timeAxis")
        time_origin = source["spec"].get("timeOriginLabel")
        events = source["spec"].get("events", [])

    sections = [
        table("Independent checks", ["Stage / check", "Status", "Evidence"], [
            [row["stage"] + "/" + row["id"], row["status"], row]
            for row in payload["checks"]
        ]),
        details("门禁详细结果", {
            "checks": payload["checks"],
            "analyzers": {name: source["result"] for name, source in sources.items()},
        }),
        details("Playback iterations", payload["iterations"]),
        details("Traffic semantics", payload["traffic_manifests"]),
    ]
    if missing:
        sections.append(details("缺失的专属曲线", missing))
    target = bundle_path(directory, "run", report_identity(payload["id"], view_name)).resolve()
    items = []
    for name, path in (view_links or {}).items():
        if name != view_name:
            items.append(dict(label=Path(name).stem, href=os.path.relpath(Path(path).resolve(), target)))
    for aggregate in payload["workload"].get("stress_aggregates", []):
        if aggregate["status"] == "GENERATED":
            items.append(dict(label="Environment " + aggregate["env_epoch"],
                              href=os.path.relpath(aggregate["report"], target)))
    if items:
        sections.append(links("其他报告视角", items))
    return dict(
        run_id=payload["id"], title=presentation["title"],
        subtitle=presentation["subtitle"],
        timeOriginLabel=time_origin, events=events,
        kpis=[
            dict(label="Execution", value=payload["status"]),
            dict(label="Validity", value=payload["workload"]["runtime_validity"]),
        ],
        panels=panels, timeAxis=time_axis, sections=sections,
    )


def write_report(directory, analysis, *, view_name=DEFAULT_VIEW, view_links=None):
    payload = copy.deepcopy(analysis)
    payload["report_view"] = view_name
    target = bundle_path(directory, "run", report_identity(payload["id"], view_name)).resolve()
    for source in payload.get("request_sources", []):
        source["path"] = os.path.relpath(source["path"], target)
    meta = run_meta(
        dict(id=payload["id"], kind="run", view=view_name),
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
        directory, "run", report_identity(payload["id"], view_name), payload,
        build_spec(analysis, directory, view_name, view_links),
        meta=meta, producer="workload",
    )


def write_views(directory, analysis, names=None):
    """Publish selected custom views, then the default report that links to them."""
    names = names or [DEFAULT_VIEW]
    paths = {
        name: bundle_path(directory, "run", report_identity(analysis["id"], name)) / "report.html"
        for name in names
    }
    for name in [name for name in names if name != DEFAULT_VIEW] + [DEFAULT_VIEW]:
        write_report(directory, analysis, view_name=name, view_links=paths)
    return paths
