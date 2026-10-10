"""Publish selected case views or the default view of all archived metrics."""

import copy
import json
from pathlib import Path

from cases.registry import view_capabilities
from reporting import bundle_path, details, load_analysis, read_bundle, write_bundle
from reporting.run_context import KPI_LABELS, canonical_spec, provenance_from
from reporting.view_config import DEFAULT_VIEW, view
from workload.report_panels import build_panels


def build_spec(payload, directory, *, name=DEFAULT_VIEW):
    presentation = view(name)
    series = payload["series"]
    spec = dict(
        subtitle=presentation["report"]["subtitle"],
        timeOriginLabel="秒；t=0 为 workload 运行开始",
        kpis=[dict(label=KPI_LABELS["execution"], value=payload["status"]),
              dict(label=KPI_LABELS["validity"], value=payload["workload"]["runtime_validity"])],
        panels=build_panels(series, payload.get("statistic_sources", {}), presentation),
        timeAxis=dict(min=0, max=max((point[0] for points in series.values()
                                    for point in points), default=1) or 1),
        sections=[],
    )
    if payload.get("unavailable_report_views"):
        spec["sections"].append(details("未生成的报告视角", payload["unavailable_report_views"]))
    from reporting.events import attach_events
    attach_events(spec, presentation, origin=payload["clock_anchor"]["epoch_s"],
                  phases=payload.get("phases", []), events=payload.get("events", []))
    return canonical_spec(spec, payload)


def write_report(directory, analysis, *, name=DEFAULT_VIEW):
    payload = copy.deepcopy(analysis)
    payload["report_view"] = name
    identity = payload["id"]
    return write_bundle(directory, "run", identity, payload,
                        build_spec(analysis, directory, name=name),
                        meta=provenance_from(payload), producer="workload")


def write_views(directory, analysis, names=None):
    """One canonical selected report, with optional explicit additional views."""
    analysis = copy.deepcopy(analysis)
    names = names or [DEFAULT_VIEW]
    paths = {}
    for name in names:
        presentation = view(name)
        if name == DEFAULT_VIEW:
            continue
        capability = view_capabilities().get(name)
        if capability is not None and capability.renderer is not None:
            paths[name] = capability.renderer(directory, analysis, presentation)
            continue
        expected = bundle_path(directory, "run", presentation["report"]["id"])
        if not expected.exists() and analysis["status"] in {"FAIL", "ERROR", "TIMEOUT", "BLOCKED"}:
            analysis.setdefault("unavailable_report_views", []).append(dict(
                view=name, producer=presentation["report"]["producer"], status="NOT_PRODUCED",
                reason="专属报告生产阶段未完成；保留失败与证据，不补算结论。",
            ))
            continue
        bundle = read_bundle(expected)
        manifest = json.loads((bundle / "manifest.json").read_text())
        if manifest.get("producer") != presentation["report"]["producer"]:
            raise ValueError("report producer mismatch for " + name)
        spec = json.loads((bundle / "report-spec.json").read_text())
        frozen = load_analysis(bundle)
        frozen["run"] = analysis
        write_bundle(directory, "run", manifest["id"], frozen,
                     _canonical_selected(spec, analysis, presentation), meta=provenance_from(analysis, spec.get("run_meta")),
                     producer=manifest["producer"], role=manifest.get("role"))
        paths[name] = bundle / manifest["entrypoint"]
    for name in names:
        if name == DEFAULT_VIEW:
            paths[name] = write_report(directory, analysis, name=name) / "report.html"
    if not paths:
        paths[DEFAULT_VIEW] = write_report(directory, analysis) / "report.html"
    # A selected report remains primary when the default metric view is also requested.
    ordered = [name for name in paths if name != DEFAULT_VIEW]
    if DEFAULT_VIEW in paths:
        ordered.append(DEFAULT_VIEW)
    return {name: paths[name] for name in ordered}


def _canonical_selected(spec, analysis, presentation):
    from reporting.events import attach_events
    return attach_events(canonical_spec(spec, analysis), presentation,
                         origin=spec["timeOriginEpochS"],
                         phases=analysis.get("phases", []), events=analysis.get("events", []))
