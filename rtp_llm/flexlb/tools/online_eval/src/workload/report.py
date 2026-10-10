"""Publish selected case views; full monitoring is an explicit diagnostic view."""

import copy
import json
from pathlib import Path

from cases.registry import VIEW_KINDS, load_capability
from reporting import bundle_path, details, load_analysis, read_bundle, write_bundle
from reporting.run_context import KPI_LABELS, canonical_spec, provenance_from
from reporting.view_config import DEFAULT_VIEW, CHECKS_VIEW, view
from workload.report_panels import build_panels


def build_spec(payload, directory, *, name=CHECKS_VIEW):
    presentation = view(name)
    series = payload["series"]
    spec = dict(
        subtitle=presentation["report"]["subtitle"],
        timeOriginLabel="秒；t=0 为 workload 运行开始",
        kpis=[dict(label=KPI_LABELS["execution"], value=payload["status"]),
              dict(label=KPI_LABELS["validity"], value=payload["workload"]["runtime_validity"])],
        panels=build_panels(series, payload.get("statistic_sources", {}), presentation)
               if name == DEFAULT_VIEW else [],
        timeAxis=dict(min=0, max=max((point[0] for points in series.values()
                                    for point in points), default=1) or 1),
        sections=[],
    )
    if payload.get("unavailable_report_views"):
        spec["sections"].append(details("未生成的报告视角", payload["unavailable_report_views"]))
    return canonical_spec(spec, payload)


def write_report(directory, analysis, *, name=CHECKS_VIEW):
    payload = copy.deepcopy(analysis)
    payload["report_view"] = name
    identity = payload["id"] + ("-metrics" if name == DEFAULT_VIEW else "")
    return write_bundle(directory, "run", identity, payload,
                        build_spec(analysis, directory, name=name),
                        meta=provenance_from(payload), producer="workload")


def write_views(directory, analysis, names=None):
    """One canonical selected report, with optional explicit additional views."""
    analysis = copy.deepcopy(analysis)
    names = names or [CHECKS_VIEW]
    paths = {}
    for name in names:
        presentation = view(name)
        if name in {DEFAULT_VIEW, CHECKS_VIEW}:
            continue
        if presentation["kind"] in VIEW_KINDS:
            renderer = load_capability(VIEW_KINDS[presentation["kind"]]["renderer"])
            paths[name] = renderer(directory, analysis, presentation)
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
                     canonical_spec(spec, analysis), meta=provenance_from(analysis, spec.get("run_meta")),
                     producer=manifest["producer"], role=manifest.get("role"))
        paths[name] = bundle / manifest["entrypoint"]
    for name in names:
        if name in {DEFAULT_VIEW, CHECKS_VIEW}:
            paths[name] = write_report(directory, analysis, name=name) / "report.html"
    if not paths or set(paths) == {DEFAULT_VIEW} and names != [DEFAULT_VIEW]:
        paths[CHECKS_VIEW] = write_report(directory, analysis) / "report.html"
    # The canonical report remains primary when full metrics are explicitly added.
    ordered = [name for name in paths if name != DEFAULT_VIEW]
    if DEFAULT_VIEW in paths:
        ordered.append(DEFAULT_VIEW)
    return {name: paths[name] for name in ordered}
