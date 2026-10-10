"""Publish selected case views or the default view of all archived metrics."""

import copy

from cases.registry import view_capabilities, ReportNotProduced
from reporting import details, write_bundle
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
        if capability is None or capability.renderer is None:
            raise ValueError("selected view has no registered renderer: " + name)
        try:
            paths[name] = capability.renderer(directory, analysis, presentation)
        except ReportNotProduced:
            if analysis["status"] not in {"FAIL", "ERROR", "TIMEOUT", "BLOCKED"}:
                raise
            analysis.setdefault("unavailable_report_views", []).append(dict(
                view=name, producer=presentation["report"]["producer"], status="NOT_PRODUCED",
                reason="专属门禁证据未完成；保留失败与证据，不补算结论。",
            ))
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
