"""Validate case-selected report views; YAML controls presentation, not analysis."""

import re
import string
from pathlib import Path

from cases.registry import VIEW_KINDS, load_capability
from scenario.loader import ScenarioError, load_document

VIEWS = Path(__file__).resolve().parents[2] / "config/report_views"
DEFAULT_VIEW = "workload.yaml"
CHECKS_VIEW = "execution.yaml"
CURVES = frozenset({"mean", "max", "detail"})


def _fail(path, message):
    raise ScenarioError(f"{path}: {message}")


def view(name):
    if type(name) is not str or not re.fullmatch(r"[a-z][a-z0-9_]*\.yaml", name):
        _fail("reports", "view must be a filename under config/report_views")
    path = VIEWS / name
    data = load_document(path)
    if name == CHECKS_VIEW:
        if set(data) != {"kind", "title", "subtitle"} or data["kind"] != "checks":
            _fail(path, "invalid execution view")
    elif name == DEFAULT_VIEW:
        required = {"kind", "title", "subtitle", "group_by", "detail_labels",
                    "summaries", "default_visible", "max_points_per_series", "presets"}
        if set(data) != required or data["kind"] != "default":
            _fail(path, "invalid default view")
        if data["group_by"] != ["epoch", "source", "metric"]:
            _fail(path, "default view must retain metric identity")
        labels = data["detail_labels"]
        if not isinstance(labels, list) or not labels or any(
            type(x) is not str or x not in {"engine_name", "pod", "engine"} for x in labels
        ) or len(set(labels)) != len(labels):
            _fail(path, "invalid detail_labels")
        for field in ("summaries", "default_visible"):
            values = data[field]
            if not isinstance(values, list) or not values or any(
                type(x) is not str or x not in CURVES for x in values
            ) or len(set(values)) != len(values):
                _fail(path, "invalid " + field)
        if set(data["summaries"]) - {"mean", "max"} or set(data["default_visible"]) - set(data["summaries"]):
            _fail(path, "default_visible must select summary curves")
        points = data["max_points_per_series"]
        if type(points) is not int or not 32 <= points <= 2048:
            _fail(path, "max_points_per_series must be 32..2048")
        presets = data["presets"]
        if not isinstance(presets, dict) or not presets or any(
            type(key) is not str or not key or not isinstance(values, list)
            or not values or any(type(item) is not str or item not in CURVES for item in values)
            for key, values in presets.items()
        ):
            _fail(path, "invalid presets")
    elif data.get("kind") in VIEW_KINDS:
        load_capability(VIEW_KINDS[data["kind"]]["validator"])(path, data, _fail)
    else:
        required = {"kind", "report", "producer", "title", "subtitle", "sections"}
        if not required <= set(data) or set(data) - required - {"panel", "panels", "time_origin", "kpis", "meta", "audit_columns", "criteria_columns", "curves", "monitoring_query_plan", "diagnostic_only", "unlisted"} or data["kind"] != "produced":
            _fail(path, "invalid produced report view")
        metrics = data.get("curves", {})
        if not isinstance(metrics, dict):
            _fail(path, "metrics must be a mapping")
        for identity, style in metrics.items():
            if type(identity) is not str or not re.fullmatch(
                r"[a-z][a-z0-9_]*(/[A-Za-z][A-Za-z0-9_]*)*", identity
            ) or not isinstance(style, dict) or not {"name", "group", "axis"} <= set(style) or set(style) - {
                "name", "group", "axis", "source_unit", "unit", "color", "hidden", "primary", "scale", "metric_id", "labels"
            }:
                _fail(path, "invalid metric presentation " + str(identity))
            if any(type(style[field]) is not str or not style[field]
                   for field in ("name", "group", "axis")):
                _fail(path, "invalid metric labels " + identity)
            if any(type(style[field]) is not str for field in ("source_unit", "unit", "color") if field in style):
                _fail(path, "invalid metric unit or color " + identity)
            if any(type(style[field]) is not bool for field in ("hidden", "primary") if field in style):
                _fail(path, "invalid metric visibility " + identity)
            if "scale" in style and (type(style["scale"]) not in (int, float)
                                     or not 0 < style["scale"] < float("inf")):
                _fail(path, "invalid metric scale " + identity)
            if style.get("scale", 1) != 1 and not {"source_unit", "unit"} <= set(style):
                _fail(path, "scaled metric requires source_unit and unit " + identity)
        if "monitoring_query_plan" in data:
            from monitoring.query_plan import load_plan

            query_plan = load_plan(data["monitoring_query_plan"])
            if "diagnostic_only" in data and any(
                not {"unit", "color", "hidden"} <= set(style)
                for style in metrics.values()
            ):
                _fail(path, "classified monitoring styles require unit, color and hidden")
            if data.get("unlisted", "error") != "error":
                _fail(path, "invalid unlisted monitoring policy")
            diagnostic = data.get("diagnostic_only", [])
            known = {f"{kind}/{metric}" for kind, queries in query_plan["sources"].items()
                     for metric in queries}
            if not isinstance(diagnostic, list) or any(
                type(identity) is not str or identity not in known for identity in diagnostic
            ) or len(diagnostic) != len(set(diagnostic)):
                _fail(path, "invalid diagnostic_only metrics")
            if data.get("unlisted", "error") == "error" and known - {style["metric_id"] for style in metrics.values()} - set(diagnostic):
                _fail(path, "monitoring metrics lack presentation or diagnostic classification")
        elif "diagnostic_only" in data or "unlisted" in data:
            _fail(path, "monitoring policy requires monitoring_query_plan")
        for field in ("report", "producer"):
            if type(data[field]) is not str or not re.fullmatch(r"[a-z][a-z0-9-]*", data[field]):
                _fail(path, "invalid " + field)
        if ("panel" in data) == ("panels" in data):
            _fail(path, "declare exactly one of panel or panels")
        panels = data.get("panels", [data.get("panel")])
        if not isinstance(panels, list) or not panels:
            _fail(path, "invalid panels")
        if "panels" in data:
            ids = []
            for panel in panels:
                if not isinstance(panel, dict) or not {"id", "curve_ids"} <= set(panel):
                    _fail(path, "panels require id and metric_ids")
                if type(panel["id"]) is not str or not panel["id"] or panel["id"] in ids:
                    _fail(path, "invalid panel id")
                ids.append(panel["id"])
                if not isinstance(panel["curve_ids"], list) or not panel["curve_ids"] or any(
                    type(name) is not str or name not in metrics for name in panel["curve_ids"]
                ):
                    _fail(path, "invalid panel metric_ids")
        for panel in panels:
            if not isinstance(panel, dict) or "title" not in panel or set(panel) - {"title", "caption", "empty_caption", "presets", "axes", "id", "curve_ids"}:
                _fail(path, "invalid panel presentation")
            for field in ("title", "caption", "empty_caption"):
                if field in panel and (type(panel[field]) is not str or not panel[field]):
                    _fail(path, "invalid panel " + field)
            presets = panel.get("presets", {})
            if not isinstance(presets, dict) or any(
                type(name) is not str or not name or not isinstance(selector, dict)
                or len(selector) != 1 or not set(selector) <= {"names", "contains", "groups", "visible"}
                or any((type(values) is not bool or values is not True) if key == "visible"
                       else (not isinstance(values, list) or not values or any(type(item) is not str or not item for item in values))
                       for key, values in selector.items())
                for name, selector in presets.items()
            ):
                _fail(path, "invalid panel presets")
            if "axes" in panel and (not isinstance(panel["axes"], dict) or any(
                not isinstance(axis, dict) or set(axis) != {"title", "position"}
                or type(axis["title"]) is not str or axis["position"] not in {"left", "right"}
                for axis in panel["axes"].values()
            )):
                _fail(path, "invalid panel axes")
            if "curve_ids" in panel and any(
                metrics[metric_id]["axis"] not in panel.get("axes", {})
                for metric_id in panel["curve_ids"]
            ):
                _fail(path, "panel metric axis is not declared")
        sections = data["sections"]
        if not isinstance(sections, dict) or not sections or any(
            type(key) is not str or type(value) is not str or not value
            for key, value in sections.items()
        ):
            _fail(path, "invalid sections")
        if "time_origin" in data and type(data["time_origin"]) is not str:
            _fail(path, "invalid time_origin")
        if "kpis" in data and (not isinstance(data["kpis"], dict) or any(
            type(key) is not str or type(value) is not str or not value
            for key, value in data["kpis"].items()
        )):
            _fail(path, "invalid kpis")
        if "meta" in data and (not isinstance(data["meta"], dict) or any(
            type(key) is not str or type(value) is not str
            for key, value in data["meta"].items()
        )):
            _fail(path, "invalid meta")
        if "audit_columns" in data and (not isinstance(data["audit_columns"], list)
            or any(type(value) is not str for value in data["audit_columns"])):
            _fail(path, "invalid audit_columns")
        if "criteria_columns" in data and (not isinstance(data["criteria_columns"], list)
            or any(type(value) is not str for value in data["criteria_columns"])):
            _fail(path, "invalid criteria_columns")
    if "curves" in data:
        from monitoring.query_plan import load_plan, definitions
        declared = definitions(load_plan(data["monitoring_query_plan"]))
        for curve_id, style in data["curves"].items():
            if (style.get("metric_id") not in declared or not isinstance(style.get("labels"), dict)
                    or any(type(k) is not str or type(v) is not str for k, v in style["labels"].items())):
                _fail(path, "curve must bind a declared metric id and labels: " + curve_id)
    if any(type(data[key]) is not str or not data[key].strip() for key in ("title", "subtitle")):
        _fail(path, "title and subtitle are required")
    if data["kind"] == "produced":
        try:
            fields = [field for _, field, _, _ in string.Formatter().parse(data["subtitle"])
                      if field is not None]
        except ValueError as exc:
            _fail(path, str(exc))
        if any(field not in {"verdict", "monitoring_status"} for field in fields):
            _fail(path, "unknown subtitle placeholder")
    return data


def declaration(value, *, kind, path="reports"):
    if kind != "workload":
        _fail(path, "report views require test.kind=workload")
    if not isinstance(value, list) or not value or any(type(name) is not str for name in value):
        _fail(path, "expected a nonempty list of view filenames")
    if len(set(value)) != len(value):
        _fail(path, "duplicate view")
    for name in value:
        view(name)
    return list(value)
