"""Validate case-selected report views; YAML controls presentation, not analysis."""

import re
import string
from pathlib import Path

from scenario.loader import ScenarioError, load_document

VIEWS = Path(__file__).resolve().parents[2] / "config/report_views"
DEFAULT_VIEW = "workload.yaml"
CURVES = frozenset({"mean", "max", "detail"})


def _fail(path, message):
    raise ScenarioError(f"{path}: {message}")


def view(name):
    if type(name) is not str or not re.fullmatch(r"[a-z][a-z0-9_]*\.yaml", name):
        _fail("reports", "view must be a filename under config/report_views")
    path = VIEWS / name
    data = load_document(path)
    if name == DEFAULT_VIEW:
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
    else:
        required = {"kind", "report", "producer", "title", "subtitle", "panel", "sections"}
        if not required <= set(data) or set(data) - required - {"time_origin", "kpis", "meta", "audit_columns", "criteria_columns", "comparison"} or data["kind"] != "produced":
            _fail(path, "invalid produced report view")
        for field in ("report", "producer"):
            if type(data[field]) is not str or not re.fullmatch(r"[a-z][a-z0-9-]*", data[field]):
                _fail(path, "invalid " + field)
        panel = data["panel"]
        if not isinstance(panel, dict) or "title" not in panel or set(panel) - {"title", "caption", "empty_caption", "presets", "axes"}:
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
        if "comparison" in data:
            comparison = data["comparison"]
            required_comparison = {"title", "subtitle", "overlay_title", "overlay_caption", "sections"}
            if not isinstance(comparison, dict) or not required_comparison <= set(comparison) or set(comparison) - required_comparison - {
                "core_metrics", "kpi_label_suffix", "metrics_columns"
            } or any(
                type(comparison[key]) is not str or not comparison[key]
                for key in ("title", "subtitle", "overlay_title", "overlay_caption")
            ) or not isinstance(comparison["sections"], dict) or any(
                type(key) is not str or type(value) is not str or not value
                for key, value in comparison["sections"].items()
            ) or "core_metrics" in comparison and (
                not isinstance(comparison["core_metrics"], list) or any(
                    type(name) is not str for name in comparison["core_metrics"]
                )
            ) or "metrics_columns" in comparison and (
                not isinstance(comparison["metrics_columns"], list) or any(
                    type(name) is not str for name in comparison["metrics_columns"]
                )
            ) or "kpi_label_suffix" in comparison and type(comparison["kpi_label_suffix"]) is not str:
                _fail(path, "invalid comparison view")
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


def select_presets(curves, definitions):
    """Select existing analyzer curves by YAML names, groups or name fragments."""
    result = {}
    for title, selector in definitions.items():
        kind, values = next(iter(selector.items()))
        result[title] = [
            curve["name"] for curve in curves
            if (not curve.get("hidden", False) if kind == "visible" else
                curve["name"] in values if kind == "names" else
                curve.get("group") in values if kind == "groups" else
                any(fragment in curve["name"] for fragment in values))
        ]
    return result


def declaration(value, *, kind, path="reports"):
    if kind != "workload":
        _fail(path, "report views require test.kind=workload")
    if not isinstance(value, list) or not value or any(type(name) is not str for name in value):
        _fail(path, "expected a nonempty list of view filenames")
    if len(set(value)) != len(value):
        _fail(path, "duplicate view")
    if DEFAULT_VIEW not in value:
        _fail(path, "include workload.yaml to retain all collected metrics")
    for name in value:
        view(name)
    return list(value)
