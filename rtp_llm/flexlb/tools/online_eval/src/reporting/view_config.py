"""Validate case-selected report views; YAML controls presentation, not analysis."""

import re
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
        if set(data) != {"kind", "title", "subtitle", "source_report", "panels"} or data["kind"] != "metrics":
            _fail(path, "invalid metrics view")
        if type(data["source_report"]) is not str or not re.fullmatch(r"[a-z][a-z0-9-]*", data["source_report"]):
            _fail(path, "invalid source_report")
        panels = data["panels"]
        if not isinstance(panels, list) or not panels or any(
            not isinstance(panel, dict) or set(panel) != {"title", "metrics"}
            or not isinstance(panel["metrics"], list) or not panel["metrics"]
            or any(type(metric) is not str or not metric for metric in panel["metrics"])
            for panel in panels
        ):
            _fail(path, "panels require titles and metric names")
    if any(type(data[key]) is not str or not data[key].strip() for key in ("title", "subtitle")):
        _fail(path, "title and subtitle are required")
    return data


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
