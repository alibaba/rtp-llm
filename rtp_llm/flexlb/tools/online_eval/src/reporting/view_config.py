"""Validated, data-only report view declarations and the shared presentation template."""

import re
from pathlib import Path

from scenario.loader import ScenarioError, load_document

VIEWS = Path(__file__).resolve().parents[2] / "config/report_views"
TEMPLATES = VIEWS / "workload.yaml"
CURVES = frozenset({"mean", "max", "detail"})


def custom_view(name, path="reports.custom"):
    if type(name) is not str or not re.fullmatch(r"[a-z][a-z0-9_]*\.yaml", name):
        _fail(path, "expected a report view filename under config/report_views")
    view = load_document(VIEWS / name)
    if set(view) != {"kind", "report"} or view["kind"] != "gate" or (
        type(view["report"]) is not str
        or not re.fullmatch(r"[a-z][a-z0-9-]*", view["report"])
    ):
        _fail(str(VIEWS / name), "expected gate view with a report bundle name")
    return view


def _fail(path, message):
    raise ScenarioError(f"{path}: {message}")


def _mapping(value, allowed, path):
    if not isinstance(value, dict) or set(value) - set(allowed):
        _fail(path, f"expected mapping with only {sorted(allowed)}")
    return value


def _named_mapping(value, path):
    if not isinstance(value, dict):
        _fail(path, "expected mapping")
    return _mapping(value, set(value), path)


def _curve_names(value, path):
    if not isinstance(value, list) or not value or any(
        type(item) is not str or item not in CURVES for item in value
    ) or len(set(value)) != len(value):
        _fail(path, "expected unique mean/max/detail names")
    return list(value)


def template(name="workload"):
    document = load_document(TEMPLATES)
    if set(document) != {"schema_version", "templates"} or document["schema_version"] != 1:
        _fail(str(TEMPLATES), "expected report template schema_version 1")
    definitions = _mapping(document["templates"], {"workload"}, str(TEMPLATES))
    if name not in definitions:
        _fail("reports.default.template", f"unknown template {name!r}")
    config = _mapping(definitions[name],
                      {"group_by", "detail_labels", "summaries", "default_visible", "max_points_per_series", "presets"},
                      str(TEMPLATES))
    if config.get("group_by") != ["epoch", "source", "metric"]:
        _fail(str(TEMPLATES), "group_by must preserve epoch/source/metric identity")
    labels = config.get("detail_labels")
    if not isinstance(labels, list) or not labels or any(
        type(label) is not str or label not in {"engine_name", "pod", "engine"}
        for label in labels
    ) or len(labels) != len(set(labels)):
        _fail(str(TEMPLATES), "detail_labels must name unique engine identity labels")
    summaries = _curve_names(config.get("summaries"), str(TEMPLATES) + ".summaries")
    if set(summaries) - {"mean", "max"}:
        _fail(str(TEMPLATES), "summaries may contain only mean/max")
    visible = _curve_names(config.get("default_visible"), str(TEMPLATES) + ".default_visible")
    if set(visible) - set(summaries):
        _fail(str(TEMPLATES), "default_visible must select summaries")
    points = config.get("max_points_per_series")
    if type(points) is not int or not 32 <= points <= 2048:
        _fail(str(TEMPLATES), "max_points_per_series must be 32..2048")
    presets = _named_mapping(config.get("presets"), str(TEMPLATES) + ".presets")
    if not presets or any(type(key) is not str or not key.strip() for key in presets):
        _fail(str(TEMPLATES), "presets require nonempty names")
    for key, names in presets.items():
        _curve_names(names, str(TEMPLATES) + ".presets." + key)
    return config


def declaration(value, *, kind, path="reports"):
    """None preserves the historical default view and legacy gate links."""
    if value is None:
        return None
    if kind != "workload":
        _fail(path, "report views require test.kind=workload")
    value = _mapping(value, {"default", "custom"}, path)
    default = value.get("default", True)
    if type(default) is bool:
        default = {"enabled": default, "template": "workload"}
    else:
        default = _mapping(default, {"enabled", "template", "default_visible", "presets"}, path + ".default")
        default = {"enabled": True, "template": "workload", **default}
    if type(default["enabled"]) is not bool:
        _fail(path + ".default.enabled", "expected boolean")
    common = template(default["template"])
    if "default_visible" in default:
        _curve_names(default["default_visible"], path + ".default.default_visible")
        if set(default["default_visible"]) - set(common["summaries"]):
            _fail(path + ".default.default_visible", "must select template summaries")
    if "presets" in default:
        presets = _named_mapping(default["presets"], path + ".default.presets")
        for name, curves in presets.items():
            if type(name) is not str or not name.strip():
                _fail(path + ".default.presets", "preset name must be nonempty")
            _curve_names(curves, path + ".default.presets." + name)
    custom = value.get("custom", [])
    if not isinstance(custom, list) or any(
        type(name) is not str for name in custom
    ) or len(set(custom)) != len(custom):
        _fail(path + ".custom", "expected unique report view filenames")
    for name in custom:
        custom_view(name, path + ".custom")
    if not default["enabled"] and not custom:
        _fail(path, "at least one view must be enabled")
    return {"default": default, "custom": list(custom)}
