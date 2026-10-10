"""Field contracts for built-in views; registered case validators own case fields."""

import re
import string

CURVES = frozenset({"mean", "max", "detail"})


def validate_checks(path, data, fail):
    if set(data) != {"kind", "title", "subtitle"} or data["kind"] != "checks":
        fail(path, "invalid execution view")


def validate_default(path, data, fail):
    required = {"kind", "title", "subtitle", "group_by", "detail_labels",
                "summaries", "default_visible", "max_points_per_series", "presets"}
    if set(data) != required or data["kind"] != "default":
        fail(path, "invalid default view")
    if data["group_by"] != ["epoch", "source", "metric"]:
        fail(path, "default view must retain metric identity")
    labels = data["detail_labels"]
    if not isinstance(labels, list) or not labels or any(
        type(x) is not str or x not in {"engine_name", "pod", "engine"} for x in labels
    ) or len(set(labels)) != len(labels):
        fail(path, "invalid detail_labels")
    for field in ("summaries", "default_visible"):
        values = data[field]
        if not isinstance(values, list) or not values or any(
            type(x) is not str or x not in CURVES for x in values
        ) or len(set(values)) != len(values):
            fail(path, "invalid " + field)
    if set(data["summaries"]) - {"mean", "max"} or set(data["default_visible"]) - set(data["summaries"]):
        fail(path, "default_visible must select summary curves")
    points = data["max_points_per_series"]
    if type(points) is not int or not 32 <= points <= 2048:
        fail(path, "max_points_per_series must be 32..2048")
    presets = data["presets"]
    if not isinstance(presets, dict) or not presets or any(
        type(key) is not str or not key or not isinstance(values, list)
        or not values or any(type(item) is not str or item not in CURVES for item in values)
        for key, values in presets.items()
    ):
        fail(path, "invalid presets")


def validate_curves(path, data, fail):
    metrics = data.get("curves", {})
    if not isinstance(metrics, dict):
        fail(path, "metrics must be a mapping")
    for identity, style in metrics.items():
        if type(identity) is not str or not re.fullmatch(
            r"[a-z][a-z0-9_]*(/[A-Za-z][A-Za-z0-9_]*)*", identity
        ) or not isinstance(style, dict) or not {"name", "group", "axis"} <= set(style) or set(style) - {
            "name", "group", "axis", "source_unit", "unit", "color", "hidden", "primary", "scale", "metric_id", "labels"
        }:
            fail(path, "invalid metric presentation " + str(identity))
        if any(type(style[field]) is not str or not style[field]
               for field in ("name", "group", "axis")):
            fail(path, "invalid metric labels " + identity)
        if any(type(style[field]) is not str for field in ("source_unit", "unit", "color") if field in style):
            fail(path, "invalid metric unit or color " + identity)
        if any(type(style[field]) is not bool for field in ("hidden", "primary") if field in style):
            fail(path, "invalid metric visibility " + identity)
        if "scale" in style and (type(style["scale"]) not in (int, float)
                                 or not 0 < style["scale"] < float("inf")):
            fail(path, "invalid metric scale " + identity)
        if style.get("scale", 1) != 1 and not {"source_unit", "unit"} <= set(style):
            fail(path, "scaled metric requires source_unit and unit " + identity)
    return metrics


def validate_panels(path, data, metrics, fail):
    if ("panel" in data) == ("panels" in data):
        fail(path, "declare exactly one of panel or panels")
    panels = data.get("panels", [data.get("panel")])
    if not isinstance(panels, list) or not panels:
        fail(path, "invalid panels")
    ids = set()
    for panel in panels:
        if "panels" in data:
            if not isinstance(panel, dict) or not {"id", "curve_ids"} <= set(panel):
                fail(path, "panels require id and metric_ids")
            if type(panel["id"]) is not str or not panel["id"] or panel["id"] in ids:
                fail(path, "invalid panel id")
            ids.add(panel["id"])
        validate_panel(path, panel, metrics, fail)


def validate_panel(path, panel, metrics, fail):
    if (not isinstance(panel, dict) or "title" not in panel or set(panel) - {
        "title", "caption", "empty_caption", "presets", "axes", "id", "curve_ids",
    }):
        fail(path, "invalid panel presentation")
    for field in ("title", "caption", "empty_caption"):
        if field in panel and (type(panel[field]) is not str or not panel[field]):
            fail(path, "invalid panel " + field)
    presets = panel.get("presets", {})
    if not isinstance(presets, dict) or any(
        type(name) is not str or not name or not _valid_preset(selector)
        for name, selector in presets.items()
    ):
        fail(path, "invalid panel presets")
    if "axes" in panel and (not isinstance(panel["axes"], dict) or any(
        not isinstance(axis, dict) or set(axis) != {"title", "position"}
        or type(axis["title"]) is not str or axis["position"] not in ("left", "right")
        for axis in panel["axes"].values()
    )):
        fail(path, "invalid panel axes")
    if "curve_ids" in panel and (not isinstance(panel["curve_ids"], list)
            or not panel["curve_ids"] or any(type(name) is not str or name not in metrics
                                            for name in panel["curve_ids"])):
        fail(path, "invalid panel metric_ids")
    if "curve_ids" in panel and any(
        metrics[metric_id]["axis"] not in panel.get("axes", {})
        for metric_id in panel["curve_ids"]
    ):
        fail(path, "panel metric axis is not declared")


def _valid_preset(selector):
    if not isinstance(selector, dict) or len(selector) != 1:
        return False
    key, values = next(iter(selector.items()))
    if key == "visible":
        return values is True
    return (key in {"names", "contains", "groups"} and isinstance(values, list)
            and bool(values) and all(type(item) is str and item for item in values))


def validate_produced(path, data, fail):
    required = {"kind", "report", "producer", "title", "subtitle", "sections"}
    if (not required <= set(data) or set(data) - required - {
        "panel", "panels", "time_origin", "kpis", "meta", "audit_columns",
        "criteria_columns", "curves", "monitoring_query_plan", "diagnostic_only", "unlisted",
    } or data["kind"] != "produced"):
        fail(path, "invalid produced report view")
    metrics = validate_curves(path, data, fail)
    for field in ("report", "producer"):
        if type(data[field]) is not str or not re.fullmatch(r"[a-z][a-z0-9-]*", data[field]):
            fail(path, "invalid " + field)
    validate_panels(path, data, metrics, fail)
    for field, require_entries, require_values in (
        ("sections", True, True), ("kpis", False, True), ("meta", False, False),
    ):
        if field not in data:
            continue
        values = data[field]
        if (not isinstance(values, dict) or (require_entries and not values) or any(
            type(key) is not str or type(value) is not str or (require_values and not value)
            for key, value in values.items()
        )):
            fail(path, "invalid " + field)
    if "time_origin" in data and type(data["time_origin"]) is not str:
        fail(path, "invalid time_origin")
    for field in ("audit_columns", "criteria_columns"):
        if field in data and (not isinstance(data[field], list)
                or any(type(value) is not str for value in data[field])):
            fail(path, "invalid " + field)


def validate_bindings(path, data, query_plan, fail):
    from monitoring.query_plan import definitions

    declared = definitions(query_plan)
    for curve_id, style in data["curves"].items():
        if (type(style.get("metric_id")) is not str or style["metric_id"] not in declared
                or not isinstance(style.get("labels"), dict)
                or any(type(k) is not str or type(v) is not str for k, v in style["labels"].items())):
            fail(path, "curve must bind a declared metric id and labels: " + curve_id)


def validate_monitoring_policy(path, data, query_plan, fail):
    if query_plan is None:
        if "diagnostic_only" in data or "unlisted" in data:
            fail(path, "monitoring policy requires monitoring_query_plan")
        return
    metrics = data.get("curves", {})
    if "diagnostic_only" in data and any(
        not {"unit", "color", "hidden"} <= set(style)
        for style in metrics.values()
    ):
        fail(path, "classified monitoring styles require unit, color and hidden")
    if data.get("unlisted", "error") != "error":
        fail(path, "invalid unlisted monitoring policy")
    diagnostic = data.get("diagnostic_only", [])
    known = {f"{kind}/{metric}" for kind, queries in query_plan["sources"].items()
             for metric in queries}
    if not isinstance(diagnostic, list) or any(
        type(identity) is not str or identity not in known for identity in diagnostic
    ) or len(diagnostic) != len(set(diagnostic)):
        fail(path, "invalid diagnostic_only metrics")
    if known - {style["metric_id"] for style in metrics.values()} - set(diagnostic):
        fail(path, "monitoring metrics lack presentation or diagnostic classification")


def validate_text(path, data, fail):
    if any(type(data[key]) is not str or not data[key].strip() for key in ("title", "subtitle")):
        fail(path, "title and subtitle are required")
    if data["kind"] != "produced":
        return
    try:
        fields = [field for _, field, _, _ in string.Formatter().parse(data["subtitle"])
                  if field is not None]
    except ValueError as exc:
        fail(path, str(exc))
    if any(field not in {"verdict", "monitoring_status"} for field in fields):
        fail(path, "unknown subtitle placeholder")
