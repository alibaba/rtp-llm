"""Field contracts for built-in views; registered case validators own case fields."""

import re
import string

from schema_contract import matches_schema

CURVES = frozenset({"mean", "max", "detail"})


def validate_structure(path, data, fail):
    required = {"report_view_schema_version", "kind", "report"}
    if (not required <= data.keys() or data.keys() - required - {"metrics", "charts", "sections"}
            or not matches_schema(data, "report_view_schema_version", 1)):
        fail(path, "invalid report view fields or version")
    report = data["report"]
    if (not isinstance(report, dict) or not {"subtitle"} <= report.keys()
            or report.keys() - {"subtitle", "id", "producer"}):
        fail(str(path) + ".report", "invalid report fields")
    if "metrics" in data:
        metrics = data["metrics"]
        if (not isinstance(metrics, dict) or "query_plan" not in metrics
                or metrics.keys() - {"query_plan", "diagnostic_only"}):
            fail(str(path) + ".metrics", "invalid metric selection fields")
    if "charts" in data and not isinstance(data["charts"], dict):
        fail(str(path) + ".charts", "expected mapping")
    if "sections" in data:
        sections = data["sections"]
        if not isinstance(sections, dict) or not sections:
            fail(str(path) + ".sections", "expected nonempty section mapping")
        for key, section in sections.items():
            location = str(path) + ".sections." + str(key)
            if (type(key) is not str or not re.fullmatch(r"[a-z][a-z0-9_]*", key)
                    or not isinstance(section, dict) or not {"title", "opened"} <= section.keys()
                    or section.keys() - {"title", "opened", "columns"}
                    or type(section["title"]) is not str or not section["title"].strip()
                    or type(section["opened"]) is not bool):
                fail(location, "invalid section fields")
            if "columns" in section and (not isinstance(section["columns"], list)
                    or not section["columns"] or any(type(column) is not str or not column.strip()
                                                    for column in section["columns"])):
                fail(location, "invalid section columns")


def validate_default(path, data, fail):
    if (set(data) != {"report_view_schema_version", "kind", "report", "charts"}
            or data["kind"] != "default" or set(data["report"]) != {"subtitle"}):
        fail(path, "invalid default view")
    charts = data["charts"]
    required = {"group_by", "detail_labels", "summaries", "default_visible", "max_points_per_series", "presets"}
    if set(charts) - {"events", "event_ids"} != required:
        fail(str(path) + ".charts", "invalid default chart fields")
    if charts["group_by"] != ["epoch", "source", "metric"]:
        fail(path, "default view must retain metric identity")
    labels = charts["detail_labels"]
    if not isinstance(labels, list) or not labels or any(
        type(x) is not str or x not in {"engine_name", "pod", "engine"} for x in labels
    ) or len(set(labels)) != len(labels):
        fail(path, "invalid detail_labels")
    for field in ("summaries", "default_visible"):
        values = charts[field]
        if not isinstance(values, list) or not values or any(
            type(x) is not str or x not in CURVES for x in values
        ) or len(set(values)) != len(values):
            fail(path, "invalid " + field)
    if set(charts["summaries"]) - {"mean", "max"} or set(charts["default_visible"]) - set(charts["summaries"]):
        fail(path, "default_visible must select summary curves")
    points = charts["max_points_per_series"]
    if type(points) is not int or not 32 <= points <= 2048:
        fail(path, "max_points_per_series must be 32..2048")
    presets = charts["presets"]
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
            "name", "group", "axis", "source_unit", "unit", "color", "hidden", "scale", "metric_id", "labels"
        }:
            fail(path, "invalid metric presentation " + str(identity))
        if any(type(style[field]) is not str or not style[field]
               for field in ("name", "group", "axis")):
            fail(path, "invalid metric labels " + identity)
        if any(type(style[field]) is not str for field in ("source_unit", "unit", "color") if field in style):
            fail(path, "invalid metric unit or color " + identity)
        if any(type(style[field]) is not bool for field in ("hidden",) if field in style):
            fail(path, "invalid metric visibility " + identity)
        if "scale" in style and (type(style["scale"]) not in (int, float)
                                 or not 0 < style["scale"] < float("inf")):
            fail(path, "invalid metric scale " + identity)
        if style.get("scale", 1) != 1 and not {"source_unit", "unit"} <= set(style):
            fail(path, "scaled metric requires source_unit and unit " + identity)
    return metrics


def validate_panels(path, data, metrics, fail):
    panels = data.get("panels")
    if not isinstance(panels, list) or not panels:
        fail(path, "invalid panels")
    ids = set()
    for panel in panels:
        if not isinstance(panel, dict) or not {"id", "curve_ids"} <= set(panel):
            fail(path, "panels require id and curve_ids")
        if type(panel["id"]) is not str or not panel["id"] or panel["id"] in ids:
            fail(path, "invalid panel id")
        ids.add(panel["id"])
        validate_panel(path, panel, metrics, fail)
    selected = {curve_id for panel in panels for curve_id in panel["curve_ids"]}
    if set(metrics) - selected:
        fail(path, "curve declarations must be referenced by a panel")


def validate_panel(path, panel, metrics, fail):
    if (not isinstance(panel, dict) or "title" not in panel or set(panel) - {
        "title", "caption", "empty_caption", "presets", "axes", "id", "curve_ids", "event_ids",
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
        not isinstance(axis, dict) or not {"title", "position"} <= set(axis)
        or set(axis) - {"title", "position", "min", "max"}
        or type(axis["title"]) is not str or axis["position"] not in ("left", "right")
        for axis in panel["axes"].values()
    )):
        fail(path, "invalid panel axes")
    for axis in panel.get("axes", {}).values():
        if any(type(axis[field]) not in (int, float) or not -float("inf") < axis[field] < float("inf")
               for field in ("min", "max") if field in axis):
            fail(path, "invalid panel axis bounds")
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


def validate_selected(path, data, fail):
    required = {"report_view_schema_version", "kind", "report", "metrics", "charts", "sections"}
    if set(data) != required or data["kind"] != "selected":
        fail(path, "invalid selected report view")
    report = data["report"]
    if set(report) != {"subtitle", "id", "producer"}:
        fail(str(path) + ".report", "selected reports require id and producer")
    for field in ("id", "producer"):
        if type(report[field]) is not str or not re.fullmatch(r"[a-z][a-z0-9-]*", report[field]):
            fail(str(path) + ".report", "invalid " + field)
    charts = data["charts"]
    if (not {"curves", "panels"} <= charts.keys()
            or charts.keys() - {"curves", "panels", "time_origin_label", "events", "event_ids"}):
        fail(str(path) + ".charts", "invalid selected chart fields")
    metrics = validate_curves(path, charts, fail)
    validate_panels(path, charts, metrics, fail)
    if "time_origin_label" in charts and (type(charts["time_origin_label"]) is not str
                                         or not charts["time_origin_label"].strip()):
        fail(str(path) + ".charts", "invalid time_origin_label")


def validate_section_contract(path, data, expected, fail):
    """Python owns section contents; YAML only names, labels and opens them."""
    sections = data["sections"]
    if sections.keys() != expected.keys():
        fail(str(path) + ".sections", "unexpected or missing sections")
    for key, column_count in expected.items():
        columns = sections[key].get("columns")
        if column_count is None:
            if columns is not None:
                fail(str(path) + ".sections." + key, "detail section cannot declare columns")
        elif columns is None or len(columns) != column_count:
            fail(str(path) + ".sections." + key, f"table requires {column_count} columns")


def validate_bindings(path, data, query_plan, fail):
    from monitoring.query_plan import definitions

    declared = definitions(query_plan)
    for curve_id, style in data["charts"]["curves"].items():
        if (type(style.get("metric_id")) is not str or style["metric_id"] not in declared
                or not isinstance(style.get("labels"), dict)
                or any(type(k) is not str or type(v) is not str for k, v in style["labels"].items())):
            fail(path, "curve must bind a declared metric id and labels: " + curve_id)


def validate_monitoring_policy(path, data, query_plan, fail):
    policy = data["metrics"]
    if "diagnostic_only" not in policy:
        fail(path, "selected views must classify diagnostic_only metrics explicitly")
    metrics = data["charts"]["curves"]
    if any(
        not {"unit", "color", "hidden"} <= set(style)
        for style in metrics.values()
    ):
        fail(path, "classified monitoring styles require unit, color and hidden")
    diagnostic = policy.get("diagnostic_only", [])
    from monitoring.query_plan import definitions
    from reporting.metric_binding import metric_classification

    known = {identity: definition for identity, definition in definitions(query_plan).items()
             if not ("promql" in definition and identity.split('/')[-1] == 'up')}
    if not isinstance(diagnostic, list) or any(
        type(identity) is not str or identity not in known for identity in diagnostic
    ) or len(diagnostic) != len(set(diagnostic)):
        fail(path, "invalid diagnostic_only metrics")
    if any(metric_classification(identity, definition, data) is None
           for identity, definition in known.items()):
        fail(path, "monitoring metrics lack presentation or diagnostic classification")


def validate_text(path, data, fail):
    subtitle = data["report"]["subtitle"]
    if type(subtitle) is not str or not subtitle.strip():
        fail(path, "subtitle is required")
    if data["kind"] != "selected":
        return
    try:
        fields = [field for _, field, _, _ in string.Formatter().parse(data["report"]["subtitle"])
                  if field is not None]
    except ValueError as exc:
        fail(path, str(exc))
    if any(field not in {"verdict", "monitoring_status"} for field in fields):
        fail(path, "unknown subtitle placeholder")
