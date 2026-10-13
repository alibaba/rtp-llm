"""Field contracts for built-in views; registered case validators own case fields."""

import re

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
            if type(key) is not str or not re.fullmatch(r"[a-z][a-z0-9_]*", key):
                fail(location, "invalid section fields")



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
        type(x) is not str or not x.strip() for x in labels
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
    for identity in metrics:
        if type(identity) is not str or not re.fullmatch(
                r"[a-z][a-z0-9_]*(/[A-Za-z][A-Za-z0-9_]*)*", identity):
            fail(path, "invalid metric presentation " + str(identity))

    return metrics


def validate_panels(path, data, fail):
    panels, curves = data.charts.panels, data.charts.curves
    if not panels:
        fail(path, 'invalid panels')
    ids, selected = set(), set()
    for panel in panels:
        if panel.id in ids:
            fail(path, 'invalid panel id')
        ids.add(panel.id)
        try:
            panel.validate_bindings(curves)
        except ValueError as exc:
            fail(path, str(exc))
        selected.update(panel.curve_ids)
    if set(curves)-selected:
        fail(path, 'curve declarations must be referenced by a panel')


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
            or charts.keys() - {"curves", "panels", "events", "event_ids"}):
        fail(str(path) + ".charts", "invalid selected chart fields")
    validate_curves(path, charts, fail)



def validate_section_contract(path, data, expected, fail):
    """Python owns section contents; YAML only names, labels and opens them."""
    sections = data.sections
    if sections.keys() != expected.keys():
        fail(str(path) + ".sections", "unexpected or missing sections")
    for key, column_count in expected.items():
        columns = sections[key].columns
        if column_count is None:
            if columns is not None:
                fail(str(path) + ".sections." + key, "detail section cannot declare columns")
        elif columns is None or len(columns) != column_count:
            fail(str(path) + ".sections." + key, f"table requires {column_count} columns")


def validate_bindings(path, data, query_plan, fail):
    from monitoring.query_plan import definitions

    declared = definitions(query_plan)
    for curve_id, binding in data.charts.curves.items():
        if binding.metric_id not in declared or binding.labels is None:
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

    known = {identity: definition for identity, definition in definitions(query_plan).items()
             if not ("promql" in definition and identity.split('/')[-1] == 'up')}
    if not isinstance(diagnostic, list) or any(
        type(identity) is not str or identity not in known for identity in diagnostic
    ) or len(diagnostic) != len(set(diagnostic)):
        fail(path, "invalid diagnostic_only metrics")



def validate_text(path, data, fail):
    subtitle = data["report"]["subtitle"]
    if type(subtitle) is not str or not subtitle.strip():
        fail(path, "subtitle is required")
