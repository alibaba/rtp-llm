"""HA view field contracts; common loading and metric binding remain generic."""


def validate(path, data, _fail):
    if (set(data) != {"report_view_schema_version", "kind", "report", "metrics", "charts", "sections"}
            or set(data["report"]) != {"subtitle"}
            or set(data["metrics"]) != {"query_plan"}
            or set(data["charts"]) != {"events", "curves", "panels"}):
        _fail(path, "invalid HA view fields")
    from reporting.view_schema import validate_section_contract
    validate_section_contract(path, data, {"sources": None}, _fail)
    charts = data["charts"]
    if not isinstance(charts["events"], dict) or not charts["events"] or any(
        type(stage) is not str or type(label) is not str or not label
        for stage, label in charts["events"].items()
    ):
        _fail(path, "invalid HA event labels")
    styles = charts["curves"]
    if not isinstance(styles, dict) or not styles:
        _fail(path, "invalid HA metric presentation")
    for identity, style in styles.items():
        if not isinstance(style, dict) or set(style) != {"name", "group", "axis", "color", "metric_id", "labels"}:
            _fail(path, "invalid HA metric presentation")
        if any(type(style[field]) is not str or not style[field] for field in ("name", "group", "axis")):
            _fail(path, "invalid HA metric labels")
        colors = style["color"]
        if not (type(colors) is str and colors or isinstance(colors, dict)
                and set(colors) == {"A", "B"} and all(type(c) is str and c for c in colors.values())):
            _fail(path, "invalid HA curve colors")
    panels = charts["panels"]
    if not isinstance(panels, list) or not panels:
        _fail(path, "invalid HA panels")
    ids = set()
    for panel in panels:
        if (not isinstance(panel, dict) or set(panel) != {"id", "title", "curve_ids", "caption"}
                or any(type(panel[field]) is not str or not panel[field] for field in ("id", "title", "caption"))
                or not isinstance(panel["curve_ids"], list) or not panel["curve_ids"]
                or any(type(identity) is not str or identity not in styles for identity in panel["curve_ids"])
                or len(set(panel["curve_ids"])) != len(panel["curve_ids"])):
            _fail(path, "invalid HA panel")
        if panel["id"] in ids:
            _fail(path, "invalid HA panels")
        ids.add(panel["id"])
