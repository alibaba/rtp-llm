#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render a validated report spec into self-contained offline HTML.

Every series owns its points. Time panels use explicit axes and share interval
selection; generic bars/scatter use the same point contract. No format adapters.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from reporting.catalog import RENDERER_THEME, series_color


def _present(value):
    """Remove absent metadata without treating zero or false as missing."""
    if isinstance(value, dict):
        cleaned = {k: item for k, v in value.items()
                   if (item := _present(v)) is not None and not k.endswith("_schema_version")}
        return cleaned or None
    if isinstance(value, list):
        cleaned = [item for v in value if (item := _present(v)) is not None]
        return cleaned or None
    return None if value is None or value == "" else value


def _text(value):
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, indent=2)
    return html.escape(str(value), quote=True)


def _brief(value, depth=0):
    if isinstance(value, list):
        if not value:
            return "0 项"
        if all(type(item) in (int, float) for item in value):
            return f"{len(value)} 项；min={min(value)}，max={max(value)}"
        return f"{len(value)} 项"
    if isinstance(value, dict):
        if depth >= 2:
            return f"{len(value)} 个字段"
        return "；".join(str(key) + "=" + _brief(item, depth + 1)
                        for key, item in list(value.items())[:5]) + ("；…" if len(value) > 5 else "")
    return str(value)[:180]


def _cell(value):
    if value is None:
        return "—"
    encoded = _text(value)
    if len(encoded) <= 240:
        return encoded
    return '<details class="cell-detail"><summary>' + _text(_brief(value)[:260]) + '</summary><pre>' + encoded + '</pre></details>'


def render_context(spec):
    """Grouped key/value cards, with long configuration values expanded on demand."""
    meta = spec.get("run_meta") or spec.get("meta") or {}
    runs = meta.get("runs") if isinstance(meta, dict) else None
    cards = runs.items() if isinstance(runs, dict) and runs else [(None, meta)]
    names = {"implementation": "代码与制品", "configuration": "配置与模型", "environment": "运行环境与拓扑",
             "workload": "流量与播放", "evidence": "证据来源"}
    rendered = []
    for label, value in cards:
        value = _present(value)
        if not value:
            continue
        groups = []
        for key, fields in value.items():
            if key in {"identity", "clock", "timeAxis", "runs"}:
                continue
            leaves = []
            def walk(item, path, depth):
                if isinstance(item, dict) and depth < (4 if key == "environment" else 2):
                    for child, entry in item.items():
                        walk(entry, path + "." + child if path else child, depth + 1)
                else:
                    leaves.append('<div class="context-field"><dt>' + _text(path) + '</dt><dd>' + _cell(item) + '</dd></div>')
            walk(fields, "", 0)
            groups.append('<article class="context-card"><h3>' + _text(names.get(key, key)) + '</h3><dl>' + ''.join(leaves) + '</dl></article>')
        if groups:
            heading = '<h3>' + _text(label) + '</h3>' if label else ''
            rendered.append(heading + '<div class="context-grid">' + ''.join(groups) + '</div>')
    if not rendered:
        return ""
    return '<details class="report-block report-context"><summary>运行信息（制品与配置）</summary><div class="block-body">' + ''.join(rendered) + '</div></details>'


def render_sections(sections):
    """All report sections use one collapsible container; tables share cell handling."""
    out = []
    for section in sections:
        title = _text(section.get("title", ""))
        kind = section["type"]
        if kind == "details":
            body = '<pre>' + _text(section["value"]) + '</pre>'
        elif kind == "table":
            heads = ''.join('<th>' + _text(c) + '</th>' for c in section["columns"])
            rows = ''.join('<tr>' + ''.join('<td>' + _cell(v) + '</td>' for v in row) + '</tr>' for row in section["rows"])
            body = '<div class="table-scroll"><table><thead><tr>' + heads + '</tr></thead><tbody>' + rows + '</tbody></table></div>'
        elif kind == "links":
            items = []
            for item in section["items"]:
                href = item["href"]
                if ":" in href or href.startswith("//"):
                    raise ValueError("report links must be relative artifact paths")
                items.append('<li><a href="' + _text(href) + '">' + _text(item["label"]) + '</a></li>')
            body = '<ul>' + ''.join(items) + '</ul>'
        else:
            raise ValueError("unsupported report section: " + kind)
        opened = section.get("opened", kind == "table")
        if type(opened) is not bool:
            raise ValueError("section opened must be boolean")
        out.append('<details class="report-block attachment"' + (' open' if opened else '') + '><summary>' + title + '</summary><div class="block-body">' + body + '</div></details>')
    return '<div class="report-sections">' + ''.join(out) + '</div>'


def render(spec):
    """Return complete HTML for a validated spec."""
    from reporting.spec import validate

    validate(spec)
    run_id = spec.get("run_id", "")
    title = spec.get("title") or ("FlexLB 压测报告 · run " + run_id)
    subtitle = spec.get("subtitle") or ""
    if isinstance(subtitle, dict):
        subtitle = {str(k): str(v) for k, v in subtitle.items() if v is not None}
    elif not isinstance(subtitle, str):
        raise TypeError("report subtitle must be a string or key-value mapping")
    kpis = spec.get("kpis") or []
    panels = spec.get("panels") or []

    payload = {
        "summary": {
            "title": title,
            "subtitle": subtitle,
            "kpis": [
                {
                    "label": k.get("label", ""),
                    "value": k.get("value", ""),
                    "tone": k.get("tone") or "",
                }
                for k in kpis
            ],
        },
        "timeAxis": spec.get("timeAxis"),
        "events": spec.get("events", []),
        "timeOriginLabel": spec.get("timeOriginLabel"),
        "meta": spec.get("meta"),
        "panels": [
            {
                "id": p["id"],
                "title": p.get("title", ""),
                "caption": p.get("caption", ""),
                "type": p.get("type", "line"),
                "bounds": p.get("bounds"),
                "axisLabels": p.get("axisLabels"),
                "events": p.get("events"),
                "timeX": bool(p.get("timeX")),
                "yMax": p.get("yMax"),
                "axes": p.get("axes", {}),
                "presets": p.get("presets", {}),
                "unit": p.get("unit", "") or "",
                "series": [
                    {
                        "metric_id": s.get("metric_id"),
                        "statistics_points": s.get("statistics_points"),
                        "provenance": s.get("provenance"),
                        "name": s.get("name", ""),
                        "points": s["points"],
                        "axis": s.get("axis", "y"),
                        "unit": s.get("unit", ""),
                        "group": s.get("group", "其他"),
                        "description": s.get("description", ""),
                        "hidden": s.get("hidden", False),
                        "dash": s.get("dash", []),
                        "color": s.get("color") or series_color(s.get("tone"), i),
                    }
                    for i, s in enumerate(p.get("series", []))
                ],
            }
            for p in panels
        ],
    }

    sections = list(spec.get("sections", []))
    page_title = html.escape(title)
    resource_dir = Path(__file__).resolve().parent / "assets"
    chartjs = (resource_dir / "chart.umd.min.js").read_text(encoding="utf-8")
    overlay = (resource_dir / "multi_curve.js").read_text(encoding="utf-8")
    interaction = (resource_dir / "legend_interaction.js").read_text(encoding="utf-8")
    template = (resource_dir / "report.html").read_text(encoding="utf-8")
    for name, color in RENDERER_THEME.items():
        template = template.replace("__REPORT_" + name + "__", color)
    return (
        template.replace("__SECTIONS__", render_sections(sections))
        .replace("__CONTEXT__", render_context(spec))
        .replace("__PAGE_TITLE__", page_title)
        .replace(
            "__SPEC_JSON__",
            json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"),
        )
        .replace("__CHARTJS_JS__", chartjs)
        .replace("__LEGEND_INTERACTION_JS__", interaction)
        .replace("__MULTI_CURVE_JS__", overlay)
    )
