"""Small shared SVG primitives for offline timeline and trade-off views."""

import html
import json
import math

COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#d97706", "#0891b2")


def esc(value):
    return html.escape(str(value), quote=True)


def bounds(values):
    values = [v for v in values if type(v) in (int, float) and math.isfinite(v)]
    if not values:
        return 0, 1
    lo, hi = min(values), max(values)
    return (lo, hi) if hi > lo else (lo - 0.5, hi + 0.5)


def chart(
    series,
    *,
    events=(),
    scatter=False,
    xbounds=None,
    ybounds=None,
    xlabel="seconds",
    ylabel="value",
):
    points = [p for s in series for p in s["points"] if p[1] is not None]
    xl, xh = xbounds or bounds(p[0] for p in points)
    yl, yh = ybounds or bounds(p[1] for p in points)
    x = lambda v: 60 + (v - xl) / (xh - xl) * 750
    y = lambda v: 265 - (v - yl) / (yh - yl) * 225
    out = [
        '<svg role="img" viewBox="0 0 880 330">',
        '<path d="M60 40V265H810" fill="none" stroke="#94a3b8"/>',
    ]
    for i in range(6):
        a, b = xl + (xh - xl) * i / 5, yl + (yh - yl) * i / 5
        out += [
            f'<text x="{x(a):.2f}" y="285" text-anchor="middle">{a:.3g}</text>',
            f'<text x="52" y="{y(b):.2f}" text-anchor="end">{b:.3g}</text>',
        ]
    out += [
        f'<text x="430" y="315" text-anchor="middle">{esc(xlabel)}</text>',
        f'<text x="60" y="23">{esc(ylabel)}</text>',
    ]
    for s in series:
        segments, segment = [], []
        for p in s["points"]:
            if p[1] is None:
                if segment:
                    segments.append(segment)
                segment = []
                continue
            if scatter:
                radius = p[2] if len(p) > 2 else 5
                out.append(
                    f'<circle cx="{x(p[0]):.2f}" cy="{y(p[1]):.2f}" r="{radius:.2f}" fill="{s["color"]}" opacity=".75"><title>{esc(s["name"])}</title></circle>'
                )
            else:
                segment.append(f"{x(p[0]):.2f},{y(p[1]):.2f}")
        if segment:
            segments.append(segment)
        for segment in segments:
            out.append(
                f'<polyline fill="none" stroke="{s["color"]}" stroke-width="2" points="{" ".join(segment)}"/>'
            )
    for i, event in enumerate(events):
        if event["t"] is None:
            continue
        at = x(event["t"])
        out.append(
            f'<path d="M{at:.2f} 40V265" stroke="#64748b" stroke-dasharray="4 3"/><text x="{at+3:.2f}" y="{47+(i%3)*15}">{esc(event["label"])}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def render(result):
    out = [
        '<!doctype html><meta charset="utf-8"><title>'
        + esc(result["title"])
        + "</title>",
        "<style>body{font:15px system-ui;color:#1e293b;background:#f8fafc;margin:32px}section{background:white;padding:20px;margin:18px 0;border:1px solid #e2e8f0;border-radius:10px}svg{width:100%;max-height:420px}svg text{font:11px system-ui}table{border-collapse:collapse;width:100%}td,th{border:1px solid #cbd5e1;padding:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere}.FEASIBLE{background:#dcfce7}.INFEASIBLE{background:#fee2e2}</style>",
        "<h1>" + esc(result["title"]) + "</h1><p>" + esc(result["verdict"]) + "</p>",
    ]
    if result["kind"] == "timeline":
        allpoints = [
            p
            for run in result["runs"]
            for panel in run["panels"]
            for points in panel["series"].values()
            for p in points
        ]
        times = [p[0] for p in allpoints] + [
            e["t"]
            for run in result["runs"]
            for e in run["events"]
            if e["t"] is not None
        ]
        xb = bounds(times)
        for index, run in enumerate(result["runs"]):
            out += [
                "<section><h2>Run "
                + str(index + 1)
                + " · "
                + esc(run["validity"])
                + "</h2><p>"
                + esc((run["identity"] or {}).get("path", "unidentified run"))
                + " · "
                + esc((run["identity"] or {}).get("sha256", "")[:12])
                + "</p>"
            ]
            for pi, panel in enumerate(run["panels"]):
                series = [
                    dict(name=k, points=v, color=COLORS[i % len(COLORS)])
                    for i, (k, v) in enumerate(panel["series"].items())
                ]
                yb = bounds(
                    p[1]
                    for other in result["runs"]
                    for points in other["panels"][pi]["series"].values()
                    for p in points
                )
                out += [
                    "<h3>" + esc(panel["title"]) + "</h3>",
                    chart(series, events=run["events"], xbounds=xb, ybounds=yb),
                    "<p>" + esc(", ".join(panel["series"])) + "</p>",
                ]
                for key, points in panel["series"].items():
                    if not points:
                        out.append("<p>MISSING_DATA: " + esc(key) + "</p>")
            out += [
                "<pre>"
                + esc(json.dumps(run["events"], ensure_ascii=False, indent=2))
                + "</pre></section>"
            ]
    else:
        spec = result["specification"]
        points = result["points"]
        facets = sorted({json.dumps(p["parameters"][spec["facet"]]) for p in points})
        colors = sorted({json.dumps(p["parameters"][spec["color"]]) for p in points})
        usable = [p for p in points if p["status"] not in {"MISSING_OR_INVALID"}]
        xb = bounds(p["metrics"][spec["x"]]["value"] for p in usable)
        yb = bounds(p["metrics"][spec["y"]]["value"] for p in usable)
        largest = max(
            [max(0, p["metrics"][spec["size"]]["value"]) for p in usable] + [1]
        )
        out.append(
            "<p>"
            + " · ".join(
                '<span style="color:'
                + COLORS[i % len(COLORS)]
                + '">● '
                + esc(spec["color"] + " = " + value)
                + "</span>"
                for i, value in enumerate(colors)
            )
            + "</p>"
        )
        for facet in facets:
            series = []
            for p in usable:
                if json.dumps(p["parameters"][spec["facet"]]) != facet:
                    continue
                v = p["metrics"]
                color = COLORS[
                    colors.index(json.dumps(p["parameters"][spec["color"]]))
                    % len(COLORS)
                ]
                series.append(
                    dict(
                        name=json.dumps(p, ensure_ascii=False),
                        color=color,
                        points=[
                            [
                                v[spec["x"]]["value"],
                                v[spec["y"]]["value"],
                                3
                                + 10
                                * math.sqrt(max(0, v[spec["size"]]["value"]) / largest),
                            ]
                        ],
                    )
                )
            out += [
                "<section><h2>" + esc(spec["facet"] + " = " + facet) + "</h2>",
                chart(
                    series,
                    scatter=True,
                    xbounds=xb,
                    ybounds=yb,
                    xlabel=spec["x"],
                    ylabel=spec["y"],
                ),
                "<p>Color: "
                + esc(spec["color"])
                + "; size: "
                + esc(spec["size"])
                + "</p></section>",
            ]
        out += [
            "<section><h2>Constraint grid · all independent runs retained</h2><table><tr><th>"
            + esc(spec["facet"])
            + "</th>"
            + "".join("<th>" + esc(c) + "</th>" for c in colors)
            + "</tr>"
        ]
        for f in facets:
            out.append("<tr><th>" + esc(f) + "</th>")
            for c in colors:
                cell = [
                    p
                    for p in points
                    if json.dumps(p["parameters"][spec["facet"]]) == f
                    and json.dumps(p["parameters"][spec["color"]]) == c
                ]
                state = (
                    cell[0]["status"]
                    if cell and len({p["status"] for p in cell}) == 1
                    else "MIXED" if cell else "NOT_RUN"
                )
                out.append(
                    '<td class="'
                    + esc(state)
                    + '">'
                    + esc(state)
                    + " ("
                    + str(len(cell))
                    + ")</td>"
                )
            out.append("</tr>")
        out.append("</table></section>")
    out.append(
        "<details><summary>Definitions, source identities, windows and evidence</summary><pre>"
        + esc(json.dumps(result, ensure_ascii=False, indent=2))
        + "</pre></details>"
    )
    return "".join(out)
