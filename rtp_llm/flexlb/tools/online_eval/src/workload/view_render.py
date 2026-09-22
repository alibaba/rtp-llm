"""Timeline and sweep layouts using the common report components and renderer."""

import json
import math
from reporting import render as render_report, details, table

COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#d97706", "#0891b2")


def bounds(values):
    values = [v for v in values if type(v) in (int, float) and math.isfinite(v)]
    if not values:
        return [0, 1]
    lo, hi = min(values), max(values)
    return [lo, hi] if hi > lo else [lo - 0.5, hi + 0.5]


def build_spec(result):
    panels, sections = [], []
    spec = dict(
        run_id=result["kind"],
        title=result["title"],
        subtitle=result["verdict"],
        panels=panels,
        sections=sections,
    )
    if result["kind"] == "timeline":
        times = [
            p[0]
            for r in result["runs"]
            for panel in r["panels"]
            for points in panel["series"].values()
            for p in points
        ]
        times += [
            e["t"] for r in result["runs"] for e in r["events"] if e["t"] is not None
        ]
        lo, hi = bounds(times)
        spec.update(
            timeAxis=dict(min=lo, max=hi), timeOriginLabel=result["time_semantics"]
        )
        for index, run in enumerate(result["runs"]):
            for pi, panel in enumerate(run["panels"]):
                axis = sorted(
                    {p[0] for points in panel["series"].values() for p in points}
                )
                yb = bounds(
                    p[1]
                    for r in result["runs"]
                    for points in r["panels"][pi]["series"].values()
                    for p in points
                )
                series = [
                    dict(
                        name=key,
                        points=[dict(x=t, y=v) for t, v in points],
                        color=COLORS[i % len(COLORS)],
                    )
                    for i, (key, points) in enumerate(panel["series"].items())
                ]
                missing = [key for key, points in panel["series"].items() if not points]
                panels.append(
                    dict(
                        id=f"run-{index}-{pi}",
                        title=f'Run {index+1} · {panel["title"]}',
                        caption=run["validity"]
                        + (" · MISSING_DATA: " + ", ".join(missing) if missing else ""),
                        type="line",
                        timeX=True,
                        xNums=axis,
                        x=[str(t) for t in axis],
                        series=series,
                        bounds=dict(y=yb),
                        events=[dict(e, name=e["label"]) for e in run["events"]],
                    )
                )
            sections.append(
                details(
                    f"Run {index+1} identity and events",
                    dict(identity=run["identity"], events=run["events"]),
                )
            )
    else:
        selection = result["specification"]
        points = result["points"]
        facets = sorted(
            {json.dumps(p["parameters"][selection["facet"]]) for p in points}
        )
        colors = sorted(
            {json.dumps(p["parameters"][selection["color"]]) for p in points}
        )
        usable = [p for p in points if p["status"] != "MISSING_OR_INVALID"]
        xb = bounds(p["metrics"][selection["x"]]["value"] for p in usable)
        yb = bounds(p["metrics"][selection["y"]]["value"] for p in usable)
        largest = max(
            [max(0, p["metrics"][selection["size"]]["value"]) for p in usable] + [1]
        )
        for index, facet in enumerate(facets):
            series = []
            for p in usable:
                if json.dumps(p["parameters"][selection["facet"]]) != facet:
                    continue
                v = p["metrics"]
                color = COLORS[
                    colors.index(json.dumps(p["parameters"][selection["color"]]))
                    % len(COLORS)
                ]
                series.append(
                    dict(
                        name=json.dumps(p, ensure_ascii=False),
                        color=color,
                        points=[
                            dict(
                                x=v[selection["x"]]["value"],
                                y=v[selection["y"]]["value"],
                                r=3
                                + 10
                                * math.sqrt(
                                    max(0, v[selection["size"]]["value"]) / largest
                                ),
                            )
                        ],
                    )
                )
            panels.append(
                dict(
                    id=f"facet-{index}",
                    type="scatter",
                    title=selection["facet"] + " = " + facet,
                    caption="Color: "
                    + selection["color"]
                    + "; size: "
                    + selection["size"],
                    series=series,
                    bounds=dict(x=xb, y=yb),
                    axisLabels=dict(x=selection["x"], y=selection["y"]),
                )
            )
        rows = []
        for f in facets:
            row = [f]
            for c in colors:
                cell = [
                    p
                    for p in points
                    if json.dumps(p["parameters"][selection["facet"]]) == f
                    and json.dumps(p["parameters"][selection["color"]]) == c
                ]
                state = (
                    cell[0]["status"]
                    if cell and len({p["status"] for p in cell}) == 1
                    else "MIXED" if cell else "NOT_RUN"
                )
                row.append(f"{state} ({len(cell)})")
            rows.append(row)
        sections.append(
            table(
                "Constraint grid · all independent runs retained",
                [selection["facet"]] + colors,
                rows,
            )
        )
    sections.append(
        details("Definitions, source identities, windows and evidence", result)
    )
    return spec


def render(result):
    return render_report(build_spec(result))
