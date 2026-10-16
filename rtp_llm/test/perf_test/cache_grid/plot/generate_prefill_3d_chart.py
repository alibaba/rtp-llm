#!/usr/bin/env python3
"""Generate a dense, dependency-free 3D prefill RT SVG.

Examples::

  python3 generate_prefill_3d_chart.py \
    --input /path/Prefill_Result.final.json \
    --output /tmp/prefill_3d.svg --batch-size 1

The plot has exactly three data coordinates: X=compute tokens, Y=cached
tokens, Z=server first_token_cost_time in milliseconds. Every usable row for the
selected batch is emitted with no point decimation. Dot colour depth shows
local query density on the compute/cache plane; TTFT remains encoded solely by
the Z coordinate.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import pathlib
from collections import defaultdict
from statistics import median
from typing import Any

from rtp_llm.test.perf_test.cache_grid.config.perf_profile import (
    extract_embedded_profile,
)
from rtp_llm.test.perf_test.cache_grid.config.perf_profile import (
    fingerprint as profile_fingerprint,
)
from rtp_llm.test.perf_test.cache_grid.config.perf_profile import (
    load_profile,
    resolve_int,
    resolve_label,
    resolve_title,
)
from rtp_llm.test.perf_test.cache_grid.runner.result_schema import current_metrics
from rtp_llm.test.perf_test.cache_grid.runner.result_schema import (
    finite_number as _number,
)
from rtp_llm.test.perf_test.cache_grid.runner.result_schema import measurement_rows

QUERY_DENSITY_PALETTE = ("#dbeafe", "#93c5fd", "#60a5fa", "#2563eb", "#1e3a8a")


def _query_density(
    rows: list[dict[str, float]], compute_max: float, cache_max: float, bins: int = 16
) -> tuple[list[int], int]:
    """Count query geometries in local compute/cache bins.

    Density deliberately ignores TTFT: colour describes where query shapes are
    concentrated, while the Z coordinate remains the sole encoding of latency.
    """
    counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    locations: list[tuple[int, int]] = []
    compute_scale = max(compute_max, 1.0)
    cache_scale = max(cache_max, 1.0)
    for row in rows:
        compute_bin = min(bins - 1, int(row["compute"] / compute_scale * bins))
        cache_bin = min(bins - 1, int(row["cache"] / cache_scale * bins))
        location = (compute_bin, cache_bin)
        locations.append(location)
        counts[location] += 1
    densities = [counts[location] for location in locations]
    return densities, max(densities, default=0)


def _query_density_color(density: int, maximum: int) -> str:
    """Map density to a log-scaled light-to-dark blue ramp."""
    if maximum <= 1:
        return QUERY_DENSITY_PALETTE[0]
    ratio = math.log(density) / math.log(maximum)
    index = round(ratio * (len(QUERY_DENSITY_PALETTE) - 1))
    return QUERY_DENSITY_PALETTE[index]


def load_rows(path: pathlib.Path, batch_size: int) -> list[dict[str, float]]:
    """Load server first-token latency from cache_grid_results.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "compute": row["compute_len"],
            "cache": row["cache_len"],
            "rt": row["prefill_rt"],
            "input": row["input_len"],
        }
        for row in measurement_rows(data, batch_size, strict=True)
    ]


def esc(value: object) -> str:
    return html.escape(str(value))


def fmt_tokens(value: float) -> str:
    if value >= 1_000_000:
        return "1M"
    if value >= 1000:
        return f"{value / 1000:.0f}K"
    return f"{value:.0f}"


def render(
    rows: list[dict[str, float]],
    source: pathlib.Path,
    batch_size: int,
    *,
    annotate_cold_threshold: int = 1_048_575,
) -> str:
    if not rows:
        raise SystemExit("no usable rows for the requested batch size")
    width, height = 2200, 1350
    origin = (340.0, 1030.0)
    vx, vy, vz = (920.0, -210.0), (340.0, 185.0), (0.0, -725.0)
    # Axis convention is intentionally X=TTFT, Y=cache, Z=compute.  Keeping
    # these names explicit prevents accidentally publishing the old
    # compute/cache/RT orientation when changing the projection.
    xmax = max(row["rt"] for row in rows)
    ymax = max(row["cache"] for row in rows)
    zmax = max(row["compute"] for row in rows)
    # A cold-only slice has no cache-axis extent.  Keep it plottable while
    # retaining the true zero values on the axis.
    xscale = max(xmax, 1.0)
    yscale = max(ymax, 1.0)
    zscale = max(zmax, 1.0)

    def project(nx: float, ny: float, nz: float) -> tuple[float, float]:
        return (
            origin[0] + vx[0] * nx + vy[0] * ny + vz[0] * nz,
            origin[1] + vx[1] * nx + vy[1] * ny + vz[1] * nz,
        )

    def line(
        a: tuple[float, float], b: tuple[float, float], stroke="#94a3b8", sw=1, dash=""
    ) -> str:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}"{extra}/>'
        )

    def polygon(
        points: list[tuple[float, float]], fill, stroke="#52718d", sw=0.6, opacity=0.34
    ) -> str:
        value = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        return (
            f'<polygon points="{value}" fill="{fill}" fill-opacity="{opacity}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def circle(x: float, y: float) -> str:
        return (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.8" fill="#204b67" '
            'fill-opacity=".74" stroke="#fff" stroke-width=".35"/>'
        )

    def polyline(points: list[tuple[float, float]], stroke: str) -> str:
        if len(points) < 2:
            return ""
        value = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        return (
            f'<polyline points="{value}" fill="none" stroke="{stroke}" '
            'stroke-width="3" stroke-opacity=".82" stroke-linejoin="round"/>'
        )

    out = [
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fff"/>
<style>text{{font-family:Arial,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;fill:#172033}}
.title{{font-size:34px;font-weight:700}} .sub{{font-size:18px;fill:#475569}}
.axis{{font-size:18px;fill:#334155}} .tick{{font-size:15px;fill:#475569}}
.paneltitle{{font-size:23px;font-weight:700}} .body{{font-size:17px;fill:#334155}}
.note{{font-size:15px;fill:#64748b}}</style>
<text x="1100" y="52" text-anchor="middle" class="title">Prefill RT — dense 3D measurement view</text>
<text x="1100" y="85" text-anchor="middle" class="sub">X = measured Server first-token latency (ms) · Y = cached tokens · Z = compute tokens; every selected geometry is plotted</text>"""
    ]

    out.append(
        polygon(
            [project(0, 0, 0), project(1, 0, 0), project(1, 1, 0), project(0, 1, 0)],
            "#f8fafc",
            "#cbd5e1",
            2,
            0.95,
        )
    )
    for i in range(1, 9):
        t = i / 9
        out.append(line(project(t, 0, 0), project(t, 1, 0), "#d6dee8"))
        out.append(line(project(0, t, 0), project(1, t, 0), "#d6dee8"))
    for t in (0.25, 0.5, 0.75, 1):
        out.append(line(project(0, 0, t), project(1, 0, t), "#d6dee8", 1, "6 6"))
        out.append(line(project(0, 0, t), project(0, 1, t), "#d6dee8", 1, "6 6"))

    # 64x64 median cells provide a continuous visual guide; all measurements
    # remain visible as dots and drop-lines below.
    cells: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        ix = min(63, int(row["rt"] / xscale * 64))
        iy = min(63, int(row["cache"] / yscale * 64))
        cells[ix, iy].append(row["compute"])
    for (ix, iy), values in sorted(
        cells.items(), key=lambda item: sum(item[0]), reverse=True
    ):
        x0, x1, y0, y1 = ix / 64, (ix + 1) / 64, iy / 64, (iy + 1) / 64
        z = median(values) / zscale
        out.append(
            polygon(
                [
                    project(x0, y0, z),
                    project(x1, y0, z),
                    project(x1, y1, z),
                    project(x0, y1, z),
                ],
                "#a9c6dc",
            )
        )

    for row in rows:  # no decimation
        nx, ny, nz = row["rt"] / xscale, row["cache"] / yscale, row["compute"] / zscale
        out.append(line(project(nx, ny, 0), project(nx, ny, nz), "#64748b", 0.45))
        out.append(circle(*project(nx, ny, nz)))

    # Trace slices so the two independent trends remain visible even when
    # thousands of points overlap in the perspective projection.  Colours
    # identify the selected slice, never the RT value (RT remains the Z axis).
    palette = (
        "#b42318",
        "#d97706",
        "#15803d",
        "#0369a1",
        "#6d28d9",
        "#be185d",
        "#475569",
        "#0f766e",
    )
    cache_counts: defaultdict[float, int] = defaultdict(int)
    compute_counts: defaultdict[float, int] = defaultdict(int)
    for row in rows:
        cache_counts[row["cache"]] += 1
        compute_counts[row["compute"]] += 1
    cache_levels = [
        level
        for level, _ in sorted(
            cache_counts.items(), key=lambda item: (-item[1], item[0])
        )[:8]
    ]
    compute_levels = [
        level
        for level, _ in sorted(
            compute_counts.items(), key=lambda item: (-item[1], item[0])
        )[:8]
    ]
    for index, level in enumerate(sorted(cache_levels)):
        subset = sorted(
            (row for row in rows if row["cache"] == level),
            key=lambda row: row["compute"],
        )
        points = [
            project(row["rt"] / xscale, row["cache"] / yscale, row["compute"] / zscale)
            for row in subset
        ]
        out.append(polyline(points, palette[index % len(palette)]))
    for index, level in enumerate(sorted(compute_levels)):
        subset = sorted(
            (row for row in rows if row["compute"] == level),
            key=lambda row: row["cache"],
        )
        points = [
            project(row["rt"] / xscale, row["cache"] / yscale, row["compute"] / zscale)
            for row in subset
        ]
        out.append(polyline(points, palette[(index + 4) % len(palette)]))

    for a, b in (
        (project(0, 0, 0), project(1, 0, 0)),
        (project(0, 0, 0), project(0, 1, 0)),
        (project(0, 0, 0), project(0, 0, 1)),
    ):
        out.append(line(a, b, "#0f172a", 3))
    for i in range(5):
        t = i / 4
        p = project(t, 0, 0)
        out.append(line((p[0], p[1]), (p[0] + 9, p[1] + 14), "#0f172a", 1.5))
        out.append(
            f'<text x="{p[0]+4:.1f}" y="{p[1]+37:.1f}" text-anchor="middle" class="tick">{xmax*t:.0f} ms</text>'
        )
        p = project(0, t, 0)
        out.append(line((p[0], p[1]), (p[0] - 8, p[1] + 14), "#0f172a", 1.5))
        out.append(
            f'<text x="{p[0]-16:.1f}" y="{p[1]+37:.1f}" text-anchor="end" class="tick">{fmt_tokens(ymax*t)}</text>'
        )
    for i in range(6):
        t = i / 5
        p = project(0, 0, t)
        out.append(line((p[0] - 9, p[1]), (p[0] - 21, p[1]), "#0f172a", 1.5))
        out.append(
            f'<text x="{p[0]-28:.1f}" y="{p[1]+5:.1f}" text-anchor="end" class="tick">{fmt_tokens(zmax*t)}</text>'
        )
    out += [
        f'<text x="{project(.55,0,0)[0]:.1f}" y="{project(.55,0,0)[1]+78:.1f}" text-anchor="middle" class="axis">Server first-token latency (X, ms)</text>',
        f'<text x="{project(0,.55,0)[0]-80:.1f}" y="{project(0,.55,0)[1]+78:.1f}" text-anchor="middle" class="axis">cached tokens (Y)</text>',
        f'<text x="{project(0,0,.58)[0]-70:.1f}" y="{project(0,0,.58)[1]:.1f}" text-anchor="middle" transform="rotate(-90 {project(0,0,.58)[0]-70:.1f},{project(0,0,.58)[1]:.1f})" class="axis">compute tokens (Z)</text>',
    ]

    one_m = next(
        (
            row
            for row in rows
            if row["input"] >= annotate_cold_threshold and row["cache"] == 0
        ),
        None,
    )
    if one_m is not None:
        x, y = project(one_m["rt"] / xscale, 0, one_m["compute"] / zscale)
        out.append(line((x, y), (x + 110, y - 70), "#b42318", 2, "5 4"))
        out.append(
            f'<rect x="{x+105:.1f}" y="{y-115:.1f}" width="330" height="78" rx="10" fill="#fff7ed" stroke="#b42318" stroke-width="2"/>'
        )
        out.append(
            f'<text x="{x+125:.1f}" y="{y-82:.1f}" class="body" fill="#991b1b">1M cold (BS={batch_size}, cache=0)</text>'
        )
        out.append(
            f'<text x="{x+125:.1f}" y="{y-53:.1f}" class="body" fill="#991b1b">TTFT = {one_m["rt"]:.1f} ms</text>'
        )

    px, py, pw, ph = 1450, 145, 690, 760
    out.append(
        f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="14" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2"/>'
    )
    out.append(f'<text x="{px+30}" y="{py+45}" class="paneltitle">Exact dataset</text>')
    details = [
        f"all rows in source = {len(data_metrics(source)):,}",
        f"plotted batch={batch_size} = {len(rows):,} (every row)",
        f"seq/input range = {min(r['input'] for r in rows):,.0f} .. {max(r['input'] for r in rows):,.0f}",
        f"TTFT range = {min(r['rt'] for r in rows):.1f} .. {xmax:.1f} ms",
        "uniform colour; coordinates are X/Y/Z only",
    ]
    for i, value in enumerate(details):
        out.append(
            f'<text x="{px+30}" y="{py+90+i*34}" class="body">{esc(value)}</text>'
        )
    out.append(
        f'<line x1="{px+30}" y1="{py+290}" x2="{px+pw-30}" y2="{py+290}" stroke="#cbd5e1"/>'
    )
    out.append(
        f'<text x="{px+30}" y="{py+335}" class="paneltitle">Interpretation</text>'
    )
    for i, value in enumerate(
        [
            "X right: larger Server first-token latency (slower).",
            "Y up: more KV reuse; cache dimension.",
            "Z up: more compute tokens (uncached work).",
            "Triangular footprint: input = compute + cache.",
        ]
    ):
        out.append(
            f'<text x="{px+30}" y="{py+378+i*35}" class="body">{esc(value)}</text>'
        )
    out.append(
        f'<text x="{px+30}" y="{py+555}" class="note">Source: {esc(source.name)} · no point decimation</text>'
    )
    out.append(
        f'<text x="{px+30}" y="{py+585}" class="note">Coloured lines: fixed-cache/fixed-compute slices; colour is not RT</text>'
    )
    out.append(
        f'<text x="{px+30}" y="{py+615}" class="note">fixed-cache: {esc(", ".join(fmt_tokens(v) for v in sorted(cache_levels)))}</text>'
    )
    out.append(
        f'<text x="{px+30}" y="{py+645}" class="note">fixed-compute: {esc(", ".join(fmt_tokens(v) for v in sorted(compute_levels)))}</text>'
    )
    out.append("</svg>")
    return "".join(out)


def data_metrics(path: pathlib.Path) -> list[Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return current_metrics(data, str(path))


def render_cold_miss_2d(
    rows: list[dict[str, float]],
    source: pathlib.Path,
    batch_size: int,
    model_label: str = "Model",
) -> str:
    """Render the cache-miss sequence-length trend as a readable 2-D SVG.

    Only rows whose observed cache is zero are included.  The main chart uses
    a linear token axis so the 1M boundary is honest; an inset expands the
    short-sequence region that would otherwise be compressed near the origin.
    """
    cold = sorted(
        (row for row in rows if row["cache"] == 0), key=lambda row: row["input"]
    )
    if not cold:
        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="420" viewBox="0 0 1200 420">
<rect width="100%" height="100%" fill="#ffffff"/>
<style>text{{font-family:Arial,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;fill:#172033}}</style>
<text x="60" y="75" font-size="32" font-weight="700">{html.escape(str(model_label))}：Cache miss 的 seq_len–RT 趋势</text>
<text x="60" y="150" font-size="21">BS={batch_size} · 当前结果没有 observed cache_len=0 的有效样本</text>
<text x="60" y="205" font-size="17" fill="#64748b">source: {html.escape(source.name)}</text>
</svg>"""
    width, height = 1800, 1050
    x0, y0, x1, y1 = 150.0, 180.0, 1430.0, 820.0
    xmax = max(row["input"] for row in cold)
    ymax = max(row["rt"] for row in cold) * 1.12

    def esc_text(value: object) -> str:
        return html.escape(str(value))

    def sx(value: float, lo: float = 0.0, hi: float = xmax) -> float:
        return x0 + (value - lo) / max(hi - lo, 1.0) * (x1 - x0)

    def sy(value: float, lo: float = 0.0, hi: float = ymax) -> float:
        return y1 - (value - lo) / max(hi - lo, 1.0) * (y1 - y0)

    def line(
        ax: float,
        ay: float,
        bx: float,
        by: float,
        stroke: str = "#cbd5e1",
        sw: float = 1.0,
        dash: str = "",
        opacity: float = 1.0,
    ) -> str:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}" stroke-opacity="{opacity}"{dash_attr}/>'
        )

    def circle(cx: float, cy: float, radius: float = 3.0) -> str:
        return (
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" '
            'fill="#2563eb" fill-opacity=".72" stroke="#ffffff" stroke-width=".55"/>'
        )

    points = [(sx(row["input"]), sy(row["rt"])) for row in cold]
    out = [
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<style>text{{font-family:Arial,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;fill:#172033}}
.title{{font-size:34px;font-weight:700}} .sub{{font-size:18px;fill:#475569}}
.axis{{font-size:20px;fill:#334155;font-weight:600}} .tick{{font-size:15px;fill:#475569}}
.paneltitle{{font-size:23px;font-weight:700}} .body{{font-size:17px;fill:#334155}}
.note{{font-size:15px;fill:#64748b}}</style>
<text x="70" y="55" class="title">{esc_text(model_label)}：Cache miss 的 seq_len–RT 趋势</text>
<text x="70" y="88" class="sub">BS={batch_size} · observed cache_len=0 · 每个 seq_len 使用完整成功测量的中位 服务端 first_token_cost_time</text>"""
    ]
    out.append(line(x0, y1, x1, y1, "#0f172a", 2))
    out.append(line(x0, y0, x0, y1, "#0f172a", 2))
    for index in range(9):
        tick = xmax * index / 8
        x = sx(tick)
        out.append(line(x, y1, x, y1 + 9, "#0f172a", 1.3))
        out.append(
            f'<text x="{x:.1f}" y="{y1 + 34:.1f}" text-anchor="middle" class="tick">{esc_text(fmt_tokens(tick))}</text>'
        )
        if x < x1 - 1:
            out.append(line(x, y0, x, y1, "#e2e8f0", 1, "5 7"))
    for index in range(7):
        value = ymax * index / 6
        y = sy(value)
        out.append(line(x0 - 8, y, x0, y, "#0f172a", 1.3))
        out.append(line(x0, y, x1, y, "#e2e8f0", 1, "5 7"))
        out.append(
            f'<text x="{x0 - 15:.1f}" y="{y + 5:.1f}" text-anchor="end" class="tick">{value:.0f}</text>'
        )
    out.append(
        '<polyline points="'
        + " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        + '" fill="none" stroke="#2563eb" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round"/>'
    )
    out.extend(circle(x, y, 2.8) for x, y in points)
    out += [
        f'<text x="{(x0 + x1) / 2:.1f}" y="{y1 + 78:.1f}" text-anchor="middle" class="axis">seq_len（tokens，线性轴）</text>',
        f'<text x="42" y="{(y0 + y1) / 2:.1f}" text-anchor="middle" transform="rotate(-90 42 {(y0 + y1) / 2:.1f})" class="axis">中位 服务端 first_token_cost_time（ms）</text>',
    ]

    # Inset for the short-sequence region, where a full 1M linear axis hides
    # useful detail.  It is a zoom of the same points, not a second dataset.
    inset_x, inset_y, inset_w, inset_h = 980.0, 225.0, 400.0, 260.0
    inset_xmax = min(131_072.0, xmax)
    inset_rows = [row for row in cold if row["input"] <= inset_xmax]
    inset_left = inset_x + 45.0
    inset_right = inset_x + inset_w - 15.0
    inset_top = inset_y + 45.0
    inset_bottom = inset_y + inset_h - 35.0

    out.append(
        f'<rect x="{inset_x:.1f}" y="{inset_y:.1f}" width="{inset_w:.1f}" height="{inset_h:.1f}" rx="10" fill="#f8fafc" stroke="#64748b" stroke-width="1.5"/>'
    )
    out.append(
        f'<text x="{inset_x + 15:.1f}" y="{inset_y + 28:.1f}" class="body" font-weight="700">放大：0–{fmt_tokens(inset_xmax)} tokens</text>'
    )
    if inset_rows:
        inset_ymin = min(row["rt"] for row in inset_rows)
        inset_ymax = max(row["rt"] for row in inset_rows) * 1.08

        def ix(value: float) -> float:
            return inset_left + value / max(inset_xmax, 1.0) * (
                inset_right - inset_left
            )

        def iy(value: float) -> float:
            return inset_bottom - (value - inset_ymin) / max(
                inset_ymax - inset_ymin, 1.0
            ) * (inset_bottom - inset_top)

        out.append(
            line(inset_left, inset_bottom, inset_right, inset_bottom, "#334155", 1.2)
        )
        out.append(
            line(inset_left, inset_top, inset_left, inset_bottom, "#334155", 1.2)
        )
        inset_points = [(ix(row["input"]), iy(row["rt"])) for row in inset_rows]
        out.append(
            '<polyline points="'
            + " ".join(f"{x:.1f},{y:.1f}" for x, y in inset_points)
            + '" fill="none" stroke="#2563eb" stroke-width="2"/>'
        )
        out.extend(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" fill="#2563eb" fill-opacity=".75"/>'
            for x, y in inset_points
        )
        out.append(
            f'<text x="{inset_x + inset_w / 2:.1f}" y="{inset_y + inset_h - 8:.1f}" text-anchor="middle" class="tick">seq_len</text>'
        )
        out.append(
            f'<text x="{inset_x + 34:.1f}" y="{inset_y + inset_h / 2:.1f}" text-anchor="middle" transform="rotate(-90 {inset_x + 34:.1f} {inset_y + inset_h / 2:.1f})" class="tick">RT</text>'
        )
    else:
        out.append(
            f'<text x="{inset_x + inset_w / 2:.1f}" y="{inset_y + inset_h / 2:.1f}" text-anchor="middle" class="note">短序列区间无样本，未绘制插图</text>'
        )

    longest = max(cold, key=lambda row: row["input"])
    out.append(
        f'<line x1="{sx(longest["input"]):.1f}" y1="{sy(longest["rt"]):.1f}" x2="{x1 - 22:.1f}" y2="{sy(longest["rt"]) - 44:.1f}" stroke="#b42318" stroke-width="1.8" stroke-dasharray="5 4"/>'
    )
    out.append(
        f'<text x="{x1 - 15:.1f}" y="{sy(longest["rt"]) - 52:.1f}" text-anchor="end" class="body" fill="#991b1b" font-weight="700">最长冷点（{fmt_tokens(longest["input"])}）：{longest["rt"]:.1f} ms</text>'
    )

    panel_x, panel_y, panel_w, panel_h = 1490.0, 150.0, 275.0, 670.0
    out.append(
        f'<rect x="{panel_x:.1f}" y="{panel_y:.1f}" width="{panel_w:.1f}" height="{panel_h:.1f}" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>'
    )
    out.append(
        f'<text x="{panel_x + 20:.1f}" y="{panel_y + 42:.1f}" class="paneltitle">怎么读</text>'
    )
    median_rt = median(row["rt"] for row in cold)
    sorted_rt = sorted(row["rt"] for row in cold)
    p95_rt = sorted_rt[min(len(sorted_rt) - 1, int(len(sorted_rt) * 0.95))]
    details = [
        f"有效冷点：{len(cold):,}",
        f"seq 范围：{fmt_tokens(min(row['input'] for row in cold))}–{fmt_tokens(xmax)}",
        f"RT 中位数：{median_rt:.1f} ms",
        f"RT P95：{p95_rt:.1f} ms",
        f"最长冷点（{fmt_tokens(longest['input'])}）：{longest['rt']:.1f} ms",
        "",
        "每个圆点 = 一个 seq_len",
        "蓝线 = 按 seq_len 排序",
        "右上插图 = 放大短序列",
        "RT 取三次成功测量中位数",
    ]
    for index, value in enumerate(details):
        out.append(
            f'<text x="{panel_x + 20:.1f}" y="{panel_y + 86 + index * 34:.1f}" class="body">{esc_text(value)}</text>'
        )
    out.append(
        f'<text x="{panel_x + 20:.1f}" y="{panel_y + panel_h - 24:.1f}" class="note">source: {esc_text(source.name)}</text>'
    )
    out.append("</svg>")
    return "".join(out)


def render_clean(
    rows: list[dict[str, float]],
    source: pathlib.Path,
    batch_size: int,
    *,
    title: str = "Model Prefill — readable 3D view",
    annotate_cold_threshold: int = 1_048_575,
) -> str:
    """Render a legible 3-D view without the old dense drop-line clutter.

    X is compute tokens, Y is observed cache reuse, and Z is TTFT.  Every
    geometry remains a dot coloured by local query density; only a handful of
    representative fixed-cache/fixed-compute slices are drawn as guide lines.
    """
    if not rows:
        raise SystemExit("no usable rows for the requested batch size")
    width, height = 2200, 1350
    # Use a true isometric projection. Tilting the compute axis left makes the
    # three coordinates visibly separate instead of looking like a flat cloud.
    origin = (360.0, 1060.0)
    vx, vy, vz = (850.0, -205.0), (430.0, 235.0), (-150.0, -710.0)
    rt_max = max(row["rt"] for row in rows)
    cache_max = max(row["cache"] for row in rows)
    compute_max = max(row["compute"] for row in rows)
    xscale, yscale, zscale = (
        max(compute_max, 1.0),
        max(cache_max, 1.0),
        max(rt_max, 1.0),
    )

    def project(nx: float, ny: float, nz: float) -> tuple[float, float]:
        return (
            origin[0] + vx[0] * nx + vy[0] * ny + vz[0] * nz,
            origin[1] + vx[1] * nx + vy[1] * ny + vz[1] * nz,
        )

    def line(
        a: tuple[float, float],
        b: tuple[float, float],
        stroke="#cbd5e1",
        sw=1,
        dash="",
        opacity=1.0,
    ) -> str:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}" stroke-opacity="{opacity}"{extra}/>'
        )

    def circle(
        point: tuple[float, float], density: int, maximum_density: int, radius=3.2
    ) -> str:
        fill = _query_density_color(density, maximum_density)
        return (
            f'<circle cx="{point[0]:.1f}" cy="{point[1]:.1f}" r="{radius}" '
            f'fill="{fill}" fill-opacity=".82" stroke="#ffffff" stroke-width=".55" '
            f'data-query-density="{density}"><title>local query density: {density}</title></circle>'
        )

    def polyline(points: list[tuple[float, float]], stroke: str, dash: str = "") -> str:
        if len(points) < 2:
            return ""
        value = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<polyline points="{value}" fill="none" stroke="{stroke}" '
            f'stroke-width="3.2" stroke-opacity=".78" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
        )

    out = [
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<style>text{{font-family:Arial,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;fill:#172033}}
.title{{font-size:34px;font-weight:700}} .sub{{font-size:18px;fill:#475569}}
.axis{{font-size:18px;fill:#334155;font-weight:600}} .tick{{font-size:15px;fill:#475569}}
.paneltitle{{font-size:23px;font-weight:700}} .body{{font-size:17px;fill:#334155}}
.note{{font-size:15px;fill:#64748b}} .legend{{font-size:16px;fill:#334155}}</style>
<text x="1100" y="52" text-anchor="middle" class="title">{esc(title)}</text>
<text x="1100" y="85" text-anchor="middle" class="sub">X = compute tokens · Y = observed cached tokens · Z = Server first-token latency (ms) · all {len(rows):,} geometries shown</text>"""
    ]

    # Ground plane and a sparse grid keep the perspective legible.
    out.append(
        f'<polygon points="{project(0,0,0)[0]:.1f},{project(0,0,0)[1]:.1f} {project(1,0,0)[0]:.1f},{project(1,0,0)[1]:.1f} {project(1,1,0)[0]:.1f},{project(1,1,0)[1]:.1f} {project(0,1,0)[0]:.1f},{project(0,1,0)[1]:.1f}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2"/>'
    )
    for i in range(1, 7):
        t = i / 7
        out.append(line(project(t, 0, 0), project(t, 1, 0), "#dbe4ee", 1, opacity=0.85))
        out.append(line(project(0, t, 0), project(1, t, 0), "#dbe4ee", 1, opacity=0.85))
    for t in (0.2, 0.4, 0.6, 0.8, 1.0):
        out.append(line(project(0, 0, t), project(1, 0, t), "#dbe4ee", 1, "5 6", 0.75))
        out.append(line(project(0, 0, t), project(0, 1, t), "#dbe4ee", 1, "5 6", 0.75))

    # Every point is kept. Colour depth represents local query-geometry
    # density on the compute/cache plane; TTFT remains encoded only by Z.
    sorted_rows = sorted(rows, key=lambda item: item["compute"])
    densities, maximum_density = _query_density(sorted_rows, compute_max, cache_max)
    for row, density in zip(sorted_rows, densities):
        out.append(
            circle(
                project(
                    row["compute"] / xscale, row["cache"] / yscale, row["rt"] / zscale
                ),
                density,
                maximum_density,
            )
        )

    # Solid warm lines are fixed-cache slices; dashed cool lines are
    # fixed-compute slices.  Keeping the two families visually distinct is
    # much easier to read than assigning an unrelated colour to every line.
    cache_palette = ("#b42318", "#d97706", "#15803d", "#0369a1")
    compute_palette = ("#7c3aed", "#c026d3", "#0369a1", "#0f766e")
    cache_values = sorted({row["cache"] for row in rows})
    compute_values = sorted({row["compute"] for row in rows})

    def nearest_levels(values: list[float], targets: list[float]) -> list[float]:
        chosen: list[float] = []
        for target in targets:
            level = min(values, key=lambda value: abs(value - target))
            if level not in chosen:
                chosen.append(level)
        return chosen

    cache_levels = nearest_levels(
        cache_values, [0, 0.33 * cache_max, 0.67 * cache_max, cache_max]
    )
    compute_levels = nearest_levels(
        compute_values, [0, 0.33 * compute_max, 0.67 * compute_max, compute_max]
    )

    def slice_points(
        subset: list[dict[str, float]], varying: str, fixed: str, fixed_value: float
    ) -> list[tuple[float, float]]:
        """Return median-binned guide points rather than a jagged raw join.

        RT has natural run-to-run noise.  Joining every raw point makes a
        slice look like a saw-tooth and hides the trend.  Binning the varying
        coordinate into 12 equal-population bins and taking median RT keeps
        the guide lines representative without changing the plotted points.
        """
        if len(subset) < 2:
            return []
        ordered = sorted(subset, key=lambda row: row[varying])
        bins = min(12, len(ordered))
        points: list[tuple[float, float]] = []
        for index in range(bins):
            lo = (index * len(ordered)) // bins
            hi = ((index + 1) * len(ordered)) // bins
            group = ordered[lo : max(hi, lo + 1)]
            varying_value = median(row[varying] for row in group)
            rt_value = median(row["rt"] for row in group)
            representative = dict(group[len(group) // 2])
            representative[varying] = varying_value
            # Draw the guide on the selected fixed-coordinate plane.  The
            # source rows come from a narrow band around that plane, which
            # avoids empty exact-level slices while preserving the intended
            # fixed-cache/fixed-compute interpretation.
            representative[fixed] = fixed_value
            representative["rt"] = rt_value
            points.append(
                project(
                    representative["compute"] / xscale,
                    representative["cache"] / yscale,
                    representative["rt"] / zscale,
                )
            )
        return points

    def near_slice(axis: str, level: float) -> list[dict[str, float]]:
        # A 2.5% band (with a 4K floor) gives enough observations at every
        # level, including the two cube boundaries, without drawing a dense
        # bundle of nearly coincident raw lines.
        tolerance = max(4_096.0, max(cache_max, compute_max) * 0.025)
        subset = [row for row in rows if abs(row[axis] - level) <= tolerance]
        if len(subset) >= 8:
            return subset
        return sorted(rows, key=lambda row: abs(row[axis] - level))[
            : max(8, min(24, len(rows)))
        ]

    for index, level in enumerate(cache_levels):
        out.append(
            polyline(
                slice_points(near_slice("cache", level), "compute", "cache", level),
                cache_palette[index],
                "",
            )
        )
    for index, level in enumerate(compute_levels):
        out.append(
            polyline(
                slice_points(near_slice("compute", level), "cache", "compute", level),
                compute_palette[index],
                "8 6",
            )
        )

    # Axes, ticks, and labels.
    for a, b in (
        (project(0, 0, 0), project(1, 0, 0)),
        (project(0, 0, 0), project(0, 1, 0)),
        (project(0, 0, 0), project(0, 0, 1)),
    ):
        out.append(line(a, b, "#0f172a", 3))
    for i in range(5):
        t = i / 4
        p = project(t, 0, 0)
        out.append(
            f'<text x="{p[0]+4:.1f}" y="{p[1]+34:.1f}" text-anchor="middle" class="tick">{fmt_tokens(compute_max*t)}</text>'
        )
        p = project(0, t, 0)
        out.append(
            f'<text x="{p[0]-15:.1f}" y="{p[1]+34:.1f}" text-anchor="end" class="tick">{fmt_tokens(cache_max*t)}</text>'
        )
    for i in range(6):
        t = i / 5
        p = project(0, 0, t)
        out.append(
            f'<text x="{p[0]-18:.1f}" y="{p[1]+5:.1f}" text-anchor="end" class="tick">{rt_max*t:.0f} ms</text>'
        )
    out += [
        f'<text x="{project(.55,0,0)[0]:.1f}" y="{project(.55,0,0)[1]+70:.1f}" text-anchor="middle" class="axis">compute tokens (X)</text>',
        f'<text x="{project(0,.55,0)[0]-75:.1f}" y="{project(0,.55,0)[1]+70:.1f}" text-anchor="middle" class="axis">cached tokens (Y)</text>',
        f'<text x="{project(0,0,.57)[0]-62:.1f}" y="{project(0,0,.57)[1]:.1f}" text-anchor="middle" transform="rotate(-90 {project(0,0,.57)[0]-62:.1f},{project(0,0,.57)[1]:.1f})" class="axis">Server first-token latency (Z, ms)</text>',
    ]

    one_m = next(
        (
            row
            for row in rows
            if row["input"] >= annotate_cold_threshold and row["cache"] == 0
        ),
        None,
    )
    if one_m is not None:
        px, py = project(one_m["compute"] / xscale, 0, one_m["rt"] / zscale)
        out.append(line((px, py), (px + 125, py - 70), "#b42318", 2, "5 4"))
        out.append(
            f'<rect x="{px+120:.1f}" y="{py-114:.1f}" width="315" height="76" rx="10" fill="#fff7ed" stroke="#b42318" stroke-width="2"/>'
        )
        out.append(
            f'<text x="{px+140:.1f}" y="{py-82:.1f}" class="body" fill="#991b1b">1M cold (cache=0)</text>'
        )
        out.append(
            f'<text x="{px+140:.1f}" y="{py-53:.1f}" class="body" fill="#991b1b">TTFT = {one_m["rt"]:.1f} ms</text>'
        )

    px, py, pw, ph = 1450, 145, 690, 790
    out.append(
        f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="14" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2"/>'
    )
    out.append(
        f'<text x="{px+30}" y="{py+45}" class="paneltitle">How to read this chart</text>'
    )
    text_lines = [
        f"source rows = {len(data_metrics(source)):,}; plotted = {len(rows):,}",
        "dots = every physical geometry; darker means denser queries",
        "X right = more uncached compute tokens",
        "Y up = more observed KV-cache reuse",
        "Z up = higher TTFT / slower prefill",
        "input length = cache + compute",
        "solid warm = fixed-cache median guides",
        "dashed cool = fixed-compute median guides",
    ]
    for i, value in enumerate(text_lines):
        out.append(
            f'<text x="{px+30}" y="{py+90+i*34}" class="body">{esc(value)}</text>'
        )
    out.append(
        f'<line x1="{px+30}" y1="{py+390}" x2="{px+pw-30}" y2="{py+390}" stroke="#cbd5e1"/>'
    )
    out.append(
        f'<text x="{px+30}" y="{py+435}" class="paneltitle">Representative cache slices</text>'
    )
    for i, level in enumerate(cache_levels):
        yy = py + 475 + i * 35
        out.append(
            f'<line x1="{px+30}" y1="{yy-6}" x2="{px+62}" y2="{yy-6}" stroke="{cache_palette[i]}" stroke-width="4" stroke-linecap="round"/>'
        )
        out.append(
            f'<text x="{px+75}" y="{yy}" class="legend">cache ≈ {fmt_tokens(level)}</text>'
        )
    out.append(
        f'<text x="{px+360}" y="{py+435}" class="paneltitle">Representative compute slices</text>'
    )
    for i, level in enumerate(compute_levels):
        yy = py + 475 + i * 35
        out.append(
            f'<line x1="{px+360}" y1="{yy-6}" x2="{px+392}" y2="{yy-6}" stroke="{compute_palette[i]}" stroke-width="4" stroke-linecap="round" stroke-dasharray="8 6"/>'
        )
        out.append(
            f'<text x="{px+405}" y="{yy}" class="legend">compute ≈ {fmt_tokens(level)}</text>'
        )
    density_y = py + 650
    out.append(
        f'<text x="{px+30}" y="{density_y}" class="paneltitle">Query distribution density</text>'
    )
    swatch_width = 82
    for index, colour in enumerate(QUERY_DENSITY_PALETTE):
        out.append(
            f'<rect x="{px+30+index*swatch_width}" y="{density_y+22}" width="{swatch_width}" height="24" fill="{colour}"/>'
        )
    out.append(
        f'<text x="{px+30}" y="{density_y+70}" class="note">sparse · local 16×16 compute/cache bins · dense (max {maximum_density})</text>'
    )
    out.append("</svg>")
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument(
        "--cold-output",
        type=pathlib.Path,
        help="Optional cache-miss 2-D SVG. Defaults to <output stem>_cold_miss.svg.",
    )
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--title", default=None, help="Override the chart title.")
    parser.add_argument(
        "--model-label", default=None, help="Model label used to build the title."
    )
    parser.add_argument(
        "--annotate-cold-threshold",
        type=int,
        default=None,
        help="Minimum input_len for the cold-point annotation (default: 1048575).",
    )
    parser.add_argument(
        "--profile", default=None, help="JSON profile for parameter defaults."
    )
    args = parser.parse_args()

    profile = None
    profile_sha256 = None
    if args.profile:
        profile = load_profile(args.profile)
        profile_sha256 = profile_fingerprint(profile)
    else:
        try:
            input_data = json.loads(args.input.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            input_data = None
        if isinstance(input_data, dict):
            embedded_sha = input_data.get("profile_sha256")
            embedded = extract_embedded_profile(input_data)
            if embedded is not None:
                profile = embedded
                profile_sha256 = (
                    embedded_sha
                    if isinstance(embedded_sha, str)
                    else profile_fingerprint(profile)
                )

    model_label = resolve_label(profile, args.model_label, "Model")
    default_title = f"{model_label} Prefill — readable 3D view"
    title = resolve_title(profile, args.title, default_title)
    cold_threshold = args.annotate_cold_threshold
    if cold_threshold is None and profile is not None:
        cold_threshold = resolve_int(
            profile, "chart", "annotate_cold_threshold", None, 1_048_575
        )
    if cold_threshold is None:
        cold_threshold = 1_048_575

    rows = load_rows(args.input, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_clean(
            rows,
            args.input,
            args.batch_size,
            title=title,
            annotate_cold_threshold=cold_threshold,
        ),
        encoding="utf-8",
    )
    cold_output = args.cold_output or args.output.with_name(
        f"{args.output.stem}_cold_miss{args.output.suffix}"
    )
    cold_output.write_text(
        render_cold_miss_2d(rows, args.input, args.batch_size, model_label),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source": str(args.input),
                "rows": len(rows),
                "all_rows": len(data_metrics(args.input)),
                "output": str(args.output),
                "cold_output": str(cold_output),
                "cold_rows": sum(row["cache"] == 0 for row in rows),
                "title": title,
                "profile": profile,
                "profile_sha256": profile_sha256,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
