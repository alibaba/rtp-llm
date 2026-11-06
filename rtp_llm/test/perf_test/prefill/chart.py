"""Offline prefill charts with optional embedded Plotly and portable SVGs."""

from __future__ import annotations

import html
import json
import statistics
from typing import Any, Mapping


def render_legacy_svgs(report: Mapping[str, Any]) -> dict[str, str]:
    """Render the legacy compute/cache/latency and cold-cache views."""
    rows = [row for row in report["observations"] if row.get("fit_eligible")]
    cold = [
        row
        for row in rows
        if row["measurements"]
        and all(
            member.get("observed_reuse_length") == 0 for member in row["measurements"]
        )
    ]
    return {
        "prefill_3d.svg": _render(rows, cold=False),
        "prefill_cold_miss.svg": _render(cold, cold=True),
    }


def _render(rows, *, cold):
    points = []
    for row in rows:
        distribution = row["request_distribution"]
        cache = sum(member["expected_reuse_length"] for member in distribution)
        compute = sum(member["input_length"] for member in distribution) - cache
        points.append((row, compute, cache, row["batch_latency_ms"]))
    xmax = max((p[1] for p in points), default=1) or 1
    cmax = max((p[2] for p in points), default=0)
    cache_scale = max(1, cmax)
    ymax = max((p[3] for p in points), default=1) or 1

    def project(compute, cache, latency):
        return (
            75 + 510 * compute / xmax + 160 * cache / cache_scale,
            530 - 100 * cache / cache_scale - 330 * latency / ymax,
        )

    batches = sorted({row["batch_size"] for row in rows})
    colors = {
        batch: f"hsl({360 * i / len(batches):.6f},70%,40%)"
        for i, batch in enumerate(batches)
    }
    height = max(640, 610 + 20 * ((len(batches) + 8) // 9))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="850" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            '<text x="55" y="25">Cold miss: observed zero reuse; color = batch</text>'
            if cold
            else '<text x="55" y="25">Prefill compute/cache; color = batch</text>'
        ),
        '<text x="55" y="45">Compute and cache: total tokens across batch members</text>',
        '<text x="55" y="65">Latency: median round-max server first-token latency</text>',
    ]
    axes = [
        (xmax, 0, 0, f"Compute: {xmax} total tokens"),
        (0, 0, ymax, f"Latency: {ymax:.3f} ms"),
    ]
    if not cold:
        axes.append((0, cmax, 0, f"Cache: {cmax} total tokens"))
    for compute, cache, latency, label in axes:
        x, y = project(compute, cache, latency)
        parts.append(f'<path d="M75 530 L{x} {y}" stroke="#64748b" fill="none"/>')
        parts.append(f'<text x="{x}" y="{y + 18}" font-size="12">{label}</text>')
    for row, compute, cache, latency in points:
        x, y = project(compute, cache, latency)
        label = html.escape(
            f'{row["grid_name"]} case {row["case_id"]} B={row["batch_size"]}, {latency:.3f} ms'
        )
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="3" fill="{colors[row["batch_size"]]}"><title>{label}</title></circle>'
        )
    for index, batch in enumerate(batches):
        label = html.escape(str(batch))
        parts.append(
            f'<text x="{55 + 80 * (index % 9)}" y="{580 + 20 * (index // 9)}" fill="{colors[batch]}">B={label}</text>'
        )
    if not points:
        parts.append('<text x="300" y="310">No valid observations</text>')
    parts.append("</svg>")
    return "".join(parts)


def _points(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    points = []
    for row in report["observations"]:
        if not row.get("fit_eligible"):
            continue
        distribution = row["request_distribution"]
        compute = sum(member["compute_length"] for member in distribution)
        cache = sum(member["expected_reuse_length"] for member in distribution)
        inputs = sum(member["input_length"] for member in distribution)
        grouped: dict[Any, list[Mapping[str, Any]]] = {}
        for member in row["measurements"]:
            grouped.setdefault(member["run"], []).append(member)
        rounds = []
        for run, members in sorted(grouped.items()):
            elapsed = members[0]["round_elapsed_ms"]
            rounds.append(
                {
                    "run": run,
                    "latency": max(m["latency_ms"] for m in members),
                    "elapsed_ms": elapsed,
                    "input_tpm": inputs * 60000 / elapsed,
                    "computed_tpm": compute * 60000 / elapsed,
                    "members": members,
                }
            )
        points.append(
            {
                "case_id": row["case_id"],
                "grid": row["grid_name"],
                "contract": row["measurement_contract"],
                "batch": row["batch_size"],
                "distribution": distribution,
                "compute": compute,
                "cache": cache,
                "latency": row["batch_latency_ms"],
                "input_tpm": statistics.mean(r["input_tpm"] for r in rounds),
                "computed_tpm": statistics.mean(r["computed_tpm"] for r in rounds),
                "rounds": rounds,
            }
        )
    return points


def render_interactive_chart(
    report: Mapping[str, Any], *, default_metric: str, fallback_svg: str
) -> str:
    """Render an offline 3D chart, or the supplied SVG when Plotly is absent."""
    points = _points(report)
    note = (
        f'<p>Report kind: {html.escape(str(report["report_kind"]))}. '
        "Historical latency-derived observations are excluded from charts.</p><ul>"
        + "".join(
            f'<li>{html.escape(str(e["case_id"]))}: {html.escape(str(e["reason"]))}</li>'
            for e in report["excluded"]
        )
        + "</ul>"
    )
    try:
        from plotly.offline import get_plotlyjs
    except ImportError:
        body = "<p>Plotly unavailable; showing static chart.</p>" + fallback_svg
    else:
        data = (
            json.dumps(points, ensure_ascii=True)
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("&", "\\u0026")
        )
        metric = (
            "computed_tpm" if default_metric in ("tpm", "computed_tpm") else "latency"
        )
        body = (
            '<div id="axis-labels" role="note" aria-live="polite"></div>'
            '<div id="chart"></div><div id="details"></div><script>'
            + get_plotlyjs()
            + "</script><script>\nconst chartData="
            + data
            + ";\n"
            + "const initialMetric="
            + json.dumps(metric)
            + ";\n"
            + _SCRIPT
            + "</script>"
        )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>Prefill 3D chart</title>"
        "<style>#axis-labels{white-space:pre-line;padding:12px 16px;"
        "margin:12px 0;background:#f3f6fa;font:14px/1.5 sans-serif}"
        "#chart{height:700px;min-height:520px}</style>"
        "</head><body><h1>Prefill 3D chart</h1>" + note + body + "</body></html>"
    )


_SCRIPT = r"""
const state = {batch: 'all', lengths: 'sum', samples: 'case', metric: initialMetric};
const batches = [...new Set(chartData.map(p => p.batch))].sort((a,b) => a-b);
function axisTitles() {
  const metric = state.metric === 'latency'
    ? (state.samples === 'case' ? 'Batch first-token latency (ms)' : 'Round maximum first-token latency (ms)')
    : state.metric === 'input_tpm'
      ? 'Total input throughput (tokens/min)'
      : 'Total computed throughput (tokens/min)';
  return {
    x: state.lengths === 'sum' ? 'Computed tokens (batch total)' : 'Computed tokens (per-request mean)',
    y: state.lengths === 'sum' ? 'Cached tokens (batch total)' : 'Cached tokens (per-request mean)',
    z: state.batch === 'all' ? 'Batch size (requests)' : metric,
    color: metric
  };
}
function updateAxisLabels() {
  const titles = axisTitles();
  document.getElementById('axis-labels').textContent =
    `X: ${titles.x}\nY: ${titles.y}\nZ: ${titles.z}\nColor: ${titles.color}`;
}
function button(label, key, value) {
  return {label, method: 'skip', args: [{key, value}]};
}
function layout() {
  const menu = (buttons, y, active) => ({type: 'dropdown', buttons, active, x: 0, y, xanchor: 'left', yanchor: 'top'});
  const titles = axisTitles();
  return {title: 'Prefill performance', scene: {
    xaxis: {title: {text: titles.x}},
    yaxis: {title: {text: titles.y}},
    zaxis: {title: {text: titles.z}}},
    coloraxis: {colorscale:'Viridis', colorbar:{title: {text: titles.color}}},
    updatemenus: [
      menu([button('All batches','batch','all'), ...batches.map(b => button('Batch '+b,'batch',b))], 1.2, state.batch === 'all' ? 0 : batches.indexOf(state.batch)+1),
      menu([button('Sum','lengths','sum'),button('Mean','lengths','mean')], 1.1, state.lengths === 'sum' ? 0 : 1),
      menu([button('Case','samples','case'),button('Round','samples','round')], 1.0, state.samples === 'case' ? 0 : 1),
      menu([button('Latency (ms)','metric','latency'),button('Total Input TPM','metric','input_tpm'),button('Total Computed TPM','metric','computed_tpm')], 0.9, ['latency','input_tpm','computed_tpm'].indexOf(state.metric))
    ], margin: {t: 100}};
}
function traces() {
  return batches.map(batch => {
    const samples = chartData.filter(p => p.batch === batch).flatMap(p =>
      state.samples === 'case' ? [{p, sample:p}] : p.rounds.map(sample => ({p,sample})));
    const divisor = p => state.lengths === 'mean' ? p.batch : 1;
    return {type:'scatter3d', mode:'markers', name:'Batch '+batch,
      visible: state.batch === 'all' || state.batch === batch,
      x:samples.map(({p}) => p.compute/divisor(p)), y:samples.map(({p}) => p.cache/divisor(p)),
      z:samples.map(({p,sample}) => state.batch === 'all' ? p.batch : sample[state.metric]), customdata:samples.map(({p,sample}) => ({case:p,sample})),
      marker:{size:5, color:samples.map(({sample}) => sample[state.metric]), coloraxis:'coloraxis'},
      hovertemplate:'Computed: %{x}<br>Cached: %{y}<br>Z: %{z}<br>Value: %{marker.color}<extra></extra>'};
  });
}
const chart = document.getElementById('chart');
Plotly.newPlot('chart', traces(), layout(), {responsive:true});
updateAxisLabels();
chart.on('plotly_buttonclicked', event => {
  const choice = event.button.args[0]; state[choice.key] = choice.value;
  Plotly.react(chart, traces(), layout(), {responsive:true});
  updateAxisLabels();
});
chart.on('plotly_click', event => {
  document.getElementById('details').textContent = JSON.stringify(event.points[0].customdata, null, 2);
});
"""


def _description(row: Mapping[str, Any]) -> str:
    return (
        f"Grid: {row['grid_name']}; case: {row['case_id']}; "
        f"Batch: {row['batch_size']}; source: {row['measurement_contract']}; "
        f"geometry: {json.dumps(row['request_distribution'], sort_keys=True)}"
    )


def _svg(rows, field, title):
    max_x = max(
        (sum(m["compute_length"] for m in r["request_distribution"]) for r in rows),
        default=1,
    )
    max_y = max((r[field] for r in rows), default=1)
    points = []
    for row in rows:
        x = (
            70
            + 670
            * sum(m["compute_length"] for m in row["request_distribution"])
            / max_x
        )
        y = 350 - 290 * row[field] / max_y
        points.append(
            f'<circle cx="{x:.3f}" cy="{y:.3f}" r="5" fill="#2563eb">'
            f"<title>{html.escape(_description(row))}; {html.escape(title)}: {row[field]}</title></circle>"
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 420" role="img">'
        f'<title>{html.escape(title)}</title><rect width="800" height="420" fill="white"/>'
        f'<text x="70" y="25">{html.escape(title)}</text>'
        '<path d="M70 50 V350 H760" fill="none" stroke="black"/>'
        f'<text x="5" y="60">{max_y:.4g}</text><text x="45" y="355">0</text>'
        f'<text x="710" y="375">{max_x}</text>'
        '<text x="250" y="405">Total computed tokens per batch</text>'
        + "".join(points)
        + "</svg>"
    )


def render_charts(report: Mapping[str, Any]) -> dict[str, str]:
    """Return deterministic artifacts; no network, browser, or engine is needed."""
    artifacts = render_legacy_svgs(report)
    rows = [row for row in report["observations"] if row["fit_eligible"]]
    for name, field, title in (
        ("latency", "batch_latency_ms", "Batch latency (ms): median round maximum"),
        (
            "tpm",
            "tokens_per_minute",
            "Computed tokens per minute: measured round wall time",
        ),
    ):
        svg = _svg(rows, field, title)
        artifacts[f"{name}.svg"] = svg
        artifacts[f"{name}.interactive.html"] = render_interactive_chart(
            report, default_metric=name, fallback_svg=artifacts["prefill_3d.svg"] + svg
        )
    return artifacts
