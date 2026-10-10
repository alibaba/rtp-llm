#!/usr/bin/env python3
"""Generate an interactive, rotatable 3D prefill chart from a result JSON.

The chart mirrors the static SVG's convention:
  X = uncached compute tokens = input_len - observed_cache_len
  Y = observed cached tokens
  Z = server first_token_cost_time (ms, including engine wait)

Every valid geometry is shown as a dot.  Coloured lines are representative
fixed-cache (warm, solid) and fixed-compute (cool, dashed) trend guides.
Duplicate geometries are collapsed by median RT.  Pass ``--all-runs`` to plot
every successful run as its own dot instead of one median dot per case.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
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
    resolve_label,
    resolve_title,
)
from rtp_llm.test.perf_test.cache_grid.config.topology import detect_cards
from rtp_llm.test.perf_test.cache_grid.runner.result_schema import measurement_rows


def load_rows(
    input_path: Path, batch_size: int, all_runs: bool = False
) -> list[dict[str, float]]:
    return measurement_rows(
        json.loads(input_path.read_text(encoding="utf-8")),
        batch_size,
        strict=False,
        all_runs=all_runs,
    )


def representative_levels(rows: list[dict[str, float]], key: str) -> list[float]:
    """Choose guide planes near 0%, 33%, 67%, and 100% of an axis."""
    values = sorted({row[key] for row in rows})
    maximum = values[-1]
    chosen: list[float] = []
    for target in (0.0, 0.33 * maximum, 0.67 * maximum, maximum):
        level = min(values, key=lambda value: abs(value - target))
        if level not in chosen:
            chosen.append(level)
    return chosen


def representative_slice(
    rows: list[dict[str, float]],
    fixed_key: str,
    fixed_value: float,
    varying_key: str,
) -> list[dict[str, float]]:
    """Build the same nearby, median-binned guide used by the static SVG."""
    axis_maximum = max(
        max(row["cache_len"] for row in rows),
        max(row["compute_len"] for row in rows),
    )
    tolerance = max(4_096.0, axis_maximum * 0.025)
    subset = [row for row in rows if abs(row[fixed_key] - fixed_value) <= tolerance]
    if len(subset) < 8:
        subset = sorted(rows, key=lambda row: abs(row[fixed_key] - fixed_value))[
            : max(8, min(24, len(rows)))
        ]

    ordered = sorted(subset, key=lambda row: row[varying_key])
    bins = min(12, len(ordered))
    points: list[dict[str, float]] = []
    for index in range(bins):
        lo = (index * len(ordered)) // bins
        hi = ((index + 1) * len(ordered)) // bins
        group = ordered[lo : max(hi, lo + 1)]
        point = dict(group[len(group) // 2])
        point[fixed_key] = fixed_value
        point[varying_key] = median(row[varying_key] for row in group)
        point["prefill_rt"] = median(row["prefill_rt"] for row in group)
        points.append(point)
    return points


def line_trace(
    rows: list[dict[str, float]],
    fixed_key: str,
    fixed_value: float,
    varying_key: str,
    color: str,
    dash: str,
    name: str,
    z_key: str = "prefill_rt",
) -> Any:
    import plotly.graph_objects as go

    subset = representative_slice(rows, fixed_key, fixed_value, varying_key)
    return go.Scatter3d(
        x=[row["compute_len"] for row in subset],
        y=[row["cache_len"] for row in subset],
        z=[row[z_key] for row in subset],
        mode="lines",
        line={"color": color, "width": 5, "dash": dash},
        name=name,
        hoverinfo="skip",
        showlegend=True,
    )


def fmt_tokens(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


Z_METRICS = {
    "rt": {
        "key": "prefill_rt",
        "label": "Server first-token latency",
        "unit": "ms",
        "derive": lambda row: row["prefill_rt"],
    },
    "tpm-compute": {
        "key": "tpm_compute",
        "label": "Compute TPM",
        "unit": "tokens/min",
        "derive": lambda row: (
            row["compute_len"] * 60000.0 / row["prefill_rt"]
            if row["prefill_rt"] > 0
            else None
        ),
    },
    "tpm-effective": {
        "key": "tpm_effective",
        "label": "Effective TPM",
        "unit": "tokens/min",
        "derive": lambda row: (
            row["input_len"] * 60000.0 / row["prefill_rt"]
            if row["prefill_rt"] > 0
            else None
        ),
    },
}


def apply_z_metric(
    rows: list[dict[str, float]],
    z_metric: str,
    cards: float = 1.0,
    rt_cards: float | None = None,
) -> list[dict[str, float]]:
    spec = Z_METRICS[z_metric]
    # rt latency is a whole-system measurement; only throughput is per card.
    divisor = rt_cards if z_metric == "rt" and rt_cards is not None else cards
    enriched: list[dict[str, float]] = []
    for row in rows:
        value = spec["derive"](row)
        if value is None or not math.isfinite(value):
            continue
        if divisor and divisor != 1.0:
            value = value / divisor
        enriched.append({**row, spec["key"]: value})
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an interactive 3D prefill chart from a result JSON file."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to cache_grid_results.json (server first_token_cost_time)",
    )
    parser.add_argument(
        "--output", type=Path, help="Output HTML path; defaults beside the input file"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size to display"
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help=(
            "Plot every successful run as its own point (raw scatter) instead "
            "of one median point per case"
        ),
    )
    parser.add_argument(
        "--z-metric",
        default="rt",
        choices=sorted(Z_METRICS),
        help=(
            "Z axis metric: rt = prefill latency (ms); tpm-compute = uncached "
            "compute tokens per minute; tpm-effective = input tokens (cache "
            "hit + compute) per minute"
        ),
    )
    parser.add_argument(
        "--cards",
        type=float,
        default=None,
        help=(
            "Accelerator count used to normalize TPM to per-card throughput. "
            "Defaults to profile engine.world_size (or TP x DP x PP); use 1 "
            "to keep the whole-system TPM."
        ),
    )
    parser.add_argument("--title", default=None, help="Chart title")
    parser.add_argument(
        "--model-label", default=None, help="Model label used to build the title."
    )
    parser.add_argument(
        "--profile", default=None, help="JSON profile for parameter defaults."
    )
    parser.add_argument(
        "--log-rt",
        action="store_true",
        help="Use log scale for the server first-token latency (Z) axis",
    )
    parser.add_argument(
        "--stretch-rt",
        type=float,
        default=None,
        help="Visual stretch factor for the server first-token latency axis (e.g. 5 makes RT changes 5x taller)",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=2.0,
        help="Measurement marker size in pixels (default: 2)",
    )
    args = parser.parse_args()

    if args.marker_size <= 0:
        parser.error("--marker-size must be greater than 0")

    if args.stretch_rt is None:
        args.stretch_rt = 1.0

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

    rows = load_rows(args.input, args.batch_size, all_runs=args.all_runs)
    if not rows:
        parser.error(f"no usable rows for batch_size={args.batch_size} in {args.input}")

    z_spec = Z_METRICS[args.z_metric]
    z_key = z_spec["key"]
    cards = 1.0
    if args.z_metric != "rt":
        try:
            cards = (
                args.cards
                if args.cards is not None
                else (detect_cards(args.input, profile) or 1.0)
            )
        except ValueError as error:
            parser.error(str(error))
        if not math.isfinite(cards) or cards <= 0 or not float(cards).is_integer():
            parser.error("--cards must be a positive integer")
    rows = apply_z_metric(rows, args.z_metric, cards=cards)
    if not rows:
        parser.error(f"no usable rows for z_metric={args.z_metric} in {args.input}")

    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as error:
        raise SystemExit(
            "Plotly is required. Install it with: python3 -m pip install --user plotly"
        ) from error

    suffix = ".interactive.html"
    if args.all_runs:
        suffix = ".all_runs.interactive.html"
    if args.z_metric != "rt":
        stem, ext = suffix.split(".", 1)
        per_card = f"_per_card{cards:g}" if cards != 1.0 else ""
        suffix = f"{stem}.{args.z_metric}{per_card}.{ext}"
    output = args.output or args.input.with_name(f"{args.input.stem}{suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)

    per_card_note = (
        f", per card (÷{cards:g} cards)"
        if args.z_metric != "rt" and cards != 1.0
        else ""
    )
    default_title = (
        f"{model_label} Prefill — interactive 3D view (batch size {args.batch_size}"
        f"{', all runs' if args.all_runs else ''}, Z = {z_spec['label']}"
        f"{per_card_note})"
    )
    title = resolve_title(profile, args.title, default_title)

    warm_palette = (
        "#b42318",
        "#d97706",
        "#15803d",
        "#0369a1",
        "#6d28d9",
        "#be185d",
        "#475569",
        "#0f766e",
    )
    cool_palette = (
        "#7c3aed",
        "#c026d3",
        "#0369a1",
        "#0f766e",
        "#b42318",
        "#d97706",
        "#15803d",
        "#475569",
    )

    cache_levels = representative_levels(rows, "cache_len")
    compute_levels = representative_levels(rows, "compute_len")

    traces: list[Any] = []

    for index, level in enumerate(cache_levels):
        traces.append(
            line_trace(
                rows,
                fixed_key="cache_len",
                fixed_value=level,
                varying_key="compute_len",
                color=warm_palette[index % len(warm_palette)],
                dash="solid",
                name=f"cache = {fmt_tokens(level)}",
                z_key=z_key,
            )
        )

    for index, level in enumerate(compute_levels):
        traces.append(
            line_trace(
                rows,
                fixed_key="compute_len",
                fixed_value=level,
                varying_key="cache_len",
                color=cool_palette[index % len(cool_palette)],
                dash="dash",
                name=f"compute = {fmt_tokens(level)}",
                z_key=z_key,
            )
        )

    show_run_index = args.all_runs and all("run_index" in row for row in rows)
    customdata = [
        [
            row["input_len"],
            row["cache_len"],
            row["compute_len"],
            row["run_index"] if show_run_index else None,
            row["prefill_rt"],
        ]
        for row in rows
    ]
    z_format = ":,.0f" if args.z_metric != "rt" else ":.2f"
    hovertemplate = (
        "Compute tokens: %{x:,.0f}<br>"
        "Cached tokens: %{y:,.0f}<br>"
        f"{z_spec['label']}: %{{z{z_format}}}{'' if args.z_metric == 'rt' else ' tokens/min'}<br>"
        "Server first-token latency: %{customdata[4]:,.2f} ms<br>"
        "Input length: %{customdata[0]:,.0f}<br>"
        + (
            "Run: %{customdata[3]}<extra></extra>"
            if show_run_index
            else "<extra></extra>"
        )
    )
    colorbar_title = (
        f"{z_spec['label']} ({z_spec['unit']}, darker = slower)"
        if args.z_metric == "rt"
        else (
            f"{z_spec['label']} per card ({z_spec['unit']}, darker = lower throughput)"
            if cards != 1.0
            else f"{z_spec['label']} ({z_spec['unit']}, darker = lower throughput)"
        )
    )

    traces.append(
        go.Scatter3d(
            x=[row["compute_len"] for row in rows],
            y=[row["cache_len"] for row in rows],
            z=[row[z_key] for row in rows],
            mode="markers",
            marker={
                "size": args.marker_size,
                "color": [row[z_key] for row in rows],
                "colorscale": "Viridis",
                # Higher latency (lower throughput) is darker; keep raw values.
                "reversescale": True,
                "colorbar": {"title": {"text": colorbar_title}},
                "opacity": 0.8,
            },
            customdata=customdata,
            hovertemplate=hovertemplate,
            name="measurements",
            showlegend=False,
        )
    )

    figure = go.Figure(data=traces)
    figure.update_layout(
        title=title,
        scene={
            "xaxis_title": "Compute tokens (X)",
            "yaxis_title": "Observed cached tokens (Y)",
            "zaxis_title": (
                f"{z_spec['label']} per card (Z, {z_spec['unit']})"
                if args.z_metric != "rt" and cards != 1.0
                else f"{z_spec['label']} (Z, {z_spec['unit']})"
            ),
            "zaxis_type": "log" if args.log_rt else "linear",
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": 1, "z": args.stretch_rt},
        },
        legend={
            "title": {"text": "Trend guides"},
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "left",
            "x": 0.01,
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
    )
    figure.write_html(output, include_plotlyjs=True, full_html=True)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "rows": len(rows),
                "aggregation": "all_runs" if args.all_runs else "median",
                "z_metric": args.z_metric,
                "cards": cards if args.z_metric != "rt" else 1.0,
                "output": str(output),
                "title": title,
                "profile": profile,
                "profile_sha256": profile_sha256,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
