"""Offline restricted FlexLB fitting for validated unified prefill observations."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import pathlib
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from rtp_llm.test.perf_test.prefill.restricted_symbolic import fit_restricted_symbolic
from rtp_llm.test.perf_test.storage import atomic_write_json

CURRENT_CONTRACTS = {"concurrent_batch_v1", "concurrent_grouped_batch_v1"}
ARTIFACTS = (
    "model.json",
    "fit_report.json",
    "predictions.csv",
    "fit_gap.svg",
    "PREFILL_TIME_FORMULA.txt",
)


@dataclass(frozen=True)
class Observation:
    batch_size: int
    input_len: int
    cache_len: int
    target_ms: float
    source: str
    requests: tuple[tuple[int, int], ...]
    geometry_key: str


def normalize_observations(observations):
    rows, excluded = [], []
    for item in observations:
        source = str(item.get("case_id", ""))
        reason = None
        if item.get("measurement_contract") not in CURRENT_CONTRACTS:
            reason = "historical or unsupported measurement contract"
        elif item.get("fit_eligible") is not True:
            reason = "observation is not fit eligible"
        distribution = item.get("request_distribution")
        batch = item.get("batch_size")
        if not reason and (
            type(batch) is not int
            or batch <= 0
            or not isinstance(distribution, list)
            or len(distribution) != batch
        ):
            reason = "invalid request distribution"
        requests = []
        if not reason:
            for member in distribution:
                if not isinstance(member, dict):
                    reason = "invalid request member"
                    break
                i, c = member.get("input_length"), member.get("expected_reuse_length")
                if type(i) is not int or type(c) is not int or not 0 <= c < i:
                    reason = "invalid request lengths"
                    break
                requests.append((i, c))
        target = item.get("batch_latency_ms")
        if not reason and (
            isinstance(target, bool)
            or not isinstance(target, (int, float))
            or not math.isfinite(target)
            or target <= 0
        ):
            reason = "invalid batch latency"
        if reason:
            excluded.append({"case_id": source, "reason": reason})
            continue
        requests = tuple(sorted(requests))
        geometry = json.dumps(requests, separators=(",", ":"))
        rows.append(
            Observation(
                batch,
                max(i for i, c in requests),
                max(c for i, c in requests),
                float(target),
                source,
                requests,
                geometry,
            )
        )
    return rows, excluded


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def error_metrics_with_predictor(
    rows: Sequence[Observation], predictor: Callable[[Observation], float]
) -> dict[str, Any]:
    errors = [abs(predictor(row) - row.target_ms) for row in rows]
    apes = [100.0 * error / row.target_ms for error, row in zip(errors, rows)]
    return {
        "n": len(rows),
        "mae_ms": statistics.mean(errors) if errors else None,
        "mape_pct": statistics.mean(apes) if apes else None,
        "p50_ape_pct": _quantile(apes, 0.50) if apes else None,
        "p95_ape_pct": _quantile(apes, 0.95) if apes else None,
        "max_ape_pct": max(apes) if apes else None,
        "p95_abs_ms": _quantile(errors, 0.95) if errors else None,
        "max_abs_ms": max(errors) if errors else None,
    }


def split_rows(rows: Sequence[Observation]) -> dict[str, list[Observation]]:
    """Keep duplicate/permuted request distributions in the same held-out split."""
    groups: dict[str, list[Observation]] = {}
    for row in rows:
        groups.setdefault(row.geometry_key or str(row.input_len), []).append(row)
    result: dict[str, list[Observation]] = {"train": [], "validation": [], "test": []}
    for seq_len, group in sorted(groups.items()):
        bucket = (
            int.from_bytes(hashlib.sha256(str(seq_len).encode()).digest()[:8], "big")
            / 2**64
        )
        result[
            "train" if bucket < 0.70 else "validation" if bucket < 0.85 else "test"
        ].extend(group)
    return result


def write_fit_gap_svg(
    predictions: Sequence[dict[str, Any]],
    path: pathlib.Path,
    model_label: str = "Model",
) -> None:
    """Write an all-point measured-vs-predicted and absolute-error chart."""
    if not predictions:
        return
    width, height = 1800, 860
    panel_width, panel_height = 670.0, 590.0
    left_x, right_x, top = 120.0, 1010.0, 150.0
    targets = [float(row["target_ms"]) for row in predictions]
    estimates = [float(row["predicted_ms"]) for row in predictions]
    errors = [abs(estimate - target) for estimate, target in zip(estimates, targets)]
    latency_max = max(max(targets), max(estimates), 1.0)
    error_max = max(max(errors), 1.0)
    p95_abs = _quantile(errors, 0.95)

    def point(panel_x: float, x: float, y: float, ymax: float) -> tuple[float, float]:
        return (
            panel_x + panel_width * x / latency_max,
            top + panel_height * (1.0 - y / ymax),
        )

    def line(x1: float, y1: float, x2: float, y2: float, **attrs: Any) -> str:
        values = " ".join(
            f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items()
        )
        return (
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
            f'y2="{y2:.1f}" {values}/>'
        )

    out = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#fff"/>',
        (
            '<style>text{font-family:Arial,"Noto Sans CJK SC","Microsoft YaHei",'
            "sans-serif;fill:#172033}.title{font-size:32px;font-weight:700}"
            ".sub{font-size:17px;fill:#475569}.panel{font-size:22px;font-weight:700}"
            ".axis{font-size:17px;fill:#334155}.tick{font-size:14px;fill:#64748b}"
            ".legend{font-size:15px;fill:#334155}</style>"
        ),
        (
            '<text x="900" y="48" text-anchor="middle" class="title">'
            f"{html.escape(model_label)}：实测 TTFT 与拟合误差</text>"
        ),
        (
            f'<text x="900" y="82" text-anchor="middle" class="sub">'
            f"{len(predictions):,} 个有效观测；"
            "每个点取每轮最大服务端首 token 延迟的跨轮中位数</text>"
        ),
        (
            f'<text x="{left_x + panel_width / 2:.1f}" y="120" '
            'text-anchor="middle" class="panel">实测值 vs 拟合值</text>'
        ),
        (
            f'<text x="{right_x + panel_width / 2:.1f}" y="120" '
            'text-anchor="middle" class="panel">绝对误差 vs 实测值</text>'
        ),
    ]
    for panel_x in (left_x, right_x):
        out.append(
            f'<rect x="{panel_x}" y="{top}" width="{panel_width}" '
            f'height="{panel_height}" fill="#f8fafc" stroke="#cbd5e1"/>'
        )
        for index in range(6):
            ratio = index / 5
            x = panel_x + ratio * panel_width
            y = top + (1.0 - ratio) * panel_height
            out.append(
                line(
                    x,
                    top,
                    x,
                    top + panel_height,
                    stroke="#e2e8f0",
                    stroke_width="1",
                )
            )
            out.append(
                line(
                    panel_x,
                    y,
                    panel_x + panel_width,
                    y,
                    stroke="#e2e8f0",
                    stroke_width="1",
                )
            )
            out.append(
                f'<text x="{x:.1f}" y="{top + panel_height + 27:.1f}" '
                f'text-anchor="middle" class="tick">{latency_max * ratio:.0f}</text>'
            )
    for index in range(6):
        ratio = index / 5
        y = top + (1.0 - ratio) * panel_height
        out.append(
            f'<text x="{left_x - 14:.1f}" y="{y + 5:.1f}" '
            f'text-anchor="end" class="tick">{latency_max * ratio:.0f}</text>'
        )
        out.append(
            f'<text x="{right_x - 14:.1f}" y="{y + 5:.1f}" '
            f'text-anchor="end" class="tick">{error_max * ratio:.0f}</text>'
        )

    out.append(
        line(
            left_x,
            top + panel_height,
            left_x + panel_width,
            top,
            stroke="#16a34a",
            stroke_width="2.5",
            stroke_dasharray="9 7",
        )
    )
    for row, target, estimate, error in zip(predictions, targets, estimates, errors):
        colour = "#2563eb" if int(row["cache_len"]) == 0 else "#d97706"
        x, y = point(left_x, target, estimate, latency_max)
        out.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.0" '
            f'fill="{colour}" fill-opacity=".42"/>'
        )
        x, y = point(right_x, target, error, error_max)
        out.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.0" '
            f'fill="{colour}" fill-opacity=".42"/>'
        )

    p95_y = top + panel_height * (1.0 - p95_abs / error_max)
    out.append(
        line(
            right_x,
            p95_y,
            right_x + panel_width,
            p95_y,
            stroke="#dc2626",
            stroke_width="2.2",
            stroke_dasharray="8 6",
        )
    )
    out.append(
        f'<text x="{right_x + panel_width - 8:.1f}" y="{p95_y - 8:.1f}" '
        f'text-anchor="end" class="legend">p95 absolute error = {p95_abs:.1f} ms</text>'
    )
    out.extend(
        [
            (
                f'<text x="{left_x + panel_width / 2:.1f}" '
                f'y="{top + panel_height + 64:.1f}" text-anchor="middle" '
                'class="axis">实测 TTFT（ms）</text>'
            ),
            (
                f'<text x="{right_x + panel_width / 2:.1f}" '
                f'y="{top + panel_height + 64:.1f}" text-anchor="middle" '
                'class="axis">实测 TTFT（ms）</text>'
            ),
            (
                f'<text x="35" y="{top + panel_height / 2:.1f}" '
                'text-anchor="middle" '
                f'transform="rotate(-90 35 {top + panel_height / 2:.1f})" '
                'class="axis">拟合 TTFT（ms）</text>'
            ),
            (
                f'<text x="925" y="{top + panel_height / 2:.1f}" '
                'text-anchor="middle" '
                f'transform="rotate(-90 925 {top + panel_height / 2:.1f})" '
                'class="axis">绝对误差（ms）</text>'
            ),
            (
                '<circle cx="690" cy="817" r="5" fill="#2563eb"/>'
                '<text x="704" y="822" class="legend">cache miss</text>'
            ),
            (
                '<circle cx="825" cy="817" r="5" fill="#d97706"/>'
                '<text x="839" y="822" class="legend">cache hit</text>'
            ),
            (
                '<line x1="980" y1="817" x2="1020" y2="817" '
                'stroke="#16a34a" stroke-width="2.5" '
                'stroke-dasharray="9 7"/>'
                '<text x="1030" y="822" class="legend">理想线 y=x</text>'
            ),
            "</svg>",
        ]
    )
    path.write_text("".join(out), encoding="utf-8")


def quality_passes(metrics):
    def passes(group):
        return (
            group["n"] > 0
            and group["mape_pct"] <= 5.0
            and group["p95_ape_pct"] <= 10.0
            and group["max_ape_pct"] <= 40.0
        )

    return passes(metrics["test"]) and all(
        all(group[name]["n"] > 0 for name in ("train", "validation", "test"))
        and passes(group["test"])
        for group in metrics["by_batch"].values()
    )


def fit_prefill_formula(observations, output_dir, *, complete):
    """Write a candidate and a separate acceptance decision; never fit legacy data.

    Expected admission failures are persisted rejections. Solver, filesystem and
    programming errors propagate to the caller, with previous artifacts removed.
    """
    output = pathlib.Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        (output / name).unlink(missing_ok=True)
    rows, excluded = normalize_observations(observations)
    splits = split_rows(rows)
    unique = len({row.geometry_key for row in rows})
    report = dict(
        schema_version=1,
        production_acceptance=False,
        candidate_generated=False,
        reason="insufficient_valid_geometries",
        complete=complete,
        unique_geometry_count=unique,
        excluded=excluded,
        measurement_contract="median_round_max_server_first_token_latency_ms",
        split={"mode": "request-distribution-hash-70-15-15"},
        split_counts={name: len(group) for name, group in splits.items()},
        thresholds={
            "min_valid_geometries": 30,
            "max_mape_pct": 5.0,
            "max_p95_ape_pct": 10.0,
            "max_ape_pct": 40.0,
        },
    )
    if unique < 30 or any(not group for group in splits.values()):
        if unique >= 30:
            report["reason"] = "insufficient_geometry_splits"
        atomic_write_json(output / "fit_report.json", report)
        return report
    model = fit_restricted_symbolic(
        splits["train"],
        splits["validation"],
        [*splits["train"], *splits["validation"]],
        token_unit=1024,
    )
    metrics = {
        name: error_metrics_with_predictor(group, model.predict)
        for name, group in {**splits, "all": rows}.items()
    }
    metrics["by_batch"] = {
        str(batch): {
            name: error_metrics_with_predictor(
                [r for r in group if r.batch_size == batch], model.predict
            )
            for name, group in {**splits, "all": rows}.items()
        }
        for batch in sorted({r.batch_size for r in rows})
    }
    quality = quality_passes(metrics)
    accepted = bool(complete and not excluded and quality)
    report.update(
        candidate_generated=True,
        production_acceptance=accepted,
        reason=(
            "accepted"
            if accepted
            else (
                "partial_run"
                if not complete
                else "excluded_observations" if excluded else "quality_gate_failed"
            )
        ),
        formula=model.formula,
        backend=model.backend,
        metrics=metrics,
        symbolic_search=model.search_report,
        production_note="Candidate existence does not imply acceptance. Validate deployment against the target FlexLB parser/runtime.",
    )
    assignment = {r.geometry_key: name for name, group in splits.items() for r in group}
    predictions = []
    for row in rows:
        predicted = model.predict(row)
        predictions.append(
            dict(
                source=row.source,
                batch_size=row.batch_size,
                target_ms=row.target_ms,
                predicted_ms=predicted,
                cache_len=row.cache_len,
                abs_error_ms=abs(predicted - row.target_ms),
                ape_pct=100 * abs(predicted - row.target_ms) / row.target_ms,
                split=assignment[row.geometry_key],
                geometry_key=row.geometry_key,
                request_distribution=json.dumps(row.requests),
            )
        )
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    write_fit_gap_svg(predictions, output / "fit_gap.svg")
    atomic_write_json(
        output / "model.json",
        dict(
            schema_version=2,
            formula=model.formula,
            token_unit=1024,
            library=model.search_report["library_version"],
            production_acceptance=accepted,
            reason=report["reason"],
            measurement_contract=report["measurement_contract"],
            terms=[
                dict(expression=t.expression, coefficient=c)
                for t, c in zip(model.terms, model.coefficients)
            ],
        ),
    )
    (output / "PREFILL_TIME_FORMULA.txt").write_text(
        "PREFILL_TIME_FORMULA=" + model.formula + "\n", encoding="utf-8"
    )
    atomic_write_json(output / "fit_report.json", report)
    return report
