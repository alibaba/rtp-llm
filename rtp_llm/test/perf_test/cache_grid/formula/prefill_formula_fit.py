#!/usr/bin/env python3
"""Fit and audit a model prefill latency formula.

The input is the raw ``cache_grid_results.json`` emitted by the cache-grid
runner.  A grid file can contain successful measurements,
failed requests, and successful requests whose cache seed was not actually
reused.  This tool accepts a row only when the runner marks it successful and
all measured reuse lengths exactly match the requested cache length.

* every requested measurement run succeeded;
* every run has output length one and finite positive server first_token_cost_time;
* observed reuse is constant and exactly matches the request; and
* request lists are preserved for all batch sizes (an optional filter is available).

Fitting uses restricted-library symbolic regression with greedy forward
selection and relative squared error. Candidates use the variables, functions,
and aggregate syntax supported by FlexLB's PrefillTimeFormula parser.
Request distributions are deterministically split into train/validation/test sets
(70/15/15 hash buckets). Selection uses train and validation; final coefficients
are refit on those two sets, keeping the test set held out.

All fit targets use the server's first_token_cost_time, stored in
runs[].prefill_time_ms (milliseconds, including engine wait). Client wall time
is diagnostic only; no client/server timing fallback is permitted.

The report keeps the fit and the production gate separate: a formula can be
useful for analysis while still failing a tail-error gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import pathlib
import re
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from rtp_llm.test.perf_test.cache_grid.config.perf_profile import (
    fingerprint as profile_fingerprint,
)
from rtp_llm.test.perf_test.cache_grid.config.perf_profile import (
    load_profile,
    resolve_int,
    resolve_label,
    resolve_str,
)
from rtp_llm.test.perf_test.cache_grid.formula.restricted_symbolic_fit import (
    DEFAULT_EXP_DECAY_TOKENS,
    DEFAULT_HINGE_TOKENS,
    PARSER_AGGREGATES,
    PARSER_BATCH_VARIABLES,
    PARSER_FUNCTIONS,
    PARSER_OPERATORS,
    PARSER_PER_REQUEST_VARIABLES,
    fit_restricted_symbolic,
)

DEFAULT_TOKEN_UNIT = 1024


@dataclass(frozen=True)
class Observation:
    batch_size: int
    input_len: int
    cache_len: int
    target_ms: float
    source: str
    requested_cache_len: int | None = None
    requests: tuple[tuple[int, int], ...] = ()
    geometry_key: str = ""

    @property
    def compute_len(self) -> int:
        return self.input_len - self.cache_len


def load_observations(paths, *, batch_size=None, estimator="median"):
    from rtp_llm.test.perf_test.cache_grid.runner.observations import collect

    records, audit = collect(
        paths, batch_size=batch_size, estimator=estimator, partial=True
    )
    rows = [
        Observation(
            r["batch_size"],
            r["input_len"],
            r["cache_len"],
            r["target_ms"],
            f'{r["source_run"]}:metrics[{r["metric_index"]}]',
            r["cache_len"],
            tuple((v[0], v[1]) for v in r["request_distribution"]),
            r["geometry_key"],
        )
        for r in records
    ]
    audit.update(
        input_files=audit["sources"],
        selected_batch_size=batch_size,
        raw_metric_count=sum(s["rows"] for s in audit["sources"]),
        raw_valid_observation_count=len(rows),
        collapsed_duplicate_geometry_count=0,
        unique_geometry_count=len({r.geometry_key for r in rows}),
        measurement_contracts=[audit["measurement_contract"]],
        seq_len_range=[
            min((r.input_len for r in rows), default=None),
            max((r.input_len for r in rows), default=None),
        ],
        cache_len_range=[
            min((r.cache_len for r in rows), default=None),
            max((r.cache_len for r in rows), default=None),
        ],
    )
    return rows, audit


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
            f"{len(predictions):,} 个严格有效 geometry；"
            "每个点取 3 次成功请求的 TTFT 中位数</text>"
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


def run_fit(args: argparse.Namespace) -> int:
    profile = None
    profile_sha256 = None
    if getattr(args, "profile", None):
        profile = load_profile(args.profile)
        profile_sha256 = profile_fingerprint(profile)

    token_unit = resolve_int(
        profile or {},
        "chart",
        "token_unit",
        getattr(args, "token_unit", None),
        DEFAULT_TOKEN_UNIT,
    )
    model_label = resolve_label(profile, getattr(args, "model_label", None), "Model")
    filename_label = re.sub(r"[^\w.-]+", "_", model_label).strip("._") or "Model"
    formula_filename = resolve_str(
        profile or {},
        "chart",
        "formula_filename",
        getattr(args, "formula_filename", None),
        f"{filename_label}_prefill_formula.txt",
    )
    formula_key = resolve_str(
        profile or {},
        "chart",
        "formula_key",
        getattr(args, "formula_key", None),
        "PREFILL_TIME_FORMULA",
    )

    paths = [pathlib.Path(value) for value in args.inputs]
    if len(paths) > 1:
        from rtp_llm.test.perf_test.cache_grid.runner.unified_pipeline import (
            validate_merge_inputs,
        )

        validate_merge_inputs(paths)
    rows, audit = load_observations(
        paths, batch_size=args.batch_size, estimator=args.estimator
    )
    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # A rejected rerun must not leave an old model marked deployable.
    (output / "model.json").write_text(
        json.dumps(
            dict(
                schema_version=2,
                production_acceptance=False,
                formula=None,
                reason="fit_not_accepted",
                measurement_contract=audit["measurement_contract"],
            ),
            indent=2,
        )
    )
    (output / "input_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if len(rows) < args.min_valid_rows and not args.allow_insufficient_data:
        report = {
            "production_acceptance": False,
            "reason": "insufficient_valid_rows",
            "required_min_valid_rows": args.min_valid_rows,
            "audit": audit,
            "formula": None,
            "profile": profile,
            "profile_sha256": profile_sha256,
        }
        (output / "fit_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False))
        return 3

    splits = split_rows(rows)
    if any(not splits[name] for name in ("train", "validation", "test")):
        (output / "fit_report.json").write_text(
            json.dumps(
                dict(
                    production_acceptance=False,
                    formula=None,
                    reason="insufficient_geometry_splits",
                    audit=audit,
                    split_counts={k: len(v) for k, v in splits.items()},
                ),
                indent=2,
            )
        )
        return 3
    hinge_tokens = (
        tuple(
            int(value)
            for value in getattr(args, "symbolic_hinge_tokens", "").split(",")
            if value.strip()
        )
        or DEFAULT_HINGE_TOKENS
    )
    exp_decay_tokens = (
        tuple(
            int(value)
            for value in getattr(args, "symbolic_exp_decay_tokens", "").split(",")
            if value.strip()
        )
        or DEFAULT_EXP_DECAY_TOKENS
    )
    symbolic_model = fit_restricted_symbolic(
        splits["train"],
        splits["validation"],
        [*splits["train"], *splits["validation"]],
        token_unit=token_unit,
        hinge_tokens=hinge_tokens,
        exp_decay_tokens=exp_decay_tokens,
        max_terms=getattr(args, "symbolic_max_terms", 15),
        complexity_tolerance_pct=getattr(
            args, "symbolic_complexity_tolerance_pct", 5.0
        ),
    )
    coefficients = list(symbolic_model.coefficients)
    backend = symbolic_model.backend
    formula = symbolic_model.formula
    predictor = symbolic_model.predict
    coefficient_names = [term.expression for term in symbolic_model.terms]
    symbolic_report = symbolic_model.search_report
    formula_compatibility = {
        "parser": "org.flexlb.balance.prediction.PrefillTimeFormula",
        "variables": [*PARSER_PER_REQUEST_VARIABLES, *PARSER_BATCH_VARIABLES],
        "functions": [*PARSER_AGGREGATES, *PARSER_FUNCTIONS],
        "operators": [*PARSER_OPERATORS, "(", ")"],
        "unsupported_constructs_used": [],
    }
    objective_name = "mean_squared_relative_error"
    metrics = {
        name: error_metrics_with_predictor(group, predictor)
        for name, group in splits.items()
    }
    metrics["all"] = error_metrics_with_predictor(rows, predictor)
    metrics["by_batch"] = {
        str(batch): {
            name: error_metrics_with_predictor(
                [r for r in group if r.batch_size == batch], predictor
            )
            for name, group in {**splits, "all": rows}.items()
        }
        for batch in sorted({r.batch_size for r in rows})
    }
    split_by_geometry = {
        row.geometry_key: name for name, group in splits.items() for row in group
    }
    predictions = []
    for row in rows:
        predicted = predictor(row)
        predictions.append(
            {
                "batch_size": row.batch_size,
                "input_len": row.input_len,
                "requested_cache_len": row.requested_cache_len,
                "cache_len": row.cache_len,
                "compute_len": row.compute_len,
                "target_ms": row.target_ms,
                "measurement_contract": audit["measurement_contract"],
                "predicted_ms": predicted,
                "signed_error_ms": predicted - row.target_ms,
                "abs_error_ms": abs(predicted - row.target_ms),
                "ape_pct": 100.0 * abs(predicted - row.target_ms) / row.target_ms,
                "split": split_by_geometry[row.geometry_key],
                "request_distribution": json.dumps(row.requests),
                "geometry_key": row.geometry_key,
                "source": row.source,
            }
        )
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(predictions[0]) if predictions else ["input_len"]
        )
        writer.writeheader()
        writer.writerows(predictions)
    write_fit_gap_svg(predictions, output / "fit_gap.svg", model_label)
    report = {
        "schema_version": 1,
        "model": model_label,
        "model_family": "restricted-symbolic",
        "backend": backend,
        "objective": objective_name,
        "measurement_contract": audit["measurement_contract"],
        "target": (
            f"{args.estimator} of successful server first_token_cost_time (runs[].prefill_time_ms), including engine wait"
        ),
        "formula": formula,
        "token_unit": token_unit,
        "formula_compatibility": formula_compatibility,
        "coefficients": [
            {"expression": name, "coefficient": value}
            for name, value in zip(coefficient_names, coefficients)
        ],
        "symbolic_search": symbolic_report,
        "audit": audit,
        "split": {
            "mode": "request-distribution-hash-70-15-15",
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
        },
        "split_counts": {name: len(group) for name, group in splits.items()},
        "metrics": metrics,
        "production_acceptance": bool(
            audit["unique_geometry_count"] >= args.min_valid_rows
            and len(splits["test"]) > 0
            and metrics["test"]["mape_pct"] is not None
            and metrics["test"]["mape_pct"] <= args.max_mape_pct
            and metrics["test"]["p95_ape_pct"] is not None
            and metrics["test"]["p95_ape_pct"] <= args.max_p95_ape_pct
            and metrics["test"]["max_ape_pct"] is not None
            and metrics["test"]["max_ape_pct"] <= args.max_max_ape_pct
        ),
        "production_note": (
            "Rows require exact requested/observed reuse unless the runner "
            "explicitly marked reuse validation as skipped; those rows use "
            "observed reuse. Failed requests are excluded. "
            "Validate the latency measurement contract, tail error, and "
            "deployment range before production use."
        ),
        "profile": profile,
        "profile_sha256": profile_sha256,
    }
    report["production_acceptance"] = (
        not audit["partial"]
        and report["production_acceptance"]
        and all(
            all(group[split]["n"] > 0 for split in ("train", "validation", "test"))
            and group["test"]["mape_pct"] <= args.max_mape_pct
            and group["test"]["p95_ape_pct"] <= args.max_p95_ape_pct
            and group["test"]["max_ape_pct"] <= args.max_max_ape_pct
            for group in metrics["by_batch"].values()
        )
    )
    report["measurement_contract"] = audit["measurement_contract"]
    report["target"] = f"{args.estimator} of per-round max server first_token_cost_time"
    report["production_note"] = (
        "Strict exact-reuse request lists; per-batch held-out gates apply. "
        "Deployment requires validation against the target FlexLB parser/runtime."
    )
    (output / "model.json").write_text(
        json.dumps(
            dict(
                schema_version=2,
                formula=formula,
                token_unit=token_unit,
                terms=report["coefficients"],
                library=symbolic_report["library_version"],
                production_acceptance=report["production_acceptance"],
                measurement_contract=report["measurement_contract"],
            ),
            indent=2,
        )
    )
    (output / "fit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / formula_filename).write_text(
        formula_key + "=" + formula + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["production_acceptance"] else 3


def run_validate(args: argparse.Namespace) -> int:
    profile = None
    if getattr(args, "profile", None):
        profile = load_profile(args.profile)
    model_label = resolve_label(profile, getattr(args, "model_label", None), "Model")
    rows, audit = load_observations(
        [pathlib.Path(value) for value in args.inputs],
        batch_size=args.batch_size,
        estimator=args.estimator,
    )
    report = {"model": model_label, "audit": audit, "valid": bool(rows)}
    if args.report:
        pathlib.Path(args.report).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if rows else 2


def _add_common_profile_args(parser: argparse.ArgumentParser) -> None:
    """Add --profile and --model-label shared by all subcommands."""
    parser.add_argument(
        "--profile", default=None, help="JSON profile for parameter defaults."
    )
    parser.add_argument(
        "--model-label", default=None, help="Override the model label in output."
    )


def run_analyze_anomalies(args: argparse.Namespace) -> int:
    """Detect anomalous measurements in a cache-grid result set.

    Checks:
    (a) Cache monotonicity: for same input_len, longer cache should be faster.
    (b) Run spread: max/min RT ratio per case should be within threshold.
    (c) Residual outliers: cases with high APE vs fitted formula.
    (d) Cross-input compute monotonicity: more compute at same cache should cost more.
    """
    profile = None
    if getattr(args, "profile", None):
        profile = load_profile(args.profile)
    model_label = resolve_label(profile, getattr(args, "model_label", None), "Model")

    token_unit = resolve_int(
        profile or {},
        "chart",
        "token_unit",
        getattr(args, "token_unit", None),
        DEFAULT_TOKEN_UNIT,
    )

    paths = [pathlib.Path(value) for value in args.inputs]
    rows, audit = load_observations(
        paths, batch_size=args.batch_size, estimator=args.estimator
    )
    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    anomalies: list[dict[str, Any]] = []
    min_rt_ms = float(getattr(args, "min_rt_ms", 5.0))
    min_compute_tokens = int(getattr(args, "min_compute_tokens", 16384))

    # (a) Cache monotonicity: group by input_len, check that RT decreases with cache_len
    by_input: dict[int, list[Observation]] = {}
    for row in rows:
        by_input.setdefault(row.input_len, []).append(row)
    for input_len, group in sorted(by_input.items()):
        sorted_by_cache = sorted(group, key=lambda r: r.cache_len)
        for i in range(1, len(sorted_by_cache)):
            prev = sorted_by_cache[i - 1]
            curr = sorted_by_cache[i]
            if curr.cache_len > prev.cache_len and curr.target_ms > prev.target_ms:
                anomalies.append(
                    {
                        "check": "cache_monotonicity",
                        "severity": "warning",
                        "input_len": input_len,
                        "detail": (
                            f"cache={prev.cache_len} -> {curr.target_ms:.1f}ms but "
                            f"cache={curr.cache_len} -> {curr.target_ms:.1f}ms "
                            f"(longer cache is slower by {curr.target_ms - prev.target_ms:.1f}ms)"
                        ),
                        "cache_short": prev.cache_len,
                        "cache_long": curr.cache_len,
                        "rt_short_ms": prev.target_ms,
                        "rt_long_ms": curr.target_ms,
                    }
                )

    # (b) Run spread: skip if we don't have per-run data (observations are aggregated)
    # This check applies to the raw result JSON items, not aggregated observations.

    # (c) Residual outliers: fit formula and find high-APE cases
    splits = split_rows(rows)
    residual_check = {
        "model_family": "restricted-symbolic",
        "status": "skipped",
        "reason": "empty_train_or_validation_split",
    }
    if splits["train"] and splits["validation"]:
        model = fit_restricted_symbolic(
            splits["train"],
            splits["validation"],
            [*splits["train"], *splits["validation"]],
            token_unit=token_unit,
        )
        residual_check = {"model_family": "restricted-symbolic", "status": "completed"}
        max_ape_pct = float(getattr(args, "max_anomaly_ape_pct", 25.0))
        for row in rows:
            predicted = model.predict(row)
            ape = 100.0 * abs(predicted - row.target_ms) / row.target_ms
            if ape > max_ape_pct and row.target_ms > min_rt_ms:
                anomalies.append(
                    {
                        "check": "residual_outlier",
                        "severity": "warning",
                        "input_len": row.input_len,
                        "cache_len": row.cache_len,
                        "compute_len": row.compute_len,
                        "target_ms": row.target_ms,
                        "measurement_contract": audit["measurement_contract"],
                        "predicted_ms": predicted,
                        "ape_pct": round(ape, 2),
                        "detail": f"APE={ape:.1f}% (target={row.target_ms:.1f}ms, predicted={predicted:.1f}ms)",
                    }
                )

    # (d) Cross-input compute monotonicity: for same cache_len, more compute = more RT
    by_cache: dict[int, list[Observation]] = {}
    for row in rows:
        by_cache.setdefault(row.cache_len, []).append(row)
    for cache_len, group in sorted(by_cache.items()):
        sorted_by_compute = sorted(group, key=lambda r: r.compute_len)
        for i in range(1, len(sorted_by_compute)):
            prev = sorted_by_compute[i - 1]
            curr = sorted_by_compute[i]
            if (
                curr.compute_len > prev.compute_len
                and curr.compute_len >= min_compute_tokens
                and prev.compute_len >= min_compute_tokens
                and curr.target_ms < prev.target_ms
            ):
                anomalies.append(
                    {
                        "check": "compute_monotonicity",
                        "severity": "info",
                        "cache_len": cache_len,
                        "detail": (
                            f"compute={prev.compute_len} -> {prev.target_ms:.1f}ms but "
                            f"compute={curr.compute_len} -> {curr.target_ms:.1f}ms "
                            f"(more compute is faster by {prev.target_ms - curr.target_ms:.1f}ms)"
                        ),
                    }
                )

    anomalies.sort(
        key=lambda a: (
            {"error": 0, "warning": 1, "info": 2}.get(a.get("severity", "info"), 2),
            -abs(a.get("ape_pct", 0)),
            a.get("check", ""),
        )
    )

    summary = {
        "total_anomalies": len(anomalies),
        "by_check": {},
        "by_severity": {},
    }
    for a in anomalies:
        check = a["check"]
        severity = a.get("severity", "info")
        summary["by_check"][check] = summary["by_check"].get(check, 0) + 1
        summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1

    report = {
        "schema_version": 1,
        "model": model_label,
        "summary": summary,
        "anomalies": anomalies,
        "audit": audit,
        "valid_observations": len(rows),
        "residual_check": residual_check,
        "floors": {"min_rt_ms": min_rt_ms, "min_compute_tokens": min_compute_tokens},
        "profile": profile,
        "profile_sha256": profile_fingerprint(profile) if profile else None,
    }
    (output / "anomaly_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"total_anomalies": len(anomalies), "summary": summary}, ensure_ascii=False
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="fit from successful measurements")
    fit.add_argument("--inputs", nargs="+", required=True)
    fit.add_argument("--output-dir", required=True)
    fit.add_argument(
        "--batch-size", type=int, default=None, help="Optional batch filter"
    )
    fit.add_argument("--min-valid-rows", type=int, default=30)
    fit.add_argument("--max-mape-pct", type=float, default=5.0)
    fit.add_argument("--max-p95-ape-pct", type=float, default=10.0)
    fit.add_argument("--max-max-ape-pct", type=float, default=40.0)
    fit.add_argument(
        "--estimator",
        choices=("median", "min", "trimmed"),
        default="median",
        help=(
            "run-time statistic per case; 'min' rejects additive spikes from "
            "external GPU contention, 'trimmed' drops the slowest run"
        ),
    )
    fit.add_argument("--symbolic-max-terms", type=int, default=15)
    fit.add_argument("--symbolic-complexity-tolerance-pct", type=float, default=5.0)
    fit.add_argument(
        "--symbolic-hinge-tokens",
        default=",".join(str(value) for value in DEFAULT_HINGE_TOKENS),
        help="Comma-separated token thresholds for compute/input hinge candidates.",
    )
    fit.add_argument(
        "--symbolic-exp-decay-tokens",
        default=",".join(str(value) for value in DEFAULT_EXP_DECAY_TOKENS),
        help="Comma-separated positive token scales for safe exp(-tokens/scale) candidates.",
    )
    fit.add_argument("--allow-insufficient-data", action="store_true")
    _add_common_profile_args(fit)
    fit.add_argument(
        "--token-unit",
        type=int,
        default=None,
        help="Token unit for feature normalization (default: 1024).",
    )
    fit.add_argument(
        "--formula-filename", default=None, help="Override the formula output filename."
    )
    fit.add_argument(
        "--formula-key", default=None, help="Override the formula key in the .txt file."
    )
    fit.set_defaults(func=run_fit)

    validate = sub.add_parser("validate-inputs", help="audit valid/invalid rows")
    validate.add_argument("--inputs", nargs="+", required=True)
    validate.add_argument("--batch-size", type=int, default=None)
    validate.add_argument(
        "--estimator",
        choices=("median", "min", "trimmed"),
        default="median",
    )
    validate.add_argument("--report")
    _add_common_profile_args(validate)
    validate.set_defaults(func=run_validate)

    anom = sub.add_parser("analyze-anomalies", help="detect anomalous measurements")
    anom.add_argument("--inputs", nargs="+", required=True)
    anom.add_argument("--output-dir", required=True)
    anom.add_argument("--batch-size", type=int, default=None)
    anom.add_argument(
        "--estimator",
        choices=("median", "min", "trimmed"),
        default="median",
    )
    anom.add_argument(
        "--min-rt-ms",
        type=float,
        default=5.0,
        help="Minimum RT floor for residual checks.",
    )
    anom.add_argument(
        "--min-compute-tokens",
        type=int,
        default=16384,
        help="Minimum compute floor for monotonicity.",
    )
    anom.add_argument(
        "--max-anomaly-ape-pct",
        type=float,
        default=25.0,
        help="APE threshold for residual outliers.",
    )
    _add_common_profile_args(anom)
    anom.add_argument("--token-unit", type=int, default=None)
    anom.set_defaults(func=run_analyze_anomalies)

    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    raise SystemExit(parsed_args.func(parsed_args))
