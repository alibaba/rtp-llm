"""Offline restricted FlexLB fitting for validated unified prefill observations."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import pathlib
import statistics
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Sequence, TypeVar

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


# ---------------------------------------------------------------------------
# Restricted-library symbolic regression search (merged from prefill/restricted_symbolic.py).
# ---------------------------------------------------------------------------

LIBRARY_VERSION = "prefill-request-list-v3"
DEFAULT_HINGE_TOKENS = (16384, 32768, 65536, 131072, 262144, 524288)
DEFAULT_EXP_DECAY_TOKENS = (16384, 65536, 262144)
PARSER_OPERATORS = ("+", "-", "*", "/", "^")
PARSER_FUNCTIONS = ("sqrt", "log", "exp", "abs", "max", "min", "pow")
PARSER_AGGREGATES = ("sum",)
PARSER_PER_REQUEST_VARIABLES = (
    "inputTokens",
    "hitCacheTokens",
    "computeTokens",
    "hasHitCache",
)
PARSER_BATCH_VARIABLES = (
    "batchSize",
    "totalInputTokens",
    "totalHitCacheTokens",
    "totalComputeTokens",
    "maxInputTokens",
    "maxComputeTokens",
)
_POWERS = (0.25, 0.75, 1.25, 1.5, 1.75, 2.5)

ObservationT = TypeVar("ObservationT")


@dataclass(frozen=True)
class CandidateTerm:
    name: str
    expression: str
    evaluate: Callable[[object], float]


@dataclass(frozen=True)
class RestrictedSymbolicModel:
    terms: tuple[CandidateTerm, ...]
    coefficients: tuple[float, ...]
    formula: str
    backend: str
    search_report: dict[str, object]

    def predict(self, row: object) -> float:
        return sum(
            coefficient * term.evaluate(row)
            for coefficient, term in zip(self.coefficients, self.terms)
        )


def _variables(row: object, token_unit: int) -> tuple[float, float, float, float]:
    input_len = float(getattr(row, "input_len"))
    cache_len = float(getattr(row, "cache_len"))
    compute_len = input_len - cache_len
    return (
        compute_len / token_unit,
        cache_len / token_unit,
        input_len / token_unit,
        float(cache_len > 0),
    )


def _requests(row):
    return getattr(row, "requests", ()) or ((row.input_len, row.cache_len),)


def _sum_term(row, token_unit, fn):
    return sum(
        fn(_variables(SimpleNamespace(input_len=i, cache_len=c), token_unit))
        for i, c in _requests(row)
    )


def build_candidate_library(
    token_unit: int,
    hinge_tokens: Sequence[int] = DEFAULT_HINGE_TOKENS,
    exp_decay_tokens: Sequence[int] = DEFAULT_EXP_DECAY_TOKENS,
) -> tuple[CandidateTerm, ...]:
    if token_unit <= 0:
        raise ValueError("token_unit must be positive")
    if any(value <= 0 for value in hinge_tokens):
        raise ValueError("hinge token thresholds must be positive")
    if any(value <= 0 for value in exp_decay_tokens):
        raise ValueError("exp decay token scales must be positive")

    u = f"(computeTokens / {token_unit}.0)"
    c = f"(hitCacheTokens / {token_unit}.0)"
    s = f"(inputTokens / {token_unit}.0)"
    h = "hasHitCache"
    terms: list[CandidateTerm] = [CandidateTerm("1", "1", lambda row: 1.0)]

    def add(
        name: str,
        expression: str,
        fn: Callable[[tuple[float, float, float, float]], float],
    ) -> None:
        terms.append(
            CandidateTerm(
                name,
                f"sum({expression})",
                lambda row, fn=fn: _sum_term(row, token_unit, fn),
            )
        )

    add("u", u, lambda values: values[0])
    add("c", c, lambda values: values[1])
    add("hasHitCache", h, lambda values: values[3])
    add("u**2", f"({u} ^ 2)", lambda values: values[0] ** 2)
    add("u*c", f"{u} * {c}", lambda values: values[0] * values[1])
    add("c**2", f"({c} ^ 2)", lambda values: values[1] ** 2)
    add("u**3", f"({u} ^ 3)", lambda values: values[0] ** 3)
    add("u**2*c", f"{u} * {u} * {c}", lambda values: values[0] ** 2 * values[1])
    add("u*c**2", f"{u} * {c} * {c}", lambda values: values[0] * values[1] ** 2)
    add("c**3", f"({c} ^ 3)", lambda values: values[1] ** 3)

    variables = (("u", u, 0), ("c", c, 1), ("s", s, 2))
    for name, expression, index in variables:
        add(
            f"sqrt({name})",
            f"sqrt({expression})",
            lambda values, index=index: math.sqrt(values[index]),
        )
        add(
            f"log1p({name})",
            f"log(1 + {expression})",
            lambda values, index=index: math.log1p(values[index]),
        )
        for power in _POWERS:
            add(
                f"{name}**{power}",
                f"pow({expression}, {power})",
                lambda values, index=index, power=power: values[index] ** power,
            )
        for scale_tokens in exp_decay_tokens:
            scale = scale_tokens / float(token_unit)
            scale_label = f"{scale:.15g}"
            add(
                f"exp(-{name}/{scale_label})",
                f"exp(-{expression} / {scale_label})",
                lambda values, index=index, scale=scale: math.exp(
                    -values[index] / scale
                ),
            )

    add("abs(u-c)", f"abs({u} - {c})", lambda values: abs(values[0] - values[1]))
    add("max(u,c)", f"max({u}, {c})", lambda values: max(values[0], values[1]))
    add("min(u,c)", f"min({u}, {c})", lambda values: min(values[0], values[1]))
    add("u/(1+c)", f"{u} / (1 + {c})", lambda values: values[0] / (1 + values[1]))
    add("c/(1+u)", f"{c} / (1 + {u})", lambda values: values[1] / (1 + values[0]))

    for name, expression, index in (("c", c, 1), ("s", s, 2), ("u", u, 0)):
        add(
            f"u*sqrt({name})",
            f"{u} * sqrt({expression})",
            lambda values, index=index: values[0] * math.sqrt(values[index]),
        )
        add(
            f"u*log1p({name})",
            f"{u} * log(1 + {expression})",
            lambda values, index=index: values[0] * math.log1p(values[index]),
        )

    for threshold_tokens in hinge_tokens:
        threshold = threshold_tokens / float(token_unit)
        label = f"{threshold:.15g}"
        add(
            f"max(u-{label},0)",
            f"max({u} - {label}, 0)",
            lambda values, threshold=threshold: max(values[0] - threshold, 0.0),
        )
        add(
            f"max(s-{label},0)",
            f"max({s} - {label}, 0)",
            lambda values, threshold=threshold: max(values[2] - threshold, 0.0),
        )
    terms.extend(
        [
            CandidateTerm("batchSize", "batchSize", lambda row: float(row.batch_size)),
            CandidateTerm(
                "maxComputeTokens",
                f"(maxComputeTokens / {token_unit}.0)",
                lambda row: max(i - c for i, c in _requests(row)) / token_unit,
            ),
            CandidateTerm(
                "maxInputTokens",
                f"(maxInputTokens / {token_unit}.0)",
                lambda row: max(i for i, c in _requests(row)) / token_unit,
            ),
        ]
    )
    return tuple(terms)


def _formula_text(terms: Sequence[CandidateTerm], coefficients: Sequence[float]) -> str:
    rendered: list[str] = []
    for term, coefficient in zip(terms, coefficients):
        if abs(coefficient) < 1e-14:
            continue
        magnitude = f"{abs(coefficient):.15g}"
        value = (
            magnitude if term.expression == "1" else f"{magnitude} * {term.expression}"
        )
        if not rendered:
            rendered.append(("-" if coefficient < 0 else "") + value)
        else:
            rendered.append((" - " if coefficient < 0 else " + ") + value)
    return "".join(rendered) if rendered else "0"


def fit_restricted_symbolic(
    train_rows: Sequence[ObservationT],
    validation_rows: Sequence[ObservationT],
    refit_rows: Sequence[ObservationT],
    *,
    token_unit: int,
    hinge_tokens: Sequence[int] = DEFAULT_HINGE_TOKENS,
    exp_decay_tokens: Sequence[int] = DEFAULT_EXP_DECAY_TOKENS,
    max_terms: int = 15,
    complexity_tolerance_pct: float = 5.0,
) -> RestrictedSymbolicModel:
    try:
        import torch  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "restricted symbolic fitting requires CPU PyTorch"
        ) from error

    if not train_rows or not validation_rows or not refit_rows:
        raise ValueError(
            "restricted symbolic fitting requires non-empty train, validation, and refit rows"
        )
    if max_terms < 1:
        raise ValueError("max_terms must be positive")
    if complexity_tolerance_pct < 0:
        raise ValueError("complexity_tolerance_pct must be nonnegative")

    terms = build_candidate_library(token_unit, hinge_tokens, exp_decay_tokens)
    max_terms = min(max_terms, len(terms))

    def matrix(rows: Sequence[ObservationT]):
        values = [[term.evaluate(row) for term in terms] for row in rows]
        for row_index, row_values in enumerate(values):
            for term, value in zip(terms, row_values):
                if not math.isfinite(value):
                    raise ValueError(
                        f"candidate {term.name!r} produced non-finite value "
                        f"at row {row_index}"
                    )
        return torch.tensor(
            values,
            dtype=torch.float64,
            device="cpu",
        )

    x_train = matrix(train_rows)
    x_validation = matrix(validation_rows)
    y_train = torch.tensor(
        [float(getattr(row, "target_ms")) for row in train_rows], dtype=torch.float64
    )
    y_validation = torch.tensor(
        [float(getattr(row, "target_ms")) for row in validation_rows],
        dtype=torch.float64,
    )
    scales = torch.sqrt(torch.mean(x_train * x_train, dim=0)).clamp_min(1.0)
    z_train = x_train / scales
    z_validation = x_validation / scales
    relative_train = z_train / y_train[:, None].clamp_min(1e-12)

    def weights(rows):
        counts = Counter(getattr(row, "batch_size", 1) for row in rows)
        geometry_counts = Counter(
            getattr(row, "geometry_key", "") or str(i) for i, row in enumerate(rows)
        )
        raw = [
            1.0 / geometry_counts[getattr(row, "geometry_key", "") or str(i)]
            for i, row in enumerate(rows)
        ]
        totals = {
            b: sum(w for w, row in zip(raw, rows) if getattr(row, "batch_size", 1) == b)
            for b in counts
        }
        return torch.tensor(
            [w / totals[getattr(row, "batch_size", 1)] for w, row in zip(raw, rows)],
            dtype=torch.float64,
        )

    train_weights = weights(train_rows)
    validation_weights = weights(validation_rows)
    relative_train = relative_train * train_weights.sqrt()[:, None]
    gram = relative_train.T @ relative_train
    rhs = relative_train.T @ train_weights.sqrt()

    def solve(z, y, indices: Sequence[int], row_weights):
        relative_design = z[:, indices] / y[:, None].clamp_min(1e-12)
        return torch.linalg.lstsq(
            relative_design * row_weights.sqrt()[:, None],
            row_weights.sqrt(),
            rcond=1e-10,
            driver="gelsd",
        ).solution

    def solve_train(indices: Sequence[int]):
        return torch.linalg.lstsq(
            gram[list(indices)][:, list(indices)],
            rhs[list(indices)],
            rcond=1e-12,
            driver="gelsd",
        ).solution

    selected = [0]
    history: list[dict[str, object]] = []
    for _ in range(max_terms):
        beta = solve_train(selected)
        train_relative_mse = float(
            torch.sum(
                train_weights
                * ((z_train[:, selected].mv(beta) - y_train) / y_train) ** 2
            )
            / train_weights.sum()
        )
        validation_relative_mse = float(
            torch.sum(
                validation_weights
                * (
                    ((z_validation[:, selected].mv(beta) - y_validation) / y_validation)
                    ** 2
                )
            )
            / validation_weights.sum()
        )
        history.append(
            {
                "term_count": len(selected),
                "term_names": [terms[index].name for index in selected],
                "train_relative_mse": train_relative_mse,
                "validation_relative_mse": validation_relative_mse,
            }
        )
        if len(selected) >= max_terms:
            break
        choices: list[tuple[float, int]] = []
        for index in range(1, len(terms)):
            if index in selected:
                continue
            candidate = [*selected, index]
            if torch.linalg.matrix_rank(z_train[:, candidate], tol=1e-8) < len(
                candidate
            ):
                continue
            candidate_beta = solve_train(candidate)
            candidate_gram = gram[candidate][:, candidate]
            candidate_rhs = rhs[candidate]
            loss = float(
                (
                    candidate_beta @ candidate_gram @ candidate_beta
                    - 2 * candidate_beta @ candidate_rhs
                    + train_weights.sum()
                )
                / train_weights.sum()
            )
            choices.append((loss, index))
        if not choices:
            break
        selected.append(min(choices, key=lambda item: (item[0], item[1]))[1])

    best_validation = min(float(item["validation_relative_mse"]) for item in history)
    limit = best_validation * (1.0 + complexity_tolerance_pct / 100.0)
    chosen = next(
        item for item in history if float(item["validation_relative_mse"]) <= limit
    )
    chosen_names = set(chosen["term_names"])
    chosen_indices = [
        index for index, term in enumerate(terms) if term.name in chosen_names
    ]

    x_refit = matrix(refit_rows) / scales
    y_refit = torch.tensor(
        [float(getattr(row, "target_ms")) for row in refit_rows], dtype=torch.float64
    )
    beta = solve(x_refit, y_refit, chosen_indices, weights(refit_rows))
    coefficients = tuple(
        float(value) for value in (beta / scales[chosen_indices]).tolist()
    )
    chosen_terms = tuple(terms[index] for index in chosen_indices)
    condition_number = float(torch.linalg.cond(x_refit[:, chosen_indices]).item())
    report: dict[str, object] = {
        "library_version": LIBRARY_VERSION,
        "candidate_count": len(terms),
        "candidate_names": [term.name for term in terms],
        "hinge_tokens": list(hinge_tokens),
        "exp_decay_tokens": list(exp_decay_tokens),
        "parser_coverage": {
            "operators": list(PARSER_OPERATORS),
            "functions": list(PARSER_FUNCTIONS),
            "aggregates": list(PARSER_AGGREGATES),
            "per_request_variables": list(PARSER_PER_REQUEST_VARIABLES),
            "batch_variables": list(PARSER_BATCH_VARIABLES),
        },
        "max_terms": max_terms,
        "complexity_tolerance_pct": complexity_tolerance_pct,
        "selection_rule": "smallest expression within tolerance of best validation relative MSE",
        "objective": "mean_squared_relative_error",
        "feature_scaling": "train RMS, clamped to a minimum of 1",
        "weighting": "equal batch-size weight; equal geometry weight within batch",
        "rank_policy": "skip linearly dependent candidate columns on training data",
        "refit_policy": "selected terms refit on train+validation",
        "test_usage": "test rows are excluded from selection and refit",
        "history": history,
        "selected_terms": [term.name for term in chosen_terms],
        "selected_validation_relative_mse": chosen["validation_relative_mse"],
        "design_condition_number": condition_number,
    }
    return RestrictedSymbolicModel(
        chosen_terms,
        coefficients,
        _formula_text(chosen_terms, coefficients),
        "torch_cpu_forward_selection_relative_wls",
        report,
    )
