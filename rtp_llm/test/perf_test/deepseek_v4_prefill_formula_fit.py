#!/usr/bin/env python3
"""Fit and audit a DeepSeek-V4-Pro prefill latency formula.

The input is the raw ``cache_grid_results.json`` emitted by the DSV4
prefill grid runner.  A grid file can contain successful measurements,
failed requests, and successful requests whose cache seed was not actually
reused.  This tool accepts a row only when the runner marks it successful and
all measured reuse lengths exactly match the requested cache length.

* every requested measurement run succeeded;
* every run has output length one and finite end-to-end TTFT;
* observed reuse is constant and exactly matches the request; and
* the selected batch size is fixed (the DSV4 Pro configuration uses 1).

The exported expression uses the names and aggregate syntax implemented by
``PrefillTimeFormula``: ``computeTokens``, ``hitCacheTokens``, ``sum()``,
numbers, arithmetic operators, and parentheses.  It does not invent an alias
such as ``tokens``.  The DSV4-Pro measurements currently use batch size one;
``sum()`` keeps the per-request terms well-defined if the same expression is
evaluated through FlexLB's batch path.

New runner output uses client HTTP wall time with ``max_new_tokens=1`` as
TTFT.  The server's ``first_token_cost_time`` is retained for diagnostics and
is used only as a backward-compatible fallback for legacy inputs.

The report keeps the fit and the production gate separate: a formula can be
useful for analysis while still failing a tail-error gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import random
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

FEATURE_NAMES = (
    "1",
    "sum(computeTokens / 1024.0)",
    "sum(hitCacheTokens / 1024.0)",
    "sum((computeTokens / 1024.0) * (computeTokens / 1024.0))",
    "sum((computeTokens / 1024.0) * (hitCacheTokens / 1024.0))",
    "sum((hitCacheTokens / 1024.0) * (hitCacheTokens / 1024.0))",
)


@dataclass(frozen=True)
class Observation:
    batch_size: int
    input_len: int
    cache_len: int
    target_ms: float
    source: str
    requested_cache_len: int | None = None

    @property
    def compute_len(self) -> int:
        return self.input_len - self.cache_len


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(result):
        return result
    return None


def _integer(value: Any) -> int | None:
    number = _finite(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _status_ok(item: dict[str, Any]) -> bool:
    status = str(item.get("status", "")).lower()
    # Fail closed. A completed HTTP forward is not a valid cache-performance
    # sample when the runner marked its reuse contract invalid.
    return not status or status in {
        "ok",
        "success",
        "passed",
    }


def _median_run_time(
    item: dict[str, Any]
) -> tuple[float | None, int | None, str | None]:
    runs = item.get("runs")
    if not isinstance(runs, list) or not runs:
        return None, None, "missing_runs"
    expected_runs = _integer(item.get("measure_runs")) or 3
    success_runs = _integer(item.get("success_runs"))
    if success_runs != expected_runs or len(runs) != expected_runs:
        return None, None, "incomplete_runs"
    values: list[float] = []
    input_len = _integer(item.get("input_len"))
    requested_cache_len = _integer(item.get("cache_len_requested")) or 0
    if item.get("reuse_exact") is False:
        return None, None, "reuse_not_exact"
    observed = item.get("cache_len_observed")
    observed_values = (
        [_integer(value) for value in observed] if isinstance(observed, list) else []
    )
    observed_values = [value for value in observed_values if value is not None]
    if not observed_values:
        observed_values = [
            _integer(run.get("reuse_len")) for run in runs if isinstance(run, dict)
        ]
        observed_values = [value for value in observed_values if value is not None]
    if len(observed_values) != expected_runs or len(set(observed_values)) != 1:
        return None, None, "observed_reuse_not_constant"
    cache_len = observed_values[0]
    if cache_len < 0 or input_len is None or cache_len >= input_len:
        return None, None, "invalid_observed_geometry"
    if cache_len != requested_cache_len:
        return None, None, "requested_reuse_mismatch"
    for run in runs:
        if not isinstance(run, dict) or run.get("success") is not True:
            return None, None, "run_failed"
        run_input = _integer(run.get("input_len"))
        output_len = _integer(run.get("output_len"))
        reuse_len = _integer(run.get("reuse_len"))
        latency = _finite(
            run.get(
                "ttft_ms",
                run.get("client_wall_time_ms", run.get("prefill_time_ms")),
            )
        )
        if run_input != input_len or output_len != 1:
            return None, None, "request_shape_mismatch"
        if reuse_len != cache_len:
            return None, None, "reuse_mismatch"
        if latency is None or latency <= 0:
            return None, None, "invalid_latency"
        values.append(latency)
    return statistics.median(values), cache_len, None


def _iter_json_metrics(path: pathlib.Path) -> Iterable[tuple[int, dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = (
        data.get("metrics", data.get("results", [])) if isinstance(data, dict) else []
    )
    if not isinstance(metrics, list):
        raise ValueError(f"{path}: expected a JSON object containing metrics[]")
    for index, item in enumerate(metrics):
        if isinstance(item, dict):
            yield index, item


def load_observations(
    paths: Sequence[pathlib.Path], *, batch_size: int = 1
) -> tuple[list[Observation], dict[str, Any]]:
    observations: list[Observation] = []
    rejected: dict[str, int] = {}
    input_files: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            source_count = len(rows)
            for index, item in enumerate(rows, 2):
                batch = _integer(item.get("batch_size")) or 1
                if batch != batch_size:
                    rejected["batch_size"] = rejected.get("batch_size", 0) + 1
                    continue
                input_len = _integer(item.get("input_len"))
                cache_len = (
                    _integer(item.get("cache_len_requested", item.get("cache_len")))
                    or 0
                )
                target = _finite(
                    item.get(
                        "avg_ttft_ms",
                        item.get(
                            "ttft_ms",
                            item.get(
                                "client_wall_time_ms",
                                item.get(
                                    "avg_prefill_time_ms",
                                    item.get(
                                        "prefill_time_ms", item.get("target_ms")
                                    ),
                                ),
                            ),
                        ),
                    )
                )
                if (
                    input_len is None
                    or cache_len < 0
                    or cache_len >= input_len
                    or target is None
                    or target <= 0
                ):
                    rejected["invalid_geometry_or_latency"] = (
                        rejected.get("invalid_geometry_or_latency", 0) + 1
                    )
                    continue
                observations.append(
                    Observation(
                        batch,
                        input_len,
                        cache_len,
                        target,
                        f"{path.name}:{index}",
                        cache_len,
                    )
                )
            input_files.append(
                {"path": str(path), "rows": source_count, "format": "csv"}
            )
            continue

        source_count = 0
        for index, item in _iter_json_metrics(path):
            source_count += 1
            if not _status_ok(item):
                rejected["status"] = rejected.get("status", 0) + 1
                continue
            batch = _integer(item.get("batch_size")) or 1
            if batch != batch_size:
                rejected["batch_size"] = rejected.get("batch_size", 0) + 1
                continue
            input_len = _integer(item.get("input_len"))
            cache_len = _integer(item.get("cache_len_requested"))
            if (
                input_len is None
                or cache_len is None
                or cache_len < 0
                or cache_len >= input_len
            ):
                rejected["invalid_geometry"] = rejected.get("invalid_geometry", 0) + 1
                continue
            target, observed_cache_len, reason = _median_run_time(item)
            if target is None:
                rejected[reason or "invalid_run"] = (
                    rejected.get(reason or "invalid_run", 0) + 1
                )
                continue
            assert observed_cache_len is not None
            observations.append(
                Observation(
                    batch,
                    input_len,
                    cache_len,
                    target,
                    f"{path.name}:metrics[{index}]",
                    cache_len,
                )
            )
        input_files.append({"path": str(path), "rows": source_count, "format": "json"})

    observations.sort(
        key=lambda row: (row.batch_size, row.input_len, row.cache_len, row.source)
    )
    raw_observation_count = len(observations)
    grouped: dict[tuple[int, int, int], list[Observation]] = {}
    for row in observations:
        grouped.setdefault((row.batch_size, row.input_len, row.cache_len), []).append(
            row
        )
    observations = [
        Observation(
            key[0],
            key[1],
            key[2],
            statistics.median(item.target_ms for item in values),
            values[0].source,
            values[0].requested_cache_len,
        )
        for key, values in sorted(grouped.items())
    ]
    unique = {(row.batch_size, row.input_len, row.cache_len) for row in observations}
    audit = {
        "input_files": input_files,
        "raw_metric_count": sum(int(item["rows"]) for item in input_files),
        "raw_valid_observation_count": raw_observation_count,
        "valid_observation_count": len(observations),
        "collapsed_duplicate_geometry_count": raw_observation_count - len(observations),
        "unique_geometry_count": len(unique),
        "rejected_counts": rejected,
        "selected_batch_size": batch_size,
        "seq_len_range": [
            min((x.input_len for x in observations), default=None),
            max((x.input_len for x in observations), default=None),
        ],
        "cache_len_range": [
            min((x.cache_len for x in observations), default=None),
            max((x.cache_len for x in observations), default=None),
        ],
    }
    return observations, audit


def feature_values(row: Observation) -> list[float]:
    compute = row.compute_len / 1024.0
    hit = row.cache_len / 1024.0
    return [
        1.0,
        compute,
        hit,
        compute * compute,
        compute * hit,
        hit * hit,
    ]


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    aug = [list(matrix[index]) + [vector[index]] for index in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(aug[row][column]))
        if abs(aug[pivot][column]) < 1e-12:
            raise ValueError(
                "singular regression matrix; collect more varied geometries"
            )
        aug[column], aug[pivot] = aug[pivot], aug[column]
        scale = aug[column][column]
        aug[column] = [value / scale for value in aug[column]]
        for row in range(n):
            if row == column:
                continue
            factor = aug[row][column]
            if factor:
                aug[row] = [a - factor * b for a, b in zip(aug[row], aug[column])]
    return [aug[index][-1] for index in range(n)]


def _weighted_median(values: Sequence[tuple[float, float]]) -> float:
    """Return a deterministic weighted median of (value, nonnegative weight)."""
    ordered = sorted(
        (value, weight)
        for value, weight in values
        if weight > 0 and math.isfinite(value)
    )
    if not ordered:
        return 0.0
    total = sum(weight for _, weight in ordered)
    threshold = total * 0.5
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def fit_lad_coefficients(
    rows: Sequence[Observation], *, max_iter: int = 2000, tol: float = 1e-10
) -> list[float]:
    """Coordinate-descent least-absolute-deviation (L1/MAE) regression.

    For one coordinate, the exact minimizer is a weighted median.  Columns
    are scaled during optimization to avoid the large dynamic range of the
    quadratic token features, then converted back to formula coefficients.
    """
    matrix = [feature_values(row) for row in rows]
    targets = [row.target_ms for row in rows]
    width = len(FEATURE_NAMES)
    scales = [1.0] * width
    for column in range(1, width):
        scales[column] = max(max(abs(values[column]) for values in matrix), 1.0)
    scaled = [
        [values[column] / scales[column] for column in range(width)]
        for values in matrix
    ]
    coefficients = [0.0] * width
    residual = list(targets)  # target - scaled_matrix @ coefficients
    for _ in range(max_iter):
        largest_delta = 0.0
        for column in range(width):
            candidates: list[tuple[float, float]] = []
            for row, x in zip(scaled, residual):
                value = row[column]
                if abs(value) > 1e-15:
                    # residual currently includes -value * old coefficient.
                    excluding = x + value * coefficients[column]
                    candidates.append((excluding / value, abs(value)))
            updated = _weighted_median(candidates)
            delta = abs(updated - coefficients[column])
            largest_delta = max(largest_delta, delta)
            if delta:
                for index, row in enumerate(scaled):
                    residual[index] -= row[column] * (updated - coefficients[column])
                coefficients[column] = updated
        if largest_delta <= tol:
            break
    return [coefficient / scale for coefficient, scale in zip(coefficients, scales)]


def fit_hybrid_coefficients(
    rows: Sequence[Observation],
    *,
    seed: int,
    steps: int = 8000,
    learning_rate: float = 0.03,
) -> list[float]:
    """Fit with torch autograd against absolute and relative error together.

    MAE is divided by the median training latency so the millisecond term and
    the relative-error term have comparable scale.  The optimized loss is::

        0.5 * mean(abs(pred-target)) / median(target)
      + 0.5 * mean(abs(pred-target) / target)

    Feature columns are scaled only while optimizing; exported coefficients
    are converted back to the original FlexLB-compatible expressions.
    """
    try:
        import torch  # type: ignore
    except ImportError as error:
        raise RuntimeError("hybrid objective requires CPU PyTorch") from error

    torch.manual_seed(seed)
    x = torch.tensor(
        [feature_values(row) for row in rows], dtype=torch.float64, device="cpu"
    )
    y = torch.tensor(
        [row.target_ms for row in rows], dtype=torch.float64, device="cpu"
    )
    scales = torch.amax(torch.abs(x), dim=0).clamp_min(1.0)
    scaled_x = x / scales
    # Start from the exact coordinate-descent MAE fit.  It is a materially
    # better starting point than least squares for the long TTFT tail, and we
    # retain it unless autograd lowers the requested combined loss.
    lad = fit_lad_coefficients(rows)
    beta = torch.nn.Parameter(
        torch.tensor(lad, dtype=torch.float64, device="cpu") * scales
    )
    optimizer = torch.optim.Adam([beta], lr=learning_rate)
    latency_scale = torch.median(y).clamp_min(1e-9)
    with torch.no_grad():
        initial_error = torch.abs(scaled_x.mv(beta) - y)
        best_loss = float(
            0.5 * initial_error.mean() / latency_scale
            + 0.5 * (initial_error / y.clamp_min(1e-9)).mean()
        )
    best_beta = beta.detach().clone()
    stale_steps = 0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            absolute_error = torch.abs(scaled_x.mv(beta) - y)
            loss = 0.5 * absolute_error.mean() / latency_scale + 0.5 * (
                absolute_error / y.clamp_min(1e-9)
            ).mean()
            loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if value + 1e-12 < best_loss:
            best_loss = value
            best_beta = beta.detach().clone()
            stale_steps = 0
        else:
            stale_steps += 1
        if stale_steps >= 1200:
            break
    return [float(value) for value in (best_beta / scales).tolist()]


def fit_coefficients(
    rows: Sequence[Observation],
    *,
    objective: str = "mae",
    ridge: float = 1e-8,
    seed: int = 20260904,
) -> tuple[list[float], str]:
    if len(rows) < len(FEATURE_NAMES):
        raise ValueError(
            f"need at least {len(FEATURE_NAMES)} valid rows, got {len(rows)}"
        )
    if objective == "mae":
        return fit_lad_coefficients(rows), "python_coordinate_descent_lad"
    if objective == "hybrid":
        return (
            fit_hybrid_coefficients(rows, seed=seed),
            "torch_cpu_autograd_hybrid_absolute_relative",
        )
    if objective != "mse":
        raise ValueError(f"unsupported objective: {objective}")
    # Prefer CPU torch when available, but keep the tool runnable in the
    # source container where torch may not be installed.
    try:
        import torch  # type: ignore

        torch.set_grad_enabled(False)
        x = torch.tensor(
            [feature_values(row) for row in rows], dtype=torch.float64, device="cpu"
        )
        y = torch.tensor(
            [row.target_ms for row in rows], dtype=torch.float64, device="cpu"
        )
        solution = torch.linalg.lstsq(x, y).solution
        return [float(value) for value in solution.tolist()], "torch_cpu_lstsq"
    except (ImportError, RuntimeError, ValueError):
        pass

    width = len(FEATURE_NAMES)
    gram = [[0.0] * width for _ in range(width)]
    rhs = [0.0] * width
    for row in rows:
        values = feature_values(row)
        for i in range(width):
            rhs[i] += values[i] * row.target_ms
            for j in range(width):
                gram[i][j] += values[i] * values[j]
    for index in range(1, width):
        gram[index][index] += ridge
    return _solve_linear_system(gram, rhs), "python_ridge_normal_equation"


def predict(coefficients: Sequence[float], row: Observation) -> float:
    return sum(
        coefficient * value
        for coefficient, value in zip(coefficients, feature_values(row))
    )


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


def error_metrics(
    rows: Sequence[Observation], coefficients: Sequence[float]
) -> dict[str, Any]:
    errors = [abs(predict(coefficients, row) - row.target_ms) for row in rows]
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


def split_rows(
    rows: Sequence[Observation],
    *,
    mode: str = "seq-hash-70-15-15",
    seed: int = 20260904,
) -> dict[str, list[Observation]]:
    if mode == "random-50-50":
        shuffled = list(rows)
        random.Random(seed).shuffle(shuffled)
        midpoint = len(shuffled) // 2
        return {
            "train": shuffled[:midpoint],
            "validation": [],
            "test": shuffled[midpoint:],
        }
    if mode != "seq-hash-70-15-15":
        raise ValueError(f"unsupported split mode: {mode}")
    groups: dict[int, list[Observation]] = {}
    for row in rows:
        groups.setdefault(row.input_len, []).append(row)
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


def formula_text(coefficients: Sequence[float]) -> str:
    terms: list[str] = []
    for index, coefficient in enumerate(coefficients):
        if abs(coefficient) < 1e-14:
            continue
        magnitude = f"{abs(coefficient):.15g}"
        expression = FEATURE_NAMES[index]
        term = magnitude if expression == "1" else f"{magnitude} * {expression}"
        if not terms:
            terms.append(("-" if coefficient < 0 else "") + term)
        else:
            terms.append((" - " if coefficient < 0 else " + ") + term)
    return "".join(terms) if terms else "0"


def write_fit_gap_svg(predictions: Sequence[dict[str, Any]], path: pathlib.Path) -> None:
    """Write an all-point measured-vs-predicted and absolute-error chart."""
    if not predictions:
        return
    width, height = 1800, 860
    panel_width, panel_height = 670.0, 590.0
    left_x, right_x, top = 120.0, 1010.0, 150.0
    targets = [float(row["target_ms"]) for row in predictions]
    estimates = [float(row["predicted_ms"]) for row in predictions]
    errors = [
        abs(estimate - target) for estimate, target in zip(estimates, targets)
    ]
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
            'sans-serif;fill:#172033}.title{font-size:32px;font-weight:700}'
            '.sub{font-size:17px;fill:#475569}.panel{font-size:22px;font-weight:700}'
            '.axis{font-size:17px;fill:#334155}.tick{font-size:14px;fill:#64748b}'
            '.legend{font-size:15px;fill:#334155}</style>'
        ),
        (
            '<text x="900" y="48" text-anchor="middle" class="title">'
            'DeepSeek-V4-Pro：实测 TTFT 与拟合误差</text>'
        ),
        (
            f'<text x="900" y="82" text-anchor="middle" class="sub">'
            f'{len(predictions):,} 个严格有效 geometry；'
            '每个点取 3 次成功请求的 TTFT 中位数</text>'
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
    paths = [pathlib.Path(value) for value in args.inputs]
    rows, audit = load_observations(paths, batch_size=args.batch_size)
    output = pathlib.Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
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
        }
        (output / "fit_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False))
        return 2

    splits = split_rows(rows, mode=args.split_mode, seed=args.split_seed)
    fit_rows = splits["train"] if len(splits["train"]) >= len(FEATURE_NAMES) else rows
    coefficients, backend = fit_coefficients(
        fit_rows, objective=args.objective, seed=args.split_seed
    )
    formula = formula_text(coefficients)
    metrics = {
        name: error_metrics(group, coefficients) for name, group in splits.items()
    }
    metrics["all"] = error_metrics(rows, coefficients)
    split_by_geometry = {
        (row.batch_size, row.input_len, row.cache_len): name
        for name, group in splits.items()
        for row in group
    }
    predictions = []
    for row in rows:
        predicted = predict(coefficients, row)
        predictions.append(
            {
                "batch_size": row.batch_size,
                "input_len": row.input_len,
                "requested_cache_len": row.requested_cache_len,
                "cache_len": row.cache_len,
                "compute_len": row.compute_len,
                "target_ms": row.target_ms,
                "predicted_ms": predicted,
                "signed_error_ms": predicted - row.target_ms,
                "abs_error_ms": abs(predicted - row.target_ms),
                "ape_pct": 100.0 * abs(predicted - row.target_ms) / row.target_ms,
                "split": split_by_geometry[
                    (row.batch_size, row.input_len, row.cache_len)
                ],
                "source": row.source,
            }
        )
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(predictions[0]) if predictions else ["input_len"]
        )
        writer.writeheader()
        writer.writerows(predictions)
    write_fit_gap_svg(predictions, output / "fit_gap.svg")
    report = {
        "schema_version": 1,
        "model": "DeepSeek-V4-Pro",
        "backend": backend,
        "objective": {
            "mae": "mean_absolute_error",
            "mse": "mean_squared_error",
            "hybrid": (
                "0.5*MAE/median(train_target_ms) + "
                "0.5*mean_absolute_percentage_error"
            ),
        }[args.objective],
        "target": (
            "median of successful client TTFT runs; falls back to "
            "server prefill_time_ms only for legacy input"
        ),
        "formula": formula,
        "formula_compatibility": {
            "parser": "org.flexlb.balance.strategy.PrefillTimeFormula",
            "variables": ["computeTokens", "hitCacheTokens"],
            "functions": ["sum"],
            "operators": ["+", "-", "*", "/", "(", ")"],
            "unsupported_constructs_used": [],
        },
        "coefficients": [
            {"expression": name, "coefficient": value}
            for name, value in zip(FEATURE_NAMES, coefficients)
        ],
        "audit": audit,
        "split": {
            "mode": args.split_mode,
            "seed": args.split_seed,
            "train_fraction": 0.5 if args.split_mode == "random-50-50" else 0.70,
            "test_fraction": 0.5 if args.split_mode == "random-50-50" else 0.15,
        },
        "split_counts": {name: len(group) for name, group in splits.items()},
        "metrics": metrics,
        "production_acceptance": bool(
            len(rows) >= args.min_valid_rows
            and len(splits["test"]) > 0
            and metrics["test"]["mape_pct"] is not None
            and metrics["test"]["mape_pct"] <= args.max_mape_pct
            and metrics["test"]["p95_ape_pct"] is not None
            and metrics["test"]["p95_ape_pct"] <= args.max_p95_ape_pct
            and metrics["test"]["max_ape_pct"] is not None
            and metrics["test"]["max_ape_pct"] <= args.max_max_ape_pct
        ),
        "production_note": (
            "Only rows whose requested and observed reuse match exactly are "
            "included. Failed requests and invalid_reuse rows are excluded. "
            "Validate the latency measurement contract, tail error, and "
            "deployment range before production use."
        ),
    }
    (output / "fit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "deepseek_v4_prefill_formula.txt").write_text(
        "PREFILL_TIME_FORMULA=" + formula + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["production_acceptance"] else 3


def run_validate(args: argparse.Namespace) -> int:
    rows, audit = load_observations(
        [pathlib.Path(value) for value in args.inputs], batch_size=args.batch_size
    )
    report = {"model": "DeepSeek-V4-Pro", "audit": audit, "valid": bool(rows)}
    if args.report:
        pathlib.Path(args.report).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if rows else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="fit from successful DSV4 measurements")
    fit.add_argument("--inputs", nargs="+", required=True)
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--batch-size", type=int, default=1)
    fit.add_argument("--min-valid-rows", type=int, default=30)
    fit.add_argument("--max-mape-pct", type=float, default=5.0)
    fit.add_argument("--max-p95-ape-pct", type=float, default=10.0)
    fit.add_argument("--max-max-ape-pct", type=float, default=40.0)
    fit.add_argument(
        "--objective",
        choices=("mae", "mse", "hybrid"),
        default="hybrid",
        help="fit objective; hybrid balances normalized absolute and relative error",
    )
    fit.add_argument(
        "--split-mode",
        choices=("random-50-50", "seq-hash-70-15-15"),
        default="random-50-50",
        help="default randomly assigns exactly half of valid geometries to training",
    )
    fit.add_argument("--split-seed", type=int, default=20260904)
    fit.add_argument("--allow-insufficient-data", action="store_true")
    fit.set_defaults(func=run_fit)
    validate = sub.add_parser("validate-inputs", help="audit valid/invalid DSV4 rows")
    validate.add_argument("--inputs", nargs="+", required=True)
    validate.add_argument("--batch-size", type=int, default=1)
    validate.add_argument("--report")
    validate.set_defaults(func=run_validate)
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    raise SystemExit(parsed_args.func(parsed_args))
