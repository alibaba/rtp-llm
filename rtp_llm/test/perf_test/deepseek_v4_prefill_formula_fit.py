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

The exported expression uses FlexLB's request-scoped ``inputTokens`` and
``hitCacheTokens`` variables inside ``sum(...)``.  Each fitted BS=1 feature is
therefore evaluated independently for every request and summed, preserving the
observed single-request fit while remaining mathematically defined for any
scheduler batch size.

Production fitting uses the runner's complete per-request client HTTP wall
measurements with ``max_new_tokens=1`` as TTFT.  Aggregate or legacy timing
fields without the corresponding successful run evidence are not auditable.

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
import statistics
from dataclasses import dataclass
from typing import Any, Sequence

FEATURE_NAMES = (
    "sum(1)",
    "sum(inputTokens / 1024.0)",
    "sum(hitCacheTokens / 1024.0)",
    "sum((inputTokens / 1024.0) * (inputTokens / 1024.0))",
    "sum((inputTokens / 1024.0) * (hitCacheTokens / 1024.0))",
    "sum((hitCacheTokens / 1024.0) * (hitCacheTokens / 1024.0))",
)
DSV4_FORMULA_BATCH_SIZE = 1


@dataclass(frozen=True)
class Observation:
    batch_size: int
    input_len: int
    cache_len: int
    target_ms: float
    source: str
    requested_cache_len: int | None = None
    production_audited: bool = True

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


def _first_nonblank(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = item.get(key)
        if value is not None and not (isinstance(value, str) and not value.strip()):
            return value
    return None


def _require_batch_size_one(batch_size: int) -> None:
    if batch_size != DSV4_FORMULA_BATCH_SIZE:
        raise ValueError(
            "DeepSeek-V4 prefill formula fitting requires --batch-size=1; "
            f"got {batch_size}"
        )


def _status_ok(item: dict[str, Any]) -> bool:
    return item.get("status") == "ok"


def _median_run_time(
    item: dict[str, Any], requested_cache_len: int | None = None
) -> tuple[float | None, int | None, str | None]:
    """Validate all evidence used by cache-grid resume, fitting, and charts."""
    if not isinstance(item, dict) or item.get("status") != "ok":
        return None, None, "status_not_ok"
    expected_runs = _integer(item.get("measure_runs"))
    success_runs = _integer(item.get("success_runs"))
    runs = item.get("runs")
    if expected_runs is None or expected_runs <= 0:
        return None, None, "missing_measure_runs"
    if (
        success_runs != expected_runs
        or not isinstance(runs, list)
        or len(runs) != expected_runs
    ):
        return None, None, "incomplete_runs"
    if item.get("reuse_exact") is not True:
        return None, None, "reuse_not_proven"
    if item.get("shape_exact") is not True:
        return None, None, "shape_not_proven"
    if item.get("timing_valid") is not True:
        return None, None, "timing_not_proven"

    input_len = _integer(item.get("input_len"))
    if requested_cache_len is None:
        requested_cache_len = _integer(
            _first_nonblank(item, "cache_len_requested", "cache_len")
        )
    if requested_cache_len is None:
        return None, None, "missing_requested_cache_len"
    if requested_cache_len > 0:
        seed = item.get("seed")
        if not isinstance(seed, dict) or seed.get("success") is not True:
            return None, None, "missing_successful_cache_seed"

    observed = item.get("cache_len_observed")
    if not isinstance(observed, list) or len(observed) != expected_runs:
        return None, None, "missing_observed_reuse"
    observed_values = [_integer(value) for value in observed]
    if any(value is None for value in observed_values) or len(
        set(observed_values)
    ) != 1:
        return None, None, "observed_reuse_not_constant"
    cache_len = observed_values[0]
    assert cache_len is not None
    if cache_len < 0 or input_len is None or cache_len >= input_len:
        return None, None, "invalid_observed_geometry"
    if cache_len != requested_cache_len:
        return None, None, "requested_reuse_mismatch"

    recorded_latencies = item.get("ttft_ms")
    if (
        not isinstance(recorded_latencies, list)
        or len(recorded_latencies) != expected_runs
    ):
        return None, None, "missing_run_timings"
    recorded_values = [_finite(value) for value in recorded_latencies]
    if any(value is None or value <= 0 for value in recorded_values):
        return None, None, "invalid_recorded_timing"

    run_values: list[float] = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict) or run.get("success") is not True:
            return None, None, "run_failed"
        if (
            _integer(run.get("input_len")) != input_len
            or _integer(run.get("output_len")) != 1
        ):
            return None, None, "request_shape_mismatch"
        if _integer(run.get("reuse_len")) != cache_len:
            return None, None, "reuse_mismatch"
        latency = _finite(run.get("ttft_ms"))
        recorded = recorded_values[index]
        if (
            latency is None
            or latency <= 0
            or recorded is None
            or not math.isclose(latency, recorded, rel_tol=1e-12, abs_tol=1e-9)
        ):
            return None, None, "invalid_latency"
        run_values.append(latency)

    expected_median = statistics.median(run_values)
    expected_average = statistics.fmean(run_values)
    median = _finite(item.get("median_ttft_ms"))
    average = _finite(item.get("avg_ttft_ms"))
    if (
        median is None
        or average is None
        or not math.isclose(median, expected_median, rel_tol=1e-12, abs_tol=1e-9)
        or not math.isclose(average, expected_average, rel_tol=1e-12, abs_tol=1e-9)
    ):
        return None, None, "aggregate_timing_mismatch"
    return expected_median, cache_len, None


def _fingerprint_config_sha256(config: Any) -> str | None:
    if not isinstance(config, dict) or not config:
        return None
    canonical = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _normalized_run_signature_sha256(config: Any) -> str | None:
    """Hash stable runtime provenance while excluding only this shard's cases."""
    if not isinstance(config, dict) or not config:
        return None
    normalized = {key: value for key, value in config.items() if key != "cases"}
    if not normalized:
        return None
    canonical = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _load_json_metrics(
    path: pathlib.Path,
) -> tuple[list[Any], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object containing metrics[]")
    metrics = data.get("metrics")
    if not isinstance(metrics, list):
        raise ValueError(f"{path}: expected a JSON object containing metrics[]")

    run_fingerprint = data.get("run_fingerprint")
    fingerprint_config = data.get("fingerprint_config")
    computed_fingerprint = _fingerprint_config_sha256(fingerprint_config)
    normalized_run_signature = _normalized_run_signature_sha256(fingerprint_config)
    total_cases = _integer(data.get("total_cases"))
    completed_cases = _integer(data.get("completed_cases"))
    provenance_checks = {
        "explicit_complete": data.get("complete") is True,
        "runner_mode": data.get("mode") == "prefix_cache_grid",
        "schema_version": (_integer(data.get("schema_version")) or 0) >= 2,
        "run_fingerprint": isinstance(run_fingerprint, str)
        and bool(run_fingerprint.strip()),
        "fingerprint_config": computed_fingerprint is not None,
        "fingerprint_sha256": computed_fingerprint is not None
        and run_fingerprint == computed_fingerprint,
        "normalized_run_signature": normalized_run_signature is not None,
        "case_counts": total_cases is not None
        and total_cases > 0
        and completed_cases == total_cases
        and len(metrics) == total_cases,
    }
    provenance = {
        "production_audited": all(provenance_checks.values()),
        "provenance_checks": provenance_checks,
        "run_fingerprint": run_fingerprint,
        "computed_run_fingerprint": computed_fingerprint,
        "normalized_run_signature": normalized_run_signature,
    }
    return metrics, provenance


def load_observations(
    paths: Sequence[pathlib.Path], *, batch_size: int = 1
) -> tuple[list[Observation], dict[str, Any]]:
    _require_batch_size_one(batch_size)
    observations: list[Observation] = []
    rejected: dict[str, int] = {}
    input_files: list[dict[str, Any]] = []
    normalized_run_signature: str | None = None
    normalized_run_signature_source: pathlib.Path | None = None
    for path in paths:
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            source_count = len(rows)
            for index, item in enumerate(rows, 2):
                raw_batch = _first_nonblank(item, "batch_size")
                batch = 1 if raw_batch is None else _integer(raw_batch)
                if batch != batch_size:
                    rejected["batch_size"] = rejected.get("batch_size", 0) + 1
                    continue
                input_len = _integer(item.get("input_len"))
                cache_len = _integer(
                    _first_nonblank(item, "cache_len_requested", "cache_len")
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
                    or cache_len is None
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
                        production_audited=False,
                    )
                )
            input_files.append(
                {
                    "path": str(path),
                    "rows": source_count,
                    "format": "csv",
                    "production_audited": False,
                }
            )
            continue

        metrics, source_provenance = _load_json_metrics(path)
        source_signature = source_provenance.get("normalized_run_signature")
        fingerprint_valid = source_provenance["provenance_checks"].get(
            "fingerprint_sha256"
        )
        if isinstance(source_signature, str) and fingerprint_valid:
            if normalized_run_signature is None:
                normalized_run_signature = source_signature
                normalized_run_signature_source = path
            elif source_signature != normalized_run_signature:
                raise ValueError(
                    f"{path}: incompatible runtime configuration; normalized run "
                    f"signature differs from {normalized_run_signature_source}"
                )
        source_count = len(metrics)
        source_valid_count = 0
        source_metric_fingerprints_match = True
        for index, item in enumerate(metrics):
            if not isinstance(item, dict):
                rejected["invalid_metric"] = rejected.get("invalid_metric", 0) + 1
                continue
            if not _status_ok(item):
                rejected["status"] = rejected.get("status", 0) + 1
                continue
            batch = _integer(item.get("batch_size"))
            if batch != batch_size:
                rejected["batch_size"] = rejected.get("batch_size", 0) + 1
                continue
            input_len = _integer(item.get("input_len"))
            cache_len = _integer(
                _first_nonblank(item, "cache_len_requested", "cache_len")
            )
            if (
                input_len is None
                or cache_len is None
                or cache_len < 0
                or cache_len >= input_len
            ):
                rejected["invalid_geometry"] = rejected.get("invalid_geometry", 0) + 1
                continue
            target, observed_cache_len, reason = _median_run_time(item, cache_len)
            if target is None:
                rejected[reason or "invalid_run"] = (
                    rejected.get(reason or "invalid_run", 0) + 1
                )
                continue
            assert observed_cache_len is not None
            metric_fingerprint_matches = (
                isinstance(item.get("run_fingerprint"), str)
                and item.get("run_fingerprint")
                == source_provenance["run_fingerprint"]
            )
            source_valid_count += 1
            source_metric_fingerprints_match &= metric_fingerprint_matches
            observations.append(
                Observation(
                    batch,
                    input_len,
                    cache_len,
                    target,
                    f"{path.name}:metrics[{index}]",
                    cache_len,
                    production_audited=bool(
                        source_provenance["production_audited"]
                        and metric_fingerprint_matches
                    ),
                )
            )
        source_provenance["valid_rows"] = source_valid_count
        source_provenance["all_metrics_valid"] = source_valid_count == source_count
        source_provenance["all_metric_fingerprints_match"] = (
            source_metric_fingerprints_match
        )
        source_provenance["production_audited"] = bool(
            source_provenance["production_audited"]
            and source_provenance["all_metrics_valid"]
            and source_metric_fingerprints_match
        )
        input_files.append(
            {
                "path": str(path),
                "rows": source_count,
                "format": "json",
                **source_provenance,
            }
        )

    observations.sort(
        key=lambda row: (row.batch_size, row.input_len, row.cache_len, row.source)
    )
    raw_observation_count = len(observations)
    raw_unaudited_observation_count = sum(
        not row.production_audited for row in observations
    )
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
            production_audited=all(item.production_audited for item in values),
        )
        for key, values in sorted(grouped.items())
    ]
    unique = {(row.batch_size, row.input_len, row.cache_len) for row in observations}
    unaudited_observation_count = sum(
        not row.production_audited for row in observations
    )
    audit = {
        "input_files": input_files,
        "raw_metric_count": sum(int(item["rows"]) for item in input_files),
        "raw_valid_observation_count": raw_observation_count,
        "raw_unaudited_observation_count": raw_unaudited_observation_count,
        "valid_observation_count": len(observations),
        "unaudited_observation_count": unaudited_observation_count,
        "production_audit_passed": bool(observations)
        and unaudited_observation_count == 0
        and bool(input_files)
        and all(item.get("production_audited") is True for item in input_files),
        "collapsed_duplicate_geometry_count": raw_observation_count - len(observations),
        "unique_geometry_count": len(unique),
        "rejected_counts": rejected,
        "selected_batch_size": batch_size,
        "normalized_run_signature": normalized_run_signature,
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
    tokens = row.input_len / 1024.0
    hit = row.cache_len / 1024.0
    return [1.0, tokens, hit, tokens * tokens, tokens * hit, hit * hit]


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


def fit_coefficients(
    rows: Sequence[Observation], *, objective: str = "mae", ridge: float = 1e-8
) -> tuple[list[float], str]:
    if len(rows) < len(FEATURE_NAMES):
        raise ValueError(
            f"need at least {len(FEATURE_NAMES)} valid rows, got {len(rows)}"
        )
    if objective == "mae":
        return fit_lad_coefficients(rows), "python_coordinate_descent_lad"
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


def split_rows(rows: Sequence[Observation]) -> dict[str, list[Observation]]:
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

    splits = split_rows(rows)
    fit_rows = splits["train"] if len(splits["train"]) >= len(FEATURE_NAMES) else rows
    coefficients, backend = fit_coefficients(fit_rows, objective=args.objective)
    formula = formula_text(coefficients)
    metrics = {
        name: error_metrics(group, coefficients) for name, group in splits.items()
    }
    metrics["all"] = error_metrics(rows, coefficients)
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
                "ape_pct": 100.0 * abs(predicted - row.target_ms) / row.target_ms,
                "source": row.source,
            }
        )
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(predictions[0]) if predictions else ["input_len"]
        )
        writer.writeheader()
        writer.writerows(predictions)
    report = {
        "schema_version": 1,
        "model": "DeepSeek-V4-Pro",
        "backend": backend,
        "objective": (
            "mean_absolute_error" if args.objective == "mae" else "mean_squared_error"
        ),
        "target": "median of complete successful client TTFT runs",
        "formula": formula,
        "formula_compatibility": {
            "parser": "org.flexlb.balance.prediction.PrefillTimeFormula",
            "variables": ["inputTokens", "hitCacheTokens"],
            "scope": "sum_of_per_request_bs1_fit_for_arbitrary_scheduler_batches",
            "training_batch_size": DSV4_FORMULA_BATCH_SIZE,
            "batch_semantics": "sum(each request's fitted latency features)",
            "operators": ["+", "-", "*", "/", "(", ")", "sum"],
            "unsupported_constructs_used": [],
        },
        "coefficients": [
            {"expression": name, "coefficient": value}
            for name, value in zip(FEATURE_NAMES, coefficients)
        ],
        "audit": audit,
        "split_counts": {name: len(group) for name, group in splits.items()},
        "metrics": metrics,
        "production_acceptance": bool(
            audit["production_audit_passed"]
            and len(rows) >= args.min_valid_rows
            and len(splits["test"]) > 0
            and metrics["test"]["mape_pct"] is not None
            and metrics["test"]["mape_pct"] <= args.max_mape_pct
            and metrics["test"]["p95_ape_pct"] is not None
            and metrics["test"]["p95_ape_pct"] <= args.max_p95_ape_pct
            and metrics["test"]["max_ape_pct"] is not None
            and metrics["test"]["max_ape_pct"] <= args.max_max_ape_pct
        ),
        "production_note": (
            "CSV observations are unaudited and may be fitted for analysis, "
            "but cannot pass production acceptance. Production requires a complete "
            "fingerprinted runner JSON source and explicit BS=1 rows with successful "
            "shape-valid runs, matching requested/observed reuse, and valid timings."
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
    valid = bool(rows) and audit["production_audit_passed"]
    report = {"model": "DeepSeek-V4-Pro", "audit": audit, "valid": valid}
    if args.report:
        pathlib.Path(args.report).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if valid else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="fit from successful DSV4 measurements")
    fit.add_argument("--inputs", nargs="+", required=True)
    fit.add_argument("--output-dir", required=True)
    fit.add_argument(
        "--batch-size",
        type=int,
        choices=(DSV4_FORMULA_BATCH_SIZE,),
        default=DSV4_FORMULA_BATCH_SIZE,
        help="Fixed at 1 because this formula is trained for single-request prefill",
    )
    fit.add_argument("--min-valid-rows", type=int, default=30)
    fit.add_argument("--max-mape-pct", type=float, default=5.0)
    fit.add_argument("--max-p95-ape-pct", type=float, default=10.0)
    fit.add_argument("--max-max-ape-pct", type=float, default=40.0)
    fit.add_argument(
        "--objective",
        choices=("mae", "mse"),
        default="mae",
        help="fit objective; default minimizes absolute error",
    )
    fit.add_argument("--allow-insufficient-data", action="store_true")
    fit.set_defaults(func=run_fit)
    validate = sub.add_parser("validate-inputs", help="audit valid/invalid DSV4 rows")
    validate.add_argument("--inputs", nargs="+", required=True)
    validate.add_argument(
        "--batch-size",
        type=int,
        choices=(DSV4_FORMULA_BATCH_SIZE,),
        default=DSV4_FORMULA_BATCH_SIZE,
        help="Fixed at 1 because this formula is trained for single-request prefill",
    )
    validate.add_argument("--report")
    validate.set_defaults(func=run_validate)
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    raise SystemExit(parsed_args.func(parsed_args))
