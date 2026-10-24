"""Validation and shared readers for the current cache-grid result schema."""

import math
from collections import defaultdict
from statistics import median
from typing import Any


class MetricFormatError(ValueError):
    """A metric cannot be represented as a single-request observation."""


SERVER_LATENCY_FIELD = "prefill_time_ms"
SERVER_LATENCY_CONTRACT = "server_first_token_cost_time_ms"


RESULT_SCHEMA_VERSION = 2
CURRENT_STATUSES = {
    "ok",
    "invalid_reuse",
    "invalid_shape",
    "invalid_timing",
    "failed",
    "error",
}


def current_metrics(data, context="result"):
    """Require the current result envelope and per-request measurements."""
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != RESULT_SCHEMA_VERSION
        or data.get("mode") != "prefix_cache_grid"
        or not isinstance(data.get("metrics"), list)
    ):
        raise ValueError(
            f"{context}: expected cache-grid result schema v2 "
            "(mode=prefix_cache_grid, metrics[])"
        )
    for index, item in enumerate(data["metrics"]):
        if (
            not isinstance(item, dict)
            or item.get("status") not in CURRENT_STATUSES
            or "batch_size" not in item
            or "input_len" not in item
        ):
            raise ValueError(f"{context}: metric {index} has an unsupported format")
        if item["status"] == "error":
            continue
        if (
            not isinstance(item.get("runs"), list)
            or "measure_runs" not in item
            or "success_runs" not in item
        ):
            raise ValueError(
                f"{context}: metric {index} requires runs/measure_runs/success_runs"
            )
        if "request_groups" not in item and "cache_len_requested" not in item:
            raise ValueError(f"{context}: metric {index} requires cache_len_requested")
        for round_ in item["runs"]:
            if not isinstance(round_, dict) or (
                "requests" in round_
                and (
                    not isinstance(round_["requests"], list)
                    or any(not isinstance(r, dict) for r in round_["requests"])
                )
            ):
                raise ValueError(f"{context}: metric {index} requires request objects")
        for run in request_runs(item):
            if type(run.get("success")) is not bool:
                raise ValueError(
                    f"{context}: metric {index} requires explicit request success"
                )
            if run.get("success") is True and SERVER_LATENCY_FIELD not in run:
                raise ValueError(
                    f"{context}: metric {index} requires per-request "
                    "prefill_time_ms (server first_token_cost_time); "
                    "client latency cannot substitute"
                )
    return data["metrics"]


def request_runs(item):
    """Iterate actual requests for timing-source auditing in either schema."""
    for run in item.get("runs", []):
        if not isinstance(run, dict):
            continue
        if "requests" in run:
            if isinstance(run["requests"], list):
                yield from (r for r in run["requests"] if isinstance(r, dict))
        else:
            yield run


def single_request_metric(item):
    """Unwrap scheduler batch=1 without changing the request latency contract.

    The batch barrier wall time is a client diagnostic, not
    a substitute for the nested request's server first_token_cost_time. Multi-request batches
    cannot be reduced to one input/cache pair and are deliberately rejected.
    The input object is never modified.
    """
    runs = item.get("runs")
    grouped = (
        "request_groups" in item
        or item.get("execution_mode") == "scheduler_fixed_batch"
    )
    if isinstance(runs, list):
        grouped = grouped or any(isinstance(r, dict) and "requests" in r for r in runs)
    if not grouped:
        return item
    if item.get("batch_size") != 1:
        raise MetricFormatError("unsupported_grouped_batch")
    groups = item.get("request_groups")
    if (
        not isinstance(groups, list)
        or len(groups) != 1
        or not isinstance(groups[0], dict)
    ):
        raise MetricFormatError("invalid_request_groups")
    group = groups[0]
    inp, cache = group.get("input_len"), group.get("cache_len")
    if (
        group.get("count") != 1
        or type(inp) is not int
        or type(cache) is not int
        or not 0 <= cache < inp
    ):
        raise MetricFormatError("invalid_request_groups")
    for key, expected in (
        ("input_len", inp),
        ("cache_len", cache),
        ("cache_len_requested", cache),
    ):
        if key in item and item[key] != expected:
            raise MetricFormatError("request_shape_mismatch")
    if not isinstance(runs, list) or not runs:
        raise MetricFormatError("missing_runs")
    flat = []
    skip_reuse = item.get("reuse_validation_skipped", False)
    for run in runs:
        requests = run.get("requests") if isinstance(run, dict) else None
        if (
            not isinstance(requests, list)
            or len(requests) != 1
            or not isinstance(requests[0], dict)
        ):
            raise MetricFormatError("invalid_grouped_run")
        request = requests[0]
        if run.get("valid") is not True or request.get("success") is not True:
            raise MetricFormatError("run_failed")
        if (
            run.get("completed_requests", 1) != 1
            or request.get("input_len") != inp
            or request.get("output_len") != 1
            or request.get("shape_exact") is False
        ):
            raise MetricFormatError("request_shape_mismatch")
        observed = request.get("reuse_len")
        if type(observed) is not int or not 0 <= observed < inp:
            raise MetricFormatError("invalid_observed_geometry")
        if not skip_reuse and (
            observed != cache or request.get("reuse_exact") is False
        ):
            raise MetricFormatError("requested_reuse_mismatch")
        latency = request.get(SERVER_LATENCY_FIELD)
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency <= 0
            or request.get("timing_valid") is False
        ):
            raise MetricFormatError("invalid_latency")
        flat.append({**request, "run_index": run.get("run_index")})
    if item.get("measure_runs", len(flat)) != len(flat) or item.get(
        "success_runs", len(flat)
    ) != len(flat):
        raise MetricFormatError("incomplete_runs")
    return {
        **item,
        "input_len": inp,
        "cache_len_requested": cache,
        "cache_len_observed": [r["reuse_len"] for r in flat],
        "measure_runs": len(flat),
        "success_runs": len(flat),
        "runs": flat,
    }


def measurement_status(
    success, shape_exact, reuse_exact, timing_valid, *, skip_reuse=False
):
    """Classify failures consistently; skipping reuse never hides shape/timing errors."""
    for valid, status in (
        (success, "failed"),
        (shape_exact, "invalid_shape"),
        (timing_valid, "invalid_timing"),
        (reuse_exact or skip_reuse, "invalid_reuse"),
    ):
        if not valid:
            return status
    return "ok"


SUCCESS_STATUSES = frozenset(("ok",))
DIAGNOSTIC_STATUSES = SUCCESS_STATUSES | {"invalid_reuse"}


def finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(value: Any) -> int | None:
    number = finite_number(value)
    return int(number) if number is not None and number.is_integer() else None


def status_ok(item, allowed=SUCCESS_STATUSES):
    return item.get("status") in allowed


def observed_reuse_values(item, parse=finite_number):
    """Read the canonical per-request observed reuse values."""
    return [
        value
        for run in request_runs(item)
        if run.get("success") is True
        and (value := parse(run.get("reuse_len"))) is not None
    ]


def consistent_value(values: list[float]) -> float | None:
    return values[0] if values and len(set(values)) == 1 else None


def observed_cache_len(item):
    values = [
        finite_number(run.get("reuse_len"))
        for run in request_runs(item)
        if run.get("success") is True
    ]
    return (
        consistent_value(values) if all(value is not None for value in values) else None
    )


def prefill_rt(item):
    values = [
        run_prefill_rt(run) for run in request_runs(item) if run.get("success") is True
    ]
    return (
        median(values)
        if values and all(value is not None for value in values)
        else None
    )


def run_prefill_rt(run):
    value = finite_number(run.get(SERVER_LATENCY_FIELD))
    return value if value is not None and value > 0 else None


def measurement_rows(data, batch_size, *, strict, all_runs=False):
    """Normalize chart geometry with explicit strict or diagnostic filtering.

    Strict plots require complete runs and requested reuse; diagnostic plots
    keep successful observations even when the requested cache was not hit.
    Fitting adds its stronger request-level checks separately.
    """
    metrics = current_metrics(data)
    rows = []
    for item in metrics:
        if not isinstance(item, dict):
            continue
        if integer(item["batch_size"]) != batch_size:
            continue
        try:
            item = single_request_metric(item)
        except MetricFormatError:
            continue
        if not status_ok(item, SUCCESS_STATUSES if strict else DIAGNOSTIC_STATUSES):
            continue
        input_len = finite_number(item["input_len"])
        if strict:
            if item.get("reuse_exact") is False and not item.get(
                "reuse_validation_skipped", False
            ):
                continue
            if item.get("success_runs") is not None:
                try:
                    if int(item["success_runs"]) != int(item["measure_runs"]):
                        continue
                except (TypeError, ValueError):
                    continue
            cache_len = observed_cache_len(item)
            if cache_len != finite_number(item.get("cache_len_requested")):
                continue
        else:
            cache_len = observed_cache_len(item)
        rt = prefill_rt(item)

        runs = item.get("runs") if isinstance(item.get("runs"), list) else []
        if all_runs and runs:
            samples = []
            for index, run in enumerate(runs, 1):
                if not isinstance(run, dict) or not run.get("success") is True:
                    continue
                samples.append(
                    (finite_number(run.get("reuse_len")), run_prefill_rt(run), index)
                )
        else:
            samples = [(cache_len, rt, None)]
        for cache, latency, index in samples:
            if (
                input_len is None
                or cache is None
                or latency is None
                or input_len < 0
                or cache < 0
                or cache > input_len
                or (strict and cache == input_len)
            ):
                continue
            row = {
                "input_len": input_len,
                "cache_len": cache,
                "compute_len": input_len - cache,
                "prefill_rt": latency,
            }
            if index is not None:
                row["run_index"] = index
            rows.append(row)
    if all_runs:
        return rows
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["input_len"], row["cache_len"], row["compute_len"])].append(row)
    return [
        {**values[0], "prefill_rt": median(item["prefill_rt"] for item in values)}
        for _, values in sorted(grouped.items())
    ]
