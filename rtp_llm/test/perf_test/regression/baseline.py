"""Strict baseline schemas, comparisons and candidate generation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

from rtp_llm.test.perf_test.plan import case_id
from rtp_llm.test.perf_test.results import parse_result_envelope
from rtp_llm.test.perf_test.storage import atomic_write_json

BASELINE_SCHEMA_VERSION = 1

_DIRECTIONS = {"higher_is_better", "lower_is_better"}

_CASE_KEY = re.compile(r"(?:prefill|decode)-[0-9a-f]{24}\Z")

_METRIC = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z")


class BaselineSchemaError(ValueError):
    """Raised when a baseline does not match the explicit versioned schema."""


@dataclass(frozen=True)
class BaselineEntry:
    case_key: str
    metric: str
    value: float
    direction: str
    threshold: float


def _finite_number(path: str, value: Any, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineSchemaError(f"{path}: expected a finite non-negative number")
    parsed = float(value)
    if (
        not math.isfinite(parsed)
        or parsed < 0
        or maximum is not None
        and parsed > maximum
    ):
        raise BaselineSchemaError(f"{path}: expected a finite non-negative number")
    return parsed


def parse_baseline(value: Any) -> Tuple[BaselineEntry, ...]:
    """Parse the baseline schema without accepting legacy baseline layouts."""
    if not isinstance(value, Mapping):
        raise BaselineSchemaError("baseline: expected an object")
    if set(value) != {"schema_version", "entries"}:
        raise BaselineSchemaError(
            "baseline: expected exactly schema_version and entries"
        )
    if value["schema_version"] != BASELINE_SCHEMA_VERSION:
        raise BaselineSchemaError("schema_version: unsupported baseline schema")
    raw_entries = value["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise BaselineSchemaError("entries: expected a non-empty array")

    parsed = []
    seen = set()
    fields = {"case_key", "metric", "value", "direction", "threshold"}
    for index, raw in enumerate(raw_entries):
        path = f"entries[{index}]"
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise BaselineSchemaError(f"{path}: expected exactly {sorted(fields)}")
        key = raw["case_key"]
        if not isinstance(key, str) or _CASE_KEY.fullmatch(key) is None:
            raise BaselineSchemaError(f"{path}.case_key: invalid stable case key")
        metric = raw["metric"]
        if not isinstance(metric, str) or _METRIC.fullmatch(metric) is None:
            raise BaselineSchemaError(f"{path}.metric: invalid metric path")
        direction = raw["direction"]
        if direction not in _DIRECTIONS:
            raise BaselineSchemaError(
                f"{path}.direction: expected higher_is_better or lower_is_better"
            )
        identity = (key, metric)
        if identity in seen:
            raise BaselineSchemaError(
                f"{path}: duplicate case_key and metric baseline entry"
            )
        seen.add(identity)
        parsed.append(
            BaselineEntry(
                case_key=key,
                metric=metric,
                value=_finite_number(f"{path}.value", raw["value"]),
                direction=direction,
                threshold=_finite_number(
                    f"{path}.threshold", raw["threshold"], maximum=1.0
                ),
            )
        )
    return tuple(parsed)


def _metric_value(metrics: Mapping[str, Any], metric: str) -> Any:
    current: Any = metrics
    for name in metric.split("."):
        if not isinstance(current, Mapping) or name not in current:
            raise KeyError(metric)
        current = current[name]
    return current


def _diagnostic(entry: BaselineEntry, actual: float, limit: float) -> str:
    relation = "at least" if entry.direction == "higher_is_better" else "at most"
    return (
        f"{entry.case_key} metric {entry.metric}: actual {actual:g}; "
        f"expected {relation} {limit:g} from baseline {entry.value:g} "
        f"with threshold {entry.threshold:g}"
    )


def compare_result_to_baseline(
    result: Mapping[str, Any], baseline: Mapping[str, Any]
) -> Dict[str, Any]:
    """Compare a completed result with every required baseline entry."""
    entries = parse_baseline(baseline)
    envelope = parse_result_envelope(result)
    if envelope.status.value != "completed":
        return {
            "schema_version": BASELINE_SCHEMA_VERSION,
            "passed": False,
            "status": "ineligible",
            "comparisons": [],
            "diagnostics": [
                f"result status {envelope.status.value} is not eligible for baseline regression"
            ],
        }

    cases = {}
    for item in envelope.cases:
        expected_key = case_id(envelope.mode, item.identity)
        if item.case_id != expected_key:
            return {
                "schema_version": BASELINE_SCHEMA_VERSION,
                "passed": False,
                "status": "invalid_result",
                "comparisons": [],
                "diagnostics": [
                    f"stable case key mismatch: result has {item.case_id}, expected {expected_key}"
                ],
            }
        if item.case_id in cases:
            return {
                "schema_version": BASELINE_SCHEMA_VERSION,
                "passed": False,
                "status": "invalid_result",
                "comparisons": [],
                "diagnostics": [f"duplicate result case key: {item.case_id}"],
            }
        cases[item.case_id] = item

    comparisons = []
    diagnostics = []
    for entry in entries:
        item = cases.get(entry.case_key)
        if item is None:
            message = f"required case key {entry.case_key} is missing for metric {entry.metric}"
            comparisons.append(
                {
                    "case_key": entry.case_key,
                    "metric": entry.metric,
                    "expected": entry.value,
                    "actual": None,
                    "direction": entry.direction,
                    "threshold": entry.threshold,
                    "limit": None,
                    "status": "missing_case",
                    "diagnostic": message,
                }
            )
            diagnostics.append(message)
            continue
        metrics = None if item.metrics is None else item.metrics.to_dict()
        try:
            actual_value = _metric_value(metrics or {}, entry.metric)
        except KeyError:
            message = (
                f"{entry.case_key} required metric {entry.metric} is missing "
                f"from case status {item.status.value}"
            )
            comparisons.append(
                {
                    "case_key": entry.case_key,
                    "metric": entry.metric,
                    "expected": entry.value,
                    "actual": None,
                    "direction": entry.direction,
                    "threshold": entry.threshold,
                    "limit": None,
                    "status": "missing_metric",
                    "diagnostic": message,
                }
            )
            diagnostics.append(message)
            continue

        try:
            actual = _finite_number("actual metric", actual_value)
        except BaselineSchemaError:
            message = (
                f"{entry.case_key} metric {entry.metric} is not a finite non-negative number: "
                f"{actual_value!r}"
            )
            comparisons.append(
                {
                    "case_key": entry.case_key,
                    "metric": entry.metric,
                    "expected": entry.value,
                    "actual": actual_value,
                    "direction": entry.direction,
                    "threshold": entry.threshold,
                    "limit": None,
                    "status": "invalid_metric",
                    "diagnostic": message,
                }
            )
            diagnostics.append(message)
            continue

        if entry.direction == "higher_is_better":
            limit = entry.value * (1.0 - entry.threshold)
            passed = actual >= limit
        else:
            limit = entry.value * (1.0 + entry.threshold)
            passed = actual <= limit
        message = None if passed else _diagnostic(entry, actual, limit)
        comparisons.append(
            {
                "case_key": entry.case_key,
                "metric": entry.metric,
                "expected": entry.value,
                "actual": actual,
                "direction": entry.direction,
                "threshold": entry.threshold,
                "limit": limit,
                "status": "passed" if passed else "regressed",
                "diagnostic": message,
            }
        )
        if message is not None:
            diagnostics.append(message)

    passed = not diagnostics
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "passed": passed,
        "status": "passed" if passed else "failed",
        "comparisons": comparisons,
        "diagnostics": diagnostics,
    }


_METRICS = {
    "prefill": (
        ("latency_ms", "lower_is_better"),
        ("throughput_tokens_per_second", "higher_is_better"),
    ),
    "decode": (
        ("latency_ms.wait", "lower_is_better"),
        ("latency_ms.prefill", "lower_is_better"),
        ("latency_ms.decode", "lower_is_better"),
        ("latency_ms.tpot", "lower_is_better"),
        ("latency_ms.total", "lower_is_better"),
        ("throughput_tokens_per_second", "higher_is_better"),
    ),
}


class BaselineCandidateError(ValueError):
    """Raised when a result cannot produce a formal baseline candidate."""


def _metric(metrics: Mapping[str, Any], path: str) -> Any:
    value: Any = metrics
    for name in path.split("."):
        value = value[name]
    return value


def build_baseline_candidate(
    result: Mapping[str, Any], *, threshold: float
) -> Dict[str, Any]:
    """Build a sorted candidate without mutating any accepted baseline."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise BaselineCandidateError("threshold must be a number from zero to one")
    threshold_value = float(threshold)
    if not 0 <= threshold_value <= 1:
        raise BaselineCandidateError("threshold must be a number from zero to one")
    envelope = parse_result_envelope(result)
    if envelope.status.value != "completed":
        raise BaselineCandidateError(
            f"baseline candidate requires completed result, got {envelope.status.value}"
        )

    entries = []
    for item in envelope.cases:
        stable_key = case_id(envelope.mode, item.identity)
        if item.case_id != stable_key:
            raise BaselineCandidateError(
                f"stable case key mismatch: result has {item.case_id}, expected {stable_key}"
            )
        if item.status.value != "ok" or item.metrics is None:
            raise BaselineCandidateError(
                f"baseline candidate requires ok case: {item.case_id}"
            )
        metrics = item.metrics.to_dict()
        for metric, direction in _METRICS[envelope.mode]:
            entries.append(
                {
                    "case_key": stable_key,
                    "metric": metric,
                    "value": _metric(metrics, metric),
                    "direction": direction,
                    "threshold": threshold_value,
                }
            )
    entries.sort(key=lambda entry: (entry["case_key"], entry["metric"]))
    candidate = {"schema_version": 1, "entries": entries}
    parse_baseline(candidate)
    return candidate


def write_baseline_candidate(
    root: Union[str, Path], result: Mapping[str, Any], *, threshold: float
) -> Path:
    """Write only reports/baseline_candidate.json under an initialized run."""
    path = Path(root) / "reports" / "baseline_candidate.json"
    atomic_write_json(path, build_baseline_candidate(result, threshold=threshold))
    return path
