#!/usr/bin/env python3
"""Summarize and validate fixed-window batch decisions from one eval run."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

DISPATCH_RE = re.compile(
    r"flexlb_batch_dispatch batch_id=(?P<batch_id>\d+) "
    r"reason=(?P<reason>\S+) batch_size=(?P<batch_size>\d+) "
    r"wait_ms=(?P<wait_ms>\d+) predicted_ms=(?P<predicted_ms>\d+) "
    r"threshold_ms=(?P<threshold_ms>\d+) fixed_wait_ms=(?P<fixed_wait_ms>\d+) "
    r"batch_size_max=(?P<batch_size_max>\d+) queue_after=(?P<queue_after>\d+) "
    r"worker=(?P<worker>\S*)"
)
COMPLETE_RE = re.compile(
    r"flexlb_batch_complete batch_id=(?P<batch_id>\d+) "
    r"predicted_ms=(?P<predicted_ms>-?\d+) actual_ms=(?P<actual_ms>-?\d+) "
    r"gap_ms=(?P<gap_ms>-?\d+) batch_size=(?P<batch_size>\d+) "
    r"engine=(?P<engine>\S+)"
)
MOCK_STAT_RE = re.compile(r"([a-z_]+)=(-?\d+(?:\.\d+)?)")
PROMETHEUS_DISPATCH_RE = re.compile(
    r"^flexlb_app_engine_balancing_master_dispatch_reason_total"
    r"\{(?P<labels>[^}]*)\}\s+(?P<value>[-+\deE.]+)\s*$"
)
PROMETHEUS_REASON_RE = re.compile(r'(?:^|,)reason="(?P<reason>[^"]+)"(?:,|$)')

INT_FIELDS = {
    "batch_id",
    "batch_size",
    "wait_ms",
    "predicted_ms",
    "threshold_ms",
    "fixed_wait_ms",
    "batch_size_max",
    "queue_after",
    "actual_ms",
    "gap_ms",
}


def _record(match: re.Match[str]) -> dict[str, int | str]:
    return {
        key: int(value) if key in INT_FIELDS else value
        for key, value in match.groupdict().items()
    }


def flexlb_log_paths(run_dir: Path) -> list[Path]:
    log_dir = run_dir / "flexlb_logs"
    paths = list(log_dir.glob("flexlb.log*")) if log_dir.is_dir() else []
    if paths:
        return sorted(paths, key=lambda path: (path.stat().st_mtime_ns, path.name))
    fallback = run_dir / "flexlb.log"
    return [fallback] if fallback.is_file() else []


def parse_log(path: Path) -> tuple[list[dict], list[dict]]:
    decisions: list[dict] = []
    completions: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            dispatch = DISPATCH_RE.search(line)
            if dispatch:
                decisions.append(_record(dispatch))
                continue
            complete = COMPLETE_RE.search(line)
            if complete:
                completions.append(_record(complete))
    return decisions, completions


def parse_mock_stats(path: Path) -> list[dict[str, int | float]]:
    if not path.is_file():
        return []
    snapshots: list[dict[str, int | float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if "java_mock_stats " not in line:
                continue
            snapshot: dict[str, int | float] = {}
            for key, raw_value in MOCK_STAT_RE.findall(line):
                snapshot[key] = float(raw_value) if "." in raw_value else int(raw_value)
            if snapshot:
                snapshots.append(snapshot)
    return snapshots


def parse_prometheus_dispatch_counts(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            metric = PROMETHEUS_DISPATCH_RE.match(line.strip())
            if not metric:
                continue
            reason = PROMETHEUS_REASON_RE.search(metric.group("labels"))
            if reason:
                counts[reason.group("reason")] += round(float(metric.group("value")))
    return dict(sorted(counts.items()))


def percentile(sorted_values: list[int], quantile: float) -> int:
    if not sorted_values:
        return 0
    index = max(0, math.ceil(quantile * len(sorted_values)) - 1)
    return sorted_values[index]


def distribution(values: Iterable[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
        }
    return {
        "count": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 3),
        "p50": percentile(ordered, 0.50),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def load_process_config(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    envs = (
        payload.get("zone_process_setting", {}).get("process_info", {}).get("envs", [])
    )
    return {
        str(item[0]): str(item[1])
        for item in envs
        if isinstance(item, list) and len(item) == 2
    }


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(run_dir: Path, master_config: Path | None) -> dict:
    decisions: list[dict] = []
    completions: list[dict] = []
    log_paths = flexlb_log_paths(run_dir)
    for log_path in log_paths:
        path_decisions, path_completions = parse_log(log_path)
        decisions.extend(path_decisions)
        completions.extend(path_completions)
    mock_stats = parse_mock_stats(run_dir / "mock_engine.log")
    prometheus_reasons = parse_prometheus_dispatch_counts(
        run_dir / "master_prometheus_after.prom"
    )
    process_env = load_process_config(master_config)
    summary = load_json(run_dir / "load_client" / "summary.json")
    server_latency = load_json(run_dir / "load_client" / "server_latency.json")

    violation_count = 0
    violations: list[dict] = []
    for decision in decisions:
        reason = decision["reason"]
        invalid = (
            (
                reason == "predict_threshold"
                and decision["predicted_ms"] < decision["threshold_ms"]
            )
            or (
                reason == "fixed_window_timeout"
                and decision["wait_ms"] + 2 < decision["fixed_wait_ms"]
            )
            or (
                reason == "batch_full"
                and decision["batch_size"] < decision["batch_size_max"]
            )
        )
        if invalid:
            violation_count += 1
            if len(violations) < 20:
                violations.append(decision)

    completion_by_batch = {item["batch_id"]: item for item in completions}
    matched = sum(1 for item in decisions if item["batch_id"] in completion_by_batch)
    slo_ms = int(process_env.get("COST_SLO_MS", "0") or 0)
    estimated_budget = [item["wait_ms"] + item["predicted_ms"] for item in decisions]
    first_decision = decisions[0] if decisions else {}
    log_reasons = dict(sorted(Counter(item["reason"] for item in decisions).items()))
    exact_decision_count = sum(prometheus_reasons.values())
    decision_count = exact_decision_count or len(decisions)

    return {
        "run_dir": str(run_dir),
        "flexlb_logs": [str(path) for path in log_paths],
        "config": {
            "algorithm": process_env.get("FLEXLB_BATCH_ALGORITHM"),
            "predict_threshold_ms": first_decision.get("threshold_ms", 0),
            "fixed_wait_ms": first_decision.get("fixed_wait_ms", 0),
            "batch_size_max": first_decision.get("batch_size_max", 0),
            "cost_slo_ms": slo_ms,
        },
        "master": {
            "actual_send_qps": summary.get(
                "actual_send_qps", summary.get("send_qps", 0.0)
            ),
            "arrival_qps": server_latency.get(
                "arrival_qps", summary.get("server_arrival_qps", 0.0)
            ),
            "completion_qps": server_latency.get(
                "completion_qps", summary.get("server_completion_qps", 0.0)
            ),
            "error_count": summary.get("error_count", summary.get("errors", 0)),
            "test_valid": summary.get("test_valid"),
            "validity_checks": summary.get("validity_checks", {}),
            "client_pacing_lag_ms": summary.get("client_pacing_lag_ms", {}),
            "client_send_peak_qps": summary.get("client_send_peak_qps", {}),
            "trace_due_peak_qps": summary.get("trace_due_peak_qps", {}),
            "schedule_latency_ms": summary.get("schedule_latency_ms", {}),
        },
        "decisions": {
            "count": decision_count,
            "source": "prometheus_counter" if prometheus_reasons else "structured_log",
            "reasons": prometheus_reasons or log_reasons,
            "log_count": len(decisions),
            "log_reasons": log_reasons,
            "log_coverage_ratio": (
                round(len(decisions) / decision_count, 6) if decision_count else 0.0
            ),
            "distribution_source": "structured_log",
            "batch_size": distribution(item["batch_size"] for item in decisions),
            "wait_ms": distribution(item["wait_ms"] for item in decisions),
            "predicted_ms": distribution(item["predicted_ms"] for item in decisions),
            "estimated_wait_plus_prefill_ms": distribution(estimated_budget),
            "estimated_over_cost_slo_count": (
                sum(value > slo_ms for value in estimated_budget) if slo_ms > 0 else 0
            ),
            "invariant_violation_count": violation_count,
            "invariant_violation_samples": violations,
        },
        "completions": {
            "count": len(completions),
            "matched_decision_count": matched,
            "actual_ms": distribution(item["actual_ms"] for item in completions),
            "prediction_gap_ms": distribution(item["gap_ms"] for item in completions),
        },
        "mock": {
            "stats_samples": len(mock_stats),
            "last": mock_stats[-1] if mock_stats else {},
            "max_observed_prefill_pending": max(
                (item.get("prefill_pending", 0) for item in mock_stats), default=0
            ),
            "max_observed_engine_prefill_pending": max(
                (item.get("max_prefill_pending", 0) for item in mock_stats), default=0
            ),
            "max_observed_heap_used_mb": max(
                (item.get("heap_used_mb", 0) for item in mock_stats), default=0
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--master-config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = analyze(args.run_dir, args.master_config)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
