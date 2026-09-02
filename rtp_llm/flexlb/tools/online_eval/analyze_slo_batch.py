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
# Same metric matcher for the consolidated layout, where the prometheus dump
# lives in master.json as a {"name{labels}": value} dict — the sample value is
# the dict value, so only the key needs to match.
PROMETHEUS_DISPATCH_KEY_RE = re.compile(
    r"^flexlb_app_engine_balancing_master_dispatch_reason_total"
    r"\{(?P<labels>[^}]*)\}\s*$"
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
    # Legacy sources win whenever they exist: a successful consolidation
    # deletes them, so a legacy file that is present means fresher data (a
    # RUN_DIR reused for a second run). The consolidated run-root master.log
    # is only the fallback for already-consolidated directories.
    log_dir = run_dir / "flexlb_logs"
    paths = list(log_dir.glob("flexlb.log*")) if log_dir.is_dir() else []
    if paths:
        return sorted(paths, key=lambda path: (path.stat().st_mtime_ns, path.name))
    fallback = run_dir / "flexlb.log"
    if fallback.is_file():
        return [fallback]
    master_log = run_dir / "master.log"
    if master_log.is_file():
        return [master_log]
    return []


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


def load_json(path: Path) -> dict:
    # Defensive loader: a missing file or a truncated/corrupt JSON (e.g. a
    # killed consolidation left a partial file) returns {} so the caller
    # falls back to the next layout instead of crashing.
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


# ---- Input entry points: legacy layout first, consolidated fallback -------
# After consolidate_run_outputs.py the run root carries mock.json / master.json
# / client.json. These loaders prefer the LEGACY files whenever they exist (a
# successful consolidation deletes them, so a legacy file that is present means
# fresher data — a RUN_DIR reused for a second run) and only fall back to the
# consolidated files, so pre-consolidation and post-consolidation run
# directories both stay analyzable. Only the entry points differ; the
# analysis below is untouched.


def load_mock_stats(run_dir: Path) -> list[dict[str, int | float]]:
    legacy = run_dir / "mock_engine.log"
    if legacy.is_file():
        return parse_mock_stats(legacy)
    mock_stats = load_json(run_dir / "mock.json").get("stats")
    if isinstance(mock_stats, list):
        return [row for row in mock_stats if isinstance(row, dict)]
    return []


def load_prometheus_dispatch_counts(run_dir: Path) -> dict[str, int]:
    legacy = run_dir / "master_prometheus_after.prom"
    if legacy.is_file():
        return parse_prometheus_dispatch_counts(legacy)
    prometheus = load_json(run_dir / "master.json").get("prometheus_after")
    if isinstance(prometheus, dict) and prometheus:
        counts: Counter[str] = Counter()
        for key, value in prometheus.items():
            metric = PROMETHEUS_DISPATCH_KEY_RE.match(key)
            if not metric:
                continue
            reason = PROMETHEUS_REASON_RE.search(metric.group("labels"))
            if reason:
                counts[reason.group("reason")] += round(float(value))
        return dict(sorted(counts.items()))
    return {}


def load_client_summary(run_dir: Path) -> dict:
    # Phase B removed load_client/summary.json (the client records raw rows
    # only); client.json is the sole source (no-backward-compat).
    return load_json(run_dir / "client.json")


def load_server_latency(run_dir: Path) -> dict:
    # server_latency.json is kept in place by consolidation, so the legacy
    # path stays the primary source; client.json's merged copy is fallback.
    legacy = load_json(run_dir / "load_client" / "server_latency.json")
    if legacy:
        return legacy
    server_latency = load_json(run_dir / "client.json").get("server_latency")
    if isinstance(server_latency, dict):
        return server_latency
    return {}


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


def load_flexlb_config(path: Path | None) -> dict:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    envs = (
        payload.get("zone_process_setting", {}).get("process_info", {}).get("envs", [])
    )
    process_env = {
        str(item[0]): str(item[1])
        for item in envs
        if isinstance(item, list) and len(item) == 2
    }
    document = process_env.get("FLEXLB_CONFIG")
    return json.loads(document) if document else {}


def analyze(run_dir: Path, master_config: Path | None) -> dict:
    decisions: list[dict] = []
    completions: list[dict] = []
    log_paths = flexlb_log_paths(run_dir)
    for log_path in log_paths:
        path_decisions, path_completions = parse_log(log_path)
        decisions.extend(path_decisions)
        completions.extend(path_completions)
    mock_stats = load_mock_stats(run_dir)
    prometheus_reasons = load_prometheus_dispatch_counts(run_dir)
    flexlb_config = load_flexlb_config(master_config)
    summary = load_client_summary(run_dir)
    server_latency = load_server_latency(run_dir)

    violation_count = 0
    violations: list[dict] = []
    for decision in decisions:
        reason = decision["reason"]
        invalid = (
            (
                # The cap only admits a member that keeps the group under
                # budget, so a multi-member group must stay below it. The
                # mandatory head is exempt: it dispatches alone at any cost.
                reason == "predicted_execution_cap"
                and decision["batch_size"] > 1
                and decision["predicted_ms"] >= decision["threshold_ms"]
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
    estimated_latency = [item["wait_ms"] + item["predicted_ms"] for item in decisions]
    first_decision = decisions[0] if decisions else {}
    scheduler = flexlb_config.get("scheduler", {})
    dispatcher = flexlb_config.get("dispatcher", {})
    log_reasons = dict(sorted(Counter(item["reason"] for item in decisions).items()))
    exact_decision_count = sum(prometheus_reasons.values())
    decision_count = exact_decision_count or len(decisions)

    return {
        "run_dir": str(run_dir),
        "flexlb_logs": [str(path) for path in log_paths],
        "config": {
            "predict_threshold_ms": first_decision.get("threshold_ms", 0),
            "fixed_wait_ms": first_decision.get("fixed_wait_ms", 0),
            "batch_size_max": first_decision.get("batch_size_max", 0),
            "scheduler_type": scheduler.get("type"),
            "ordering_type": scheduler.get("ordering", {}).get("type"),
            "dispatcher_type": dispatcher.get("type"),
        },
        "master": {
            "actual_send_qps": summary.get(
                "actual_send_qps", summary.get("send_qps", 0.0)
            ),
            "arrival_qps": server_latency.get(
                "arrival_qps", summary.get("server_arrival_qps", 0.0)
            ),
            "completion_qps": server_latency.get("completion_qps", 0.0),
            "error_count": summary.get("error_count", summary.get("errors", 0)),
            "test_valid": summary.get("test_valid"),
            "validity_checks": summary.get("validity_checks", {}),
            "client_pacing_lag_ms": summary.get("client_pacing_lag_ms", {}),
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
            "estimated_wait_plus_prefill_ms": distribution(estimated_latency),
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
            # Fall back to the legacy field names (prefill_pending /
            # max_prefill_pending) so logs produced by older mock engines
            # remain analyzable.
            "max_observed_prefill_waiting": max(
                (
                    item.get("prefill_waiting", item.get("prefill_pending", 0))
                    for item in mock_stats
                ),
                default=0,
            ),
            "max_observed_engine_prefill_waiting": max(
                (
                    item.get("max_prefill_waiting", item.get("max_prefill_pending", 0))
                    for item in mock_stats
                ),
                default=0,
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
