"""Markdown report generation for FlexLB online evaluation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def write_markdown_report(
    *,
    summary: Mapping[str, object],
    results: Sequence[Mapping[str, object]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FlexLB Online Evaluation Report",
        "",
        "## Overview",
        "",
        f"- Trace: `{summary.get('trace', '')}`",
        f"- Total requests: {summary.get('total_requests', 0)}",
        f"- Scheduled: {summary.get('scheduled', 0)}",
        f"- Completed: {summary.get('completed', 0)}",
        f"- Errors: {summary.get('errors', 0)}",
        f"- Offered QPS: {summary.get('offered_qps', 0)}",
        f"- Completed QPS: {summary.get('completed_qps', 0)}",
        f"- SLA TTFT: {summary.get('sla_ttft_ms', 0)} ms",
        f"- SLA violations: {summary.get('sla_violations', 0)} ({summary.get('sla_violation_rate', 0)})",
        "",
        "## Latency",
        "",
        _latency_table(
            [
                ("Schedule", _mapping(summary.get("schedule_latency_ms"))),
                ("TTFT", _mapping(summary.get("ttft_ms"))),
                ("Total", _mapping(summary.get("total_ms"))),
            ]
        ),
        "",
        "## Load Balance",
        "",
        _balance_section("Prefill", _mapping(summary.get("prefill_balance"))),
        "",
        _balance_section("Decode", _mapping(summary.get("decode_balance"))),
        "",
        "## Status Counts",
        "",
        _kv_table(_mapping(summary.get("status_counts"))),
        "",
        "## Top Errors",
        "",
        _error_table(results),
        "",
        "## Capacity Reading",
        "",
        "- Use completed QPS as the throughput signal.",
        "- A healthy point requires completed QPS to track offered QPS while TTFT p99 stays under the SLA.",
        "- Treat rising timeout/error rate or skewed prefill/decode distribution as the knee of the curve.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _latency_table(rows: Iterable[tuple[str, Mapping[str, object]]]) -> str:
    lines = [
        "| Metric | Count | P50 | P90 | P95 | P99 | Max | Mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in rows:
        lines.append(
            "| {name} | {count} | {p50} | {p90} | {p95} | {p99} | {maxv} | {mean} |".format(
                name=name,
                count=row.get("count", 0),
                p50=row.get("p50", 0),
                p90=row.get("p90", 0),
                p95=row.get("p95", 0),
                p99=row.get("p99", 0),
                maxv=row.get("max", 0),
                mean=row.get("mean", 0),
            )
        )
    return "\n".join(lines)


def _balance_section(name: str, balance: Mapping[str, object]) -> str:
    counts = _mapping(balance.get("counts"))
    lines = [
        f"### {name}",
        "",
        f"- Stddev: {balance.get('stddev', 0)}",
        f"- Max over avg: {balance.get('max_over_avg', 0)}",
        "",
        _kv_table(counts),
    ]
    return "\n".join(lines)


def _kv_table(values: Mapping[str, object]) -> str:
    if not values:
        return "_empty_"
    lines = ["| Key | Value |", "|---|---:|"]
    for key, value in sorted(values.items(), key=lambda item: str(item[0])):
        lines.append(f"| `{key}` | {value} |")
    return "\n".join(lines)


def _error_table(results: Sequence[Mapping[str, object]]) -> str:
    errors = Counter(
        str(row.get("error") or row.get("status") or "unknown")
        for row in results
        if str(row.get("status", "")) not in ("ok", "scheduled")
    )
    if not errors:
        return "_none_"
    lines = ["| Error | Count |", "|---|---:|"]
    for error, count in errors.most_common(10):
        compact = error.replace("\n", " ")[:240]
        lines.append(f"| `{compact}` | {count} |")
    return "\n".join(lines)


def load_summary(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
