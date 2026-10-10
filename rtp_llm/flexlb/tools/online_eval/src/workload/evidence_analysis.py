"""Audit evidence and derive workload statistics before report generation."""

import json
import math
from pathlib import Path

from monitoring.metrics import parse_prometheus_samples


def journal_rows(path):
    rows, issues = [], []
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        return [], [str(exc)]
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("epoch_s"), (int, float))
                or not math.isfinite(row["epoch_s"])
            ):
                raise ValueError("invalid sample timestamp")
            if "ended_epoch_s" in row and (
                type(row["ended_epoch_s"]) not in (int, float)
                or not math.isfinite(row["ended_epoch_s"])
                or row["ended_epoch_s"] < row["epoch_s"]
            ):
                raise ValueError("invalid sample completion timestamp")
            rows.append(row)
        except (ValueError, TypeError) as exc:
            issues.append(f"line {number}: {exc}")
    return rows, issues


def audit_journals(directory, expected_sources, max_gap_s=None, windows=None):
    issues = []
    for source in expected_sources:
        epoch, name = source.split("/", 1)
        raw = Path(directory) / "telemetry" / epoch / (name + ".prom")
        journal = (
            raw.with_name("mock-samples.jsonl")
            if name == "mock"
            else Path(str(raw) + ".samples.jsonl")
        )
        rows, errors = journal_rows(journal)
        if not rows:
            errors.append("no sampling rounds")
        if windows is not None:
            window = windows.get(source, {})
            start, end = window.get("started_epoch_s"), window.get("ended_epoch_s")
            if start is None or end is None or end < start:
                errors.append("missing or invalid collector lifetime")
            elif rows and max_gap_s is not None:
                if rows[0]["epoch_s"] - start > max_gap_s:
                    errors.append("sampling began too late for collector lifetime")
                if end - rows[-1]["epoch_s"] > max_gap_s:
                    errors.append("sampling ended too early for collector lifetime")
        previous = None
        for index, row in enumerate(rows, 1):
            if row.get("sequence") != index:
                errors.append(f"sampling sequence discontinuity at round {index}")
            if previous is not None and row["epoch_s"] < previous:
                errors.append("sample clock moved backwards")
            if (
                previous is not None
                and max_gap_s is not None
                and row["epoch_s"] - previous > max_gap_s
            ):
                errors.append(
                    f"sampling gap exceeds {max_gap_s}s: {previous}..{row['epoch_s']}"
                )
            previous = row["epoch_s"]
        try:
            raw_times = [
                int(line[5:])
                for line in raw.read_text().splitlines()
                if line.startswith("# ts=")
            ]
            successful = [
                int(row["epoch_s"] * 1000) for row in rows if row.get("error") is None
            ]
            if raw_times != successful:
                errors.append("raw samples do not match successful journal rounds")
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
        issues.extend(dict(source=source, error=error) for error in errors)
    return issues


def read_series(root, epoch_s, max_gap_s=None):
    series = {}
    for path in sorted(Path(root).glob("telemetry/*/*.prom")):
        timestamp = None
        for line in path.read_text().splitlines():
            if line.startswith("# ts="):
                timestamp = int(line[5:]) / 1000 - epoch_s
                continue
            if timestamp is None:
                continue
            for name, labels, value in parse_prometheus_samples(line, ""):
                if not math.isfinite(value):
                    continue
                key = "/".join(
                    (
                        path.parent.name,
                        path.stem,
                        name,
                        json.dumps(labels, sort_keys=True),
                    )
                )
                series.setdefault(key, []).append([timestamp, value])
    for path in sorted(Path(root).glob("telemetry/*/*samples.jsonl")):
        source = path.name.replace(".prom.samples.jsonl", "").replace(
            "-samples.jsonl", ""
        )
        prefix = f"{path.parent.name}/{source}/"
        gaps = [
            [int(row["epoch_s"] * 1000) / 1000 - epoch_s, None]
            for row in journal_rows(path)[0]
            if row.get("error") is not None
        ]
        for key in series:
            if key.startswith(prefix):
                series[key] = sorted(series[key] + gaps, key=lambda point: point[0])
    if max_gap_s is not None:
        for key, points in series.items():
            gaps = [
                [(left[0] + right[0]) / 2, None]
                for left, right in zip(points, points[1:])
                if right[0] - left[0] > max_gap_s
            ]
            series[key] = sorted(points + gaps, key=lambda point: point[0])
    return series


def classify_gaps(gaps, evidence):
    """Only confirmed injections exempt the named source during that interval."""
    anchor = evidence["clock_anchor"]["epoch_s"]
    teardown = [
        e
        for e in evidence.get("phases", [])
        if e["action"] == "teardown" and e["event"] == "start"
    ]
    expected, unexpected = {}, {}
    for key, times in gaps.items():
        source = "/".join(key.split("/")[:2])
        for timestamp in times:
            epoch = timestamp + anchor
            allowed = any(
                o["source"] == source
                and o["started_epoch_s"] <= epoch <= o["ended_epoch_s"]
                for o in evidence.get("expected_outages", [])
            )
            allowed = allowed or any(
                str(e["env_epoch"]) == key.split("/")[0] and epoch >= e["epoch_s"]
                for e in teardown
            )
            (expected if allowed else unexpected).setdefault(key, []).append(timestamp)
    return expected, unexpected


def collection_gaps(directory, anchor, max_gap_s):
    """Source availability comes from scrape journals, not sparse label series."""
    gaps = {}
    for path in Path(directory).glob("telemetry/*/*samples.jsonl"):
        source = path.name.replace(".prom.samples.jsonl", "").replace(
            "-samples.jsonl", ""
        )
        rows = journal_rows(path)[0]
        times = [int(row["epoch_s"] * 1000) / 1000 - anchor for row in rows]
        # HTTP may begin before an injected kill and fail after it. Attribute
        # failed scrapes to their observed completion, without extending any
        # declared fault window. Old journals retain their original timestamp.
        missing = [
            row.get("ended_epoch_s", row["epoch_s"]) - anchor
            for row in rows
            if row.get("error") is not None
        ]
        if max_gap_s is not None:
            missing.extend(
                (left + right) / 2
                for left, right in zip(times, times[1:])
                if right - left > max_gap_s
            )
        if missing:
            gaps[f"{path.parent.name}/{source}/collection"] = missing
    return gaps


def bounded_collection_gaps(series, gaps, max_gap_s):
    """Tolerate only gaps bracketed by successful scrapes within the budget."""
    tolerated, fatal = {}, {}
    for key, times in gaps.items():
        source = "/".join(key.split("/")[:2])
        healthy = sorted(t for name, points in series.items()
                         if name.startswith(source + "/up/")
                         for t, value in points if value == 1)
        for t in times:
            left = max((x for x in healthy if x < t), default=None)
            right = min((x for x in healthy if x > t), default=None)
            bounded = (left is not None and right is not None
                       and 0 < right - left <= max_gap_s)
            (tolerated if bounded else fatal).setdefault(key, []).append(t)
    return tolerated, fatal


def analyze_report(directory, result, evidence):
    directory = Path(directory)
    from monitoring.session import archived_series

    series, statistic_sources, source_gaps, monitor_errors = archived_series(
        directory, evidence["clock_anchor"]["epoch_s"]
    )
    # Logs, request journals and debug API snapshots are dedicated-test evidence.
    # They cannot silently supply a missing performance curve.
    gaps = {
        key: [t for t, value in points if value is None]
        for key, points in series.items()
        if any(value is None for _, value in points)
    }
    expected_gaps, unexpected_gaps = classify_gaps(source_gaps, evidence)
    result["workload"]["telemetry_gaps"] = gaps
    result["workload"]["collection_gaps"] = source_gaps
    result["workload"]["expected_telemetry_gaps"] = expected_gaps
    result["workload"]["unexpected_telemetry_gaps"] = unexpected_gaps
    result["workload"]["telemetry_completeness"] = (
        "PARTIAL" if source_gaps else "COMPLETE"
    )
    budget = result["workload"].get("runtime_configuration", {}).get("max_sample_gap_s", 0)
    tolerated, fatal = bounded_collection_gaps(series, unexpected_gaps, budget)
    result["workload"]["tolerated_telemetry_gaps"] = tolerated
    result["workload"]["fatal_telemetry_gaps"] = fatal
    result["workload"]["telemetry_warnings"] = [
        dict(source=key, times=times, max_gap_s=budget,
             reason="Short collection gap bracketed by successful scrapes")
        for key, times in tolerated.items()
    ]
    if fatal:
        result["workload"]["runtime_validity"] = "INVALID"
    checks = [
        dict(stage=s["id"], **check) for s in result["stages"] for check in s["checks"]
    ]
    # Query archives retain absent optional curves for diagnosis. Only a
    # failed query or a series explicitly required by the collector invalidates
    # workload evidence; source coverage is checked separately below.
    diagnostics = [error for error in monitor_errors
                   if error.get("severity") == "diagnostic"]
    journal_issues = [error for error in monitor_errors
                      if error.get("severity") != "diagnostic"]
    result["workload"]["telemetry_diagnostics"] = diagnostics
    result["workload"]["telemetry_integrity_errors"] = journal_issues
    if journal_issues or diagnostics:
        result["workload"]["telemetry_completeness"] = "PARTIAL"
    if journal_issues:
        result["workload"]["runtime_validity"] = "INVALID"
    missing = []
    if result.get("workload", {}).get("capture_metrics"):
        present = {"/".join(key.split("/")[:2]) for key in series}
        expected = evidence.get("expected_telemetry", [])
        missing = sorted(set(expected) - present)
        if not expected:
            missing.append("expected telemetry inventory absent")
    if missing:
        result["workload"]["runtime_validity"] = "INVALID"
        result["workload"]["missing_telemetry"] = missing
        result["workload"]["prior_status"] = result["status"]
        result["status"] = "ERROR"
        result["error"] = result["error"] or "missing workload telemetry: " + ", ".join(
            missing
        )
    if result["workload"]["runtime_validity"] == "INVALID" and result["status"] in {
        "PASS",
        "FINDING-CONFIRMED",
        "FINDING-RESOLVED",
    }:
        # A contract or expected-failure detector cannot bless incomplete evidence.
        result["workload"]["prior_status"] = result["status"]
        result["status"] = "ERROR"
        result["error"] = (
            result.get("error")
            or "workload evidence is invalid; inspect workload diagnostics"
        )
    traffic = []
    iterations = []
    for flow_input in sorted(directory.glob("flows/*/flow-input.json")):
        flow = json.loads(flow_input.read_text())
        traffic.append(flow.get("trace", {}))
    payload = dict(
        workload_analysis_schema_version=1,
        traffic_manifests=traffic,
        iterations=iterations,
        configuration=result.get("implementation", {}).get("configuration"),
        implementation=result.get("implementation", {}),
        runtime_provenance=evidence.get("runtime_provenance"),
        id=result["id"],
        status=result["status"],
        workload=result["workload"],
        checks=checks,
        stages=result["stages"],
        clock_anchor=evidence["clock_anchor"],
        phases=evidence["phases"],
        events=evidence.get("events", []),
        series=series,
        statistic_sources=statistic_sources,
        request_sources=[],
        configuration_sha256=result.get("implementation", {}).get(
            "configuration_sha256"
        ),
    )
    from reporting import discover_reports

    payload["gate_reports"] = discover_reports(directory, role="gate")
    payload["gate_report"] = next(iter(payload["gate_reports"]), None)
    return payload
