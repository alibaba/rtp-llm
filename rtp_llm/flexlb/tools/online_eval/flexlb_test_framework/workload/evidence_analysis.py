"""Audit evidence and derive workload statistics before report generation."""

import hashlib
import json
import math
from pathlib import Path

from online_eval.metrics import parse_prometheus_samples
from online_eval.playback import iteration_windows


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


def mature_series(aggregates, workload_epoch_s):
    """Reuse the exact derived curves rendered by the mature stress report."""
    series, sources, issues = {}, {}, []
    for entry in aggregates:
        if entry["status"] != "GENERATED":
            continue
        path = Path(entry["path"])
        try:
            aggregate = json.loads(path.read_text())
            rows = [
                json.loads(line)
                for line in (path.parent / "client_events.jsonl")
                .read_text()
                .splitlines()
                if line.strip()
            ]
            times = [
                r["send_start_epoch_ms"] for r in rows if r.get("send_start_epoch_ms")
            ]
            if not times:
                raise ValueError("mature statistics lack their request time origin")
            offset = min(times) / 1000 - workload_epoch_s
            for row in aggregate.get("per_second", []):
                for metric, value in row.items():
                    if metric == "t" or (
                        value is not None and type(value) not in (int, float)
                    ):
                        continue
                    key = f"statistics/{entry['env_epoch']}/per_second/{metric}"
                    series.setdefault(key, []).append([offset + row["t"], value])
                    sources[key] = dict(
                        path=str(path),
                        field="per_second." + metric,
                        time_basis="request send second",
                        window_statistic="mean of per-second derived values; not a pooled percentile",
                    )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(dict(path=str(path), error=str(exc)))
    return series, sources, issues


def analyze_report(directory, result, evidence):
    directory = Path(directory)
    max_gap_s = (
        result.get("workload", {})
        .get("runtime_configuration", {})
        .get("max_sample_gap_s")
    )
    series = read_series(directory, evidence["clock_anchor"]["epoch_s"], max_gap_s)
    derived, statistic_sources, statistic_issues = mature_series(
        result["workload"].get("stress_aggregates", []),
        evidence["clock_anchor"]["epoch_s"],
    )
    series.update(derived)
    from .cache_gate import report_series

    gate_series, gate_sources = report_series(
        directory, evidence["clock_anchor"]["epoch_s"]
    )
    series.update(gate_series)
    statistic_sources.update(gate_sources)
    if statistic_issues:
        result["workload"]["statistics_issues"] = statistic_issues
        result["workload"]["runtime_validity"] = "INVALID"
    gaps = {
        key: [t for t, value in points if value is None]
        for key, points in series.items()
        if any(value is None for _, value in points)
    }
    source_gaps = collection_gaps(
        directory, evidence["clock_anchor"]["epoch_s"], max_gap_s
    )
    expected_gaps, unexpected_gaps = classify_gaps(source_gaps, evidence)
    result["workload"]["telemetry_gaps"] = gaps
    result["workload"]["collection_gaps"] = source_gaps
    result["workload"]["expected_telemetry_gaps"] = expected_gaps
    result["workload"]["unexpected_telemetry_gaps"] = unexpected_gaps
    result["workload"]["telemetry_completeness"] = (
        "PARTIAL" if source_gaps else "COMPLETE"
    )
    if unexpected_gaps:
        result["workload"]["runtime_validity"] = "INVALID"
    checks = [
        dict(stage=s["id"], **check) for s in result["stages"] for check in s["checks"]
    ]
    journal_issues = (
        audit_journals(
            directory,
            evidence.get("expected_telemetry", []),
            max_gap_s,
            evidence.get("telemetry_windows"),
        )
        if result.get("workload", {}).get("capture_metrics")
        else []
    )
    result["workload"]["telemetry_integrity_errors"] = journal_issues
    if journal_issues:
        result["workload"]["runtime_validity"] = "INVALID"
        result["workload"]["telemetry_completeness"] = "PARTIAL"
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
    if gate_series:
        # Audit every raw source above, but avoid duplicating hundreds of MB of
        # per-engine telemetry into the gate's JSON and thousands of HTML panels.
        # Full raw samples remain available for independent reanalysis.
        result["workload"]["report_series_scope"] = "derived_and_cache_gate"
        result["workload"]["raw_telemetry_sources"] = [
            dict(path=str(path), bytes=path.stat().st_size)
            for path in sorted(directory.glob("telemetry/*/*.prom"))
        ]
        series = {**derived, **gate_series}
    traffic = []
    iterations = []
    for flow_input in sorted(directory.glob("flows/*/flow-input.json")):
        flow = json.loads(flow_input.read_text())
        traffic.append(flow.get("trace", {}))
        journal = flow_input.parent / "client_lifecycle.jsonl"
        if journal.exists():
            issued = (json.loads(line) for line in journal.open() if line.strip())
            laps = iteration_windows(
                row for row in issued if row.get("event") == "issued"
            )
            iterations.extend(dict(group=flow.get("group_id"), **lap) for lap in laps)
    payload = dict(
        schema_version=1,
        traffic_manifests=traffic,
        iterations=iterations,
        configuration=result.get("implementation", {}).get("configuration"),
        implementation=result.get("implementation", {}),
        id=result["id"],
        status=result["status"],
        workload=result["workload"],
        checks=checks,
        stages=result["stages"],
        clock_anchor=evidence["clock_anchor"],
        phases=evidence["phases"],
        series=series,
        statistic_sources=statistic_sources,
        request_sources=[
            dict(
                path=str(Path(a["path"]).parent / "client_events.jsonl"),
                sha256=hashlib.sha256(
                    (Path(a["path"]).parent / "client_events.jsonl").read_bytes()
                ).hexdigest(),
                env_epoch=a["env_epoch"],
            )
            for a in result["workload"].get("stress_aggregates", [])
            if a["status"] == "GENERATED"
            and (Path(a["path"]).parent / "client_events.jsonl").is_file()
        ],
        configuration_sha256=result.get("implementation", {}).get(
            "configuration_sha256"
        ),
    )
    payload["gate_report"] = None
    from online_eval.reporting import bundle_path

    gate = bundle_path(directory, "run", "cache-scale-in") / "report.html"
    if gate.is_file():
        payload["gate_report"] = str(gate.resolve())
    return payload
