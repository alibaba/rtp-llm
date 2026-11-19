"""Pure offline report generation for persisted prefill runs."""

from __future__ import annotations

import html
import json
import math
import os
import statistics
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.plan import PlannedCase, thaw
from rtp_llm.test.perf_test.prefill.measurement import (
    validate_concurrent_prefill_metrics,
)
from rtp_llm.test.perf_test.run_directory import RunDirectoryLock, inspect_run_directory
from rtp_llm.test.perf_test.storage import RunStore, atomic_write_json

REPORT_SCHEMA_VERSION = 1
_VALIDATION_KEYS = ("shape", "reuse", "timing")


def _positive_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _non_negative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _exclude(case_id: str, status: str, reason: str) -> Dict[str, str]:
    return {"case_id": case_id, "status": status, "reason": reason}


def _observation(
    planned: PlannedCase, committed: Optional[Mapping[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, str]]]:
    if committed is None:
        return None, _exclude(planned.case_id, "missing", "case result is missing")

    status = committed.get("status")
    if status != "ok":
        name = status if isinstance(status, str) and status else "invalid_result"
        return None, _exclude(
            planned.case_id, name, "case did not complete successfully"
        )

    identity = thaw(planned.identity)
    if committed.get("identity") != identity:
        return None, _exclude(
            planned.case_id,
            "invalid_result",
            "case identity does not match the execution plan",
        )

    metrics = committed.get("metrics")
    if not isinstance(metrics, Mapping):
        return None, _exclude(
            planned.case_id, "invalid_result", "prefill metrics are missing"
        )

    validation = metrics.get("validation")
    if not isinstance(validation, Mapping) or any(
        validation.get(name) is not True for name in _VALIDATION_KEYS
    ):
        return None, _exclude(
            planned.case_id,
            "invalid_validation",
            "shape, reuse, and timing validation must all pass",
        )

    input_length = metrics.get("input_length")
    expected_reuse = metrics.get("expected_reuse_length")
    observed_reuse = metrics.get("observed_reuse_length")
    compute_length = metrics.get("compute_length")
    latency_ms = metrics.get("latency_ms")
    throughput = metrics.get("throughput_tokens_per_second")
    measurements = metrics.get("measurements")
    contract = identity.get("measurement_contract", "legacy_latency_derived_v1")
    grouped = contract == "concurrent_grouped_batch_v1"
    groups = (
        identity.get("request_groups", [])
        if grouped
        else [
            {
                "count": identity["batch_size"],
                "input_len": identity["input_len"],
                "cache_len": identity["cache_len"],
            }
        ]
    )
    distribution = []
    for group_index, group in enumerate(groups):
        for _ in range(group["count"]):
            distribution.append(
                {
                    "request_index": len(distribution),
                    "group_index": group_index,
                    "input_length": group["input_len"],
                    "expected_reuse_length": group["cache_len"],
                    "compute_length": group["input_len"] - group["cache_len"],
                }
            )
    planned_input = (
        sum(m["input_length"] for m in distribution)
        if grouped
        else identity.get("input_len")
    )
    planned_reuse = (
        sum(m["expected_reuse_length"] for m in distribution)
        if grouped
        else identity.get("cache_len")
    )
    planned_compute = planned_input - planned_reuse
    lengths_valid = (
        type(input_length) is int
        and input_length > 0
        and _non_negative_int(expected_reuse)
        and _non_negative_int(observed_reuse)
        and type(compute_length) is int
        and compute_length > 0
        and expected_reuse + compute_length == input_length
        and expected_reuse == planned_reuse
        and compute_length == planned_compute
        and input_length == planned_input
    )
    if not lengths_valid:
        return None, _exclude(
            planned.case_id,
            "invalid_shape",
            "prefill lengths are inconsistent with the planned case",
        )
    if not _positive_number(latency_ms) or not _positive_number(throughput):
        return None, _exclude(
            planned.case_id,
            "invalid_timing",
            "latency and throughput must be finite positive numbers",
        )
    if not isinstance(measurements, list) or any(
        not isinstance(item, Mapping) for item in measurements
    ):
        return None, _exclude(
            planned.case_id,
            "invalid_result",
            "measurements must be an array of objects",
        )

    try:
        validate_concurrent_prefill_metrics(identity, metrics, committed.get("runs"))
    except ValueError as error:
        return None, _exclude(planned.case_id, "invalid_measurement", str(error))

    eligible = contract in ("concurrent_batch_v1", "concurrent_grouped_batch_v1")
    round_maxima = {}
    if eligible:
        for member in measurements:
            run = member["run"]
            round_maxima[run] = max(round_maxima.get(run, 0), member["latency_ms"])
    return {
        "case_id": planned.case_id,
        "grid_name": identity.get("grid_name", "default"),
        "request_distribution": distribution,
        "batch_latency_ms": (
            statistics.median(round_maxima.values()) if eligible else None
        ),
        "tokens_per_minute": throughput * 60,
        "fit_eligible": eligible,
        "fit_exclusion_reason": (
            None if eligible else "historical latency-derived measurement contract"
        ),
        "measurement_contract": identity.get(
            "measurement_contract", "legacy_latency_derived_v1"
        ),
        "batch_size": identity["batch_size"],
        "input_length": input_length,
        "expected_reuse_length": expected_reuse,
        "observed_reuse_length": observed_reuse,
        "compute_length": compute_length,
        "latency_ms": latency_ms,
        "throughput_tokens_per_second": throughput,
        "measurements": [dict(item) for item in measurements],
    }, None


def _atomic_write_text(path: Path, value: str) -> None:
    if not path.parent.is_dir():
        raise FileNotFoundError(
            f"artifact parent directory does not exist: {path.parent}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    replaced = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        replaced = True
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if not replaced:
            temporary.unlink(missing_ok=True)


def _render_html(report: Mapping[str, Any]) -> str:
    rows = []
    for item in report["observations"]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['case_id'])}</td>"
            f"<td>{html.escape(item['measurement_contract'])}</td>"
            f"<td>{item['batch_size']}</td>"
            f"<td>{item['input_length']}</td>"
            f"<td>{item['observed_reuse_length']}</td>"
            f"<td>{item['compute_length']}</td>"
            f"<td>{item['latency_ms']}</td>"
            f"<td>{item['throughput_tokens_per_second']}</td>"
            "</tr>"
        )
    excluded = "".join(
        "<li>"
        f"{html.escape(item['case_id'])}: {html.escape(item['status'])} - "
        f"{html.escape(item['reason'])}"
        "</li>"
        for item in report["excluded"]
    )
    sampling = ""
    if report["traffic_sampling"]:
        details = html.escape(
            json.dumps(report["traffic_sampling"], indent=2, sort_keys=True)
        )
        sampling = (
            "<h2>Traffic sampling provenance</h2>"
            "<p>These are sampled request geometries; "
            "they do not replay production arrivals.</p>"
            f"<pre>{details}</pre>"
        )
    return """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Prefill performance report</title></head>
<body>
<h1>Prefill performance report</h1>
<p>Report kind: {kind}</p>
<p>Server first-token latency is measured in milliseconds.</p>
<p>concurrent_batch_v1 throughput uses measured round wall time;
legacy_latency_derived_v1 identifies historical latency-derived measurements.</p>
{sampling}
<table>
<thead><tr><th>Case</th><th>Measurement contract</th><th>Batch</th><th>Input</th><th>Reused</th><th>Computed</th><th>Latency (ms)</th><th>Tokens/s</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<h2>Excluded cases</h2><ul>{excluded}</ul>
</body>
</html>
""".format(
        kind=html.escape(report["report_kind"]),
        rows="".join(rows),
        excluded=excluded,
        sampling=sampling,
    )


def generate_prefill_report(
    root: Union[str, Path], *, acquire_lock: bool = True
) -> Dict[str, Any]:
    """Generate deterministic report artifacts using only persisted run data."""
    root_path = Path(root)
    lock = (
        RunDirectoryLock(root_path, action="report") if acquire_lock else nullcontext()
    )
    with lock:
        snapshot = inspect_run_directory(root_path)
        plan = snapshot.execution_plan
        if plan.mode != "prefill":
            raise PlanningError("prefill report requires a prefill execution plan")
        if (root_path / "results" / "cases").exists():
            raise PlanningError(
                "legacy prefill results/cases layout is unsupported; start a new run"
            )
        committed = RunStore(root_path).read_prefill_cases(plan)
        observations: List[Dict[str, Any]] = []
        excluded: List[Dict[str, str]] = []
        for planned in plan.cases:
            observation, rejection = _observation(
                planned, committed.get(planned.case_id)
            )
            if observation is not None:
                observations.append(observation)
            if rejection is not None:
                excluded.append(rejection)

        workload = thaw(plan.resolved_config)["workload"]
        sampling = {}
        if "traffic_sampling" in workload:
            sampling["default"] = {
                "seed": workload["seed"],
                "sampling": workload["traffic_sampling"],
            }
        for grid in workload.get("grids", []):
            if "traffic_sampling" in grid["workload"]:
                child = grid["workload"]
                sampling[grid["name"]] = {
                    "seed": child["seed"],
                    "sampling": child["traffic_sampling"],
                }
        report: Dict[str, Any] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "mode": "prefill",
            "traffic_sampling": sampling,
            "report_kind": "formal" if not excluded else "diagnostic",
            "config_fingerprint": plan.config_fingerprint,
            "workload_fingerprint": plan.workload_fingerprint,
            "counts": {
                "planned": len(plan.cases),
                "valid": len(observations),
                "excluded": len(excluded),
            },
            "observations": observations,
            "excluded": excluded,
            "formula_input": {
                "measurement_contract": "median_round_max_server_first_token_latency_ms",
                "rows": [row for row in observations if row["fit_eligible"]],
                "excluded": [
                    {"case_id": row["case_id"], "reason": row["fit_exclusion_reason"]}
                    for row in observations
                    if not row["fit_eligible"]
                ],
            },
            "chart": {
                "x": "total_compute_tokens",
                "latency_y": "batch_latency_ms",
                "tpm_y": "tokens_per_minute",
            },
        }
        from rtp_llm.test.perf_test.prefill.formula import fit_prefill_formula

        report["formula_fit"] = fit_prefill_formula(
            observations, snapshot.paths.reports / "formula", complete=not excluded
        )
        atomic_write_json(snapshot.paths.reports / "prefill_report.json", report)
        atomic_write_json(
            snapshot.paths.reports / "prefill_observations.json",
            {
                "schema_version": REPORT_SCHEMA_VERSION,
                "observations": observations,
                "excluded": excluded,
            },
        )
        _atomic_write_text(
            snapshot.paths.reports / "prefill_report.html", _render_html(report)
        )
        from rtp_llm.test.perf_test.prefill.chart import render_charts

        for name, content in render_charts(report).items():
            _atomic_write_text(snapshot.paths.reports / name, content)
        return report
