"""Derive ordered decode observations and input-by-output matrix views."""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Sequence

from rtp_llm.test.perf_test.results import (
    RESULT_SCHEMA_VERSION,
    ResultSchemaError,
    parse_case_result,
)


def _observation(index: int, result: Mapping[str, Any]) -> Dict[str, Any]:
    identity = result.get("identity") or {}
    status = result.get("status", "missing")
    errors = copy.deepcopy(result.get("errors", []))
    valid = False
    try:
        parsed = parse_case_result(index, "decode", result)
        valid = parsed.status.value == "ok"
    except ResultSchemaError as error:
        if status == "ok":
            status = "invalid_result"
        errors.append({"code": "invalid_result", "message": str(error)})
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        metrics = {}
    search = copy.deepcopy(metrics.get("search"))
    if search is None:
        for error in errors:
            details = error.get("details") if isinstance(error, Mapping) else None
            if isinstance(details, Mapping) and "search" in details:
                search = copy.deepcopy(details["search"])
                break
    policy = copy.deepcopy(identity.get("batch_policy", {}))
    lengths = list(identity.get("input_lengths", []))
    row = {
        "case_id": result.get("case_id"),
        "status": status,
        "workload_kind": identity.get("workload_kind"),
        "input_lengths": copy.deepcopy(metrics.get("input_lengths", lengths)),
        "input_length": lengths[0] if lengths else None,
        "output_length": identity.get("output_length"),
        "batch_policy": policy,
        "batch_size": policy.get("size"),
        "tpot_ms": None,
        "throughput_tokens_per_second": None,
        "total_latency_ms": None,
        "success_rate": None,
        "errors": errors,
        "search": search,
        "profiling": copy.deepcopy(metrics.get("profiling")),
        "distribution": copy.deepcopy(
            metrics.get("distribution", identity.get("distribution"))
        ),
    }
    if valid:
        total = metrics["successful_requests"] + metrics["failed_requests"]
        row.update(
            batch_size=metrics["global_batch_size"],
            tpot_ms=metrics["latency_ms"]["tpot"],
            throughput_tokens_per_second=metrics["throughput_tokens_per_second"],
            total_latency_ms=metrics["latency_ms"]["total"],
            success_rate=metrics["successful_requests"] / total,
        )
    if policy.get("strategy") == "target_tpot":
        search_record = search if isinstance(search, Mapping) else {}
        row.update(
            best_batch_size=search_record.get("best_batch_size") if valid else None,
            search_outcome=search_record.get("outcome"),
        )
    return row


def generate_decode_report(case_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Preserve every supplied case; only successful validated cases have metrics."""
    observations = [
        _observation(index, result) for index, result in enumerate(case_results)
    ]
    valid = sum(row["status"] == "ok" for row in observations)
    all_valid = bool(observations) and valid == len(observations)
    groups: Dict[tuple, Dict[str, Any]] = {}
    exact_axes = {"input_lengths": [], "output_lengths": []}
    distribution_rows = []
    for row in observations:
        if row["workload_kind"] == "distribution":
            distribution_rows.append(copy.deepcopy(row))
            continue
        if row["workload_kind"] != "grid":
            continue
        policy = row["batch_policy"]
        key = tuple(sorted(policy.items()))
        if key not in groups:
            groups[key] = {
                "batch_policy": copy.deepcopy(policy),
                "batch_size": policy.get("size"),
                "input_lengths": [],
                "output_lengths": [],
                "cells": [],
            }
        matrix = groups[key]
        for name, value in (
            ("input_lengths", row["input_length"]),
            ("output_lengths", row["output_length"]),
        ):
            if value not in matrix[name]:
                matrix[name].append(value)
            if policy.get("strategy") == "fixed" and value not in exact_axes[name]:
                exact_axes[name].append(value)
        matrix["cells"].append(copy.deepcopy(row))

    for matrix in groups.values():
        if matrix["batch_policy"].get("strategy") == "fixed":
            matrix.update(copy.deepcopy(exact_axes))
        positions = {}
        for cell in matrix["cells"]:
            position = (cell["input_length"], cell["output_length"])
            if position in positions:
                all_valid = False
            positions.setdefault(position, []).append(cell)
        matrix["cells"] = []
        for input_length in matrix["input_lengths"]:
            for output_length in matrix["output_lengths"]:
                if (input_length, output_length) not in positions:
                    all_valid = False
                    missing = {
                        "case_id": None,
                        "status": "missing",
                        "input_length": input_length,
                        "output_length": output_length,
                        "tpot_ms": None,
                        "throughput_tokens_per_second": None,
                        "total_latency_ms": None,
                        "success_rate": None,
                        "errors": [
                            {
                                "code": "missing",
                                "message": "grid cell result is missing",
                            }
                        ],
                    }
                    if matrix["batch_policy"].get("strategy") == "target_tpot":
                        missing.update(best_batch_size=None, search_outcome=None)
                    matrix["cells"].append(missing)
                else:
                    matrix["cells"].extend(positions[(input_length, output_length)])
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "mode": "decode",
        "report_kind": "formal" if all_valid else "diagnostic",
        "counts": {
            "planned": len(observations),
            "valid": valid,
            "excluded": len(observations) - valid,
        },
        "observations": observations,
        "matrices": list(groups.values()),
        "distribution_cases": distribution_rows,
    }
