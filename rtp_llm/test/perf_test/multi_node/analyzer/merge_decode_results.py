#!/usr/bin/env python3
"""Merge collected Task_*/[run/]results/result.json decode v2 cases into CSV."""

import csv
import json
import re
from pathlib import Path
from typing import Optional

from rtp_llm.test.perf_test.results import (
    ResultSchemaError,
    parse_case_result,
    parse_result_envelope,
)

DECODE_FIELDS = [
    "input_length",
    "output_length",
    "batch_strategy",
    "global_batch_size",
    "best_batch_size",
    "success_rate",
    "wait_ms",
    "prefill_ms",
    "decode_ms",
    "tpot_ms",
    "total_ms",
    "throughput_tokens_per_second",
    "status",
]


def extract_task_params_from_path(task_dir_name: str) -> dict:
    params = {}
    for name in ("dp_size", "hack_layer_num"):
        match = re.search(rf"{name}-(\d+)", task_dir_name)
        if match:
            params[name] = int(match.group(1))
    if params:
        return params
    match = re.search(r"(\w+)-(\d+)", task_dir_name)
    return {match[1]: int(match[2])} if match else {"task_name": task_dir_name}


def scan_result_files(benchmark_dir: str) -> list:
    result_files = []
    seen_paths = set()
    for task in sorted(Path(benchmark_dir).glob("Task_*")):
        if not task.is_dir():
            continue
        for run in (task, task / "run"):
            path = run / "results" / "result.json"
            if path.is_file() and path.resolve() not in seen_paths:
                seen_paths.add(path.resolve())
                result_files.append(
                    (extract_task_params_from_path(task.name), str(path))
                )
    result_files.sort(
        key=lambda item: (
            item[0].get("dp_size", 999999),
            item[0].get("hack_layer_num", 999999),
            item[1],
        )
    )
    return result_files


def load_result_json(json_file: str) -> Optional[dict]:
    try:
        with open(json_file, encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError) as error:
        print(f"Cannot load {json_file}: {error}")
        return None


def _validate_envelope_header(envelope):
    """Validate metadata and summary without rejecting unrelated valid cases."""
    if not isinstance(envelope, dict):
        raise ResultSchemaError("result: expected an object")
    if (
        type(envelope.get("schema_version")) is not int
        or envelope.get("mode") != "decode"
    ):
        raise ResultSchemaError("expected a decode schema_version=2 envelope")
    cases = envelope.get("cases")
    if not isinstance(cases, list):
        raise ResultSchemaError("cases: expected an array")
    fields = {
        "planned_cases",
        "successful_cases",
        "failed_cases",
        "profiling_failed_cases",
    }
    summary = envelope.get("summary")
    if not isinstance(summary, dict) or set(summary) != fields:
        raise ResultSchemaError("summary: expected decode case counts")
    for field, count in summary.items():
        if type(count) is not int or count < 0:
            raise ResultSchemaError(f"summary.{field}: expected a non-negative integer")
    if (
        summary["planned_cases"] != len(cases)
        or summary["successful_cases"] + summary["failed_cases"] != len(cases)
        or summary["profiling_failed_cases"] > summary["successful_cases"]
    ):
        raise ResultSchemaError("summary: case counts are inconsistent")
    # Empty cases permit the shared parser to validate all metadata independently.
    return parse_result_envelope(
        {
            **envelope,
            "cases": [],
            "summary": {field: 0 for field in fields},
        }
    )


def merge_all_results(result_files: list) -> list:
    """Return (task parameters, normalized case rows); print excluded diagnostics."""
    merged = []
    seen_cases = set()
    for params, path in result_files:
        envelope = load_result_json(path)
        try:
            header = _validate_envelope_header(envelope)
        except (ResultSchemaError, TypeError, OverflowError) as error:
            print(f"Skipping {path}: {error}")
            continue
        try:
            parse_result_envelope(envelope)
        except (ResultSchemaError, TypeError, OverflowError) as error:
            print(f"Diagnostic {path}: {error}; validating cases independently")
        run_identity = (
            tuple(sorted(params.items())),
            header.run_id,
            header.config_fingerprint,
            header.workload_fingerprint,
        )
        rows = []
        for index, value in enumerate(envelope["cases"]):
            try:
                case = parse_case_result(index, "decode", value).to_dict()
            except (ResultSchemaError, TypeError, OverflowError) as error:
                print(f"Skipping {path}: cases[{index}]: {error}")
                continue
            key = (run_identity, case["case_id"])
            if key in seen_cases:
                print(
                    f"Skipping duplicate {path}: case={case['case_id']} run={header.run_id}"
                )
                continue
            metrics = case.get("metrics")
            if case.get("errors") or not metrics:
                print(
                    f"{path}: case={case['case_id']} status={case['status']} "
                    f"errors={json.dumps(case.get('errors', []), ensure_ascii=False)}"
                )
            if not metrics:
                continue
            seen_cases.add(key)
            profiling = metrics.get("profiling") or {}
            if profiling.get("status") == "failed":
                print(
                    f"{path}: case={case['case_id']} profiling=failed "
                    f"error={profiling.get('error')}"
                )
            inputs = metrics["input_lengths"]
            strategy = case["identity"]["batch_policy"]["strategy"]
            search = metrics.get("search") or {}
            successful = metrics["successful_requests"]
            requests = successful + metrics["failed_requests"]
            row = {
                "input_length": (
                    inputs[0] if len(set(inputs)) == 1 else json.dumps(inputs)
                ),
                "output_length": metrics["output_length"],
                "batch_strategy": strategy,
                "global_batch_size": metrics["global_batch_size"],
                "best_batch_size": (
                    metrics["global_batch_size"]
                    if strategy == "fixed"
                    else search.get("best_batch_size")
                ),
                "success_rate": successful / requests if requests else 0.0,
                "throughput_tokens_per_second": metrics["throughput_tokens_per_second"],
                "status": case["status"],
            }
            row.update(
                {
                    f"{name}_ms": metrics["latency_ms"][name]
                    for name in ("wait", "prefill", "decode", "tpot", "total")
                }
            )
            rows.append(row)
        # Retain the task even when every case has only diagnostics.
        merged.append((params, rows))
    return merged


def _param_keys(merged_data: list) -> list:
    keys = {key for params, _ in merged_data for key in params}
    return sorted(
        keys,
        key=lambda key: (
            0 if key == "dp_size" else 1 if key == "hack_layer_num" else 2,
            key,
        ),
    )


def save_merged_csv(merged_data: list, output_file: str):
    if not merged_data:
        return
    fields = _param_keys(merged_data) + DECODE_FIELDS
    with open(output_file, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for params, metrics in merged_data:
            for metric in metrics:
                writer.writerow({**params, **metric})


def print_summary_table(merged_data: list):
    fields = _param_keys(merged_data) + DECODE_FIELDS
    print("\t".join(fields))
    for params, metrics in merged_data:
        for metric in metrics:
            row = {**params, **metric}
            print("\t".join(str(row.get(field, "")) for field in fields))


def print_statistics_summary(merged_data: list):
    for params, metrics in merged_data:
        success_rate = (
            sum(item["success_rate"] for item in metrics) / len(metrics)
            if metrics
            else 0.0
        )
        print(
            f"{params}: metric_cases={len(metrics)}, mean_success_rate={success_rate:.4f}"
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir", help="Directory containing Task_* runs")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--no-print", action="store_true")
    args = parser.parse_args()
    if not Path(args.benchmark_dir).is_dir():
        parser.error(f"Directory does not exist: {args.benchmark_dir}")
    merged = merge_all_results(scan_result_files(args.benchmark_dir))
    if not merged:
        parser.error("No decode v2 result files found")
    output = args.output or f"{Path(args.benchmark_dir).name}_merged.csv"
    save_merged_csv(merged, output)
    if not args.no_print:
        print_statistics_summary(merged)
        print_summary_table(merged)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
