#!/usr/bin/env python3
"""Summarize existing traffic evidence without changing or rerunning its gates."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def read_json(path, warnings, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        warnings.append(f"{path.name}: {error}")
        return default


def percentile(values, quantile):
    values = sorted(values)
    return values[max(0, math.ceil(quantile * len(values)) - 1)] if values else None


def read_events(directory, warnings):
    events = []
    for path in sorted((directory / "environment").rglob("engine_events.jsonl")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                event = json.loads(line)
                if event.get("event") == "prefill_done":
                    events.append(event)
            except ValueError as error:
                warnings.append(f"{path}:{number}: {error}")
    return events


def complete_trace(directory, manifest, requests, warnings):
    path = directory / "trace.json"
    trace = read_json(path, warnings)
    if not isinstance(trace, dict):
        return False
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest.get("trace_sha256"):
        warnings.append("trace.json SHA differs from manifest")
        return False
    expected = [row for rows in trace.values() for row in rows]
    actual = {row.get("request_id"): row for row in requests}
    expected_ids = {row["request_id"] for row in expected}
    if (
        len(actual) != len(requests)
        or len(expected_ids) != len(expected)
        or expected_ids != set(actual)
    ):
        return False
    return all(
        row["request_id"] in actual
        and all(
            actual[row["request_id"]].get(key) == value for key, value in row.items()
        )
        for row in expected
    )


def describe_phase(stage, requests, events):
    """Descriptive fallback only; missing/ambiguous Engine evidence stays missing."""
    name = stage.get("stage", stage.get("name"))
    rows = [row for row in requests if row.get("stage") == name]
    successes = [row for row in rows if row.get("success") is True]
    by_id = defaultdict(list)
    for event in events:
        by_id[event.get("rid")].append(event)
    matched = []
    for row in successes:
        matches = by_id[row["request_id"]]
        if len(matches) == 1:
            event = matches[0]
            hit = event.get("cache_hit_tokens")
            if (
                event.get("cancelled") is False
                and event.get("input_len") == row["input_tokens"]
                and isinstance(hit, int)
                and 0 <= hit <= row["input_tokens"]
            ):
                matched.append(event)
    input_tokens = sum(row["input_tokens"] for row in successes)
    hit_tokens = sum(event["cache_hit_tokens"] for event in matched)
    coherent = len(matched) == len(successes) and bool(successes)
    elapsed = None
    if "start_s" in stage and "drained_s" in stage:
        elapsed = max(stage["seconds"], stage["drained_s"] - stage["start_s"])
    latencies = [row["ttft_s"] for row in successes if "ttft_s" in row]
    tail = []
    if stage.get("require_tail_zero_rejections") and "start_s" in stage:
        tail = [
            row
            for row in rows
            if row.get("planned_s", 0) - stage["start_s"] >= stage["settle_seconds"]
        ]
    return {
        "stage": name,
        "arrivals": len(rows),
        "completed": len(successes),
        "errors": len(rows) - len(successes),
        "success_fraction": len(successes) / len(rows) if rows else None,
        "success_token_hit_rate": hit_tokens / input_tokens if coherent else None,
        "input_tpm_including_drain": input_tokens * 60 / elapsed if elapsed else None,
        "compute_tpm_including_drain": (
            (input_tokens - hit_tokens) * 60 / elapsed if coherent and elapsed else None
        ),
        "ttft_samples": len(latencies),
        "ttft_p99_s": percentile(latencies, 0.99),
        "max_pacing_lag_s": max(
            (row.get("pacing_lag_s", 0) for row in rows), default=None
        ),
        "recovery_tail_arrivals": (
            len(tail) if stage.get("require_tail_zero_rejections") else None
        ),
        "recovery_tail_errors": (
            sum(row.get("success") is not True for row in tail)
            if stage.get("require_tail_zero_rejections")
            else None
        ),
        "successful_prefill_evidence": len(matched),
    }


def summarize_run(matrix, result, complete):
    directory = matrix / result["directory"]
    warnings = []
    manifest = read_json(directory / "manifest.json", warnings, {})
    report = read_json(directory / "report.json", warnings, {})
    failure = read_json(directory / "failure.json", warnings, {})
    requests = read_json(directory / "requests.json", warnings, [])
    stages = read_json(directory / "stages.json", warnings, [])
    events = read_events(directory, warnings)
    exit_code = result.get("exit_code")
    status = "INCOMPLETE"
    if (
        failure
        or report.get("status") == "FAIL"
        or (exit_code is not None and exit_code != 0)
    ):
        status = "FAIL"
    elif exit_code == 0 and report.get("status") == "PASS":
        status = "PASS"
    run = {
        "matrix": str(matrix),
        "directory": str(directory),
        "matrix_status": "COMPLETE" if complete else "INCOMPLETE",
        "case": result.get("case", manifest.get("case", {}).get("name")),
        "dispatch": result.get("dispatch", manifest.get("dispatch")),
        "jitter": result.get("jitter", manifest.get("jitter")),
        "seed": result.get("seed", manifest.get("seed")),
        "status": status,
        "exit_code": exit_code,
        "failure": failure.get("error", ""),
        "trace_sha256": manifest.get("trace_sha256"),
        "complete_request_trace": complete_trace(
            directory, manifest, requests, warnings
        ),
        "performance_checked_batches": report.get("performance", {}).get(
            "checked_batches"
        ),
        "warnings": warnings,
        "manifest": manifest,
    }
    official = {stage["stage"]: stage for stage in report.get("stages", [])}
    if not stages:
        stages = list(official.values()) or manifest.get("case", {}).get("phases", [])
    phases = []
    for stage in stages:
        description = describe_phase(stage, requests, events)
        name = description["stage"]
        use_official = name in official
        metrics = official[name] if use_official else description
        row = {
            key: value
            for key, value in run.items()
            if key not in {"manifest", "warnings"}
        }
        row["metrics_source"] = "OFFICIAL_REPORT" if use_official else "RAW_UNVALIDATED"
        for key in description:
            row[key] = metrics.get(key, description[key])
        window = (stage.get("start_epoch_ms"), stage.get("end_epoch_ms"))
        if all(value is not None for value in window):
            completed = [
                event
                for event in events
                if window[0] <= event.get("prefill_done_ms", -1) < window[1]
            ]
            row["observed_execution_batches"] = len(
                {
                    (
                        event.get("engine_name"),
                        event.get("engine_incarnation"),
                        event.get("prefill_done_ms"),
                    )
                    for event in completed
                }
            )
            evidence = report.get("performance", {}).get("batch_evidence")
            row["phase_performance_checked_batches"] = (
                sum(
                    window[0] <= event["done_wall_ms"] < window[1]
                    and not event["ambiguous_transition"]
                    for event in evidence
                )
                if evidence is not None
                else None
            )
        phases.append(row)
    return run, phases


def summarize_matrix(matrix):
    if not matrix.is_dir():
        raise ValueError(f"matrix directory does not exist: {matrix}")
    warnings = []
    results = read_json(matrix / "results.json", warnings)
    complete = isinstance(results, list)
    if not complete:
        results = []
        for slot in sorted(matrix.glob("results-slot-*.json")):
            results.extend(read_json(slot, warnings, []))
    directories = [row["directory"] for row in results]
    if len(directories) != len(set(directories)):
        raise ValueError(f"duplicate result entries in {matrix}")
    observed = {
        path.parent.name
        for filename in ("manifest.json", "report.json", "failure.json")
        for path in matrix.glob(f"*/{filename}")
    }
    for directory in sorted(observed - set(directories)):
        if directory not in directories:
            results.append({"directory": directory})
            complete = False
    runs, phases = [], []
    for result in results:
        run, rows = summarize_run(matrix, result, complete)
        runs.append(run)
        phases.extend(rows)
    return (
        {
            "directory": str(matrix),
            "status": "COMPLETE" if complete else "INCOMPLETE",
            "warnings": warnings,
        },
        runs,
        phases,
    )


def comparisons(runs, phases):
    indexed = defaultdict(list)
    for run in runs:
        indexed[
            (run["matrix"], run["case"], run["dispatch"], run["seed"], run["jitter"])
        ].append(run)
    output = []
    for run in runs:
        if run["jitter"] == "stable":
            continue
        candidates = indexed[
            (run["matrix"], run["case"], run["dispatch"], run["seed"], "stable")
        ]
        control = candidates[0] if len(candidates) == 1 else None
        matched = bool(
            control
            and control["complete_request_trace"]
            and run["complete_request_trace"]
            and control["trace_sha256"] == run["trace_sha256"]
        )
        compatible = bool(
            control
            and all(
                control["manifest"].get(key)
                and control["manifest"].get(key) == run["manifest"].get(key)
                for key in (
                    "case",
                    "master_sha256",
                    "mock_sha256",
                    "source_sha256",
                    "schedule_proto_sha256",
                )
            )
        )
        reason = "MATCHED" if matched and compatible else "NO_COMPLETE_MATCHED_CONTROL"
        if reason == "MATCHED" and (
            run["status"] != "PASS" or control["status"] != "PASS"
        ):
            reason = "MATCHED_UNVALIDATED"
        base_rows = {
            row["stage"]: row
            for row in phases
            if control and row["directory"] == control["directory"]
        }
        for row in (row for row in phases if row["directory"] == run["directory"]):
            entry = {
                key: row[key]
                for key in (
                    "matrix",
                    "matrix_status",
                    "case",
                    "dispatch",
                    "jitter",
                    "seed",
                    "stage",
                    "status",
                )
            }
            entry.update(
                comparison_status=reason,
                control_status=control["status"] if control else None,
                trace_sha256=run["trace_sha256"],
            )
            for metric in (
                "success_fraction",
                "success_token_hit_rate",
                "input_tpm_including_drain",
                "compute_tpm_including_drain",
                "ttft_p99_s",
            ):
                before = base_rows.get(row["stage"], {}).get(metric)
                after = row.get(metric)
                entry[f"delta_{metric}"] = (
                    after - before
                    if matched
                    and compatible
                    and before is not None
                    and after is not None
                    else None
                )
            output.append(entry)
    return output


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def display(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(lines, headers, rows):
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend(
        "| " + " | ".join(display(value) for value in row) + " |" for row in rows
    )
    lines.append("")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrices", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if len({directory.resolve() for directory in args.matrices}) != len(args.matrices):
        parser.error("duplicate matrix input")
    matrices, runs, phases = [], [], []
    for directory in args.matrices:
        matrix, found, rows = summarize_matrix(directory.resolve())
        matrices.append(matrix)
        runs.extend(found)
        phases.extend(rows)
    paired = comparisons(runs, phases)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "matrices": matrices,
        "runs": runs,
        "phases": phases,
        "comparisons": paired,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    write_csv(args.out_dir / "phases.csv", phases)
    write_csv(args.out_dir / "comparisons.csv", paired)
    lines = [
        "# Prefill 流量证据汇总",
        "",
        "状态沿用原始执行结果；本工具不重定义门槛。INCOMPLETE 为未完成矩阵快照，RAW_UNVALIDATED 仅是失败或缺报告运行的描述性统计。报告生成成功不代表测试通过。",
        "",
        "TPM 为成功请求 cohort（含排空时间），KV 命中率按成功请求 token 加权；P99 附成功样本数。批次数按完成时间所在阶段统计，性能核验数来自原始审计。差值=波动−stable，仅在完整请求 trace、代码及配置匹配时计算；不代表统计显著性。",
        "",
    ]
    markdown_table(
        lines,
        ["矩阵", "状态", "原始运行状态"],
        [
            (
                matrix["directory"],
                matrix["status"],
                dict(
                    Counter(
                        run["status"]
                        for run in runs
                        if run["matrix"] == matrix["directory"]
                    )
                ),
            )
            for matrix in matrices
        ],
    )
    for seed in sorted({run["seed"] for run in runs}, key=str):
        lines.extend([f"## Seed {seed}", ""])
        for run in (run for run in runs if run["seed"] == seed):
            lines.extend(
                [
                    f"### {run['case']} / {run['dispatch']} / {run['jitter']} — {run['status']}",
                    "",
                    f"目录：{run['directory']}；矩阵：{run['matrix_status']}；完整实发输入：{run['complete_request_trace']}。",
                    "",
                ]
            )
            if run["failure"] or run["warnings"]:
                lines.extend(
                    [
                        f"错误/证据问题：{display(run['failure'])} {display('; '.join(run['warnings']))}",
                        "",
                    ]
                )
            rows = [row for row in phases if row["directory"] == run["directory"]]
            keys = (
                "stage",
                "metrics_source",
                "arrivals",
                "completed",
                "errors",
                "success_token_hit_rate",
                "input_tpm_including_drain",
                "compute_tpm_including_drain",
                "ttft_p99_s",
                "ttft_samples",
                "max_pacing_lag_s",
                "recovery_tail_arrivals",
                "recovery_tail_errors",
                "observed_execution_batches",
                "phase_performance_checked_batches",
            )
            markdown_table(
                lines,
                [
                    "阶段",
                    "来源",
                    "到达",
                    "完成",
                    "错误",
                    "KV hit",
                    "Input TPM",
                    "Compute TPM",
                    "P99(s)",
                    "n",
                    "lag(s)",
                    "尾到达",
                    "尾错误",
                    "批次",
                    "perf核验",
                ],
                [[row.get(key) for key in keys] for row in rows],
            )
        lines.extend(["### 与 stable 配对差值", ""])
        keys = (
            "case",
            "dispatch",
            "jitter",
            "stage",
            "comparison_status",
            "delta_success_fraction",
            "delta_success_token_hit_rate",
            "delta_input_tpm_including_drain",
            "delta_compute_tpm_including_drain",
            "delta_ttft_p99_s",
        )
        markdown_table(
            lines,
            [
                "场景",
                "模式",
                "波动",
                "阶段",
                "配对",
                "Δ成功率",
                "ΔKV hit",
                "ΔInput TPM",
                "ΔCompute TPM",
                "ΔP99(s)",
            ],
            [[row.get(key) for key in keys] for row in paired if row["seed"] == seed],
        )
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "matrices": matrices,
                "run_statuses": dict(Counter(run["status"] for run in runs)),
            }
        )
    )


if __name__ == "__main__":
    main()
