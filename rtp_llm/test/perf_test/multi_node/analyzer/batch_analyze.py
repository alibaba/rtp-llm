"""Analyze listed Kineto traces from a new perf-test run or a tree of runs."""

import argparse
import csv
import json
import os
import runpy
import sys
from pathlib import Path

from rtp_llm.test.perf_test.errors import PerfTestError
from rtp_llm.test.perf_test.multi_node.analyzer.configs.module_base import (
    TestSetUp,
    normalize_module_config,
)
from rtp_llm.test.perf_test.multi_node.analyzer.trace_analyze import analyze_trace
from rtp_llm.test.perf_test.plan import ExecutionPlan, thaw
from rtp_llm.test.perf_test.profiling import parse_profile_manifest, profile_trace_path
from rtp_llm.test.perf_test.storage import atomic_write_json


def load_module_config(path):
    try:
        namespace = runpy.run_path(str(Path(path).resolve()))
        return normalize_module_config(namespace.get("module_config"))
    except Exception as error:
        raise ValueError(f"cannot load module config {path}: {error}") from error


def module_setup(trace):
    identity, topology = trace["case_identity"], trace["topology"]
    mode = trace.get("phase") or trace["mode"]
    tp, dp = topology["tp"], topology["dp"]
    try:
        if trace["mode"] == "decode":
            lengths = identity["input_lengths"]
            batch = len(lengths)
            if len(set(lengths)) != 1:
                raise ValueError(
                    "heterogeneous decode lengths have no single module formula shape"
                )
            length = lengths[0]
        else:
            batch = identity["batch_size"]
            groups = identity.get("request_groups")
            shapes = (
                {(group["input_len"], group["cache_len"]) for group in groups}
                if groups
                else {(identity["input_len"], identity.get("cache_len", 0))}
            )
            if len(shapes) != 1:
                raise ValueError(
                    "heterogeneous prefill groups have no single module formula shape"
                )
            length, cache = next(iter(shapes))
            if cache:
                raise ValueError(
                    "cache-reuse prefill needs a cache-aware formula; estimates omitted"
                )
        if batch % dp or batch <= 0 or length <= 0:
            raise ValueError("invalid per-DP batch/length for module formula")
    except (KeyError, TypeError, ValueError) as error:
        return TestSetUp(dp_size=dp, tp_size=tp, mode=mode, estimation_error=str(error))
    steps = None
    if trace["mode"] == "prefill" and mode == "prefill":
        steps = 1
    elif trace.get("phase") == "decode" and identity.get("profiling", {}).get(
        "enabled"
    ):
        steps = identity["profiling"]["steps"]
    return TestSetUp(
        dp_size=dp,
        tp_size=tp,
        batch_size=batch // dp,
        seq_len=length,
        mode=mode,
        steps=steps,
    )


def analyze_runs(directory, *, min_occurrences=1, module_config=None):
    """Read frozen case identities and current manifests without modifying inputs."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError(f"run directory does not exist: {directory}")
    if type(min_occurrences) is not int or min_occurrences <= 0:
        raise ValueError("min_occurrences must be a positive integer")
    report = {"schema_version": 1, "traces": [], "skipped": []}
    plans = []
    for current, children, files in os.walk(directory, followlinks=False):
        children.sort()
        if "execution_plan.json" in files:
            plans.append(Path(current) / "execution_plan.json")
            # A run owns its internal member snapshots and collected copies.
            children[:] = []
    if not plans:
        raise ValueError(
            "no execution_plan.json found; expected a new perf-test run or tree of runs"
        )
    for path in plans:
        root = path.parent
        try:
            plan = ExecutionPlan.from_dict(json.loads(path.read_text()))
        except (OSError, ValueError, KeyError, TypeError, PerfTestError) as error:
            report["skipped"].append({"path": str(path), "reason": str(error)})
            continue
        cases = {case.case_id: thaw(case.identity) for case in plan.cases}
        topology = thaw(plan.resolved_config)["model"]["topology"]
        diagnostics = root / "workload" / "trace_diagnostics"
        if diagnostics.is_dir() and any(diagnostics.rglob("*.json")):
            report["skipped"].append(
                {
                    "path": str(diagnostics),
                    "reason": "unassociated member traces retained for diagnosis; no matching completed profile case",
                }
            )
        manifests = sorted((root / "profiles").glob("*/manifest.json"))
        if not manifests:
            report["skipped"].append(
                {
                    "path": str(root),
                    "reason": "no profile sessions; profiling must be captured before trace analysis",
                }
            )
        for manifest_path in manifests:
            try:
                if manifest_path.is_symlink() or manifest_path.parent.is_symlink():
                    raise ValueError("profile manifest symlink is forbidden")
                manifest = parse_profile_manifest(json.loads(manifest_path.read_text()))
                if (
                    manifest["session_id"] != manifest_path.parent.name
                    or manifest["mode"] != plan.mode
                ):
                    raise ValueError("profile session/mode does not match its run")
                if manifest["status"] in ("running", "interrupted"):
                    raise ValueError(f"profile session is {manifest['status']}")
                for case in manifest["cases"]:
                    if case["case_id"] not in cases:
                        raise ValueError(
                            f"profile case not present in frozen plan: {case['case_id']}"
                        )
            except (OSError, ValueError, KeyError, TypeError, PerfTestError) as error:
                report["skipped"].append(
                    {"path": str(manifest_path), "reason": str(error)}
                )
                continue
            for case in manifest["cases"]:
                context = {
                    "run_id": Path(plan.result_directory).name,
                    "run_directory": str(root),
                    "mode": plan.mode,
                    "session_id": manifest["session_id"],
                    "case_id": case["case_id"],
                    "case_identity": cases[case["case_id"]],
                    "topology": topology,
                    "capture_scope": manifest["capture_scope"],
                }
                if case["status"] != "completed":
                    report["skipped"].append(
                        {
                            **context,
                            "reason": case["error"]
                            or f"profile case is {case['status']}",
                        }
                    )
                    continue
                for artifact in case["artifacts"]:
                    try:
                        if artifact["format"] != "chrome_trace":
                            raise ValueError(
                                f"unsupported trace format for JSON analysis: {artifact['format']}"
                            )
                        trace = profile_trace_path(
                            root,
                            manifest["session_id"],
                            case["case_id"],
                            artifact["path"],
                        )
                        stats = analyze_trace(
                            json.loads(trace.read_text()),
                            min_occurrences=min_occurrences,
                            module_config=module_config,
                            setup=(
                                module_setup({**context, "phase": artifact["phase"]})
                                if module_config is not None
                                else None
                            ),
                        )
                    except (
                        OSError,
                        ValueError,
                        KeyError,
                        TypeError,
                        PerfTestError,
                    ) as error:
                        report["skipped"].append(
                            {**context, "path": artifact["path"], "reason": str(error)}
                        )
                        continue
                    report["traces"].append(
                        {
                            **context,
                            "trace_file": artifact["path"],
                            "rank": artifact["rank"],
                            "phase": artifact["phase"],
                            "statistics": stats,
                        }
                    )
    return report


def write_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "analysis.json", report)
    fields = [
        "run_id",
        "run_directory",
        "mode",
        "session_id",
        "case_id",
        "rank",
        "phase",
        "trace_file",
        "capture_scope",
        "topology",
        "case_identity",
        "kernel_count",
        "kernel_time_us",
        "kernel_span_us",
        "kernel_name",
        "count",
        "total_duration_us",
        "avg_duration_us",
        "min_duration_us",
        "max_duration_us",
        "median_duration_us",
        "std_dev_us",
    ]
    module_fields = [
        "module_name",
        "kernel_names",
        "count",
        "total_duration_us",
        "kernel_time_percentage",
        "estimated_flops_per_invocation",
        "estimated_io_bytes_per_invocation",
        "estimated_tflops",
        "estimated_bandwidth_gbs",
        "estimation_error",
    ]
    context_fields = fields[: fields.index("kernel_name")]
    tables = [("kernels.csv", "kernels", fields)]
    if any("modules" in trace["statistics"] for trace in report["traces"]):
        tables.append(("modules.csv", "modules", context_fields + module_fields))
    else:
        (output / "modules.csv").unlink(missing_ok=True)
    for filename, key, columns in tables:
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for trace in report["traces"]:
                context = {
                    name: value for name, value in trace.items() if name != "statistics"
                }
                context["topology"] = json.dumps(context["topology"], sort_keys=True)
                context["case_identity"] = json.dumps(
                    context["case_identity"], sort_keys=True
                )
                context.update(
                    {
                        name: value
                        for name, value in trace["statistics"].items()
                        if name not in ("kernels", "modules")
                    }
                )
                for entry in trace["statistics"].get(key, []):
                    row = {**context, **entry}
                    if key == "modules":
                        row["kernel_names"] = json.dumps(row["kernel_names"])
                    writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", type=Path, help="One run, experiment, or directory containing runs"
    )
    parser.add_argument("-o", "--output-dir", type=Path)
    parser.add_argument("--min-occurrences", type=int, default=1)
    parser.add_argument(
        "--module-conf",
        type=Path,
        help="Python module_config mapping; optional module timing and model estimates",
    )
    args = parser.parse_args(argv)
    try:
        config = load_module_config(args.module_conf) if args.module_conf else None
        report = analyze_runs(
            args.run_dir, min_occurrences=args.min_occurrences, module_config=config
        )
        output = args.output_dir or args.run_dir / "reports" / "trace"
        write_report(report, output)
    except (OSError, ValueError) as error:
        print(f"trace analysis failed: {error}", file=sys.stderr)
        return 1
    print(
        f"Analyzed {len(report['traces'])} traces; skipped {len(report['skipped'])}; report: {output}"
    )
    for skipped in report["skipped"]:
        print(f"Skipped: {skipped['reason']}", file=sys.stderr)
    return 0 if report["traces"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
