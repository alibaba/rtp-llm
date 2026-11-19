"""Local Bazel test entry retaining benchmark outputs as test artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from rtp_llm.test.perf_test.entry import main as entry_main
from rtp_llm.test.perf_test.errors import (
    CaseExecutionError,
    ConfigurationError,
    PerfTestError,
)


def worker_executors(selected_case_ids=()):
    from rtp_llm.test.perf_test.execution.local import create_default_local_executor

    return {"local": create_default_local_executor(selected_case_ids=selected_case_ids)}


def perf_main(arguments):
    """Compatibility for direct Bazel callers; never re-submit a local test."""
    from rtp_llm.test.perf_test.cli import parse_cli
    from rtp_llm.test.perf_test.entry import _source_config
    from rtp_llm.test.perf_test.launch_config import prepare_config
    from rtp_llm.test.perf_test.run_directory import inspect_run_directory

    try:
        request = parse_cli(arguments)
        backend = (
            prepare_config(_source_config(request.config), request.overrides)[
                "execution"
            ]["backend"]
            if request.config
            else inspect_run_directory(request.run_dir).execution_plan.backend
        )
        if backend != "local":
            raise ConfigurationError("perf_test_local requires execution.backend=local")
    except (PerfTestError, ValueError, OSError) as error:
        sys.stderr.write(f"perf_test: {error}\n")
        return getattr(error, "exit_code", 1)
    return entry_main(
        arguments, executor_factory=lambda: worker_executors(request.cases)
    )


def _run_bundle(arguments):
    from rtp_llm.test.perf_test.cli import CliRequest
    from rtp_llm.test.perf_test.entry import execute_plan
    from rtp_llm.test.perf_test.plan import ExecutionPlan
    from rtp_llm.test.perf_test.storage import atomic_write_json

    parser = argparse.ArgumentParser(
        description="Execute a frozen local benchmark plan"
    )
    parser.add_argument("--run-bundle", required=True)
    args = parser.parse_args(arguments)
    bundle_path = Path(args.run_bundle).resolve()
    payload = json.loads(bundle_path.read_text())
    if (
        payload.get("schema_version") != 1
        or payload.get("plan", {}).get("backend") != "local"
    ):
        raise ConfigurationError(
            "local worker requires a versioned local execution plan"
        )
    plan = ExecutionPlan.from_dict(payload["plan"])
    request = CliRequest(**payload["request"])
    if (
        request.dry_run
        or request.mode not in (None, plan.mode)
        or request.action != plan.action
    ):
        raise ConfigurationError("local worker request does not match frozen plan")
    if request.broker_host:
        raise ConfigurationError("local workers cannot submit AOne jobs")
    # Launch flags belong to the controller. The worker executes only this plan.
    from dataclasses import replace

    request = replace(request, bazel_args=(), bazel_startup_args=(), overrides=())
    try:
        result = execute_plan(
            plan,
            payload["source"],
            request,
            executor_factory=lambda: worker_executors(request.cases),
        )
        values = (
            dict(result)
            if isinstance(result, dict)
            else {
                "exit_code": getattr(result, "exit_code", 0),
            }
        )
        for key in ("state", "signal_number"):
            value = getattr(result, key, values.get(key))
            if value is not None:
                values[key] = getattr(value, "value", value)
    except PerfTestError as error:
        sys.stderr.write(f"perf_test_local: {error}\n")
        values = {"exit_code": error.exit_code, "state": "failed", "error": str(error)}
    except Exception as error:
        from rtp_llm.test.perf_test.errors import error_message

        values = {
            "exit_code": CaseExecutionError.exit_code,
            "state": "failed",
            "error": error_message(error),
        }
    except KeyboardInterrupt:
        values = {"exit_code": 130, "state": "interrupted"}
    code = int(values.get("exit_code", 0))
    if plan.action == "profile" and values.get("status") == "failed":
        code = CaseExecutionError.exit_code
    root = Path(plan.result_directory)
    evidence_root = root if plan.action == "run" else bundle_path.parent
    if evidence_root.is_dir():
        atomic_write_json(
            evidence_root / "launch_environment.json",
            {
                "platform": plan.resolved_config["execution"].get("platform", "cuda13"),
                "environment": {
                    key: os.environ[key]
                    for key in (
                        "CC",
                        "CXX",
                        "CUDAHOSTCXX",
                        "NVCC_PREPEND_FLAGS",
                        "LD_LIBRARY_PATH",
                        "PATH",
                        "TRITON_PTXAS_PATH",
                        "DG_JIT_CPP_STANDARD",
                    )
                    if key in os.environ
                },
            },
        )
        atomic_write_json(evidence_root / "launch_request.json", payload["request"])
    atomic_write_json(payload["result_path"], {**values, "exit_code": code})
    return code


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(
        arg == "--run-bundle" or arg.startswith("--run-bundle=") for arg in arguments
    ):
        try:
            return _run_bundle(arguments)
        except PerfTestError as error:
            sys.stderr.write(f"perf_test_local: {error}\n")
            return error.exit_code
        except (ValueError, OSError, TypeError, KeyError) as error:
            sys.stderr.write(f"perf_test_local: {error}\n")
            return ConfigurationError.exit_code
    original_directory = Path.cwd()
    # Bazel starts tests in runfiles. Resolve inputs there before moving to the
    # writable artifact directory; absolute output roots in JSON stay absolute.
    path_options = ("--config", "--run-dir", "--run-context")
    for index, argument in enumerate(arguments):
        if argument in path_options:
            if (
                index + 1 < len(arguments)
                and arguments[index + 1]
                and not arguments[index + 1].startswith("-")
            ):
                arguments[index + 1] = str(Path(arguments[index + 1]).absolute())
        else:
            option, separator, value = argument.partition("=")
            if option in path_options and separator and value:
                arguments[index] = option + "=" + str(Path(value).absolute())
    output_directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    try:
        if output_directory:
            os.chdir(output_directory)
        return perf_main(arguments)
    finally:
        os.chdir(original_directory)


if __name__ == "__main__":
    raise SystemExit(main())
