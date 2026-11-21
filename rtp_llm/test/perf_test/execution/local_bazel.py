"""Submit one frozen local plan into the platform's Bazel test environment."""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from dataclasses import asdict
from pathlib import Path

from rtp_llm.test.perf_test.cli import CliRequest
from rtp_llm.test.perf_test.errors import ConfigurationError
from rtp_llm.test.perf_test.launch_config import validate_bazel_arguments
from rtp_llm.test.perf_test.plan import LocalResourcePlan
from rtp_llm.test.perf_test.run_context import read_run_context
from rtp_llm.test.perf_test.storage import atomic_write_json

WORKSPACE = Path(__file__).resolve().parents[4]
TARGET = "//rtp_llm/test/perf_test:perf_test_local"


def local_command(plan, request, bundle_path):
    if plan.backend != "local" or not isinstance(plan.resource_plan, LocalResourcePlan):
        raise ConfigurationError("local Bazel submission requires a local plan")
    request = request or CliRequest(plan.mode, plan.action, None, None, (), False)
    validate_bazel_arguments(request.bazel_args, local=True)
    execution = plan.resolved_config["execution"]
    # Device IDs index the GPU wrapper's allocation. Acquire enough devices to
    # cover explicitly selected indices, as well as the model's world size.
    count = max(plan.resource_plan.devices) + 1
    timeout = execution.get("test_timeout_seconds", 7000)
    startup = list(request.bazel_startup_args)
    if not any(
        arg == "--output_user_root" or arg.startswith("--output_user_root=")
        for arg in startup
    ):
        identity = hashlib.sha256(str(WORKSPACE).encode()).hexdigest()[:12]
        cache = (
            Path.home()
            / ".cache"
            / ("bazel_perf_" + identity + "_" + execution.get("platform", "cuda13"))
        )
        startup.insert(0, "--output_user_root=" + str(cache))
    return [
        "bazelisk",
        *startup,
        "test",
        TARGET,
        "--config=" + execution.get("platform", "cuda13"),
        *request.bazel_args,
        "--test_strategy=local",
        "--strategy=TestRunner=local",
        "--nocache_test_results",
        "--test_output=streamed",
        f"--test_timeout={timeout}",
        f"--test_env=GPU_COUNT={count}",
        f"--test_env=WORLD_SIZE={count}",
        f"--test_env=RTP_TEST_WATCHDOG_SECONDS={timeout}",
        "--test_arg=--run-bundle=" + str(bundle_path),
    ]


def local_preview(plan, request=None):
    if plan.action == "report":
        return {"commands": []}
    bundle = (
        Path(plan.result_directory).parent
        / ".launch"
        / Path(plan.result_directory).name
        / "bundle.json"
    )
    return {
        "commands": [
            {"stage": "local-test", "argv": local_command(plan, request, bundle)}
        ]
    }


def execute_local_bazel(plan, source, request=None):
    request = request or CliRequest(plan.mode, plan.action, None, None, (), False)
    # Validate before creating any files or invoking the toolchain.
    local_command(plan, request, Path("bundle.json"))
    context_data = read_run_context(request.run_context)
    launch_dir = (
        Path(plan.result_directory).parent
        / ".launch"
        / Path(plan.result_directory).name
        / (plan.action + "-" + uuid.uuid4().hex)
    )
    launch_dir.mkdir(parents=True, exist_ok=False)
    bundle = launch_dir / "bundle.json"
    arguments = asdict(request)
    if request.run_context:
        context = launch_dir / "run_context.json"
        atomic_write_json(context, context_data)
        arguments["run_context"] = str(context)
    result_path = launch_dir / "worker_result.json"
    atomic_write_json(
        bundle,
        {
            "schema_version": 1,
            "plan": plan.to_dict(),
            "source": source,
            "request": arguments,
            "result_path": str(result_path),
        },
    )
    command = local_command(plan, request, bundle)
    atomic_write_json(
        launch_dir / "submission.json", {"argv": command, "cwd": str(WORKSPACE)}
    )
    completed = subprocess.run(command, cwd=WORKSPACE, check=False)
    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if not isinstance(result, dict) or type(result.get("exit_code")) is not int:
            raise ConfigurationError("invalid local worker result")
        # A passing worker cannot mask a Bazel/wrapper failure after completion.
        if completed.returncode and result["exit_code"] == 0:
            result["exit_code"] = completed.returncode
    else:
        result = {"exit_code": completed.returncode or 1}
    result.update(run_directory=plan.result_directory, launch_directory=str(launch_dir))
    return result
