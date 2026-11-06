import argparse
import copy
import itertools
import json
import logging
import math
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Support the existing standalone invocation from the multi_node directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from rtp_llm.test.perf_test.config_resolver import canonical_json
from rtp_llm.test.perf_test.decode.scheduler import requires_distributed_decode
from rtp_llm.test.perf_test.errors import (
    ConfigurationError,
    PartialRunError,
    PerfTestError,
)
from rtp_llm.test.perf_test.multi_node.decode_runtime import build_external_decode_plan
from rtp_llm.test.perf_test.multi_node.prefill_runtime import (
    build_external_prefill_plan,
)
from rtp_llm.test.perf_test.plan import thaw
from rtp_llm.test.perf_test.results import parse_result_envelope

# Global variables
logger = None
test_output_dir = None
experiment_output_dir = None

benchmark_separator = "=" * 120
task_separator = "-" * 120
SCRIPT_DIRECTORY = Path(__file__).resolve().parent


def get_args():
    parser = argparse.ArgumentParser(description="Automated Benchmarking Script")
    parser.add_argument(
        "-c",
        "--config",
        default=str(SCRIPT_DIRECTORY / "multi_benchmark_config.yaml"),
        type=str,
        help="Path to the configuration file",
    )
    args = parser.parse_args()
    return args


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    machine = config.get("machine_config") or {}
    build = config.get("build_config") or {}
    for key in machine.keys() & build.keys():
        if machine[key] != build[key]:
            raise ConfigurationError(
                f"{key} conflicts between machine_config and build_config"
            )
    # Validate every expanded task before main can build, clean, or contact hosts.
    for benchmark in config.get("benchmarks", []):
        fixed = benchmark.get("fixed_config")
        if not isinstance(fixed, dict):
            raise ConfigurationError("each benchmark requires fixed_config")
        path = fixed.get("perf_test_config")
        if not isinstance(path, str) or not path.strip():
            raise ConfigurationError("each benchmark requires perf_test_config")
        fixed["perf_test_config"] = str(
            (Path(config_path).resolve().parent / path).resolve()
        )
        common = {}
        for section in ("machine_config", "build_config", "common_config"):
            common.update(config.get(section) or {})
        # Build and cleanup happen once for the experiment. Every task must
        # inherit the same host and checkout configuration prepared there.
        shared = {
            **(config.get("machine_config") or {}),
            **(config.get("build_config") or {}),
        }
        shared_keys = shared.keys() | {
            "ip_lists",
            "run_user",
            "ssh_port",
            "git_repo_url",
            "git_checkout_ref",
            "open_source_url",
            "open_source_ref",
            "ft_sub_dir",
            "ft_work_dir",
            "bazel_build_args",
            "bazel_targets",
        }
        for scope in (config.get("common_config") or {}, fixed):
            for key in shared_keys & scope.keys():
                if scope[key] != shared.get(key):
                    raise ConfigurationError(
                        f"{key} must be configured globally in machine_config/build_config"
                    )
        common.update(fixed)
        iterative = benchmark.get("iterative_search_config") or {}
        recursive = benchmark.get("recursive_search_config") or {}
        for key in shared_keys & (iterative.keys() | recursive.keys()):
            raise ConfigurationError(
                f"{key} cannot be a search axis; configure it globally"
            )
        check_benchmark_iterative_search_config(iterative)
        check_benchmark_recursive_search_config(recursive)
        iterations = (
            [dict(zip(iterative, values)) for values in zip(*iterative.values())]
            if iterative
            else [{}]
        )
        for iteration in iterations:
            for values in itertools.product(*recursive.values()):
                task = {**common, **iteration, **dict(zip(recursive, values))}
                validate_benchmark(task)
    return config


def validate_benchmark(task_config, result_directory="preflight"):
    path = task_config.get("perf_test_config")
    if not path:
        raise ConfigurationError("each benchmark requires perf_test_config")
    tp = int(task_config["tp_size"])
    dp = int(task_config["dp_size"])
    source = json.loads(Path(path).read_text(encoding="utf-8"))
    kind = source.get("workload", {}).get("kind")
    if kind == "cache_grid":
        builder = build_external_prefill_plan
    elif kind in ("grid", "distribution"):
        builder = build_external_decode_plan
        requires_distributed_decode(
            {"tp": tp, "dp": dp, "pp": int(task_config.get("pp_size", 1))},
            int(task_config.get("local_world_size", 8)),
        )
    else:
        raise ConfigurationError(f"unsupported perf workload kind: {kind!r}")
    return builder(
        path,
        result_directory=result_directory,
        **(
            {"resolved_workload_path": task_config["resolved_workload_path"]}
            if kind == "cache_grid" and task_config.get("resolved_workload_path")
            else {}
        ),
        actual_topology={
            "tp": tp,
            "dp": dp,
            "pp": int(task_config.get("pp_size", 1)),
            "ep": int(task_config.get("ep_size", tp * dp)),
        },
    )


def validate_decode_benchmark(task_config, result_directory="preflight"):
    """Compatibility name for callers that still preflight decode tasks."""
    plan, source = validate_benchmark(task_config, result_directory)
    if plan.mode != "decode":
        raise ConfigurationError("decode benchmark requires a decode workload")
    return plan, source


def convert_to_env(benchmark_config: Dict[str, Any]) -> Dict[str, str]:
    env = {}
    for key, value in benchmark_config.items():
        if not isinstance(value, (int, float, str, bool)):
            continue
        if isinstance(value, bool):
            env[key.upper()] = "1" if value else "0"
        else:
            env[key.upper()] = str(value)
    if "perf_test_config" in benchmark_config:
        env["PERF_TEST_CONFIG_PATH"] = str(benchmark_config["perf_test_config"])
    return env


def ensure_test_output_dir():
    """Ensure test_output directory exists"""
    global test_output_dir
    if test_output_dir is None:
        test_output_dir = os.path.join(os.getcwd(), "test_output")
        os.makedirs(test_output_dir, exist_ok=True)
        test_output_dir = os.path.abspath(test_output_dir)
    return test_output_dir


def ensure_experiment_output_dir(experiment_name: str):
    """Create timestamped output directory for each experiment"""
    global experiment_output_dir
    if experiment_output_dir is None:
        now = datetime.now()
        timestamp = now.strftime(f"%Y%m%d-%H%M%S-{now.microsecond:06d}")
        experiment_output_dir = os.path.join(
            ensure_test_output_dir(), f"Experiment_{experiment_name}_{timestamp}"
        )
        os.makedirs(experiment_output_dir, exist_ok=True)
        experiment_output_dir = os.path.abspath(experiment_output_dir)
    return experiment_output_dir


def setup_logging():
    """Setup logging configuration with file output to experiment_output_dir/host.log"""
    global logger
    if experiment_output_dir is None:
        raise RuntimeError(
            "experiment_output_dir must be initialized before setting up logging"
        )
    # Log file path: experiment_output_dir/host.log
    log_file_path = os.path.join(experiment_output_dir, "host.log")
    log_file_path = os.path.abspath(log_file_path)
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(),  # Also output to console
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file_path}")
    return log_file_path


def log_stage(message: str, stage: str = ""):
    """Unified logging for different stages"""
    stage_markers = {
        "BENCHMARK": "[BENCHMARK]",
        "TASK": "[TASK]",
        "BUILD": "[BUILD]",
        "KILL": "[KILL]",
        "COPY": "[COPY]",
        "CLEAN": "[CLEAN]",
        "INFO": "[INFO]",
        "WARNING": "[WARNING]",
        "ERROR": "[ERROR]",
    }
    if stage == "":
        logger.info(f"{message}")
    else:
        stage_marker = stage_markers.get(stage, f"[{stage}]")
        logger.info(f"{stage_marker} {message}")


def extract_run_state(log_file_path: str, *, plan, run_id):
    """Find this master's terminal marker; collected artifacts remain authoritative."""
    if not os.path.exists(log_file_path):
        return None
    state = None
    with open(log_file_path, encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith(f"PERF_TEST_{plan.mode.upper()}_RESULT "):
                continue
            try:
                value = json.loads(line.split(" ", 1)[1])
            except json.JSONDecodeError:
                continue
            expected = {
                "run_id": run_id,
                "world_rank": 0,
                "config_fingerprint": plan.config_fingerprint,
                "workload_fingerprint": plan.workload_fingerprint,
            }
            if (
                isinstance(value, dict)
                and type(value.get("world_rank")) is int
                and value.get("state") in ("completed", "partial")
                and all(value.get(key) == item for key, item in expected.items())
            ):
                state = value["state"]
    return state


def extract_decode_state(log_file_path: str, *, plan, run_id):
    return extract_run_state(log_file_path, plan=plan, run_id=run_id)


def verify_collected_run(root, *, plan, run_id, state):
    """Accept only terminal artifacts copied from this attempt's master path."""
    root = Path(root)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        envelope = json.loads(
            (root / "results/result.json").read_text(encoding="utf-8")
        )
        if not isinstance(manifest, dict):
            raise ValueError("manifest must contain an object")
        resolved = json.loads(
            (root / "resolved_config.json").read_text(encoding="utf-8")
        )
        persisted_plan = json.loads(
            (root / "execution_plan.json").read_text(encoding="utf-8")
        )
        if canonical_json(resolved) != canonical_json(thaw(plan.resolved_config)):
            raise ValueError("persisted normalized config differs from the plan")
        if (
            not isinstance(persisted_plan, dict)
            or not isinstance(persisted_plan.get("result_directory"), str)
            or not persisted_plan["result_directory"].strip()
        ):
            raise ValueError("persisted execution plan requires a result directory")
        expected_plan = plan.to_dict()
        # The master's run root differs from both preflight and collected roots.
        expected_plan["result_directory"] = persisted_plan["result_directory"]
        if canonical_json(persisted_plan) != canonical_json(expected_plan):
            raise ValueError("persisted execution plan differs from the plan")
        parse_result_envelope(envelope)
        expected = {
            "run_id": run_id,
            "mode": plan.mode,
            "backend": "local",
            "config_fingerprint": plan.config_fingerprint,
            "workload_fingerprint": plan.workload_fingerprint,
        }
        if any(
            manifest.get(key) != value or envelope.get(key) != value
            for key, value in expected.items()
        ):
            raise ValueError("run identity or fingerprints differ")
        if manifest.get("state") != state or envelope.get("status") != state:
            raise ValueError("terminal states differ")
        identities = {case.case_id: thaw(case.identity) for case in plan.cases}
        cases = envelope["cases"]
        if (
            manifest.get("case_ids") != [case.case_id for case in plan.cases]
            or len(cases) != len(identities)
            or {case["case_id"]: case["identity"] for case in cases} != identities
        ):
            raise ValueError("persisted cases differ from the plan")
        actual_state = (
            "completed" if all(case["status"] == "ok" for case in cases) else "partial"
        )
        if actual_state != state:
            raise ValueError("case statuses differ from terminal state")
    except (
        OSError,
        UnicodeError,
        ValueError,
        TypeError,
        KeyError,
        PerfTestError,
    ) as error:
        raise RuntimeError(
            f"persisted {plan.mode} run verification failed: {error}"
        ) from error


def verify_collected_decode_run(root, *, plan, run_id, state):
    return verify_collected_run(root, plan=plan, run_id=run_id, state=state)


def execute_task(
    task_config: Dict[str, Any],
    benchmark_output_dir: str,
    variable_dict: Dict[str, Any],
    copy_test_result: bool,
    num_retry_times: int,
) -> None:
    plan, source = validate_benchmark(task_config)
    # Create task output directory
    task_output_name = "Task_" + "_".join(f"{k}-{v}" for k, v in variable_dict.items())
    task_output_dir = os.path.join(benchmark_output_dir, task_output_name)
    os.makedirs(task_output_dir, exist_ok=False)
    task_output_dir = os.path.abspath(task_output_dir)
    task_test_output_path = os.path.abspath(os.path.join(task_output_dir, "test.log"))
    task_kill_output_path = os.path.abspath(os.path.join(task_output_dir, "kill.log"))
    task_copy_output_path = os.path.abspath(os.path.join(task_output_dir, "copy.log"))
    task_clean_output_path = os.path.abspath(os.path.join(task_output_dir, "clean.log"))
    task_trace_files_dir = os.path.abspath(os.path.join(task_output_dir, "trace_files"))
    os.makedirs(task_trace_files_dir, exist_ok=False)
    task_process_logs_dir = os.path.abspath(
        os.path.join(task_output_dir, "process_logs")
    )
    os.makedirs(task_process_logs_dir, exist_ok=False)
    # Log task information
    log_stage(f"{task_separator} Task Running: {task_output_name} {task_separator}")
    log_stage(f"Task Output Directory: {task_output_dir}", stage="TASK")
    log_stage(f"Task Test Output Path: {task_test_output_path}", stage="TASK")
    log_stage(f"Task Kill Output Path: {task_kill_output_path}", stage="TASK")
    log_stage(f"Task Copy Output Path: {task_copy_output_path}", stage="TASK")
    log_stage(f"Task Clean Output Path: {task_clean_output_path}", stage="TASK")
    log_stage(f"Task Trace Files Directory: {task_trace_files_dir}", stage="TASK")
    log_stage(f"Task Process Logs Directory: {task_process_logs_dir}", stage="TASK")
    log_stage(f"Task Variable Dict: {variable_dict}", stage="TASK")
    # Convert task config to environment variables
    multi_runner_env = convert_to_env(task_config)
    multi_runner_env["PERF_TEST_MODE"] = plan.mode
    config_dir = Path(task_output_dir) / "config"
    config_dir.mkdir()
    if plan.mode == "prefill":
        # Freeze traffic distributions on the launcher; remote hosts do not
        # need the original external distribution file.
        from rtp_llm.test.perf_test.prefill.traffic_batch import (
            materialize_traffic_source,
        )

        source = materialize_traffic_source(
            source, thaw(plan.resolved_config["workload"])
        )
        resolved_workload_path = config_dir / "resolved_workload.json"
        resolved_workload_path.write_text(
            json.dumps(thaw(plan.resolved_config["workload"])), encoding="utf-8"
        )
        multi_runner_env["PERF_TEST_RESOLVED_WORKLOAD_PATH"] = str(
            resolved_workload_path
        )
    if source["workload"]["kind"] == "distribution":
        # Keep the delivered config self-contained; relative distributions must
        # not point back to the launcher's filesystem on a remote rank.
        distribution_path = config_dir / "distribution.json"
        distribution_path.write_text(
            json.dumps(thaw(plan.resolved_config)["workload"]["distribution"]),
            encoding="utf-8",
        )
        source["workload"]["distribution_file"] = "distribution.json"
        multi_runner_env["PERF_TEST_DISTRIBUTION_PATH"] = str(distribution_path)
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps(source), encoding="utf-8")
    plan, source = validate_benchmark(
        {
            **task_config,
            "perf_test_config": str(config_path),
            "resolved_workload_path": multi_runner_env.get(
                "PERF_TEST_RESOLVED_WORKLOAD_PATH"
            ),
        }
    )
    multi_runner_env["PERF_TEST_CONFIG_PATH"] = str(config_path)
    log_stage(f"Task Config: {multi_runner_env}", stage="TASK")
    multi_runner_args = [
        "bash",
        str(SCRIPT_DIRECTORY / "multi_runner.sh"),
    ]
    # Retry test
    terminal_state = None
    for i in range(num_retry_times + 1):
        multi_runner_env["PERF_TEST_RUN_ID"] = uuid.uuid4().hex
        # Run subprocess and redirect both stdout and stderr to file
        with open(task_test_output_path, "a", encoding="utf-8") as f:
            f.write(f"=== OUTPUT {i} ===\n")
            f.flush()
            result = subprocess.run(
                multi_runner_args,
                cwd=SCRIPT_DIRECTORY,
                env=multi_runner_env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                # Preserve launcher failures for terminal-state validation below.
                # A bound marker triggers collection; persisted artifacts prove completion.
                check=False,
            )
            f.write(f"\n\n=== Return Code: {result.returncode} ===\n")
            f.flush()
        # kill task processes
        multi_kill_script(
            {
                "ip_lists": task_config["ip_lists"],
                "run_user": task_config["run_user"],
                "ssh_port": task_config["ssh_port"],
            },
            kill_log_path=task_kill_output_path,
        )
        # A partial run still has diagnostic artifacts that must be collected.
        state = extract_run_state(
            task_test_output_path,
            plan=plan,
            run_id=multi_runner_env["PERF_TEST_RUN_ID"],
        )
        if state:
            multi_copy_script(
                {
                    "ip_lists": task_config["ip_lists"],
                    "run_user": task_config["run_user"],
                    "ssh_port": task_config["ssh_port"],
                    "ft_sub_dir": task_config["ft_sub_dir"],
                    "task_output_dir": task_output_dir,
                    "perf_test_run_id": multi_runner_env["PERF_TEST_RUN_ID"],
                    "local_world_size": task_config.get("local_world_size", 8),
                    "collect_diagnostics": (
                        copy_test_result or state == "partial" or result.returncode != 0
                    ),
                },
                copy_log_path=task_copy_output_path,
            )
            verify_collected_run(
                Path(task_output_dir) / "run",
                plan=plan,
                run_id=multi_runner_env["PERF_TEST_RUN_ID"],
                state=state,
            )
            log_stage(f"{plan.mode} run finalized: {state}", stage="TASK")
            terminal_state = state
            break
        else:
            # Early failures may have logs/manifests but no complete result tree.
            # Never delete remote evidence if even best-effort collection fails.
            try:
                multi_copy_script(
                    {
                        "ip_lists": task_config["ip_lists"],
                        "run_user": task_config["run_user"],
                        "ssh_port": task_config["ssh_port"],
                        "ft_sub_dir": task_config["ft_sub_dir"],
                        "task_output_dir": str(
                            Path(task_output_dir)
                            / "failed_attempts"
                            / multi_runner_env["PERF_TEST_RUN_ID"]
                        ),
                        "perf_test_run_id": multi_runner_env["PERF_TEST_RUN_ID"],
                        "local_world_size": task_config.get("local_world_size", 8),
                        "collect_diagnostics": True,
                        "diagnostics_only": True,
                    },
                    copy_log_path=task_copy_output_path,
                )
            except (OSError, subprocess.CalledProcessError):
                raise RuntimeError(
                    "diagnostic collection failed; remote output retained"
                )
            multi_clean_script(
                {
                    "ip_lists": task_config["ip_lists"],
                    "run_user": task_config["run_user"],
                    "ssh_port": task_config["ssh_port"],
                    "ft_sub_dir": task_config["ft_sub_dir"],
                },
                clean_log_path=task_clean_output_path,
            )
            if i < num_retry_times:
                log_stage(
                    f"{plan.mode} run did not finalize; check log file, retry times: {i+1}",
                    stage="WARNING",
                )
            else:
                log_stage(
                    f"{plan.mode} run did not finalize; check log file\n\n",
                    stage="ERROR",
                )
    if terminal_state is None:
        raise RuntimeError(f"{plan.mode} run did not finalize; see task test.log")
    # Copy test result and clean task result files
    if copy_test_result or terminal_state == "partial" or result.returncode != 0:
        # clean task result files
        multi_clean_script(
            {
                "ip_lists": task_config["ip_lists"],
                "run_user": task_config["run_user"],
                "ssh_port": task_config["ssh_port"],
                "ft_sub_dir": task_config["ft_sub_dir"],
            },
            clean_log_path=task_clean_output_path,
        )
    if terminal_state == "partial":
        raise PartialRunError(
            f"{plan.mode} run is partial; diagnostic artifacts were collected"
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"{plan.mode} rank failure: SSH runner exited {result.returncode}; "
            "diagnostic artifacts were collected"
        )


def search_recursive_config(
    fixed_config: Dict[str, Any],
    iterative_config: Dict[str, Any],
    recursive_search_config: Dict[str, Any],
    variable_dict: Dict[str, Any],
    benchmark_output_dir: str,
    copy_test_result: bool,
    num_retry_times: int,
) -> None:
    # Search parameter groups
    for param_key in recursive_search_config.keys():
        if isinstance(recursive_search_config[param_key], list):
            params_values = recursive_search_config[param_key]
            for param_value in params_values:
                recursive_search_config[param_key] = param_value
                variable_dict[param_key] = param_value
                search_recursive_config(
                    fixed_config,
                    iterative_config,
                    recursive_search_config,
                    variable_dict,
                    benchmark_output_dir,
                    copy_test_result,
                    num_retry_times,
                )
            variable_dict.pop(param_key)
            recursive_search_config[param_key] = params_values
            return
    # Merge task config
    task_config = copy.deepcopy(fixed_config)
    task_config.update(iterative_config)
    task_config.update(recursive_search_config)
    # Process parallel config
    dp_size = task_config.get("dp_size", None)
    tp_size = task_config.get("tp_size", None)
    assert (
        dp_size is not None and tp_size is not None
    ), "dp_size and tp_size must be provided"
    ep_size = task_config.get("ep_size", dp_size * tp_size)
    world_size = dp_size * tp_size * task_config.get("pp_size", 1)
    local_world_size = task_config.get("local_world_size", 8)
    local_world_size = world_size if world_size < local_world_size else local_world_size
    used_ip_lists = get_used_ip_lists(
        task_config.pop("ip_lists"), world_size, local_world_size
    )
    if world_size % local_world_size != 0:
        raise ValueError(
            f"World size {world_size} must be a multiple of local world size {local_world_size}"
        )
    if len(used_ip_lists) < math.ceil(world_size / local_world_size):
        raise ValueError(
            f"Not enough IP addresses provided for world size {world_size} and local world size {local_world_size}"
        )
    if local_world_size not in [1, 2, 4, 8]:
        raise ValueError(
            f"Local world size {local_world_size} must be one of [1, 2, 4, 8]"
        )
    task_config["ep_size"] = ep_size
    task_config["world_size"] = world_size
    task_config["local_world_size"] = local_world_size
    task_config["ip_lists"] = ",".join(used_ip_lists)
    # Execute task
    execute_task(
        task_config,
        benchmark_output_dir,
        variable_dict,
        copy_test_result=copy_test_result,
        num_retry_times=num_retry_times,
    )


def search_iterative_config(
    fixed_config: Dict[str, Any],
    iterative_search_config: Dict[str, Any],
    recursive_search_config: Dict[str, Any],
    benchmark_output_dir: str,
    copy_test_result: bool,
    num_retry_times: int,
) -> None:
    if len(iterative_search_config) == 0:
        search_recursive_config(
            fixed_config,
            {},
            recursive_search_config,
            {},
            benchmark_output_dir,
            copy_test_result=copy_test_result,
            num_retry_times=num_retry_times,
        )
    else:
        for idx in range(len(list(iterative_search_config.values())[0])):
            iterative_config = {k: v[idx] for k, v in iterative_search_config.items()}
            variable_dict = copy.deepcopy(iterative_config)
            search_recursive_config(
                fixed_config,
                iterative_config,
                recursive_search_config,
                variable_dict,
                benchmark_output_dir,
                copy_test_result=copy_test_result,
                num_retry_times=num_retry_times,
            )


def check_build_success(log_file_path: str) -> bool:
    """Check if build log contains success message"""
    if not os.path.exists(log_file_path):
        return False
    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Check for build success message
        return "Build completed successfully" in content
    except Exception as e:
        log_stage(f"Error reading build log file {log_file_path}: {e}", stage="WARNING")
        return False


def multi_build_script(
    machine_config: Dict[str, Any],
    build_config: Dict[str, Any],
    build_log_path: str,
    build_from_scratch: int,
    num_retry_times: int,
) -> None:
    # Prepare build config
    copy_build_config = copy.deepcopy(build_config)
    copy_build_config.update(machine_config)
    copy_build_config["sub_cmd"] = "build"
    copy_build_config["build_from_scratch"] = build_from_scratch
    copy_build_config["ip_lists"] = ",".join(copy_build_config["ip_lists"])
    # Convert build config to environment variables
    multi_runner_env = convert_to_env(copy_build_config)
    multi_runner_args = ["bash", str(SCRIPT_DIRECTORY / "multi_runner.sh")]
    # Log build information
    log_stage(f"Build environment | Log: {build_log_path}", stage="BUILD")
    # Run build command and redirect both stdout and stderr to build log file
    for i in range(num_retry_times + 1):
        with open(build_log_path, "a", encoding="utf-8") as f:
            f.write(f"=== BUILD OUTPUT {i} ===\n")
            f.flush()
            result = subprocess.run(
                multi_runner_args,
                cwd=SCRIPT_DIRECTORY,
                env=multi_runner_env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
            f.write(f"\n\n=== Build Return Code: {result.returncode} ===\n")
            f.flush()
        # Check if build was successful
        if check_build_success(build_log_path):
            log_stage("Environment build completed successfully\n", stage="BUILD")
            break
        else:
            if i < num_retry_times:
                log_stage(
                    f"Build may have failed - no success message found in log, please check log file, retry times: {i+1}......",
                    stage="WARNING",
                )
            else:
                log_stage(
                    "Build failed - no success message found in log, please check log file\n",
                    stage="ERROR",
                )
                raise RuntimeError("Test environment build failed")


def multi_kill_script(copy_kill_config: Dict[str, Any], kill_log_path: str) -> None:
    # Prepare kill config
    copy_kill_config["sub_cmd"] = "kill"
    if isinstance(copy_kill_config["ip_lists"], list):
        copy_kill_config["ip_lists"] = ",".join(copy_kill_config["ip_lists"])
    # Convert kill config to environment variables
    multi_runner_env = convert_to_env(copy_kill_config)
    multi_runner_args = ["bash", str(SCRIPT_DIRECTORY / "multi_runner.sh")]
    # Log kill information
    log_stage(f"Kill processes | Log: {kill_log_path}", stage="KILL")
    # Run kill command and redirect both stdout and stderr to file
    with open(kill_log_path, "a", encoding="utf-8") as f:
        f.write("=== KILL OUTPUT ===\n")
        f.flush()
        result = subprocess.run(
            multi_runner_args,
            cwd=SCRIPT_DIRECTORY,
            env=multi_runner_env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        f.write(f"\n\n=== Kill Return Code: {result.returncode} ===\n")
        f.flush()
    log_stage("Processes killed successfully", stage="KILL")


def multi_copy_script(copy_copy_config: Dict[str, Any], copy_log_path: str) -> None:
    # Prepare clean config
    copy_copy_config["sub_cmd"] = "copy"
    if isinstance(copy_copy_config["ip_lists"], list):
        copy_copy_config["ip_lists"] = ",".join(copy_copy_config["ip_lists"])
    # Convert clean config to environment variables
    multi_runner_env = convert_to_env(copy_copy_config)
    multi_runner_args = ["bash", str(SCRIPT_DIRECTORY / "multi_runner.sh")]
    # Log copy test results information
    log_stage(f"Copy test results | Log: {copy_log_path}", stage="COPY")
    # Run copy test results command and redirect both stdout and stderr to file
    with open(copy_log_path, "a", encoding="utf-8") as f:
        f.write("=== COPY TEST RESULTS OUTPUT ===\n")
        f.flush()
        result = subprocess.run(
            multi_runner_args,
            cwd=SCRIPT_DIRECTORY,
            env=multi_runner_env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        f.write(f"\n\n=== Copy Test Results Return Code: {result.returncode} ===\n")
        f.flush()
    log_stage("Test results copied successfully", stage="COPY")


def multi_clean_script(copy_clean_config: Dict[str, Any], clean_log_path: str) -> None:
    # Prepare clean config
    copy_clean_config["sub_cmd"] = "clean"
    if isinstance(copy_clean_config["ip_lists"], list):
        copy_clean_config["ip_lists"] = ",".join(copy_clean_config["ip_lists"])
    # Convert clean config to environment variables
    multi_runner_env = convert_to_env(copy_clean_config)
    multi_runner_args = ["bash", str(SCRIPT_DIRECTORY / "multi_runner.sh")]
    # Log clean information
    log_stage(f"Clean log files | Log: {clean_log_path}", stage="CLEAN")
    # Run clean command and redirect both stdout and stderr to file
    with open(clean_log_path, "a", encoding="utf-8") as f:
        f.write("=== CLEAN OUTPUT ===\n")
        f.flush()
        result = subprocess.run(
            multi_runner_args,
            cwd=SCRIPT_DIRECTORY,
            env=multi_runner_env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        f.write(f"\n\n=== Clean Return Code: {result.returncode} ===\n")
        f.flush()
    log_stage("Log files cleaned successfully\n\n", stage="CLEAN")


def multi_test_script(
    fixed_config: Dict[str, Any],
    iterative_search_config: Dict[str, Any],
    recursive_search_config: Dict[str, Any],
    copy_test_result: bool,
    num_retry_times: int,
) -> None:
    def _convert_int_to_list(value: Any) -> List[int]:
        if isinstance(value, int):
            return [value]
        elif isinstance(value, list):
            return value
        else:
            raise ValueError(f"Invalid type for parameter '{value}': {type(value)}")

    # Create output directory for current benchmark
    global experiment_output_dir
    benchmark_name = fixed_config.pop("benchmark_name", "unknown_benchmark")
    now = datetime.now()
    timestamp = now.strftime(f"%Y%m%d-%H%M%S-{now.microsecond:06d}")
    benchmark_output_dir = os.path.join(
        experiment_output_dir, f"Benchmark_{benchmark_name}_{timestamp}"
    )
    os.makedirs(benchmark_output_dir, exist_ok=False)
    benchmark_output_dir = os.path.abspath(benchmark_output_dir)

    # Log benchmark information
    log_stage(
        f"{benchmark_separator} Benchmark Running: {benchmark_name} {benchmark_separator}"
    )
    log_stage(f"Benchmark Output Directory: {benchmark_output_dir}", stage="BENCHMARK")

    # Calculate total number of test tasks
    num_test_tasks = 1
    num_test_tasks *= (
        len(list(iterative_search_config.values())[0])
        if len(iterative_search_config) > 0
        else 1
    )
    num_test_tasks *= math.prod(len(lst) for lst in recursive_search_config.values())
    log_stage(f"Total number of test tasks: {num_test_tasks}", stage="BENCHMARK")

    # Search iterative config
    search_iterative_config(
        fixed_config,
        iterative_search_config,
        recursive_search_config,
        benchmark_output_dir,
        copy_test_result=copy_test_result,
        num_retry_times=num_retry_times,
    )
    log_stage(f"Benchmark Completed: {benchmark_name}\n\n\n\n", stage="BENCHMARK")


def get_used_ip_lists(
    ip_lists: List[str], world_size: int, local_world_size: int
) -> List[str]:
    num_all_nodes = len(ip_lists)
    num_used_nodes = math.ceil(world_size / local_world_size)
    assert (
        num_used_nodes <= num_all_nodes
    ), f"Number of used nodes {num_used_nodes} must not be greater than number of all nodes {num_all_nodes}"
    return ip_lists[:num_used_nodes]


def check_machine_and_build_config(
    machine_config: Dict[str, Any], build_config: Dict[str, Any]
) -> None:
    assert "ip_lists" in machine_config, "ip_lists must be provided"
    assert (
        len(machine_config["ip_lists"]) > 0
    ), "at least one useful machine must be provided"
    assert "run_user" in machine_config, "run_user must be provided"
    assert "ssh_port" in machine_config, "ssh_port must be provided"
    assert (
        "git_repo_url" in build_config
        and "git_checkout_ref" in build_config
        and "open_source_url" not in build_config
    ) or (
        "git_repo_url" not in build_config
        and "git_checkout_ref" not in build_config
        and "open_source_url" in build_config
    ), "git_repo_url, git_checkout_ref, open_source_url and open_source_ref settings are unreasonable"
    assert "open_source_ref" in build_config, "open_source_ref must be provided"
    assert "ft_sub_dir" in build_config, "ft_sub_dir must be provided"
    assert "bazel_build_args" in build_config, "bazel_build_args must be provided"


def check_benchmark_fixed_config(fixed_config: Dict[str, Any]) -> None:
    for config_key, config_value in fixed_config.items():
        assert isinstance(config_value, (int, float, str, bool)) or config_key in [
            "ip_lists"
        ], f"Invalid type for parameter '{config_key}': {type(config_value)}"


def check_benchmark_iterative_search_config(
    iterative_search_config: Dict[str, Any]
) -> None:
    assert all(
        [
            isinstance(config_value, list)
            for config_value in iterative_search_config.values()
        ]
    ), "all values in iterative_search_config must be a list"
    if len(iterative_search_config) == 0:
        return
    iterative_search_config_len = len(list(iterative_search_config.values())[0])
    assert all(
        [
            len(config_value) == iterative_search_config_len
            for config_value in iterative_search_config.values()
        ]
    ), "all values in iterative_search_config must have the same length"


def check_benchmark_recursive_search_config(
    recursive_search_config: Dict[str, Any]
) -> None:
    assert all(
        [
            isinstance(config_value, list)
            for config_value in recursive_search_config.values()
        ]
    ), "all values in recursive_search_config must be a list"


def main():
    # Parse args
    args = get_args()
    # Load experiment config
    config = load_config(args.config)
    # Parse experiment parameters from config
    experiment_name = config.get("experiment_name", "unknown_experiment")
    test_config = config.get("test_config") or {}
    num_retry_times = test_config.get("num_retry_times", 3)
    build_from_scratch = test_config.get("build_from_scratch", 2)
    copy_test_result = test_config.get("copy_test_result", True)
    machine_config = config.get("machine_config") or {}
    build_config = config.get("build_config") or {}
    common_config = config.get("common_config") or {}
    benchmarks = config.get("benchmarks", [])
    assert len(benchmarks) > 0, "experiment must have at least one benchmark"
    # Check config legitimacy
    check_machine_and_build_config(machine_config, build_config)

    # Make sure test output directory and benchmark output directory are created
    ensure_test_output_dir()
    ensure_experiment_output_dir(experiment_name)
    # Setup logging
    log_file_path = setup_logging()
    # Create build, kill and clean log files
    build_log_path = os.path.join(experiment_output_dir, "build.log")
    kill_log_path = os.path.join(experiment_output_dir, "kill.log")
    clean_log_path = os.path.join(experiment_output_dir, "clean.log")

    # Print directory of test output and benchmark output and log file
    log_stage(f"Test output directory: {test_output_dir}", stage="INFO")
    log_stage(f"Experiment output directory: {experiment_output_dir}", stage="INFO")
    log_stage(f"Host log file: {log_file_path}", stage="INFO")
    log_stage(f"Build log file: {build_log_path}", stage="INFO")
    log_stage(f"Kill log file: {kill_log_path}", stage="INFO")
    log_stage(f"Clean log file: {clean_log_path}", stage="INFO")

    # Build test environment before running all benchmarks
    if build_from_scratch > 0:
        multi_build_script(
            machine_config,
            build_config,
            build_log_path=build_log_path,
            build_from_scratch=build_from_scratch,
            num_retry_times=num_retry_times,
        )
    # Make sure all processes are killed before running benchmarks
    copy_kill_config = copy.deepcopy(machine_config)
    multi_kill_script(copy_kill_config, kill_log_path=kill_log_path)
    # Make sure all test result files are cleaned before running benchmarks
    copy_clean_config = copy.deepcopy(machine_config)
    copy_clean_config["ft_sub_dir"] = build_config["ft_sub_dir"]
    multi_clean_script(copy_clean_config, clean_log_path=clean_log_path)
    # Run all benchmarks
    exit_code = 0
    for benchmark_config in benchmarks:
        # Get benchmark name, fixed config, iterative search config and recursive search config
        benchmark_name = benchmark_config.get("benchmark_name", "unknown_benchmark")
        benchmark_fixed = benchmark_config.get("fixed_config") or {}
        fixed_config = {}
        iterative_search_config = benchmark_config.get("iterative_search_config") or {}
        recursive_search_config = benchmark_config.get("recursive_search_config") or {}
        # Merge single fixed config
        fixed_config.update(machine_config)
        fixed_config.update(build_config)
        fixed_config.update(common_config)
        fixed_config.update(benchmark_fixed)
        fixed_config["sub_cmd"] = "test"
        fixed_config["build_from_scratch"] = 0
        fixed_config["benchmark_name"] = benchmark_name
        # Check config legitimacy
        check_benchmark_fixed_config(fixed_config)
        check_benchmark_iterative_search_config(iterative_search_config)
        check_benchmark_recursive_search_config(recursive_search_config)
        try:
            time.sleep(10)  # wait for all machines to be ready
            multi_test_script(
                fixed_config,
                iterative_search_config,
                recursive_search_config,
                copy_test_result=copy_test_result,
                num_retry_times=num_retry_times,
            )
        except Exception as e:
            log_stage(f"Benchmark {benchmark_name} failed: {e}", stage="ERROR")
            exit_code = max(exit_code, getattr(e, "exit_code", 1))
            continue
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
