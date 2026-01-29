import argparse
import copy
import logging
import math
import os
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List

import yaml

# Global variables
logger = None
test_output_dir = None
experiment_output_dir = None

benchmark_separator = "=" * 120
task_separator = "-" * 120


def get_args():
    parser = argparse.ArgumentParser(description="Automated Benchmarking Script")
    parser.add_argument(
        "-c",
        "--config",
        default="./multi_benchmark_config.yaml",
        type=str,
        help="Path to the configuration file",
    )
    args = parser.parse_args()
    return args


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def convert_to_env(benchmark_config: Dict[str, Any]) -> Dict[str, str]:
    env = {}
    for key, value in benchmark_config.items():
        if not isinstance(value, (int, float, str, bool)):
            continue
        if isinstance(value, bool):
            env[key.upper()] = "1" if value else "0"
        else:
            env[key.upper()] = str(value)
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


def extract_table_from_log(log_file_path: str) -> str:
    """Extract table content from log file using improved method"""
    if not os.path.exists(log_file_path):
        return ""

    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        log_stage(f"Error reading log file {log_file_path}: {e}", stage="WARNING")
        return ""

    # Find table start position - ensure it's within table structure
    table_start = -1
    for i, line in enumerate(lines):
        # Check if line contains table title and has table borders
        if ("Decode Result" in line or "Prefill Result" in line) and "|" in line:
            # Check if previous line is table border
            if i > 0 and lines[i - 1].strip().startswith("+") and "-" in lines[i - 1]:
                table_start = i - 1  # Include border line
                break
            # Check if next line is table border
            elif (
                i + 1 < len(lines)
                and lines[i + 1].strip().startswith("+")
                and "-" in lines[i + 1]
            ):
                table_start = i
                break

    if table_start == -1:
        return ""

    # Find table end position
    table_end = -1
    for i in range(table_start, len(lines)):
        line = lines[i].strip()
        # Find lines starting with "+" and containing "-"
        if line.startswith("+") and "-" in line:
            # Check if next line is still table content
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # If next line is not table content (not starting with | or +), current line is end border
                if not next_line.startswith("|") and not next_line.startswith("+"):
                    table_end = i
                    break
            else:
                # If it's the last line, treat as end
                table_end = i
                break

    if table_end == -1:
        return ""

    # Extract table content
    table_lines = lines[table_start : table_end + 1]

    # Return table content, remove trailing newlines and recombine
    return "".join(line.rstrip() + "\n" for line in table_lines).rstrip()


def execute_task(
    task_config: Dict[str, Any],
    benchmark_output_dir: str,
    variable_dict: Dict[str, Any],
    copy_test_result: bool,
    num_retry_times: int,
) -> None:
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
    log_stage(f"Task Config: {multi_runner_env}", stage="TASK")
    multi_runner_args = [
        "sh",
        "./multi_runner.sh",
    ]
    # Retry test
    for i in range(num_retry_times + 1):
        # Run subprocess and redirect both stdout and stderr to file
        with open(task_test_output_path, "a", encoding="utf-8") as f:
            f.write(f"=== OUTPUT {i} ===\n")
            f.flush()
            result = subprocess.run(
                multi_runner_args,
                env=multi_runner_env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
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
        # Check if test was successful and extract table
        table_content = extract_table_from_log(task_test_output_path)
        if table_content and "N/A" not in table_content:
            log_stage("Test results table:", stage="TASK")
            log_stage("\n" + table_content, stage="TASK")
            # Copy test result and clean task result files
            if copy_test_result:
                # copy test result to local
                multi_copy_script(
                    {
                        "ip_lists": task_config["ip_lists"],
                        "run_user": task_config["run_user"],
                        "ssh_port": task_config["ssh_port"],
                        "ft_sub_dir": task_config["ft_sub_dir"],
                        "task_output_dir": task_output_dir,
                    },
                    copy_log_path=task_copy_output_path,
                )
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
            break
        else:
            if i < num_retry_times:
                log_stage(
                    f"Test may have failed - no table pattern found in log, please check log file, retry times: {i+1}......",
                    stage="WARNING",
                )
                time.sleep(10)
            else:
                log_stage(
                    "Test failed - no table pattern found in log, please check log file\n\n",
                    stage="ERROR",
                )
        # Copy test result and clean task result files
        if copy_test_result:
            # copy test result to local
            multi_copy_script(
                {
                    "ip_lists": task_config["ip_lists"],
                    "run_user": task_config["run_user"],
                    "ssh_port": task_config["ssh_port"],
                    "ft_sub_dir": task_config["ft_sub_dir"],
                    "task_output_dir": task_output_dir,
                },
                copy_log_path=task_copy_output_path,
            )
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
    ep_size = dp_size * tp_size
    world_size = ep_size
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
    multi_runner_args = ["sh", "./multi_runner.sh"]
    # Log build information
    log_stage(f"Build environment | Log: {build_log_path}", stage="BUILD")
    # Run build command and redirect both stdout and stderr to build log file
    for i in range(num_retry_times + 1):
        with open(build_log_path, "a", encoding="utf-8") as f:
            f.write(f"=== BUILD OUTPUT {i} ===\n")
            f.flush()
            result = subprocess.run(
                multi_runner_args,
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
    multi_runner_args = ["sh", "./multi_runner.sh"]
    # Log kill information
    log_stage(f"Kill processes | Log: {kill_log_path}", stage="KILL")
    # Run kill command and redirect both stdout and stderr to file
    with open(kill_log_path, "a", encoding="utf-8") as f:
        f.write("=== KILL OUTPUT ===\n")
        f.flush()
        result = subprocess.run(
            multi_runner_args,
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
    multi_runner_args = ["sh", "./multi_runner.sh"]
    # Log copy test results information
    log_stage(f"Copy test results | Log: {copy_log_path}", stage="COPY")
    # Run copy test results command and redirect both stdout and stderr to file
    with open(copy_log_path, "a", encoding="utf-8") as f:
        f.write("=== COPY TEST RESULTS OUTPUT ===\n")
        f.flush()
        result = subprocess.run(
            multi_runner_args,
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
    multi_runner_args = ["sh", "./multi_runner.sh"]
    # Log clean information
    log_stage(f"Clean log files | Log: {clean_log_path}", stage="CLEAN")
    # Run clean command and redirect both stdout and stderr to file
    with open(clean_log_path, "a", encoding="utf-8") as f:
        f.write("=== CLEAN OUTPUT ===\n")
        f.flush()
        result = subprocess.run(
            multi_runner_args,
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
    for benchmark_config in benchmarks:
        # Get benchmark name, fixed config, iterative search config and recursive search config
        benchmark_name = benchmark_config.get("benchmark_name", "unknown_benchmark")
        fixed_config = benchmark_config.get("fixed_config") or {}
        iterative_search_config = benchmark_config.get("iterative_search_config") or {}
        recursive_search_config = benchmark_config.get("recursive_search_config") or {}
        # Merge single fixed config
        fixed_config.update(machine_config)
        fixed_config.update(build_config)
        fixed_config.update(common_config)
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
            continue


if __name__ == "__main__":
    main()
