import argparse
import copy
import datetime
import json
import logging
import math
import os
import pprint
import re
import subprocess
from typing import Any, Dict, List

import yaml

# Configure logging - simplified format
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Global variables for output directories
test_output_dir = None
benchmark_output_dir = None


def get_args():
    parser = argparse.ArgumentParser(description="Automated Benchmarking Script")
    parser.add_argument(
        "-m",
        "--mode",
        default="run",
        type=str,
        help="Mode",
        choices=["run", "clean"],
    )
    parser.add_argument(
        "-s",
        "--scratch",
        action="store_true",
        help="Run with scratch, doing some init job like install pip, make softlink",
    )
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
    env = os.environ.copy()
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
    return test_output_dir


def ensure_benchmark_output_dir(benchmark_name: str):
    """Create timestamped output directory for each benchmark"""
    global benchmark_output_dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_output_dir = os.path.join(
        ensure_test_output_dir(), f"{benchmark_name}_{timestamp}"
    )
    os.makedirs(benchmark_output_dir, exist_ok=True)
    return benchmark_output_dir


def log_stage(stage: str, message: str = ""):
    """Unified logging for different stages"""
    stage_markers = {
        "START": "[START]",
        "RUNNING": "[RUNNING]",
        "COMPLETED": "[COMPLETED]",
        "ERROR": "[ERROR]",
        "INFO": "[INFO]",
        "WARNING": "[WARNING]",
    }
    marker = stage_markers.get(stage, f"[{stage}]")
    if message:
        logger.info(f"{marker} {message}")
    else:
        logger.info(f"{marker}")


def extract_table_from_log(log_file_path: str) -> str:
    """Extract table content from log file using improved method"""
    if not os.path.exists(log_file_path):
        return ""

    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        log_stage("WARNING", f"Error reading log file {log_file_path}: {e}")
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


def search_parameter_groups(
    benchmark_config: Dict[str, Any], benchmark_name: str
) -> None:
    multi_runner_env = convert_to_env(benchmark_config)
    multi_runner_args = [
        "sh",
        "./multi_runner.sh",
    ]

    # Create output filename with tp_size and other parameters
    tp_size = benchmark_config.get("tp_size", "1")
    dp_size = benchmark_config.get("dp_size", "1")
    ep_size = benchmark_config.get("ep_size", "1")
    world_size = benchmark_config.get("world_size", "1")

    output_filename = f"output_tp{tp_size}_dp{dp_size}_ep{ep_size}_ws{world_size}.log"
    output_filepath = os.path.join(benchmark_output_dir, output_filename)
    output_filepath_abs = os.path.abspath(output_filepath)

    # Unified progress display with absolute path
    log_stage(
        "RUNNING",
        f"Test: TP={tp_size}, DP={dp_size}, EP={ep_size}, WS={world_size} | Log: {output_filepath_abs}",
    )

    # Run subprocess and redirect both stdout and stderr to file
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write("=== OUTPUT ===\n")
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

    # Check if test was successful and extract table
    table_content = extract_table_from_log(output_filepath)
    if table_content:
        log_stage("INFO", "Test results table:")
        print(table_content)
    else:
        log_stage(
            "WARNING",
            f"Test may have failed - no table pattern found in log, please check log file",
        )


def search_combined_parameter_groups(benchmark_config: Dict[str, Any]) -> None:

    def _convert_int_to_list(value: Any) -> List[int]:
        if isinstance(value, int):
            return [value]
        elif isinstance(value, list):
            return value
        else:
            raise ValueError(f"Invalid type for parameter '{value}': {type(value)}")

    benchmark_name = benchmark_config.pop("name", "unknown_benchmark")
    benchmark_description = benchmark_config.pop("description", "")

    # Create output directory for current benchmark
    ensure_benchmark_output_dir(benchmark_name)

    log_stage(
        "START", f"Benchmark: {benchmark_name}, Description: {benchmark_description}"
    )
    ip_lists = benchmark_config.pop("ip_lists", [])
    dp_size_list = _convert_int_to_list(benchmark_config.pop("dp_size", []))
    tp_size_list = _convert_int_to_list(benchmark_config.pop("tp_size", []))

    if len(dp_size_list) != len(tp_size_list):
        raise ValueError("Inconsistent parameter list lengths")

    total_tests = len(tp_size_list)
    log_stage("INFO", f"Planning to run {total_tests} test configurations")

    for i, (tp_size, dp_size) in enumerate(zip(tp_size_list, dp_size_list), 1):
        ep_size = dp_size * tp_size
        world_size = ep_size
        local_world_size = 8 if world_size >= 8 else world_size
        if world_size % local_world_size != 0:
            raise ValueError(
                f"World size {world_size} must be a multiple of local world size {local_world_size}"
            )

        if len(ip_lists) < math.ceil(world_size / local_world_size):
            raise ValueError(
                f"Not enough IP addresses provided for world size {world_size} and local world size {local_world_size}"
            )

        if tp_size * dp_size != ep_size:
            raise ValueError(
                f"TP size {tp_size} and DP size {dp_size} must match EP size {ep_size}"
            )

        if world_size != ep_size:
            raise ValueError(f"World size {world_size} must match EP size {ep_size}")

        if local_world_size not in [1, 2, 4, 8]:
            raise ValueError(
                f"Local world size {local_world_size} must be one of [1, 2, 4, 8]"
            )

        benchmark_config["tp_size"] = tp_size
        benchmark_config["dp_size"] = dp_size
        benchmark_config["ep_size"] = ep_size
        benchmark_config["world_size"] = world_size
        benchmark_config["local_world_size"] = local_world_size
        benchmark_config["ip_lists"] = ",".join(
            ip_lists[: math.ceil(world_size / local_world_size)]
        )
        benchmark_config["build_from_scratch"] = 1
        benchmark_config["sub_cmd"] = "test"

        search_parameter_groups(benchmark_config, benchmark_name)

    log_stage("COMPLETED", f"Benchmark: {benchmark_name}")


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
        log_stage("WARNING", f"Error reading build log file {log_file_path}: {e}")
        return False


def multi_build_script(origin_config: Dict[str, Any], ip_lists: List[str]) -> None:
    config = copy.deepcopy(origin_config)
    config["sub_cmd"] = "build"
    config["build_from_scratch"] = 2
    multi_runner_env = convert_to_env(config)
    multi_runner_env["IP_LISTS"] = ",".join(ip_lists)
    multi_runner_args = ["sh", "./multi_runner.sh"]

    # Ensure test_output directory exists
    ensure_test_output_dir()
    build_log_path = os.path.join(test_output_dir, "build.log")
    build_log_path_abs = os.path.abspath(build_log_path)

    log_stage("START", f"Test environment build | Log: {build_log_path_abs}")

    # Run build command and redirect both stdout and stderr to file
    with open(build_log_path, "w", encoding="utf-8") as f:
        f.write("=== BUILD OUTPUT ===\n")
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

    # Check if build was successful
    if check_build_success(build_log_path):
        log_stage("COMPLETED", f"Test environment build completed successfully")
    else:
        log_stage(
            "ERROR",
            f"Build may have failed - 'Build completed successfully' not found in log, please check log file",
        )


def multi_kill_script(origin_config: Dict[str, Any], ip_lists: List[str]) -> None:
    config = copy.deepcopy(origin_config)
    config["sub_cmd"] = "kill"
    multi_runner_env = convert_to_env(config)
    multi_runner_env["IP_LISTS"] = ",".join(ip_lists)
    multi_runner_args = ["sh", "./multi_runner.sh"]

    # Ensure test_output directory exists
    ensure_test_output_dir()
    kill_log_path = os.path.join(test_output_dir, "kill.log")
    kill_log_path_abs = os.path.abspath(kill_log_path)

    log_stage("START", f"Test processes cleanup | Log: {kill_log_path_abs}")

    # Run kill command and redirect both stdout and stderr to file
    with open(kill_log_path, "w", encoding="utf-8") as f:
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

    log_stage("COMPLETED", f"Test processes cleanup, log saved to: {kill_log_path}")


def multi_clean_script(origin_config: Dict[str, Any], ip_lists: List[str]) -> None:
    config = copy.deepcopy(origin_config)
    config["sub_cmd"] = "clean"
    multi_runner_env = convert_to_env(config)
    multi_runner_env["IP_LISTS"] = ",".join(ip_lists)
    multi_runner_args = ["sh", "./multi_runner.sh"]

    # Ensure test_output directory exists
    ensure_test_output_dir()
    clean_log_path = os.path.join(test_output_dir, "clean.log")
    clean_log_path_abs = os.path.abspath(clean_log_path)

    log_stage("START", f"Test log files cleanup | Log: {clean_log_path_abs}")

    # Run clean command and redirect both stdout and stderr to file
    with open(clean_log_path, "w", encoding="utf-8") as f:
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

    log_stage("COMPLETED", f"Test log files cleanup")


def multi_test_script(origin_config: Dict[str, Any]) -> None:
    config = copy.deepcopy(origin_config)
    search_combined_parameter_groups(config)


def main():
    args = get_args()
    config = load_config(args.config)

    log_stage("INFO", f"Mode: {args.mode}")
    log_stage("INFO", f"Output directory: {ensure_test_output_dir()}")

    if args.mode == "run":
        for item in config.get("benchmarks", []):
            ip_lists = item["ip_lists"]
            if args.scratch:
                multi_build_script(item, item["ip_lists"])
            multi_test_script(item)
            multi_kill_script(item, ip_lists)
    elif args.mode == "clean":
        for item in config.get("benchmarks", []):
            ip_lists = item["ip_lists"]
            multi_clean_script(item, ip_lists)


if __name__ == "__main__":
    main()
