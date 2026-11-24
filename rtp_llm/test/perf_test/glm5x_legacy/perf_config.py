"""Perf test configuration: argument parsing, path resolution, PerfTestConfig generation."""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

from rtp_llm.test.perf_test.engine_paths import extract_arg, resolve_engine_paths
from rtp_llm.test.perf_test.glm5x_legacy.dataclass import PerfTestConfig
from rtp_llm.test.perf_test.glm5x_legacy.perf_utils import auto_generate_bs_list


def parse_args(
    argv: Optional[List[str]] = None,
) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="RTP-LLM batch decode performance test runner. "
        "Unrecognized arguments are forwarded to the engine server.",
    )

    perf = parser.add_argument_group("perf test configuration")
    perf.add_argument(
        "--batch_size",
        type=str,
        default="1,8,16",
        help="Comma-separated batch sizes for grid mode",
    )
    perf.add_argument(
        "--input_len",
        type=str,
        default="1024,4096",
        help="Comma-separated input lengths for grid mode",
    )
    perf.add_argument(
        "--partial",
        type=int,
        default=1,
        choices=[1, 2],
        help="1: decode only (default), 2: prefill only (grid mode only, not supported in distribution mode)",
    )
    perf.add_argument("--generate_config", type=str, default="{}")
    perf.add_argument(
        "--result_dir",
        type=str,
        default=os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "./perf_results"),
    )
    perf.add_argument("--decode_test_length", type=int, default=10)
    perf.add_argument(
        "--num_measures",
        type=int,
        default=5,
        help="Number of measurements per BS. Trim min/max and average the rest.",
    )
    perf.add_argument(
        "--warmup_runs",
        type=int,
        default=None,
        help="Override PERF_FORMAL_WARMUP_RUNS for every case.",
    )
    perf.add_argument(
        "--measure_runs",
        type=int,
        default=None,
        help="Override PERF_MEASURE_RUNS for every case.",
    )
    perf.add_argument(
        "--profile_runs",
        type=int,
        default=None,
        help="Override PERF_PROFILE_RUNS for every case.",
    )
    perf.add_argument(
        "--engine_arg",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Repeatable engine argument shorthand, e.g. tp_size=8.",
    )
    perf.add_argument(
        "--engine_env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "Repeatable engine environment override; explicit engine CLI arguments "
            "take precedence when both configure the same setting."
        ),
    )

    engine = parser.add_argument_group(
        "engine args consumed by perf test (also forwarded to server)"
    )
    engine.add_argument("--dp_size", type=int, default=1)
    engine.add_argument("--max_seq_len", type=int, default=8192)
    engine.add_argument("--concurrency_limit", type=int, default=64)

    parsed_argv = sys.argv[1:] if argv is None else argv
    unsupported = {
        "--target_tpot",
        "--dataset_name",
        "--dataset_path",
        "--dataset",
        "--test_json",
        "--cache_grid_json",
        "--cache_measure_runs",
        "--cache_request_timeout",
        "--cache_commit_tail_tokens",
    }
    if any(item.split("=", 1)[0] in unsupported for item in parsed_argv):
        parser.error("GLM5.x compatibility supports the dedicated grid suites only")
    # Empty metadata fields retain the upstream grid report shape.
    parser.set_defaults(
        dataset_name="",
        dataset_path="",
        dataset="",
        test_json="",
        cache_grid_json="",
        target_tpot=0,
    )
    args, remaining = parser.parse_known_args(parsed_argv)
    args.batch_size_explicit = any(
        item == "--batch_size" or item.startswith("--batch_size=")
        for item in parsed_argv
    )
    return args, remaining


def _parse_name_value(value: str, option: str) -> Tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"{option} expects NAME=VALUE, got {value!r}")
    name, parsed = value.split("=", 1)
    name = name.strip().lstrip("-")
    if not name or any(ch.isspace() for ch in name):
        raise ValueError(f"{option} has invalid name in {value!r}")
    return name, parsed


def _engine_arg_argv(engine_args: List[str]) -> List[str]:
    result: List[str] = []
    for item in engine_args:
        name, value = _parse_name_value(item, "--engine_arg")
        result.extend([f"--{name}", value])
    return result


def _apply_engine_env(engine_env: List[str]) -> List[str]:
    names: List[str] = []
    for item in engine_env:
        name, value = _parse_name_value(item, "--engine_env")
        os.environ[name] = value
        names.append(name)
    return sorted(set(names))


def _apply_run_overrides(args: argparse.Namespace) -> None:
    for option, env_name in (
        ("warmup_runs", "PERF_FORMAL_WARMUP_RUNS"),
        ("measure_runs", "PERF_MEASURE_RUNS"),
        ("profile_runs", "PERF_PROFILE_RUNS"),
    ):
        value = getattr(args, option)
        if value is None:
            continue
        if value < 0:
            raise ValueError(f"--{option} must be >= 0, got {value}")
        os.environ[env_name] = str(value)


def _replace_cli_value(argv: List[str], key: str, new_value: str) -> None:
    flag = f"--{key}"
    prefix = f"--{key}="
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            argv[i + 1] = new_value
            return
        if arg.startswith(prefix):
            argv[i] = prefix + new_value
            return
    raise ValueError(
        f"perf_test: missing {flag} in argv, cannot replace with local path"
    )


def resolve_perf_engine_paths(remaining: List[str]) -> List[str]:
    """Resolve Hub links / repo ids to local paths before forwarding to engine."""
    return resolve_engine_paths(remaining)


def prepare_config(args: argparse.Namespace, remaining: List[str]) -> PerfTestConfig:
    """Build the upstream GLM fixed-grid configuration."""
    batch_size_explicit = getattr(
        args,
        "batch_size_explicit",
        any(a.startswith("--batch_size") for a in sys.argv[1:]),
    )
    # Grid mode
    input_len_list = [int(x) for x in args.input_len.split(",")]

    # Prefill mode (partial=2): always BS=1, no need for large BS list
    if args.partial == 2:
        batch_size_list = [1]
    elif batch_size_explicit:
        batch_size_list = [int(x) for x in args.batch_size.split(",")]
    else:
        batch_size_list = auto_generate_bs_list(args.concurrency_limit)

    effective_max_concurrency = max(batch_size_list)
    if args.target_tpot > 0:
        effective_max_concurrency = max(
            effective_max_concurrency, args.concurrency_limit
        )

    return PerfTestConfig(
        is_distribution=False,
        all_seq_lens=input_len_list,
        batch_size_list=batch_size_list,
        input_len_list=input_len_list,
        max_seq_len=max(input_len_list) + args.decode_test_length,
        max_concurrency=effective_max_concurrency,
    )
