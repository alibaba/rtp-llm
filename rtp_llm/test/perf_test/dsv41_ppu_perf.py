"""CP8 prefill / DP8 decode grids over the standard batch_decode_test runner.

Configuration, dry runs and result analysis require only the Python standard
library. Device runs use the checked runtime and its tokenizer dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rtp_llm.config.v41_runtime import env_bool

SCHEMA_VERSION = 1
WORLD_SIZE = 8


def sizes(value):
    """Parse a positive, unique comma-separated grid; K means 1024."""
    try:
        result = []
        for item in value.split(","):
            item = item.strip().lower()
            multiplier = 1024 if item.endswith("k") else 1
            result.append(int(item[:-1] if multiplier != 1 else item) * multiplier)
        if not result or min(result) <= 0 or len(set(result)) != len(result):
            raise ValueError
        return result
    except ValueError:
        raise argparse.ArgumentTypeError(
            "expected unique positive sizes, e.g. 1,8,16 or 2k,8k"
        )


def env_boolean(value):
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError("FAKE_BALANCE_EXPERT must be a boolean")


def parse_args(argv=None):
    env = os.environ
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        nargs="?",
        choices=("prefill", "decode", "all", "graph"),
        default="decode",
    )
    parser.add_argument("--checkpoint-path", default=env.get("RTP_V41_TEST_CHECKPOINT"))
    parser.add_argument("--runtime-root", default=env.get("RTP_V41_TEST_RUNTIME"))
    parser.add_argument("--result-dir", default=env.get("TEST_UNDECLARED_OUTPUTS_DIR"))
    parser.add_argument(
        "--batches",
        type=sizes,
        default=env.get("RTP_V41_PERF_BATCHES"),
        help="Global request counts, shared by both phases",
    )
    parser.add_argument("--prefill-batches", type=sizes)
    parser.add_argument("--decode-batches", type=sizes)
    parser.add_argument(
        "--context-lengths",
        type=sizes,
        default=env.get("RTP_V41_PERF_CONTEXT_LENGTHS"),
        help="Prefill input tokens / Decode initial synthetic KV tokens; K=1024",
    )
    parser.add_argument("--decode-tokens", type=int, default=None)
    parser.add_argument("--greedy", action="store_true", help="Use top_k=1 in the native request config")
    parser.add_argument(
        "--warmup-runs", type=int, default=int(env.get("RTP_V41_PERF_WARMUP_RUNS", "1"))
    )
    parser.add_argument("--measure-runs", type=int, default=None)
    parser.add_argument(
        "--profile-runs",
        type=int,
        default=int(env.get("RTP_V41_PERF_PROFILE_RUNS", "0")),
    )
    parser.add_argument("--request-timeout", type=int, default=7200)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--kv-cache-mem-mb", type=int, default=8192)
    parser.add_argument(
        "--decode-graph",
        action="store_true",
        default=env.get("RTP_V41_TEST_DECODE_GRAPH") == "1",
    )
    parser.add_argument(
        "--graph-mega-mhc",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the PPU Mega-mHC transition (default for ordinary Decode Graph)",
    )
    parser.add_argument(
        "--graph-packed-attention", action=argparse.BooleanOptionalAction,
        default=None,
        help="Read packed SWA and compressed KV directly (default with a FlashMLA receipt)",
    )
    parser.add_argument(
        "--graph-visible-indexer", action=argparse.BooleanOptionalAction,
        default=None,
        help="Use visible-length scoring and selection (default for Decode Graph)",
    )
    parser.add_argument(
        "--capture-batches",
        type=sizes,
        default=env.get("RTP_V41_GRAPH_CAPTURE_BATCHES"),
        help="Per-DP Graph capacities; default derives them from global batches",
    )
    parser.add_argument(
        "--moe-backend", choices=("grouped", "megamoe"), default="megamoe"
    )
    parser.add_argument(
        "--fake-balance-expert",
        action=argparse.BooleanOptionalAction,
        default=env_boolean(env.get("FAKE_BALANCE_EXPERT", "0")),
        help="Override real expert routes for performance measurements; default off",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the complete plan without loading a model or writing files",
    )
    args = parser.parse_args(argv)
    if args.phase == "graph":
        args.phase, args.decode_graph = "decode", True
        graph_defaults = True
    else:
        graph_defaults = False
    args.decode_tokens = (
        args.decode_tokens
        if args.decode_tokens is not None
        else int(env.get("RTP_V41_PERF_STEPS", "8" if graph_defaults else "129"))
    )
    args.measure_runs = (
        args.measure_runs
        if args.measure_runs is not None
        else int(env.get("RTP_V41_PERF_MEASURE_RUNS", "1" if graph_defaults else "2"))
    )
    args.prefill_batches = args.prefill_batches or args.batches or [1, 2, 4]
    args.decode_batches = (
        args.decode_batches
        or args.batches
        or ([1, 8, 16] if graph_defaults else [1, 8, 16, 32, 64, 128, 256])
    )
    if args.context_lengths is None:
        old_input = sizes(env.get("RTP_V41_PERF_INPUT_LEN", "2049"))
        args.context_lengths = (
            old_input
            if args.phase == "prefill" and "RTP_V41_PERF_INPUT_LEN" in env
            else [n - 1 for n in old_input]
        )
    if not args.checkpoint_path:
        parser.error("--checkpoint-path or RTP_V41_TEST_CHECKPOINT is required")
    if not args.result_dir:
        args.result_dir = str(
            Path("perf_results/dsv41_ppu")
            / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
    if args.warmup_runs < 0 or args.profile_runs < 0 or args.measure_runs <= 0:
        parser.error("warmup/profile runs must be >= 0 and measure runs must be > 0")
    if args.decode_tokens < 2 or args.request_timeout <= 0 or args.kv_cache_mem_mb <= 0:
        parser.error(
            "decode tokens must be >= 2; timeout and KV memory must be positive"
        )
    if args.decode_graph and (args.phase == "prefill" or args.moe_backend != "megamoe"):
        parser.error("Graph requires DP8 Decode with MegaMoE/W4A4")
    if args.graph_mega_mhc and not args.decode_graph:
        parser.error("--graph-mega-mhc requires --decode-graph")
    if (args.graph_packed_attention or args.graph_visible_indexer) and not args.decode_graph:
        parser.error("Graph attention switches require --decode-graph")
    try:
        if args.graph_mega_mhc is None:
            mhc_backend = env.get("RTP_V41_PPU_MHC_BACKEND", "mega").strip().lower()
            args.graph_mega_mhc = args.decode_graph and env_bool(
                env, "RTP_V41_PPU_GRAPH_MEGA_MHC", default=mhc_backend == "mega"
            )
        if args.graph_packed_attention is None:
            args.graph_packed_attention = args.decode_graph and env_bool(
                env, "RTP_V41_PPU_GRAPH_PACKED_ATTENTION",
                default=bool(env.get("RTP_V41_PPU_PACKED_DEPENDENCY")),
            )
        if args.graph_visible_indexer is None:
            args.graph_visible_indexer = args.decode_graph and env_bool(
                env, "RTP_V41_PPU_GRAPH_VISIBLE_INDEXER", default=True
            )
        build_plan(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def build_plan(args):
    output = Path(args.result_dir).resolve()
    phases = ("prefill", "decode") if args.phase == "all" else (args.phase,)
    jobs = []
    for phase in phases:
        decode = phase == "decode"
        dp = WORLD_SIZE if decode else 1
        batches = args.decode_batches if decode else args.prefill_batches
        if decode and any(b >= dp and b % dp for b in batches):
            raise ValueError(
                "DP8 global batches must be < 8 or divisible by 8; uneven nonempty owners cannot use the batch scheduler"
            )
        if min(args.context_lengths) <= 0 or (
            decode and any(n % 2 for n in args.context_lengths)
        ):
            raise ValueError(
                "contexts must be positive; synthetic Decode KV lengths must be even (complete compression pairs)"
            )
        inputs = [n + int(decode) for n in args.context_lengths]
        output_tokens = args.decode_tokens if decode else 1
        needed = max(inputs) + output_tokens
        maximum = (
            args.max_seq_len
            if args.max_seq_len is not None
            else (needed + 255) // 256 * 256
        )
        if maximum < needed:
            raise ValueError(
                f"{phase} needs {needed} sequence slots (input + output); "
                f"--max-seq-len {maximum} is too small"
            )
        local_batches = sorted({math.ceil(b / dp) for b in batches})
        captures = args.capture_batches or local_batches
        if decode and args.decode_graph and max(captures) < max(local_batches):
            raise ValueError("capture batches do not cover the largest per-DP batch")
        directory = output / phase
        environment = {
            "PERF_GLOBAL_BATCH": "1",
            "PERF_DETAILED_LATENCIES": "1",
            "PERF_NATIVE_HTTP": "1",
            "PERF_GRID_WARMUP_RUNS": "1",
            "PERF_TEST": "1",
            "PERF_REQUEST_TIMEOUT": str(args.request_timeout),
            "PERF_FORMAL_WARMUP_RUNS": str(args.warmup_runs),
            "PERF_MEASURE_RUNS": str(args.measure_runs),
            "PERF_PROFILE_RUNS": "0",
            "DISABLE_COMPILE_PRINT": "1",
            "FAKE_BALANCE_EXPERT": str(int(args.fake_balance_expert)),
            "MAX_SEQ_LEN": str(maximum),
            "KV_CACHE_MEM_MB": str(args.kv_cache_mem_mb),
            "ENABLE_CUDA_GRAPH": str(int(decode and args.decode_graph)),
            "RTP_V41_PPU_GRAPH_MEGA_MHC": str(int(decode and args.graph_mega_mhc)),
            "RTP_V41_PPU_GRAPH_PACKED_ATTENTION": str(int(decode and args.graph_packed_attention)),
            "RTP_V41_PPU_GRAPH_VISIBLE_INDEXER": str(int(decode and args.graph_visible_indexer)),
            "RTP_V41_PPU_MOE_BACKEND": args.moe_backend,
            "RTP_V41_PPU_MOE_PRECISION": (
                "w4a4" if args.moe_backend == "megamoe" else "w4a8"
            ),
            "RTP_TEST_GRACEFUL_STOP_TIMEOUT": "180",
            "RTP_V41_PERF_EVIDENCE_DIR": str(directory / "rank-evidence"),
            "MAGA_SERVER_WORK_DIR": str(directory),
            "RTP_PYTHON_BIN": sys.executable,
        }
        command = [
            sys.executable,
            "-B",
            "-m",
            "rtp_llm.test.perf_test.batch_decode_test",
            "--model_type",
            "deepseek_v41",
            "--checkpoint_path",
            str(Path(args.checkpoint_path).resolve()),
            "--tokenizer_path",
            str(Path(args.checkpoint_path).resolve()),
            "--input_len",
            ",".join(map(str, inputs)),
            "--batch_size",
            ",".join(map(str, batches)),
            "--dp_size",
            str(dp),
            "--partial",
            "1" if decode else "2",
            "--decode_test_length",
            str(output_tokens),
            "--num_measures",
            str(args.measure_runs),
            "--warmup_runs",
            str(args.warmup_runs),
            "--measure_runs",
            str(args.measure_runs),
            "--profile_runs",
            "0",
            "--result_dir",
            str(directory),
            "--max_seq_len",
            str(maximum),
            "--max_context_batch_size",
            "1" if decode else str(max(batches)),
            "--max_batch_tokens_size",
            str(
                max(max(inputs), max(local_batches))
                if decode
                else max(inputs) * max(batches)
            ),
            "--fake_balance_expert",
            str(int(args.fake_balance_expert)),
            "--warm_up",
            "0",
            "--model_warm_up",
            "0",
            "--generate_config",
            json.dumps({"force_sp_accept": False, **({"top_k": 1} if args.greedy else {})}),
            "--frontend_pre_stop_drain_seconds",
            "0",
            "--dash_sc_grpc_pre_stop_drain_seconds",
            "0",
            "--backend_post_frontend_drain_seconds",
            "0",
        ]
        if decode and args.decode_graph:
            command.extend(
                [
                    "--enable_cuda_graph",
                    "1",
                    "--decode_capture_config",
                    ",".join(map(str, captures)),
                ]
            )
        cases = [
            {
                "case_id": f"{phase}_b{batch}_ctx{context}",
                "phase": phase,
                "batch_size": batch,
                "context_length": context,
                "input_len": input_len,
                "cached_tokens_per_request": context if decode else 0,
                "executed_prefill_tokens_per_request": 0 if decode else context,
                "output_tokens_per_request": output_tokens,
                "fake_balance_expert": args.fake_balance_expert,
                "raw_path": str(
                    directory
                    / "raw_latencies"
                    / f"bs{batch}_seq{input_len}_{phase}.json"
                ),
                "dp_request_counts": [
                    sum(i % dp == rank for i in range(batch)) for rank in range(dp)
                ],
            }
            for batch in batches
            for context, input_len in zip(args.context_lengths, inputs)
        ]
        jobs.append(
            {
                "phase": phase,
                "directory": str(directory),
                "command": command,
                "environment": environment,
                "cases": cases,
                "topology": {
                    "world_size": 8,
                    "physical_tp": 1 if decode else 8,
                    "attention_cp": 1 if decode else 8,
                    "attention_dp": dp,
                    "attention_tp": 1,
                    "ep": 8,
                    "enp": 8,
                },
                "synthetic_kv": decode,
                "speculative": False,
                "fake_balance_expert": args.fake_balance_expert,
                "prefix_reuse": False,
                "graph": decode and args.decode_graph,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "model_type": "deepseek_v41",
        "jobs": jobs,
        "measurement": {
            "warmup_runs": args.warmup_runs,
            "measure_runs": args.measure_runs,
            "profile_runs": args.profile_runs,
            "profile_services_are_separate": True,
        },
    }


def statistics_ms(values):
    if not values:
        return None
    ordered = sorted(values)

    def percentile(q):
        index = (len(ordered) - 1) * q
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": max(values),
    }


def summarize_case(case, expected_runs):
    try:
        return _summarize_case(case, expected_runs)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return {
            **case,
            "status": "failed",
            "errors": [f"invalid raw evidence: {error}"],
            "latency_ms": None,
            "throughput": None,
            "trimmed": False,
        }


def _summarize_case(case, expected_runs):
    result = {
        **case,
        "status": "failed",
        "errors": [],
        "latency_ms": None,
        "throughput": None,
        "trimmed": False,
    }
    try:
        data = json.loads(Path(case["raw_path"]).read_text())
    except (OSError, ValueError) as error:
        result["errors"].append(f"missing or invalid raw evidence: {error}")
        return result
    if not isinstance(data, dict):
        raise ValueError("raw evidence must be an object")
    errors = result["errors"]
    decode = case["phase"] == "decode"
    runs = data.get("measurement_runs", [])
    if not isinstance(runs, list):
        raise ValueError("measurement_runs must be a list")
    if (
        len(runs) != expected_runs
        or data.get("global_batch_size") != case["batch_size"]
        or data.get("dp_size") != len(case["dp_request_counts"])
    ):
        errors.append(
            "measurement run count or global batch differs from the requested grid"
        )
    totals = Counter()
    wall, ttft, tpot, intervals, wait, request_ms = [], [], [], [], [], []
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("measurement round must be an object")
        duration = (run["ended_ns"] - run["started_ns"]) / 1e6
        responses = run.get("responses", [])
        if (
            not math.isfinite(duration)
            or duration <= 0
            or len(responses) != case["batch_size"]
        ):
            errors.append(
                f"run {run.get('run')}: invalid wall duration or request count"
            )
            continue
        wall.append(duration)
        owners = Counter()
        for response in responses:
            try:
                if not response.get("success") or response.get("parse_error"):
                    raise ValueError("unsuccessful response or stream parse error")
                aux = response["metric_response"]["aux_info"]
                if (
                    int(aux["input_len"]) != case["input_len"]
                    or int(aux["output_len"]) != case["output_tokens_per_request"]
                ):
                    raise ValueError(
                        "actual input/output token counts differ from the requested case"
                    )
                if not decode and int(aux.get("reuse_len", 0)) != 0:
                    raise ValueError("prefill unexpectedly reused prefix KV")
                cost, first, queued = (
                    (
                        float(aux[key])
                        for key in ("cost_time", "first_token_cost_time", "wait_time")
                    )
                    if "wait_time" in aux
                    else (
                        float(aux["cost_time"]),
                        float(aux["first_token_cost_time"]),
                        0.0,
                    )
                )
                if (
                    not all(math.isfinite(v) for v in (cost, first, queued))
                    or not 0 <= queued <= first <= cost
                ):
                    raise ValueError("invalid cumulative engine timing")
                owners[int(response["owner_rank"])] += 1
                events = response["events"]
                previous_count, previous_cost = 0, 0.0
                token_intervals = []
                for event in events:
                    current = event["aux_info"]
                    count, timestamp = int(current["output_len"]), float(
                        current["cost_time"]
                    )
                    if (
                        not math.isfinite(timestamp)
                        or count < previous_count
                        or timestamp < previous_cost
                        or count - previous_count > 1
                    ):
                        raise ValueError("missing tokens or non-monotone engine timing")
                    if previous_count and count > previous_count:
                        token_intervals.append(timestamp - previous_cost)
                    if count > previous_count:
                        previous_count, previous_cost = count, timestamp
                if (
                    previous_count != case["output_tokens_per_request"]
                    or len(token_intervals) != previous_count - 1
                ):
                    raise ValueError("incomplete per-token latency evidence")
                totals["requests"] += 1
                totals["input_tokens"] += case["executed_prefill_tokens_per_request"]
                totals["output_tokens"] += int(aux["output_len"])
                ttft.append(first - queued)
                request_ms.append(cost - queued)
                wait.append(queued)
                if decode:
                    tpot.append((cost - first) / (int(aux["output_len"]) - 1))
                    intervals.extend(token_intervals)
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"run {run.get('run')}: {error}")
        if [owners[rank] for rank in range(len(case["dp_request_counts"]))] != case[
            "dp_request_counts"
        ] or sum(owners.values()) != len(responses):
            errors.append(
                f"run {run.get('run')}: request distribution does not match the requested DP owners"
            )
    result["counts"] = dict(totals, measurement_runs=len(runs))
    if errors:
        return result
    seconds = sum(wall) / 1000
    result.update(
        status="ok",
        latency_ms={
            "batch_wall": statistics_ms(wall),
            "request": statistics_ms(request_ms),
            "wait": statistics_ms(wait),
            "ttft": statistics_ms(ttft),
            "tpot": statistics_ms(tpot),
            "token_interval": statistics_ms(intervals),
        },
        throughput={
            "measurement_seconds": seconds,
            "requests_per_s": totals["requests"] / seconds,
            "input_tokens_per_s": (
                totals["input_tokens"] / seconds if not decode else None
            ),
            "input_tokens_per_s_per_device": (
                totals["input_tokens"] / seconds / WORLD_SIZE if not decode else None
            ),
            "output_tokens_per_s": totals["output_tokens"] / seconds,
            "output_tokens_per_s_per_device": totals["output_tokens"]
            / seconds
            / WORLD_SIZE,
        },
    )
    return result


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def phase_environment(args, job):
    from rtp_llm.config.v41_runtime import V41RuntimeConfig
    from rtp_llm.tools.v41.launch import preset_env

    parent = dict(os.environ)
    for name in ("MODEL_SERVICE_CONFIG", "REMOTE_RPC_SERVER_IP", "REMOTE_SERVER_PORT"):
        parent.pop(name, None)
    parent.update(job["environment"])
    env = preset_env(
        job["phase"],
        args.checkpoint_path,
        parent=parent,
        fake_balance_expert=args.fake_balance_expert,
    )
    env.update(job["environment"])
    V41RuntimeConfig.from_env(env)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(Path(args.runtime_root).resolve()), parent.get("PYTHONPATH")))
    )
    return env


def check_graph_evidence(job):
    directory = Path(job["environment"]["RTP_V41_PERF_EVIDENCE_DIR"])
    graph_mega_mhc = job["environment"].get("RTP_V41_PPU_GRAPH_MEGA_MHC") == "1"
    attention_switches = {
        route: job["environment"].get(f"RTP_V41_PPU_GRAPH_{route.upper()}") == "1"
        for route in ("packed_attention", "visible_indexer")
    }
    assignments = 0
    for rank in range(WORLD_SIZE):
        report = json.loads((directory / f"moe-rank{rank}.json").read_text())
        engines = report["engines"]
        if (
            report["rank"] != rank
            or report["world_size"] != 8
            or report["model_layers"] != [40]
            or report["decode_graph_replays"] <= 0
        ):
            raise ValueError(
                f"rank {rank}: missing full-model DP8 Graph replay evidence"
            )
        if sum(e["layers"] for e in engines if e["layers"]) != 40 or not engines:
            raise ValueError(f"rank {rank}: incomplete layer evidence")
        if any(
            e["backend"] != "megamoe"
            or e["precision"] != "w4a4"
            or e["grouped_gemm_calls"]
            or (e["layers"] and e["mega_moe_calls"] <= 0)
            for e in engines
        ):
            raise ValueError(
                f"rank {rank}: unexpected MoE backend or missing native calls"
            )
        mhc = report.get("mega_mhc", [])
        captured = sum(stats["graph_capture_calls"] for stats in mhc)
        if graph_mega_mhc and (
            captured < 40
            or captured % 40
            or any(
                name.startswith("graph:")
                for stats in mhc
                for name in stats["fallbacks"]
            )
        ):
            raise ValueError(f"rank {rank}: incomplete Mega-mHC Graph capture: {mhc}")
        if not graph_mega_mhc and captured:
            raise ValueError(f"rank {rank}: Mega-mHC ran in the Graph control arm")
        attention = report.get("attention_graph_capture_calls", [])
        for route, enabled in attention_switches.items():
            calls = sum(stats.get(route, 0) for stats in attention)
            if (calls > 0) != enabled or (route == "packed_attention" and calls % 40):
                raise ValueError(f"rank {rank}: wrong {route} Graph capture count: {calls}")
        assignments += sum(e["mega_received_assignments"] for e in engines)
    if not assignments:
        raise ValueError("Graph reports contain no received expert assignments")


def prepare_token_map(args, directory):
    from rtp_llm.models_py.modules.dsv41.ppu.engram import load_token_map

    path = directory / "token-map.json"
    load_token_map(str(Path(args.checkpoint_path) / "tokenizer.json"), str(path))
    return str(path)


def execute_phase(args, job, directory=None, profile_runs=0):
    command = list(job["command"])
    env = phase_environment(args, job)
    directory = Path(directory or job["directory"])
    directory.mkdir(parents=True)
    env["RTP_V41_TOKEN_MAP_CACHE"] = prepare_token_map(args, directory)
    if profile_runs:
        for flag, value in (
            ("--result_dir", str(directory)),
            ("--profile_runs", str(profile_runs)),
            ("--measure_runs", "1"),
        ):
            command[command.index(flag) + 1] = value
        env.update(
            PERF_MEASURE_RUNS="1",
            PERF_PROFILE_RUNS=str(profile_runs),
            MAGA_SERVER_WORK_DIR=str(directory),
            RTP_V41_PERF_EVIDENCE_DIR=str(directory / "rank-evidence"),
        )
    with (directory / "driver.log").open("w") as log:
        completed = subprocess.run(
            command,
            cwd=args.runtime_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"{job['phase']} runner exited {completed.returncode}; see {directory / 'driver.log'}"
        )
    if job["graph"] and not profile_runs:
        check_graph_evidence(job)


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if (
        not args.runtime_root
        or not (Path(args.runtime_root) / "native-runtime.json").is_file()
    ):
        raise ValueError(
            "--runtime-root / RTP_V41_TEST_RUNTIME must identify a checked native runtime (native-runtime.json)"
        )
    output = Path(args.result_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "run.json"
    # Never overwrite measurements or silently pick up an earlier run's raw files.
    with manifest.open("x") as stream:
        json.dump(plan, stream, indent=2)
    cases, phase_errors = [], []
    summary = {
        "schema_version": SCHEMA_VERSION,
        "model_type": "deepseek_v41",
        "status": "running",
        "timing": {
            "throughput": "Actual executed prefill / emitted output tokens divided by summed unprofiled client batch windows, including HTTP and client dispatch; warmup and profile are excluded",
            "latency": "Engine cumulative milliseconds; TTFT excludes queue wait, TPOT excludes first token, token intervals retain every sample",
            "decode_context": "Synthetic KV length before the first decode; HTTP input_len=context+1; no actual prefill or PD transfer is timed",
            "batch": "Global requests; per-DP counts are explicit",
            "trimmed": False,
        },
        "runtime": json.loads(
            (Path(args.runtime_root) / "native-runtime.json").read_text()
        ),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": cases,
        "errors": phase_errors,
    }
    save(output / "summary.json", summary)
    for job in plan["jobs"]:
        try:
            execute_phase(args, job)
        except Exception as error:
            phase_errors.append(str(error))
        for case in job["cases"]:
            result = summarize_case(case, args.measure_runs)
            result.update(
                topology=job["topology"],
                synthetic_kv=job["synthetic_kv"],
                graph=job["graph"],
                graph_mega_mhc=job["environment"]["RTP_V41_PPU_GRAPH_MEGA_MHC"] == "1",
                graph_packed_attention=job["environment"]["RTP_V41_PPU_GRAPH_PACKED_ATTENTION"] == "1",
                graph_visible_indexer=job["environment"]["RTP_V41_PPU_GRAPH_VISIBLE_INDEXER"] == "1",
                moe_backend=args.moe_backend,
                moe_precision=job["environment"]["RTP_V41_PPU_MOE_PRECISION"],
                max_seq_len=int(job["environment"]["MAX_SEQ_LEN"]),
            )
            cases.append(result)
            latency = result["latency_ms"]
            throughput = result["throughput"]
            timing = ""
            if throughput:
                timing = f"  input={throughput['input_tokens_per_s']} output={throughput['output_tokens_per_s']:.3f} tok/s  TTFT={latency['ttft']['mean']:.3f} ms"
                if latency["tpot"]:
                    timing += f"  TPOT={latency['tpot']['mean']:.3f} ms (p99={latency['token_interval']['p99']:.3f})"
            print(
                f"{case['case_id']}: {result['status']}"
                + (timing if throughput else f"  {result['errors']}"),
                flush=True,
            )
        summary["status"] = (
            "failed"
            if phase_errors or any(c["status"] != "ok" for c in cases)
            else "ok"
        )
        save(output / "summary.json", summary)
        (output / "results.jsonl").write_text(
            "".join(json.dumps(c, allow_nan=False) + "\n" for c in cases)
        )
    # Profiling is a separate service invocation after ALL measured cases, so
    # profiler teardown cannot contaminate a later shape's measurement window.
    if args.profile_runs:
        for job in plan["jobs"]:
            try:
                execute_phase(
                    args, job, output / "profiles" / job["phase"], args.profile_runs
                )
            except Exception as error:
                phase_errors.append(f"profile: {error}")
        if phase_errors:
            summary["status"] = "failed"
        save(output / "summary.json", summary)
    print(f"Results: {output / 'summary.json'}", flush=True)
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
