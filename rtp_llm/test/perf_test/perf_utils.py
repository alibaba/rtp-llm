"""Utility functions for perf test: BS generation, KV cache filtering, engine status query, timeline collection."""

import argparse
import glob
import json
import logging
import os
import shutil
import time
from typing import Any, Dict, List

import requests

from rtp_llm.test.perf_test.dataclass import PerfTestConfig


def auto_generate_bs_list(concurrency_limit: int) -> List[int]:
    """Auto-generate BS list: <64 step=8, >=64 step=64."""
    bs_list = [1]
    bs = 8
    while bs <= concurrency_limit:
        bs_list.append(bs)
        if bs < 64:
            bs += 8
        else:
            bs += 64
    return bs_list


def filter_bs_by_kvcache(
    bs_list: List[int], input_len: int, max_kv_tokens_per_dp: float
) -> List[int]:
    """Filter BS list by KV cache capacity. BS is per-DP."""
    return [bs for bs in bs_list if bs * input_len <= max_kv_tokens_per_dp]


def query_engine_status(port: int) -> Dict[str, Any]:
    """Query /cache_status and /worker_status, return unified engine info.

    Returns dict with keys: max_kv_tokens, total_kv_cache, block_size,
    concurrency_limit, dp_size.  Returns empty dict on failure.
    """
    result: Dict[str, Any] = {}
    try:
        cache = requests.get(f"http://127.0.0.1:{port}/cache_status", timeout=10).json()
        worker = requests.get(
            f"http://127.0.0.1:{port}/worker_status", timeout=10
        ).json()
        if "error" in cache or "error" in worker:
            logging.warning(f"Engine status error: cache={cache}, worker={worker}")
            return result

        cache_results = cache.get("results", [cache])
        per_dp_kv = [
            int(c.get("total_kv_cache", 0)) * int(c.get("block_size", 1))
            for c in cache_results
        ]
        result["max_kv_tokens"] = min(per_dp_kv) if per_dp_kv else 0
        result["total_kv_cache"] = int(cache.get("total_kv_cache", 0))
        result["block_size"] = int(cache.get("block_size", 1))
        result["dp_size"] = int(cache.get("dp_size", 1))
        result["concurrency_limit"] = int(worker.get("frontend_concurrency_limit", 0))
    except Exception as e:
        logging.warning(f"Failed to query engine status: {e}")
    return result


def collect_timeline_files(result_dir: str) -> None:
    """Collect profiler timeline JSON files into a timelines/ subdirectory."""
    time.sleep(3)
    timeline_dir = os.path.join(result_dir, "timelines")
    pattern = os.path.join(result_dir, "*.json")
    timeline_files = [
        f
        for f in glob.glob(pattern)
        if os.path.basename(f).startswith(("profiler_ts", "profiler_"))
        or "_wr" in os.path.basename(f)
    ]
    if timeline_files:
        os.makedirs(timeline_dir, exist_ok=True)
        for f in timeline_files:
            dst = os.path.join(timeline_dir, os.path.basename(f))
            shutil.move(f, dst)
            logging.debug(f"Collected timeline: {dst}")
    else:
        logging.info("No timeline files found in %s", result_dir)


def write_test_info(args: argparse.Namespace, remaining_args: List[str]) -> None:
    """Persist test configuration to result_dir for downstream consumers."""
    from rtp_llm.test.perf_test.dataset import extract_arg

    info = {
        "model_type": os.environ.get("MODEL_TYPE"),
        "checkpoint_path": os.environ.get("CHECKPOINT_PATH"),
        "tokenizer_path": os.environ.get("TOKENIZER_PATH"),
        "tp_size": extract_arg(remaining_args, "tp_size", "1"),
        "dp_size": args.dp_size,
        "max_seq_len": args.max_seq_len,
        "concurrency_limit": args.concurrency_limit,
        "decode_test_length": args.decode_test_length,
        "dataset_name": args.dataset_name or None,
        "dataset_path": args.dataset_path or args.dataset or None,
    }
    path = os.path.join(args.result_dir, "test_info.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    logging.info(f"Wrote test info to {path}")


def print_config_table(
    args: argparse.Namespace,
    config: PerfTestConfig,
    engine_status: Dict[str, Any],
    remaining: List[str],
) -> None:
    """Print test config, engine info, and per-input_len filtered BS."""
    from prettytable import PrettyTable

    from rtp_llm.test.perf_test.dataset import extract_arg

    mode = "distribution" if config.is_distribution else "grid"
    if args.target_tpot > 0:
        mode += "+tps"

    model_table = PrettyTable()
    model_table.title = "Model & Engine Info"
    model_table.field_names = ["Parameter", "Value"]
    model_table.align = "l"
    model_table.add_row(
        [
            "Model Type",
            os.environ.get("MODEL_TYPE", extract_arg(remaining, "model_type", "N/A")),
        ]
    )
    model_table.add_row(
        [
            "Checkpoint",
            os.environ.get(
                "CHECKPOINT_PATH", extract_arg(remaining, "checkpoint_path", "N/A")
            ),
        ]
    )
    model_table.add_row(["TP Size", extract_arg(remaining, "tp_size", "1")])
    model_table.add_row(["DP Size", args.dp_size])
    if engine_status:
        model_table.add_row(
            ["KV Cache Tokens (per DP)", engine_status.get("max_kv_tokens", "N/A")]
        )
        model_table.add_row(
            ["KV Cache Blocks (per DP)", engine_status.get("total_kv_cache", "N/A")]
        )
        model_table.add_row(["Block Size", engine_status.get("block_size", "N/A")])
        model_table.add_row(
            [
                "Concurrency Limit",
                engine_status.get("concurrency_limit", args.concurrency_limit),
            ]
        )
    logging.info("Model & engine info:\n" + str(model_table))

    table = PrettyTable()
    table.title = "Perf Test Configuration"
    table.field_names = ["Parameter", "Value"]
    table.align = "l"
    table.add_row(["Mode", mode])
    table.add_row(["Target TPOT (ms)", args.target_tpot or "N/A"])
    table.add_row(["Batch Sizes", config.batch_size_list or "N/A"])
    table.add_row(["Input Lengths", config.input_len_list or "N/A"])
    table.add_row(["Decode Test Length", args.decode_test_length])
    table.add_row(["Partial", args.partial])
    table.add_row(["Generate Config", args.generate_config])
    logging.info("Test configuration:\n" + str(table))

    if config.input_len_list and not config.is_distribution and args.partial != 2:
        max_kv = (
            float(engine_status.get("max_kv_tokens", float("inf")))
            if engine_status
            else float("inf")
        )
        bs_table = PrettyTable()
        bs_table.title = "Effective BS per Input Length (after KV cache filter)"
        bs_table.field_names = ["Input Len", "Max BS (KV cache)", "Candidate BS List"]
        bs_table.align = "l"
        for il in config.input_len_list:
            filtered = filter_bs_by_kvcache(config.batch_size_list, il, max_kv)
            max_bs = filtered[-1] if filtered else 0
            bs_table.add_row([il, max_bs, filtered or "N/A"])
        logging.info("Effective BS per input length:\n" + str(bs_table))
