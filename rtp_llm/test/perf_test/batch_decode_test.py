import argparse
import glob
import json
import logging
import os
import shutil
import time
from typing import Any, Dict, List, Optional

from rtp_llm.test.perf_test.cache_grid_runner import CacheGridRunner
from rtp_llm.test.perf_test.dataset import KNOWN_DATASETS, extract_arg
from rtp_llm.test.perf_test.distribution_runner import DistributionRunner
from rtp_llm.test.perf_test.grid_runner import GridRunner
from rtp_llm.test.perf_test.hub_download import (
    needs_perf_hub_resolve,
    resolve_checkpoint_or_tokenizer_for_perf,
)
from rtp_llm.test.perf_test.sampling import prepare_distribution_config
from rtp_llm.test.perf_test.server import EngineServer
from rtp_llm.test.perf_test.test_util import create_query


def run_single(
    port: int,
    dp_size: int,
    batch_size_list: List[int],
    input_len_list: List[int],
    input_query_dict: Dict[int, str],
    is_decode: bool = True,
    dump_json_path: str = ".",
    decode_test_length: int = 20,
    tp_size: int = 1,
    generate_config: Optional[Dict[str, Any]] = None,
):
    """Backward-compatible wrapper — delegates to GridRunner."""
    return GridRunner(
        port,
        dp_size,
        batch_size_list,
        input_len_list,
        input_query_dict,
        is_decode=is_decode,
        dump_json_path=dump_json_path,
        decode_test_length=decode_test_length,
        tp_size=tp_size,
        generate_config=generate_config,
    ).run()


def parse_args():
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
    dataset_group = perf.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset_name",
        type=str,
        default="",
        help="Known dataset name (auto-downloads via ModelScope / HF). "
        f"Choices: {list(KNOWN_DATASETS.keys())}",
    )
    dataset_group.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Local path to dataset JSON (conversation or prompt format).",
    )
    perf.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Path to distribution.csv for runtime stratified sampling. "
        "Requires --max_seq_len and --concurrency_limit.",
    )
    perf.add_argument(
        "--test_json",
        type=str,
        default="",
        help="Path to previously saved test config JSON for replay",
    )
    perf.add_argument(
        "--partial",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="0: test all, 1: decode only, 2: prefill only",
    )
    perf.add_argument("--generate_config", type=str, default="{}")
    perf.add_argument(
        "--result_dir",
        type=str,
        default=os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "./perf_results"),
    )
    perf.add_argument("--decode_test_length", type=int, default=10)
    perf.add_argument(
        "--cache_grid_json",
        type=str,
        default="",
        help=(
            "JSON describing an explicit total-seq x prefix-cache grid. "
            "Each case inserts a prefix, then verifies aux_info.reuse_len."
        ),
    )
    perf.add_argument(
        "--cache_measure_runs",
        type=int,
        default=3,
        help="Measured requests per cache-grid case (default: 3)",
    )
    perf.add_argument(
        "--cache_request_timeout",
        type=int,
        default=int(os.environ.get("PERF_REQUEST_TIMEOUT", "7200")),
        help="Per-request timeout in seconds for cache-grid mode (default: 7200)",
    )
    perf.add_argument(
        "--cache_commit_tail_tokens",
        type=int,
        default=int(os.environ.get("CACHE_COMMIT_TAIL_TOKENS", "4096")),
        help=(
            "Extra seed tokens used to commit the requested cache prefix "
            "before measurement (default: 4096)"
        ),
    )

    engine = parser.add_argument_group(
        "engine args consumed by perf test (also forwarded to server)"
    )
    engine.add_argument("--dp_size", type=int, default=1)
    engine.add_argument("--max_seq_len", type=int, default=8192)
    engine.add_argument("--concurrency_limit", type=int, default=64)

    args, remaining = parser.parse_known_args()
    return args, remaining


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
    """在转发给引擎前，将 Hub 链接 / repo id 等解析为本地路径并写回 argv（同一远程引用只下载一次）。"""
    out = list(remaining)
    resolved_cache: Dict[str, str] = {}
    for k in ("checkpoint_path", "tokenizer_path"):
        val = extract_arg(out, k)
        if not val or not needs_perf_hub_resolve(val):
            continue
        if val not in resolved_cache:
            local = resolve_checkpoint_or_tokenizer_for_perf(val)
            logging.info(f"perf_test: resolved --{k} -> {local}")
            resolved_cache[val] = local
        _replace_cli_value(out, k, resolved_cache[val])
    return out


def _load_cache_grid_cases(path: str) -> List[Dict[str, int]]:
    """Load and validate an explicit total-sequence × cache-length grid.

    The cache runner measures batch=1 only.  Cache length is a request-level
    workload dimension, not an engine CLI argument, so the case file is kept
    separate from the forwarded engine args.
    """
    with open(path, encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("cache grid must be a JSON object")
    explicit_cases = "cases" in config
    if explicit_cases:
        raw_cases = config["cases"]
    else:
        seq_lens = config.get("seq_lens")
        if seq_lens is None:
            generation = config.get("seq_generation", {})
            if generation.get("kind") != "linear_with_dense_prefix":
                raise ValueError(
                    "cache grid requires cases, seq_lens, or "
                    "seq_generation.kind=linear_with_dense_prefix"
                )
            count = int(generation.get("count", 489))
            max_seq_len = int(generation.get("max_seq_len", 1048575))
            seq_block = int(config.get("seq_block_size", 256))
            if count < 2 or max_seq_len <= seq_block:
                raise ValueError("invalid seq_generation bounds")
            values = set(range(seq_block, min(16384, max_seq_len), seq_block))
            target_nonmax = count - 1
            i = 0
            while len(values) < target_nonmax:
                raw = seq_block + round(
                    i * (max_seq_len - 2 * seq_block) / max(1, target_nonmax - 1)
                )
                aligned = max(
                    seq_block,
                    min(
                        max_seq_len - seq_block,
                        round(raw / seq_block) * seq_block,
                    ),
                )
                values.add(aligned)
                i += 1
                if i > target_nonmax * 20:
                    raise ValueError("unable to generate unique seq lengths")
            seq_lens = sorted(values)[:target_nonmax] + [max_seq_len]
            if len(seq_lens) != count or len(set(seq_lens)) != count:
                raise ValueError("generated sequence lengths are not unique")
        ratios = [
            float(x)
            for x in config.get(
                "cache_ratios",
                [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.95],
            )
        ]
        block = int(config.get("cache_block_size", 4096))
        if block <= 0 or any(ratio < 0.0 or ratio >= 1.0 for ratio in ratios):
            raise ValueError("cache_ratios must be in [0, 1) and block must be positive")
        raw_cases = []
        case_id = 0
        for seq_len in seq_lens:
            seq_len = int(seq_len)
            max_cache_len = max(0, ((seq_len - block) // block) * block)
            for ratio in ratios:
                cache_len = int((max(0, seq_len - 1) * ratio) // block) * block
                raw_cases.append(
                    {
                        "case_id": case_id,
                        "batch_size": 1,
                        "input_len": seq_len,
                        "cache_len": min(cache_len, max_cache_len),
                    }
                )
                case_id += 1
            if max_cache_len > 0:
                raw_cases.append(
                    {
                        "case_id": case_id,
                        "batch_size": 1,
                        "input_len": seq_len,
                        "cache_len": max_cache_len,
                    }
                )
                case_id += 1

    if not isinstance(raw_cases, list):
        raise ValueError("cache grid cases must be a list")
    cases: List[Dict[str, int]] = []
    seen = set()
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"cache grid case {index} must be an object")
        case = {
            "case_id": int(raw.get("case_id", index)),
            "batch_size": int(raw.get("batch_size", 1)),
            "input_len": int(raw["input_len"]),
            "cache_len": int(raw.get("cache_len", 0)),
        }
        if case["batch_size"] != 1:
            raise ValueError("cache grid currently requires batch_size=1")
        if case["input_len"] <= 0 or not 0 <= case["cache_len"] < case["input_len"]:
            raise ValueError(f"invalid cache grid case: {case}")
        key = (case["batch_size"], case["input_len"], case["cache_len"])
        if key in seen:
            if not explicit_cases:
                continue
            raise ValueError(f"duplicate cache grid case: {case}")
        seen.add(key)
        cases.append(case)
    if not cases:
        raise ValueError(f"cache grid {path} contains no cases")
    return cases


def _collect_timeline_files(result_dir: str) -> None:
    """Wait for async profiler saves and collect timeline JSON files into a timelines/ subdirectory."""
    # C++ engine's ProfilerSaveWorker writes timeline JSONs asynchronously in a
    # background thread. Sleep to allow pending writes to flush before we move files.
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
            logging.info(f"Collected timeline: {dst}")
    else:
        logging.info("No timeline files found in %s", result_dir)


def _write_test_info(args: argparse.Namespace, remaining_args: List[str]) -> None:
    """Persist test configuration to result_dir for downstream consumers."""
    info = {
        "model_type": os.environ.get("MODEL_TYPE"),
        "checkpoint_path": os.environ.get("CHECKPOINT_PATH"),
        "tokenizer_path": os.environ.get("TOKENIZER_PATH"),
        "tp_size": extract_arg(remaining_args, "tp_size", "1"),
        "dp_size": args.dp_size,
        "max_seq_len": args.max_seq_len,
        "concurrency_limit": args.concurrency_limit,
        "decode_test_length": args.decode_test_length,
        "cache_grid_json": args.cache_grid_json or None,
        "cache_measure_runs": (
            args.cache_measure_runs if args.cache_grid_json else None
        ),
        "cache_request_timeout": (
            args.cache_request_timeout if args.cache_grid_json else None
        ),
        "cache_commit_tail_tokens": (
            args.cache_commit_tail_tokens if args.cache_grid_json else None
        ),
        "dataset_name": args.dataset_name or None,
        "dataset_path": args.dataset_path or args.dataset or None,
    }
    path = os.path.join(args.result_dir, "test_info.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    logging.info(f"Wrote test info to {path}")


def _effective_grid_max_seq_len(
    args: argparse.Namespace, input_len_list: List[int]
) -> int:
    needed_seq_len = max(input_len_list) + args.decode_test_length
    return max(needed_seq_len, args.max_seq_len)


def main() -> str:
    from rtp_llm.config.log_config import setup_logging

    setup_logging()

    args, remaining = parse_args()
    remaining = resolve_perf_engine_paths(remaining)
    generate_config = json.loads(args.generate_config)
    os.makedirs(args.result_dir, exist_ok=True)
    EngineServer.propagate_engine_env(remaining)

    logging.info(f"Result directory: {args.result_dir}")
    logging.info(f"Engine args forwarded to server: {remaining}")

    if args.cache_grid_json:
        if args.partial != 2:
            raise ValueError("--cache_grid_json is prefill-only; use --partial=2")
        if args.cache_measure_runs <= 0:
            raise ValueError("--cache_measure_runs must be positive")
        if args.cache_request_timeout <= 0:
            raise ValueError("--cache_request_timeout must be positive")
        if args.cache_commit_tail_tokens <= 0:
            raise ValueError("--cache_commit_tail_tokens must be positive")

        cases = _load_cache_grid_cases(args.cache_grid_json)
        for case in cases:
            cache_len = int(case["cache_len"])
            input_len = int(case["input_len"])
            if cache_len and cache_len % args.cache_commit_tail_tokens:
                raise ValueError(
                    "cache-grid cache_len must align to "
                    f"--cache_commit_tail_tokens={args.cache_commit_tail_tokens}: "
                    f"{case}"
                )
            if cache_len and cache_len + args.cache_commit_tail_tokens > input_len:
                raise ValueError(
                    "cache-grid cache_len must leave one commit tail before "
                    f"server startup: {case}"
                )
        max_input_len = max(int(case["input_len"]) for case in cases)
        max_batch_size = max(int(case["batch_size"]) for case in cases)
        tokenizer_path = (
            extract_arg(remaining, "tokenizer_path")
            or extract_arg(remaining, "checkpoint_path")
            or os.environ.get("TOKENIZER_PATH", "")
        )
        if not tokenizer_path:
            raise ValueError(
                "cache-grid mode requires --tokenizer_path or --checkpoint_path"
            )

        server = EngineServer(args, remaining)
        server.start(
            max_seq_len=max(max_input_len + args.decode_test_length, args.max_seq_len),
            max_concurrency=max_batch_size,
        )
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True
            )
            CacheGridRunner(
                server.port,
                tokenizer,
                cases,
                args.result_dir,
                request_timeout=args.cache_request_timeout,
                measure_runs=args.cache_measure_runs,
                cache_commit_tail_tokens=args.cache_commit_tail_tokens,
            ).run()
            _collect_timeline_files(args.result_dir)
        finally:
            server.stop()
        _write_test_info(args, remaining)
        return args.result_dir

    distribution_mode = (
        args.dataset_name or args.dataset_path or args.dataset or args.test_json
    )
    if distribution_mode:
        tokenizer_path = (
            extract_arg(remaining, "tokenizer_path")
            or extract_arg(remaining, "checkpoint_path")
            or os.environ.get("TOKENIZER_PATH", "")
        )
        test_config = prepare_distribution_config(
            tokenizer_path=tokenizer_path,
            max_seq_len=args.max_seq_len,
            max_concurrency=args.concurrency_limit * args.dp_size,
            result_dir=args.result_dir,
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
            dataset_csv=args.dataset,
            test_json=args.test_json,
        )

        batch_seq_len_map = test_config["batch_seq_len_map"]
        all_seq_lens = sorted(
            set(sl for sls in batch_seq_len_map.values() for sl in sls)
        )

        needed_seq_len = max(all_seq_lens) + args.decode_test_length
        effective_max_seq_len = max(needed_seq_len, args.max_seq_len)

        server = EngineServer(args, remaining)
        server.start(
            max_seq_len=effective_max_seq_len,
            max_concurrency=max(int(k) for k in batch_seq_len_map),
        )

        input_query_dict = create_query(input_len_list=all_seq_lens)
        DistributionRunner(
            server.port,
            args.dp_size,
            test_config,
            input_query_dict,
            dump_json_path=args.result_dir,
            decode_test_length=args.decode_test_length,
            generate_config=generate_config,
        ).run()
        _collect_timeline_files(args.result_dir)
        server.stop()
    else:
        batch_size_list = [int(x) for x in args.batch_size.split(",")]
        input_len_list = [int(x) for x in args.input_len.split(",")]
        needed_seq_len = max(input_len_list) + args.decode_test_length
        effective_max_seq_len = max(needed_seq_len, args.max_seq_len)

        server = EngineServer(args, remaining)
        server.start(
            max_seq_len=_effective_grid_max_seq_len(args, input_len_list),
            max_concurrency=max(batch_size_list),
        )

        input_query_dict = create_query(input_len_list=input_len_list)

        if args.partial in (0, 1):
            GridRunner(
                server.port,
                args.dp_size,
                batch_size_list,
                input_len_list,
                input_query_dict,
                is_decode=True,
                dump_json_path=args.result_dir,
                decode_test_length=args.decode_test_length,
                generate_config=generate_config,
            ).run()
        if args.partial in (0, 2):
            GridRunner(
                server.port,
                args.dp_size,
                batch_size_list,
                input_len_list,
                input_query_dict,
                is_decode=False,
                dump_json_path=args.result_dir,
                decode_test_length=args.decode_test_length,
                generate_config=generate_config,
            ).run()
        _collect_timeline_files(args.result_dir)
        server.stop()

    _write_test_info(args, remaining)
    return args.result_dir


if __name__ == "__main__":
    main()
