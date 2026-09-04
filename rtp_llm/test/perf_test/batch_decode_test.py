"""RTP-LLM batch decode performance test — main entry point.

Three-phase flow:
  1. Configure — parse args, resolve paths, build PerfTestConfig
  2. Serve    — start engine, query engine status, print config tables
  3. Run      — dispatch to prefill or decode runner, collect timelines
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from rtp_llm.test.perf_test.cache_grid_runner import CacheGridRunner
from rtp_llm.test.perf_test.dataclass import PerfTestConfig
from rtp_llm.test.perf_test.dataset import extract_arg
from rtp_llm.test.perf_test.distribution_runner import DistributionRunner
from rtp_llm.test.perf_test.grid_runner import GridRunner
from rtp_llm.test.perf_test.perf_config import (
    parse_args,
    prepare_config,
    resolve_perf_engine_paths,
)
from rtp_llm.test.perf_test.perf_utils import (
    collect_timeline_files,
    filter_bs_by_kvcache,
    print_config_table,
    query_engine_status,
    write_test_info,
)
from rtp_llm.test.perf_test.server import EngineServer
from rtp_llm.test.perf_test.test_util import create_query
from rtp_llm.test.perf_test.tps_runner import TpsBinarySearchRunner
from rtp_llm.test.utils.coredump_util import summarize_and_cleanup_coredumps

# ---------------------------------------------------------------------------
#  Backward-compatible wrapper (used by external callers)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
#  Grid-mode helpers
# ---------------------------------------------------------------------------


def _load_cache_grid_cases(path: str) -> List[Dict[str, int]]:
    """Load and validate an explicit total-sequence × cache-length grid."""
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


def _effective_grid_max_seq_len(
    args: argparse.Namespace, input_len_list: List[int]
) -> int:
    """Grid-mode max_seq_len: decode headroom, but never below explicit --max_seq_len.

    prepare_config() sizes grid max_seq_len as max(input_len) + decode_test_length.
    DSv4 perf targets size the KV pool from an explicitly larger --max_seq_len
    (e.g. --input_len 65536 --decode_test_length 100 --max_seq_len 65664), so that
    request must win.  Distribution mode already does this in prepare_config().
    """
    needed_seq_len = max(input_len_list) + args.decode_test_length
    return max(needed_seq_len, args.max_seq_len)


def _explicit_batch_size_list(args: argparse.Namespace) -> Optional[List[int]]:
    """--batch_size as given on the command line, or None when it was defaulted."""
    if not any(a.startswith("--batch_size") for a in sys.argv[1:]):
        return None
    return [int(x) for x in args.batch_size.split(",")]


# ---------------------------------------------------------------------------
#  Phase 3: Run — prefill / decode dispatch
# ---------------------------------------------------------------------------


def _run_prefill(
    port: int,
    dp_size: int,
    config: PerfTestConfig,
    input_query_dict: Dict[int, str],
    batch_size_list: Optional[List[int]] = None,
    **kwargs: Any,
) -> None:
    """Prefill grid run.

    Defaults to BS=1 (prefill measures single-request TTFT); prepare_config()
    pins config.batch_size_list to [1] for --partial 2 as well.  DSv4 prefill
    targets (e.g. v4_flash_cp4_ep4_prefill_64k_perf) sweep prefill at
    batch_size > 1, so main() forwards an explicitly requested --batch_size.
    """
    if not config.input_len_list:
        return
    GridRunner(
        port,
        dp_size,
        batch_size_list or [1],
        config.input_len_list,
        input_query_dict,
        is_decode=False,
        **kwargs,
    ).run()


def _run_decode(
    port: int,
    dp_size: int,
    args: argparse.Namespace,
    config: PerfTestConfig,
    input_query_dict: Dict[int, str],
    engine_status: Dict[str, Any],
    **kwargs: Any,
) -> None:
    max_kv = (
        float(engine_status.get("max_kv_tokens", float("inf")))
        if engine_status
        else float("inf")
    )

    if args.target_tpot > 0:
        runner = TpsBinarySearchRunner(
            port,
            dp_size,
            args.target_tpot,
            max_bs=args.concurrency_limit,
            **kwargs,
        )
        if config.is_distribution:
            assert config.test_config is not None
            runner.run_distribution(config.test_config, input_query_dict)
        else:
            max_bs_per_len = {
                il: max(
                    [bs for bs in config.batch_size_list if bs * il <= max_kv] or [1]
                )
                for il in config.input_len_list
            }
            runner.run_grid(config.input_len_list, input_query_dict, max_bs_per_len)
    else:
        if config.is_distribution:
            assert config.test_config is not None
            DistributionRunner(
                port,
                dp_size,
                config.test_config,
                input_query_dict,
                **kwargs,
            ).run()
        else:
            for input_len in config.input_len_list:
                filtered_bs = filter_bs_by_kvcache(
                    config.batch_size_list, input_len, max_kv
                )
                if not filtered_bs:
                    logging.warning(
                        f"No BS fits KV cache for input_len={input_len}, skipping"
                    )
                    continue
                GridRunner(
                    port,
                    dp_size,
                    filtered_bs,
                    [input_len],
                    input_query_dict,
                    is_decode=True,
                    **kwargs,
                ).run()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> str:
    from rtp_llm.config.log_config import setup_logging

    setup_logging()

    args, remaining = parse_args()
    remaining = resolve_perf_engine_paths(remaining)
    # batch_decode_test always needs BatchDecodeScheduler
    if extract_arg(remaining, "use_batch_decode_scheduler") is None:
        remaining.extend(["--use_batch_decode_scheduler", "1"])
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
        try:
            server.start(
                max_seq_len=max(
                    max_input_len + args.decode_test_length, args.max_seq_len
                ),
                max_concurrency=max_batch_size,
                use_batch_decode_scheduler=True,
            )
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
            collect_timeline_files(args.result_dir)
            write_test_info(args, remaining)
        finally:
            server.stop()
            summarize_and_cleanup_coredumps(args.result_dir)
        return args.result_dir

    # Phase 1: Configure
    config = prepare_config(args, remaining)
    if not config.is_distribution:
        config.max_seq_len = _effective_grid_max_seq_len(args, config.input_len_list)

    # Phase 2: Serve
    server = EngineServer(args, remaining)
    try:
        server.start(
            max_seq_len=config.max_seq_len,
            max_concurrency=config.max_concurrency,
            use_batch_decode_scheduler=True,
        )
        engine_status = query_engine_status(server.port)
        print_config_table(args, config, engine_status, remaining)

        # Phase 3: Run
        input_query_dict = create_query(input_len_list=config.all_seq_lens)
        runner_kwargs = dict(
            dump_json_path=args.result_dir,
            decode_test_length=args.decode_test_length,
            generate_config=generate_config,
            num_measures=args.num_measures,
        )

        if args.partial == 2:
            _run_prefill(
                server.port,
                args.dp_size,
                config,
                input_query_dict,
                batch_size_list=_explicit_batch_size_list(args),
                **runner_kwargs,
            )

        if args.partial == 1:
            _run_decode(
                server.port,
                args.dp_size,
                args,
                config,
                input_query_dict,
                engine_status,
                **runner_kwargs,
            )

        # Cleanup
        collect_timeline_files(args.result_dir)
        server.stop()
        write_test_info(args, remaining)

        if args.partial != 2:
            from rtp_llm.test.perf_test.visualization import plot_decode_results

            try:
                plot_decode_results(args.result_dir)
            except Exception as e:
                logging.warning(f"plot_decode_results failed: {e}")
    finally:
        summarize_and_cleanup_coredumps(args.result_dir)

    return args.result_dir


if __name__ == "__main__":
    main()
