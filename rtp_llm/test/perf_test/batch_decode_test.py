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
from typing import Any, Dict, List, Optional

from rtp_llm.test.perf_test.dataclass import PerfTestConfig
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
#  Phase 3: Run — prefill / decode dispatch
# ---------------------------------------------------------------------------


def _run_prefill(
    port: int,
    dp_size: int,
    config: PerfTestConfig,
    input_query_dict: Dict[int, str],
    **kwargs: Any,
) -> None:
    if not config.input_len_list:
        return
    GridRunner(
        port,
        dp_size,
        [1],
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
    generate_config = json.loads(args.generate_config)
    os.makedirs(args.result_dir, exist_ok=True)
    EngineServer.propagate_engine_env(remaining)

    logging.info(f"Result directory: {args.result_dir}")
    logging.info(f"Engine args forwarded to server: {remaining}")

    # Phase 1: Configure
    config = prepare_config(args, remaining)

    # Phase 2: Serve
    server = EngineServer(args, remaining)
    try:
        server.start(
            max_seq_len=config.max_seq_len, max_concurrency=config.max_concurrency
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
                server.port, args.dp_size, config, input_query_dict, **runner_kwargs
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
