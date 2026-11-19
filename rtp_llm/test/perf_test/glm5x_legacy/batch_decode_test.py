"""Private upstream GLM5.x fixed-grid compatibility runner.

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

from rtp_llm.test.perf_test.engine_paths import extract_arg
from rtp_llm.test.perf_test.glm5x_legacy.dataclass import (
    PerfTestConfig,
    TableType,
    create_metrics_table,
)
from rtp_llm.test.perf_test.glm5x_legacy.grid_runner import GridRunner
from rtp_llm.test.perf_test.glm5x_legacy.perf_config import (
    _apply_engine_env,
    _apply_run_overrides,
    _engine_arg_argv,
    _parse_name_value,
    parse_args,
    prepare_config,
    resolve_perf_engine_paths,
)
from rtp_llm.test.perf_test.glm5x_legacy.perf_utils import (
    _is_sensitive_name,
    _redact_argv,
    _sanitize_provenance_value,
    collect_timeline_files,
    filter_bs_by_kvcache,
    print_config_table,
    query_engine_status,
    write_test_info,
)
from rtp_llm.test.perf_test.server import EngineServer
from rtp_llm.test.perf_test.test_util import create_query
from rtp_llm.test.utils.coredump_util import summarize_and_cleanup_coredumps


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
    batch_size_explicit = getattr(
        args,
        "batch_size_explicit",
        any(a.startswith("--batch_size") for a in sys.argv[1:]),
    )
    if not batch_size_explicit:
        return None
    return [int(x) for x in args.batch_size.split(",")]


_PERFORMANCE_ENV_NAMES = {
    "ACT_TYPE",
    "CACHE_CONFIG",
    "CACHE_STORE_TYPE",
    "CHECKPOINT_PATH",
    "CONCURRENCY_LIMIT",
    "CP_ROTATE_METHOD",
    "DEVICE_NAME",
    "DEVICE_RESERVE_MEMORY_BYTES",
    "DP_SIZE",
    "DSV4_CHUNK_TOKENS",
    "DSV4_FIXED_POOL_BLOCKS",
    "ENABLE_CUDA_GRAPH",
    "EP_SIZE",
    "FP8_KV_CACHE",
    "GEN_NUM_PER_CYCLE",
    "INT8_MODE",
    "KV_CACHE_MEM_BYTES",
    "KV_CACHE_MEM_MB",
    "LOAD_METHOD",
    "LOCAL_WORLD_SIZE",
    "MAX_BATCH_SIZE",
    "MAX_BATCH_TOKENS_SIZE",
    "MAX_CONTEXT_BATCH_SIZE",
    "MAX_SEQ_LEN",
    "MODEL_TYPE",
    "PREFILL_CP_KV_CACHE_SHARDED",
    "QUANTIZATION",
    "RESERVER_RUNTIME_MEM_MB",
    "SEQ_SIZE_PER_BLOCK",
    "SP_ACT_TYPE",
    "SP_CHECKPOINT_PATH",
    "SP_MODEL_TYPE",
    "SP_TYPE",
    "TOKENIZER_PATH",
    "TP_SIZE",
    "USE_DEEPEP_LOW_LATENCY",
    "USE_DEEPEP_MOE",
    "WORLD_SIZE",
}
_PERFORMANCE_ENV_PREFIXES = (
    "CACHE_",
    "CUDA_",
    "DEEP_EP_",
    "DG_JIT_",
    "DSV4_",
    "ENABLE_",
    "FP8_",
    "GEN_TIMELINE_",
    "INT8_",
    "KV_CACHE_",
    "LOAD_",
    "MODEL_",
    "MOE_",
    "NCCL_",
    "PERF_",
    "PREFILL_",
    "QUANTIZATION_",
    "RTP_LLM_",
    "SP_",
    "TORCH_",
    "USE_DEEP",
)


def _is_performance_runtime_name(name: str, explicit_names: set[str]) -> bool:
    return (
        name in explicit_names
        or name in _PERFORMANCE_ENV_NAMES
        or name.startswith(_PERFORMANCE_ENV_PREFIXES)
    )


def _fingerprint_engine_env(names: List[str]) -> Dict[str, str]:
    """Capture runtime env with plaintext limited to the reviewed safe allowlist."""
    explicit_names = set(names)
    relevant_names = {
        name
        for name in os.environ
        if _is_performance_runtime_name(name, explicit_names)
        and not _is_sensitive_name(name)
    }
    return {
        name: str(
            _sanitize_provenance_value(
                name,
                os.environ[name],
                allow_plaintext=name in _PERFORMANCE_ENV_NAMES,
            )
        )
        for name in sorted(relevant_names)
    }


def _effective_performance_config(
    engine_args: List[str],
    engine_env_names: List[str],
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, str]]:
    """Resolve effective runtime values with CLI taking precedence over env."""
    values = _fingerprint_engine_env(engine_env_names)
    explicit_names = set(engine_env_names)
    sources = {
        name: "engine_env" if name in explicit_names else "environment"
        for name in values
    }
    index = 0
    while index < len(engine_args):
        argument = engine_args[index]
        index += 1
        if not argument.startswith("--"):
            continue
        option = argument[2:]
        if "=" in option:
            option, value = option.split("=", 1)
        elif index < len(engine_args) and not engine_args[index].startswith("--"):
            value = engine_args[index]
            index += 1
        else:
            value = "1"
        name = option.replace("-", "_").upper()
        if not _is_performance_runtime_name(name, explicit_names):
            continue
        if _is_sensitive_name(name):
            values.pop(name, None)
            sources.pop(name, None)
            continue
        values[name] = str(
            _sanitize_provenance_value(
                name,
                value,
                allow_plaintext=name in _PERFORMANCE_ENV_NAMES,
            )
        )
        sources[name] = "cli"

    for name, value in (cli_overrides or {}).items():
        normalized = name.replace("-", "_").upper()
        if _is_sensitive_name(normalized):
            continue
        values[normalized] = str(
            _sanitize_provenance_value(
                normalized,
                value,
                allow_plaintext=normalized in _PERFORMANCE_ENV_NAMES,
            )
        )
        sources[normalized] = "cli"
    return {
        "values": dict(sorted(values.items())),
        "sources": dict(sorted(sources.items())),
    }


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

    grid_metrics = []
    for input_len in config.input_len_list:
        filtered_bs = filter_bs_by_kvcache(config.batch_size_list, input_len, max_kv)
        if not filtered_bs:
            logging.warning(f"No BS fits KV cache for input_len={input_len}, skipping")
            continue
        metrics = GridRunner(
            port,
            dp_size,
            filtered_bs,
            [input_len],
            input_query_dict,
            is_decode=True,
            **kwargs,
        ).run()
        grid_metrics.extend(metrics)
    if grid_metrics:
        create_metrics_table(
            TableType.Decode,
            grid_metrics,
            kwargs.get("dump_json_path", "."),
            {"dp_size": dp_size, "tp_size": kwargs.get("tp_size", 1)},
            "Decode Result",
            kwargs.get("generate_config"),
        )


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> str:
    from rtp_llm.config.log_config import setup_logging

    setup_logging()

    args, remaining = parse_args()
    engine_env_names = _apply_engine_env(args.engine_env)
    _apply_run_overrides(args)
    remaining.extend(_engine_arg_argv(args.engine_arg))
    remaining = resolve_perf_engine_paths(remaining)
    # batch_decode_test always needs BatchDecodeScheduler
    if extract_arg(remaining, "use_batch_decode_scheduler") is None:
        remaining.extend(["--use_batch_decode_scheduler", "1"])
    generate_config = json.loads(args.generate_config)
    os.makedirs(args.result_dir, exist_ok=True)
    EngineServer.propagate_engine_env(remaining)

    logging.info(f"Result directory: {args.result_dir}")
    logging.info(f"Engine args forwarded to server: {_redact_argv(remaining)}")

    # Phase 1: Configure
    config = prepare_config(args, remaining)
    if not config.is_distribution:
        config.max_seq_len = _effective_grid_max_seq_len(args, config.input_len_list)
    effective_runtime_config = _effective_performance_config(
        remaining,
        engine_env_names,
        {
            "DP_SIZE": args.dp_size,
            "MAX_SEQ_LEN": config.max_seq_len,
            "CONCURRENCY_LIMIT": config.max_concurrency,
        },
    )
    write_test_info(
        args,
        remaining,
        engine_env_names,
        status="running",
        effective_max_seq_len=config.max_seq_len,
        service_concurrency_limit=config.max_concurrency,
        effective_runtime_config=effective_runtime_config,
    )

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
        write_test_info(
            args,
            remaining,
            engine_env_names,
            status="completed",
            effective_max_seq_len=config.max_seq_len,
            service_concurrency_limit=config.max_concurrency,
            effective_runtime_config=effective_runtime_config,
        )

        if args.partial != 2:
            from rtp_llm.test.perf_test.glm5x_legacy.visualization import (
                plot_decode_results,
            )

            try:
                plot_decode_results(args.result_dir)
            except Exception as e:
                logging.warning(f"plot_decode_results failed: {e}")
    finally:
        server.stop()
        summarize_and_cleanup_coredumps(args.result_dir)

    return args.result_dir
