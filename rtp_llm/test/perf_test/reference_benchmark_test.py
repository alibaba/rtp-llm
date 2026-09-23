#!/usr/bin/env python3
"""Start the standard CEP4PP2 server and measure the exact reference prompts."""
from __future__ import annotations

import json
import logging
import os
import pathlib
import time

from rtp_llm.test.perf_test.reference_ttft_client import (
    REFERENCE_MANIFEST_SHA256,
    run_reference_benchmark,
)


def _required_protocol_environment(environ=os.environ):
    expected = {
        "PERF_GRID_WARMUP_RUNS": "2",
        "PERF_MEASURE_RUNS": "8",
        "PERF_FORMAL_WARMUP_RUNS": "0",
        "PERF_PROFILE_RUNS": "0",
        "DSV4_FWD_PROFILE": "0",
    }
    bad = {
        key: (environ.get(key), value)
        for key, value in expected.items()
        if environ.get(key) != value
    }
    if bad:
        raise RuntimeError("reference protocol environment mismatch: %r" % bad)
    return expected


def _manifest_path() -> pathlib.Path:
    raw = os.environ.get("RTP_LLM_REFERENCE_MANIFEST")
    if not raw:
        raise RuntimeError(
            "RTP_LLM_REFERENCE_MANIFEST must name the frozen reference input"
        )
    path = pathlib.Path(raw)
    if not path.is_file():
        raise RuntimeError("reference manifest is absent from runfiles: %s" % path)
    return path


def main() -> str:
    # Import the existing standard lifecycle only after the pure CPU protocol
    # checks above are importable/testable without RTP/Torch/CUDA.
    from rtp_llm.config.log_config import setup_logging
    from rtp_llm.test.perf_test.batch_decode_test import (
        _configure_scheduler,
        _effective_grid_max_seq_len,
        _ensure_default_role_type,
        _unset_tpsync_for_timing,
    )
    from rtp_llm.test.perf_test.perf_config import (
        parse_args,
        prepare_config,
        resolve_perf_engine_paths,
    )
    from rtp_llm.test.perf_test.perf_utils import (
        print_config_table,
        query_engine_status,
        write_test_info,
    )
    from rtp_llm.test.perf_test.server import EngineServer
    from rtp_llm.test.utils.coredump_util import summarize_and_cleanup_coredumps

    _required_protocol_environment()
    _unset_tpsync_for_timing()
    setup_logging()
    args, remaining = parse_args()
    remaining = resolve_perf_engine_paths(remaining)
    _ensure_default_role_type(remaining)
    generate_config = json.loads(args.generate_config)
    if generate_config:
        raise RuntimeError("reference target owns its exact generation config")
    use_batch_decode_scheduler = _configure_scheduler(args, remaining, generate_config)
    config = prepare_config(args, remaining)
    if (
        args.partial != 2
        or args.dp_size != 1
        or args.batch_size != "1"
        or config.input_len_list != [32768]
        or args.decode_test_length != 1
        or config.is_distribution
    ):
        raise RuntimeError(
            "reference target requires PP prefill BS1 32768/1 grid identity"
        )
    config.max_seq_len = _effective_grid_max_seq_len(args, config.input_len_list)
    os.makedirs(args.result_dir, exist_ok=True)
    EngineServer.propagate_engine_env(remaining)
    server = EngineServer(args, remaining)
    started = time.time()
    result = None
    try:
        server.start(
            max_seq_len=config.max_seq_len,
            max_concurrency=1,
            use_batch_decode_scheduler=use_batch_decode_scheduler,
        )
        engine_status = query_engine_status(server.port)
        print_config_table(args, config, engine_status, remaining)
        result = run_reference_benchmark(
            port=server.port,
            manifest_path=_manifest_path(),
            output_path=pathlib.Path(args.result_dir) / "reference_benchmark.json",
            timeout=1800.0,
            warmup_runs=2,
            measure_runs=8,
            expected_iter_count=8,
        )
        if not result.get("valid_run"):
            raise RuntimeError("reference benchmark returned an invalid/incomplete run")
        logging.info(
            "reference benchmark summary: %s",
            json.dumps(result["statistics_successes_only"]),
        )
    finally:
        try:
            server.stop()
        finally:
            summarize_and_cleanup_coredumps(args.result_dir)
    write_test_info(args, remaining)
    pathlib.Path(args.result_dir, "reference_protocol.json").write_text(
        json.dumps(
            {
                "manifest_sha256": REFERENCE_MANIFEST_SHA256,
                "warmups": 2,
                "measures": 8,
                "client_metric": "first token-bearing SSE event",
                "server_metric": "first_token_cost_time and queue-excluded derivative",
                "started_at_epoch": started,
            },
            indent=2,
        )
        + "\n"
    )
    return args.result_dir


if __name__ == "__main__":
    main()
