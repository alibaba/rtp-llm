"""Offline throughput benchmark — standalone entry point.

Flow:
  1. Parse benchmark and engine arguments
  2. Start engine server
  3. Run benchmark and collect metrics
"""

import logging
import os

from rtp_llm.test.perf_test.dataset import extract_arg
from rtp_llm.test.perf_test.offline_runner import OfflineBenchConfig, OfflineRunner
from rtp_llm.test.perf_test.perf_config import resolve_perf_engine_paths
from rtp_llm.test.perf_test.server import EngineServer
from rtp_llm.test.utils.coredump_util import summarize_and_cleanup_coredumps


def parse_offline_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="RTP-LLM offline throughput benchmark. "
        "Unrecognized arguments are forwarded to the engine server.",
    )

    bench = parser.add_argument_group("benchmark")
    bench.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Duration in seconds to dispatch requests (duration mode)",
    )
    bench.add_argument(
        "--total_requests",
        type=int,
        default=-1,
        help="Fixed-workload mode: submit exactly N requests then drain. "
        "-1 = 2 * manually configured concurrency_limit, "
        "0 = duration mode.",
    )
    bench.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for workload generation (default 42)",
    )
    bench.add_argument(
        "--drain_timeout",
        type=int,
        default=0,
        help="Max seconds to wait for in-flight requests after dispatch ends. "
        "0 = wait forever (default). "
        "When exceeded, remaining requests are cancelled and report is generated.",
    )
    bench.add_argument("--input_len_min", type=int, default=512)
    bench.add_argument("--input_len_max", type=int, default=2048)
    bench.add_argument("--output_len_min", type=int, default=64)
    bench.add_argument("--output_len_max", type=int, default=512)
    bench.add_argument("--prefix_groups", type=int, default=1)
    bench.add_argument("--prefix_len", type=int, default=0)
    bench.add_argument("--num_return_sequences", type=int, default=1)
    bench.add_argument("--dump_workload", type=str, default="")
    bench.add_argument(
        "--result_dir",
        type=str,
        default=os.environ.get(
            "TEST_UNDECLARED_OUTPUTS_DIR", "./offline_bench_results"
        ),
    )
    bench.add_argument(
        "--max_seq_len",
        type=int,
        default=32768,
        help="Max sequence length (context + output). Defaults to 32768.",
    )
    bench.add_argument(
        "--concurrency_limit",
        type=int,
        required=True,
        help="Manually configured engine concurrency limit",
    )
    bench.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Enable GPU timeline profiling during steady state",
    )
    bench.add_argument(
        "--profile_steps",
        type=int,
        default=50,
        help="Number of steps to profile (default 50)",
    )
    bench.add_argument("--dp_size", type=int, default=1)
    bench.add_argument("--partial", type=int, default=1)
    bench.add_argument(
        "--checkpoint_path", type=str, default="", help="Model checkpoint path"
    )
    bench.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="Tokenizer path (defaults to checkpoint_path)",
    )
    bench.add_argument("--tp_size", type=int, default=1)
    bench.add_argument(
        "--server_start_timeout",
        type=int,
        default=1600,
        help="Max seconds to wait for engine server health check during startup",
    )

    args, remaining = parser.parse_known_args()
    return args, remaining


def main() -> str:
    from rtp_llm.config.log_config import setup_logging

    setup_logging()

    args, remaining = parse_offline_args()
    os.makedirs(args.result_dir, exist_ok=True)

    checkpoint_path = args.checkpoint_path or os.environ.get("CHECKPOINT_PATH", "")
    tokenizer_path = (
        args.tokenizer_path or os.environ.get("TOKENIZER_PATH", "") or checkpoint_path
    )
    tp_size = args.tp_size or 1

    if checkpoint_path and not extract_arg(remaining, "checkpoint_path"):
        remaining.extend(["--checkpoint_path", checkpoint_path])
    if tokenizer_path and not extract_arg(remaining, "tokenizer_path"):
        remaining.extend(["--tokenizer_path", tokenizer_path])
    if args.tp_size and not extract_arg(remaining, "tp_size"):
        remaining.extend(["--tp_size", str(tp_size)])

    # Resolve Hub IDs (e.g. "Qwen/Qwen3.5-35B-A3B") to local paths
    remaining = resolve_perf_engine_paths(remaining)
    checkpoint_path = extract_arg(remaining, "checkpoint_path") or checkpoint_path
    tokenizer_path = extract_arg(remaining, "tokenizer_path") or tokenizer_path
    model_type = extract_arg(remaining, "model_type", "") or os.environ.get(
        "MODEL_TYPE", ""
    )

    EngineServer.propagate_engine_env(remaining)

    dump_workload = args.dump_workload
    if dump_workload and not os.path.isabs(dump_workload):
        dump_workload = os.path.join(args.result_dir, dump_workload)

    total_requests = args.total_requests
    if total_requests == -1:
        total_requests = args.concurrency_limit * 2
        logging.info(
            f"Default total_requests: {total_requests} "
            f"(2 * concurrency_limit={args.concurrency_limit})"
        )

    config = OfflineBenchConfig(
        input_len_min=args.input_len_min,
        input_len_max=args.input_len_max,
        output_len_min=args.output_len_min,
        output_len_max=args.output_len_max,
        prefix_groups=args.prefix_groups,
        prefix_len=args.prefix_len,
        num_return_sequences=args.num_return_sequences,
        duration_s=args.duration,
        drain_timeout_s=args.drain_timeout,
        concurrency_limit=args.concurrency_limit,
        total_requests=total_requests,
        seed=args.seed,
        dump_workload=dump_workload,
    )
    config.validate()

    # Start engine and run benchmark
    server = EngineServer(args, remaining)
    try:
        server.start(
            max_seq_len=args.max_seq_len,
            max_concurrency=args.concurrency_limit,
            server_start_timeout=args.server_start_timeout,
            use_batch_decode_scheduler=False,
        )

        runner = OfflineRunner(
            port=server.port,
            config=config,
            tokenizer_path=tokenizer_path,
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            result_dir=args.result_dir,
            profile=args.profile,
            profile_steps=args.profile_steps,
        )

        runner.run()
    finally:
        try:
            server.stop()
        finally:
            summarize_and_cleanup_coredumps(args.result_dir)

    return args.result_dir


if __name__ == "__main__":
    main()
