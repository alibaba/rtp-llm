#!/usr/bin/env python3
"""Run one K3 single-role performance workload through RTP's perf framework."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from rtp_llm.test.perf_test.batch_perf_impl import BatchPerfImpl
from rtp_llm.test.perf_test.dataset import extract_arg
from rtp_llm.test.perf_test.server import EngineServer
from rtp_llm.test.perf_test.test_util import _load_tokenizer


def _unique_query_worker(args):
    tokenizer_path, input_len, index = args
    tokenizer = _load_tokenizer(tokenizer_path)
    marker = f"k3-independent-prefix-{index:08d} "
    source = marker + ("hello " * (input_len + 32))
    left, right = 0, len(source)
    best = marker
    while left < right:
        mid = (left + right) // 2
        candidate = source[:mid]
        length = len(tokenizer.encode(candidate))
        if length == input_len:
            return index, candidate
        if length < input_len:
            best = candidate
            left = mid + 1
        else:
            right = mid
    while len(tokenizer.encode(best)) < input_len:
        best += " x"
    while len(tokenizer.encode(best)) > input_len and len(best) > len(marker):
        best = best[:-1]
    if len(tokenizer.encode(best)) != input_len:
        raise ValueError(f"cannot construct exact prefix {index} at length {input_len}")
    return index, best


def create_unique_queries(tokenizer_path: str, input_len: int, count: int):
    if input_len <= 0 or count <= 0:
        raise ValueError("input_len and count must be positive")
    args = [(tokenizer_path, input_len, index) for index in range(count)]
    with ProcessPoolExecutor(max_workers=min(8, count)) as executor:
        result = list(executor.map(_unique_query_worker, args))
    return [query for _, query in sorted(result)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prefill", "decode"), required=True)
    parser.add_argument("--input-len", type=int, default=960000)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument(
        "--workset-size",
        type=int,
        default=0,
        help="number of independent prefixes; defaults to batch-size",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--decode-test-length", type=int, default=512)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-trace-name", default="k3")
    parser.add_argument("--dp-size", type=int, default=1)
    args, remaining = parser.parse_known_args()
    if args.input_len <= 0 or args.batch_size <= 0 or args.rounds <= 0:
        parser.error("input length, batch size and rounds must be positive")
    if args.workset_size < 0:
        parser.error("workset size must be non-negative")
    if (args.workset_size or args.batch_size) % args.batch_size:
        parser.error("workset size must be a multiple of batch size")
    if args.mode == "decode" and args.decode_test_length <= 0:
        parser.error("decode test length must be positive")
    if args.mode == "prefill" and (args.batch_size != 1 or args.dp_size != 1):
        parser.error("prefill working-set comparison uses sequential batch-size=1, dp-size=1")
    return args, remaining


def main() -> int:
    args, remaining = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.partial = 2 if args.mode == "prefill" else 1
    args.profile_runs = 1 if args.profile else 0
    args.unique_prefixes = True
    args.decode_test_length = 1 if args.mode == "prefill" else args.decode_test_length
    # PDFUSION also initializes Prefill workspaces during a Decode-only run.
    # Bound them by the normal K3 model chunk, not the historical KV length.
    os.environ.setdefault("KIMI_K3_PREFILL_CHUNK_TOKENS", "65536")
    if args.mode == "prefill":
        tp_size = int(extract_arg(remaining, "tp_size", "1"))
        os.environ.setdefault("KIMI_K3_SHARED_EXPERT_WEIGHT_SHARD", str(int(tp_size % 2 == 0)))
        # BatchDecodeScheduler resets the clock after cache loading. Use the
        # normal Prefill lifecycle to measure both host reuse and queueing.
        remaining += ["--use_batch_decode_scheduler", "0", "--role_type", "PREFILL", "--reuse_cache", "1"]
    EngineServer.propagate_engine_env(remaining)
    # K3's Python hidden-state producer still selects MTP/Eagle via SP_TYPE,
    # while the executor reads the parsed CLI configuration.
    sp_type = extract_arg(remaining, "sp_type", os.environ.get("SP_TYPE", ""))
    if sp_type:
        os.environ["SP_TYPE"] = sp_type
    tokenizer_path = os.environ.get(
        "TOKENIZER_PATH", os.environ.get("CHECKPOINT_PATH", "")
    )
    for index, value in enumerate(remaining[:-1]):
        if value in ("--tokenizer_path", "--checkpoint_path"):
            tokenizer_path = remaining[index + 1]
            break
    workset_size = args.workset_size or args.batch_size
    queries = create_unique_queries(tokenizer_path, args.input_len, workset_size)
    server = EngineServer(args, remaining)
    # GenerateStream::maxTokenNum subtracts speculative output-buffer reserve;
    # async execution reserves 2 * propose_step + 1 tokens beyond the output.
    propose_step = int(extract_arg(remaining, "gen_num_per_cycle", "3"))
    server.start(
        max_seq_len=args.input_len + args.decode_test_length + 2 * propose_step + 1,
        max_concurrency=args.batch_size,
    )
    results = []
    try:
        # Prefill needs both a fill and a revisit to warm the cache-hit path.
        # Traverse the whole working set so warmup cannot hide eviction.
        warmup_rounds = 2 if args.mode == "prefill" else 1
        for round_index in range(-warmup_rounds, args.rounds):
            round_metrics = []
            for start in range(0, workset_size, args.batch_size):
                batch = queries[start : start + args.batch_size]
                if len(batch) < args.batch_size:
                    continue
                metric = BatchPerfImpl(
                    server.port,
                    args.dp_size,
                    args.batch_size,
                    batch,
                    is_decode=args.mode == "decode",
                    wait_time=1800,
                    decode_test_length=args.decode_test_length,
                    profile=False,
                    profile_trace_name=f"{args.profile_trace_name}_r{round_index}",
                    warmup_runs=0,
                    measure_runs=1,
                    profile_runs=0,
                ).run()
                row = vars(metric).copy()
                if args.mode == "prefill":
                    row["avg_ttft_ms"] = metric.avg_prefill_time + metric.avg_wait_time
                round_metrics.append(row)
            if round_index >= 0:
                results.append({"round": round_index, "metrics": round_metrics})
        # Profiling also touches cache, so do it only after all timed visits.
        if args.profile:
            BatchPerfImpl(
                server.port,
                args.dp_size,
                args.batch_size,
                queries[:args.batch_size],
                is_decode=args.mode == "decode",
                wait_time=1800,
                decode_test_length=args.decode_test_length,
                profile=True,
                profile_trace_name=args.profile_trace_name,
                warmup_runs=0,
                measure_runs=0,
                profile_runs=1,
            ).run()
    finally:
        server.stop()
    output = {
        "mode": args.mode,
        "input_len": args.input_len,
        "batch_size": args.batch_size,
        "workset_size": workset_size,
        "rounds": args.rounds,
        "warmup_rounds": warmup_rounds,
        "metrics": results,
    }
    (args.result_dir / "k3_perf.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
