#!/usr/bin/env python3
"""Start a single mock engine as a standalone process.

Used by engine_kill_restart_test.sh to start a single engine that can be
independently killed and restarted, separate from the mock_engine_cluster.py
process which manages the remaining engines.

Since mock_engine_cluster.py starts all engines within a single Python process
(one asyncio event loop, one PID), it is impossible to kill a single engine
without killing the entire cluster.  This script creates a MockEngineCluster
with exactly one engine, giving it its own PID, gRPC port, and HTTP control
API so that it can be killed and restarted independently.

Usage:
    python3 run_single_engine.py \\
        --grpc-port 55301 \\
        --role prefill \\
        --name prefill-1 \\
        --performance perf.json \\
        --cache-blocks 6000
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal

from online_eval.mock_engine import MockEngineCluster
from online_eval.proto_utils import ensure_proto_modules
from online_eval.rt_model import (
    PerformanceModel,
    extract_prefill_formula_from_master_config,
    load_performance_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--grpc-port", type=int, required=True)
    parser.add_argument("--role", required=True, choices=["prefill", "decode"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--performance", help="performance model JSON")
    parser.add_argument(
        "--master-config",
        default=None,
        help="Master config JSON (read PREFILL_TIME_FORMULA for the mock engine)",
    )
    parser.add_argument("--cache-blocks", type=int, default=6000)
    parser.add_argument("--total-kv-tokens", type=int, default=6_291_456)
    parser.add_argument("--block-size", type=int, default=1024)
    return parser.parse_args()


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    args = parse_args()
    pb2, pb2_grpc = ensure_proto_modules()
    perf_cfg = load_performance_config(args.performance)
    perf_cfg.setdefault("block_size", args.block_size)
    formula_str = extract_prefill_formula_from_master_config(args.master_config)
    if formula_str:
        perf_cfg.setdefault("prefill", {})["formula_str"] = formula_str
    performance = PerformanceModel(perf_cfg)

    http_port = args.grpc_port - 1
    cluster = MockEngineCluster(pb2, pb2_grpc, performance, base_http_port=http_port)

    await cluster.add_engine(
        name=args.name,
        role=args.role,
        host=args.host,
        port=args.grpc_port,
        cache_capacity_blocks=args.cache_blocks,
        total_kv_tokens=args.total_kv_tokens,
        block_size=args.block_size,
    )

    await cluster.start_http_server(http_port)

    print(
        f"single engine started: {args.name} ({args.role}) "
        f"at {args.host}:{args.grpc_port} (http={http_port})",
        flush=True,
    )
    print("press Ctrl-C to stop", flush=True)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    await stop_event.wait()
    await cluster.stop()


if __name__ == "__main__":
    asyncio.run(main())
