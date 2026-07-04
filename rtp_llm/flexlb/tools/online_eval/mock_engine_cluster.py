#!/usr/bin/env python3
"""Start a mock rtp-llm engine cluster for FlexLB online evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
from pathlib import Path

from online_eval.mock_engine import MockEngineCluster
from online_eval.proto_utils import ensure_proto_modules
from online_eval.rt_model import PerformanceModel, load_performance_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-grpc-port", type=int, default=50051)
    parser.add_argument("--n-prefill", type=int, default=2)
    parser.add_argument("--n-decode", type=int, default=4)
    parser.add_argument("--performance", help="performance model JSON")
    parser.add_argument("--prefill-cache-blocks", type=int, default=6000)
    parser.add_argument("--decode-cache-blocks", type=int, default=3000)
    parser.add_argument("--prefill-total-kv-tokens", type=int, default=6_291_456)
    parser.add_argument("--decode-total-kv-tokens", type=int, default=6_291_456)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--prefill-domain", default="mock.prefill.hosts.address")
    parser.add_argument("--decode-domain", default="mock.decode.hosts.address")
    parser.add_argument(
        "--endpoint-file", default="rtp_llm/flexlb/tools/online_eval/run/endpoints.json"
    )
    parser.add_argument(
        "--env-file", default="rtp_llm/flexlb/tools/online_eval/run/flexlb_env.txt"
    )
    parser.add_argument("--snapshot-interval-s", type=float, default=5.0)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    pb2, pb2_grpc = ensure_proto_modules()
    perf_cfg = load_performance_config(args.performance)
    perf_cfg.setdefault("block_size", args.block_size)
    performance = PerformanceModel(perf_cfg)
    cluster = MockEngineCluster(pb2, pb2_grpc, performance)

    port = args.base_grpc_port
    for i in range(args.n_prefill):
        await cluster.add_engine(
            name=f"prefill-{i}",
            role="prefill",
            host=args.host,
            port=port,
            cache_capacity_blocks=args.prefill_cache_blocks,
            total_kv_tokens=args.prefill_total_kv_tokens,
            block_size=args.block_size,
        )
        port += 1
    for i in range(args.n_decode):
        await cluster.add_engine(
            name=f"decode-{i}",
            role="decode",
            host=args.host,
            port=port,
            cache_capacity_blocks=args.decode_cache_blocks,
            total_kv_tokens=args.decode_total_kv_tokens,
            block_size=args.block_size,
        )
        port += 1

    await write_outputs(cluster, args)
    print(
        f"mock engine cluster started: {args.n_prefill} prefill, {args.n_decode} decode"
    )
    print(f"endpoint file: {args.endpoint_file}")
    print(f"flexlb env file: {args.env_file}")
    print("press Ctrl-C to stop")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    snapshot_task = asyncio.create_task(
        snapshot_loop(cluster, args.snapshot_interval_s)
    )
    try:
        await stop_event.wait()
    finally:
        snapshot_task.cancel()
        await cluster.stop()


async def write_outputs(cluster: MockEngineCluster, args: argparse.Namespace) -> None:
    endpoint_path = Path(args.endpoint_file)
    endpoint_path.parent.mkdir(parents=True, exist_ok=True)
    env_path = Path(args.env_file)
    env_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = await cluster.snapshot()
    env = cluster.service_discovery_env(args.prefill_domain, args.decode_domain)
    payload = {
        "prefill_domain": args.prefill_domain,
        "decode_domain": args.decode_domain,
        "env": env,
        **snapshot,
    }
    endpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    env_lines = [
        "# Start flexlb-api with these environment variables.",
        "# DOMAIN_ADDRESS:* contains ':' and cannot be exported by bash directly;",
        "# pass it via env as shown below.",
        "",
        "env \\",
    ]
    for key, value in env.items():
        env_lines.append(f"  '{key}={value}' \\")
    env_lines.append("  <your-flexlb-api-start-command>")
    env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")


async def snapshot_loop(cluster: MockEngineCluster, interval_s: float) -> None:
    if interval_s <= 0:
        return
    while True:
        await asyncio.sleep(interval_s)
        snapshot = await cluster.snapshot()
        compact = [
            f"{e['name']} running={e['running']} completed={e['completed']} "
            f"cache={e['cache_keys']} avail_kv={e['available_kv_tokens']}"
            for e in snapshot["engines"]
        ]
        print(" | ".join(compact), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
