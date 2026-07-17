#!/usr/bin/env python3
"""Start a mock rtp-llm engine cluster for FlexLB online evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
from pathlib import Path

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
    parser.add_argument("--base-grpc-port", type=int, default=50051)
    parser.add_argument("--n-prefill", type=int, default=2)
    parser.add_argument("--n-decode", type=int, default=4)
    parser.add_argument("--performance", help="performance model JSON")
    parser.add_argument(
        "--master-config",
        default=None,
        help="Master config JSON (read PREFILL_TIME_FORMULA from here "
        "so the mock engine uses the same formula as the Master, "
        "eliminating coefficient duplication)",
    )
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
    parser.add_argument(
        "--per-engine-perf",
        default=None,
        help="JSON list of per-engine performance overrides, e.g. "
        '[{"name":"prefill-0","prefill_ms":200}]',
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Current shard ID (default 0)",
    )
    parser.add_argument(
        "--total-shards",
        type=int,
        default=1,
        help="Total number of shards (default 1)",
    )
    parser.add_argument(
        "--partial-endpoint-file",
        default=None,
        help="Path to write partial endpoint file for this shard "
        "(for multi-process sharding; default: not written)",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=None,
        help="Override HTTP control port (default: auto-computed from "
        "base_grpc_port and shard_id)",
    )
    return parser.parse_args()


def build_per_engine_perf(
    base_cfg: dict, per_engine_json: str | None
) -> dict[str, PerformanceModel]:
    """Parse --per-engine-perf JSON into {engine_name: PerformanceModel}.

    Supports two formats:
    1. JSON list:  [{"name":"prefill-0","prefill_ms":200}]
    2. JSON dict:  {"prefill-0": {"prefill_fixed_ms": 100}, "prefill-1": {"prefill_fixed_ms": 200}}
    """
    if not per_engine_json:
        return {}
    import copy

    parsed = json.loads(per_engine_json)
    result: dict[str, PerformanceModel] = {}

    if isinstance(parsed, dict):
        # Dict format: {engine_name: {override_keys}}
        for name, overrides in parsed.items():
            if not name or not isinstance(overrides, dict):
                continue
            cfg = copy.deepcopy(base_cfg)
            if "prefill_fixed_ms" in overrides:
                cfg.setdefault("prefill", {})["fixed_ms"] = overrides[
                    "prefill_fixed_ms"
                ]
            if "decode_scale" in overrides:
                cfg.setdefault("decode", {})["scale"] = overrides["decode_scale"]
            for key in ("prefill", "decode", "sleep_scale", "block_size"):
                if key in overrides:
                    if isinstance(cfg.get(key), dict) and isinstance(
                        overrides[key], dict
                    ):
                        cfg[key].update(overrides[key])
                    else:
                        cfg[key] = overrides[key]
            result[name] = PerformanceModel(cfg)
    elif isinstance(parsed, list):
        # List format (existing): [{"name": "prefill-0", "prefill_ms": 200}]
        for entry in parsed:
            name = entry.get("name")
            if not name:
                continue
            cfg = copy.deepcopy(base_cfg)
            if "prefill_ms" in entry:
                cfg.setdefault("prefill", {})["fixed_ms"] = entry["prefill_ms"]
            for key in ("prefill", "decode", "sleep_scale", "block_size"):
                if key in entry:
                    if isinstance(cfg.get(key), dict) and isinstance(entry[key], dict):
                        cfg[key].update(entry[key])
                    else:
                        cfg[key] = entry[key]
            result[name] = PerformanceModel(cfg)
    return result


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    args = parse_args()
    pb2, pb2_grpc = ensure_proto_modules()
    perf_cfg = load_performance_config(args.performance)
    perf_cfg.setdefault("block_size", args.block_size)
    # Read PREFILL_TIME_FORMULA from the master config so the mock engine
    # evaluates the same formula as the Java Master — single source of truth.
    formula_str = extract_prefill_formula_from_master_config(args.master_config)
    if formula_str:
        perf_cfg.setdefault("prefill", {})["formula_str"] = formula_str
    performance = PerformanceModel(perf_cfg)
    per_engine_perf = build_per_engine_perf(perf_cfg, args.per_engine_perf)
    # HTTP control port: unique per shard when sharding.
    # In multi-shard mode, placed well above the gRPC engine port range to
    # avoid collisions with the Linux kernel ephemeral port range (32768-60999).
    # In single-process mode, ``base - 1`` is kept for backward compatibility
    # with standalone test scripts that compute ``MOCK_BASE_GRPC_PORT - 1``.
    if args.http_port is not None:
        http_port = args.http_port
    elif args.total_shards > 1:
        http_port = (
            args.base_grpc_port + args.n_prefill + args.n_decode + 100 + args.shard_id
        )
    else:
        http_port = args.base_grpc_port - 1

    cluster = MockEngineCluster(pb2, pb2_grpc, performance, base_http_port=http_port)

    # Calculate this shard's engine index ranges.
    # Engine names and ports are global (based on the global engine index)
    # so that merging endpoints from all shards produces globally unique
    # names and ports.
    pf_start = args.shard_id * args.n_prefill // args.total_shards
    pf_end = (args.shard_id + 1) * args.n_prefill // args.total_shards
    dc_start = args.shard_id * args.n_decode // args.total_shards
    dc_end = (args.shard_id + 1) * args.n_decode // args.total_shards

    for i in range(pf_start, pf_end):
        name = f"prefill-{i}"
        port = args.base_grpc_port + i
        await cluster.add_engine(
            name=name,
            role="prefill",
            host=args.host,
            port=port,
            cache_capacity_blocks=args.prefill_cache_blocks,
            total_kv_tokens=args.prefill_total_kv_tokens,
            block_size=args.block_size,
            performance_override=per_engine_perf.get(name),
        )
    for i in range(dc_start, dc_end):
        name = f"decode-{i}"
        port = args.base_grpc_port + args.n_prefill + i
        await cluster.add_engine(
            name=name,
            role="decode",
            host=args.host,
            port=port,
            cache_capacity_blocks=args.decode_cache_blocks,
            total_kv_tokens=args.decode_total_kv_tokens,
            block_size=args.block_size,
            performance_override=per_engine_perf.get(name),
        )

    # Write partial endpoint file BEFORE starting the HTTP server so that
    # if the HTTP server crashes, the launcher still knows which engines
    # this shard intended to host.
    if args.partial_endpoint_file:
        write_partial_endpoints(cluster, args, http_port)
    try:
        await cluster.start_http_server(http_port)
    except OSError as exc:
        logging.error(
            "shard %d: failed to start HTTP server on port %d: %s",
            args.shard_id,
            http_port,
            exc,
        )
        raise
    if args.total_shards <= 1:
        await write_outputs(cluster, args)

    n_pf = pf_end - pf_start
    n_dc = dc_end - dc_start
    print(f"mock engine cluster started: {n_pf} prefill, " f"{n_dc} decode")
    print(f"endpoint file: {args.endpoint_file}")
    print(f"flexlb env file: {args.env_file}")
    print(f"http control API on port {http_port}")
    if args.total_shards > 1:
        print(f"shard {args.shard_id}/{args.total_shards}")
    if args.partial_endpoint_file:
        print(f"partial endpoint file: {args.partial_endpoint_file}")
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


def write_partial_endpoints(
    cluster: MockEngineCluster, args: argparse.Namespace, http_port: int
) -> None:
    """Write a partial endpoint file for this shard.

    The file contains this shard's engine info in a format suitable for
    merging by mock_engine_shard_launcher.py.
    """
    engines = []
    for state in cluster.states:
        engines.append(
            {
                "name": state.name,
                "ip": state.host,
                "grpc_port": state.grpc_port,
                "http_port": state.http_port,
                "role": state.role,
            }
        )
    payload = {
        "engines": engines,
        "shard_id": args.shard_id,
        "http_port": http_port,
    }
    path = Path(args.partial_endpoint_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


async def snapshot_loop(cluster: MockEngineCluster, interval_s: float) -> None:
    if interval_s <= 0:
        return
    while True:
        await asyncio.sleep(interval_s)
        snapshot = await cluster.snapshot()
        # per-engine detailed log
        for e in snapshot["engines"]:
            rpc = e.get("rpc_counts", {})
            print(
                f"[{e['name']}] role={e['role']} running={e['running']} "
                f"accepted={e['accepted']} completed={e['completed']} "
                f"cancelled={e.get('cancelled_count', 0)} "
                f"rpc={rpc} "
                f"cache={e['cache_keys']} avail_kv={e['available_kv_tokens']}",
                flush=True,
            )
        # cluster summary
        total_running = sum(e["running"] for e in snapshot["engines"])
        total_accepted = sum(e["accepted"] for e in snapshot["engines"])
        total_completed = sum(e["completed"] for e in snapshot["engines"])
        total_cancelled = sum(e.get("cancelled_count", 0) for e in snapshot["engines"])
        print(
            f"[CLUSTER] engines={len(snapshot['engines'])} "
            f"total_running={total_running} total_accepted={total_accepted} "
            f"total_completed={total_completed} total_cancelled={total_cancelled}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
