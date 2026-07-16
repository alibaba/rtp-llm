#!/usr/bin/env python3
"""Multi-process shard launcher for mock engine cluster.

Starts *n_shards* ``mock_engine_cluster.py`` subprocesses, merges their
partial endpoint files into a single ``endpoints.json`` / ``flexlb_env.txt``,
and runs an HTTP proxy that fans out control requests across all shards.

The proxy listens on ``base_grpc_port + n_prefill + n_decode + 100 + n_shards``
(well above the gRPC engine port range, safely outside the kernel ephemeral
range 32768-60999).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import web
from online_eval.mock_engine import generate_aggregated_prometheus_metrics

logger = logging.getLogger("shard_launcher")

CLUSTER_SCRIPT = str(Path(__file__).resolve().parent / "mock_engine_cluster.py")

# POST endpoints that target a specific engine and must be routed to the
# shard that owns that engine.
_ENGINE_ROUTES = {
    "/inject",
    "/clear_inject",
    "/set_perf",
    "/set_kv_pressure",
    "/set_queue_depth",
    "/stop_engine",
    "/start_engine",
}

# Temp directory for partial endpoint files.
_TMP_DIR = Path("/tmp")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
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
        help="Master config JSON (read PREFILL_TIME_FORMULA for the mock engine)",
    )
    parser.add_argument("--prefill-cache-blocks", type=int, default=6000)
    parser.add_argument("--decode-cache-blocks", type=int, default=3000)
    parser.add_argument("--prefill-total-kv-tokens", type=int, default=6_291_456)
    parser.add_argument("--decode-total-kv-tokens", type=int, default=6_291_456)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--prefill-domain", default="mock.prefill.hosts.address")
    parser.add_argument("--decode-domain", default="mock.decode.hosts.address")
    parser.add_argument(
        "--endpoint-file",
        default="rtp_llm/flexlb/tools/online_eval/run/endpoints.json",
    )
    parser.add_argument(
        "--env-file",
        default="rtp_llm/flexlb/tools/online_eval/run/flexlb_env.txt",
    )
    parser.add_argument("--snapshot-interval-s", type=float, default=5.0)
    parser.add_argument("--per-engine-perf", default=None)
    parser.add_argument("--n-shards", type=int, default=2, help="Number of shards")
    parser.add_argument(
        "--shard-ready-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds to wait for all shards to become ready",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Shard subprocess management
# ---------------------------------------------------------------------------
def shard_http_port(
    base_grpc_port: int, shard_id: int, n_prefill: int = 0, n_decode: int = 0
) -> int:
    """HTTP control port for a shard.

    Placed well above the gRPC engine port range
    (``base_grpc_port + n_prefill + n_decode + 100 + shard_id``) so that
    it never collides with the Linux kernel ephemeral port range
    (32768-60999) when ``base_grpc_port`` is itself above that range.
    """
    return base_grpc_port + n_prefill + n_decode + 100 + shard_id


def proxy_http_port(
    base_grpc_port: int, n_prefill: int = 0, n_decode: int = 0, n_shards: int = 1
) -> int:
    """HTTP port for the shard-proxy server.

    One above the highest shard HTTP port
    (``base_grpc_port + n_prefill + n_decode + 100 + n_shards``).
    """
    return base_grpc_port + n_prefill + n_decode + 100 + n_shards


def partial_file_path(shard_id: int) -> Path:
    return _TMP_DIR / f"mock_partial_shard_{shard_id}.json"


def build_shard_command(args: argparse.Namespace, shard_id: int) -> List[str]:
    """Build the subprocess command for one shard."""
    cmd: List[str] = [
        sys.executable,
        CLUSTER_SCRIPT,
        "--host",
        args.host,
        "--base-grpc-port",
        str(args.base_grpc_port),
        "--n-prefill",
        str(args.n_prefill),
        "--n-decode",
        str(args.n_decode),
        "--prefill-cache-blocks",
        str(args.prefill_cache_blocks),
        "--decode-cache-blocks",
        str(args.decode_cache_blocks),
        "--prefill-total-kv-tokens",
        str(args.prefill_total_kv_tokens),
        "--decode-total-kv-tokens",
        str(args.decode_total_kv_tokens),
        "--block-size",
        str(args.block_size),
        "--prefill-domain",
        args.prefill_domain,
        "--decode-domain",
        args.decode_domain,
        "--shard-id",
        str(shard_id),
        "--total-shards",
        str(args.n_shards),
        "--partial-endpoint-file",
        str(partial_file_path(shard_id)),
        "--http-port",
        str(
            shard_http_port(
                args.base_grpc_port, shard_id, args.n_prefill, args.n_decode
            )
        ),
        # Launcher handles aggregated snapshots.
        "--snapshot-interval-s",
        "0",
    ]
    if args.performance:
        cmd += ["--performance", args.performance]
    if args.master_config:
        cmd += ["--master-config", args.master_config]
    if args.per_engine_perf:
        cmd += ["--per-engine-perf", args.per_engine_perf]
    return cmd


async def _check_shard_health(session: aiohttp.ClientSession, url: str) -> bool:
    """Check a single shard's /health endpoint."""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
            return resp.status == 200
    except Exception:
        return False


async def wait_for_shards(
    host: str,
    base_grpc_port: int,
    n_shards: int,
    timeout_s: float = 120.0,
    n_prefill: int = 0,
    n_decode: int = 0,
) -> bool:
    """Poll each shard's /health endpoint until all are ready (parallel checks).

    Logs which shards failed their health check for easier diagnosis.
    """
    async with aiohttp.ClientSession() as session:
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            shard_ids = list(range(n_shards))
            urls = [
                f"http://{host}:{shard_http_port(base_grpc_port, sid, n_prefill, n_decode)}/health"
                for sid in shard_ids
            ]
            results = await asyncio.gather(
                *[_check_shard_health(session, url) for url in urls]
            )
            if all(results):
                return True
            # Log which shards failed for diagnostics.
            failed = [
                (sid, shard_http_port(base_grpc_port, sid, n_prefill, n_decode))
                for sid, ok in zip(shard_ids, results)
                if not ok
            ]
            for sid, port in failed:
                logger.warning(
                    "shard %d health check FAILED on HTTP port %d "
                    "(http://%s:%d/health) — shard may still be starting "
                    "or crashed during startup",
                    sid,
                    port,
                    host,
                    port,
                )
            await asyncio.sleep(0.5)
        return False


# ---------------------------------------------------------------------------
# Endpoint merging
# ---------------------------------------------------------------------------
async def load_partial_endpoints(n_shards: int) -> List[dict]:
    """Read all shard partial endpoint files (with retry)."""
    for _ in range(20):
        partials = []
        ok = True
        for sid in range(n_shards):
            path = partial_file_path(sid)
            if path.exists():
                partials.append(json.loads(path.read_text(encoding="utf-8")))
            else:
                ok = False
                break
        if ok:
            return partials
        await asyncio.sleep(0.2)
    raise RuntimeError("Not all partial endpoint files appeared in time")


def merge_endpoints(args: argparse.Namespace, partials: List[dict]) -> Dict[str, str]:
    """Merge partial endpoint files into endpoints.json and flexlb_env.txt.

    Returns ``{engine_name: shard_http_url}`` routing table.
    """
    all_engines: List[dict] = []
    engine_routes: Dict[str, str] = {}
    for p in partials:
        shard_url = f"http://{args.host}:{p['http_port']}"
        for eng in p["engines"]:
            all_engines.append(eng)
            engine_routes[eng["name"]] = shard_url

    # Build env (mirrors MockEngineCluster.service_discovery_env).
    prefill_addrs = ",".join(
        f"{e['ip']}:{e['http_port']}" for e in all_engines if e["role"] == "prefill"
    )
    decode_addrs = ",".join(
        f"{e['ip']}:{e['http_port']}" for e in all_engines if e["role"] == "decode"
    )
    model_service_config = {
        "service_id": "aigc.text-generation.generation.engine_service",
        "load_balance": True,
        "role_endpoints": [
            {
                "group": "mock",
                "prefill_endpoint": {
                    "address": args.prefill_domain,
                    "protocol": "http",
                    "path": "/",
                },
                "decode_endpoint": {
                    "address": args.decode_domain,
                    "protocol": "http",
                    "path": "/",
                },
            }
        ],
    }
    env = {
        "MODEL_SERVICE_CONFIG": json.dumps(model_service_config, separators=(",", ":")),
        f"DOMAIN_ADDRESS:{args.prefill_domain}": prefill_addrs,
        f"DOMAIN_ADDRESS:{args.decode_domain}": decode_addrs,
    }

    # Write endpoints.json
    endpoint_path = Path(args.endpoint_file)
    endpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prefill_domain": args.prefill_domain,
        "decode_domain": args.decode_domain,
        "env": env,
        "engines": all_engines,
    }
    endpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Write flexlb_env.txt
    env_path = Path(args.env_file)
    env_path.parent.mkdir(parents=True, exist_ok=True)
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

    return engine_routes


# ---------------------------------------------------------------------------
# HTTP proxy server
# ---------------------------------------------------------------------------
class ShardProxy:
    """HTTP proxy that fans out control requests across all shards."""

    def __init__(
        self,
        host: str,
        proxy_port: int,
        shard_ports: List[int],
        engine_routes: Dict[str, str],
    ) -> None:
        self.host = host
        self.proxy_port = proxy_port
        self.shard_ports = shard_ports
        self.engine_routes = engine_routes
        self._session: Optional[aiohttp.ClientSession] = None
        self._runner: Optional[web.AppRunner] = None

    @property
    def shard_urls(self) -> List[str]:
        return [f"http://{self.host}:{p}" for p in self.shard_ports]

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        app = web.Application()
        app.router.add_get("/snapshot", self._handle_snapshot)
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/requests", self._handle_requests)
        app.router.add_get("/metrics", self._handle_metrics)
        for path in _ENGINE_ROUTES:
            app.router.add_post(path, self._handle_engine_route)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self.proxy_port)
        await site.start()

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
        if self._session:
            await self._session.close()

    async def _fetch_json(self, url: str, timeout: float = 5.0) -> Any:
        try:
            async with self._session.get(
                url, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status != 200:
                    return None
                return await resp.json()
        except Exception:
            return None

    async def _fetch_text(self, url: str, timeout: float = 5.0) -> str:
        try:
            async with self._session.get(
                url, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status != 200:
                    return ""
                return await resp.text()
        except Exception:
            return ""

    async def _handle_health(self, request: web.Request) -> web.Response:
        results = await asyncio.gather(
            *(self._fetch_json(f"{u}/health") for u in self.shard_urls)
        )
        all_ok = all(r and r.get("status") == "ok" for r in results)
        return web.json_response(
            {"status": "ok" if all_ok else "degraded"},
            status=200 if all_ok else 503,
        )

    async def _handle_snapshot(self, request: web.Request) -> web.Response:
        results = await asyncio.gather(
            *(self._fetch_json(f"{u}/snapshot") for u in self.shard_urls)
        )
        all_engines: List[dict] = []
        for r in results:
            if r and isinstance(r, dict):
                all_engines.extend(r.get("engines", []))
        return web.json_response({"engines": all_engines})

    async def _handle_requests(self, request: web.Request) -> web.Response:
        results = await asyncio.gather(
            *(self._fetch_json(f"{u}/requests") for u in self.shard_urls)
        )
        merged: Dict[str, Any] = {}
        for r in results:
            if r and isinstance(r, dict):
                merged.update(r)
        return web.json_response(merged)

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        """GET /metrics — Prometheus metrics from all shards.

        By default, returns role-aggregated metrics (sum/avg/max by role).
        Use ?per_engine=true to get per-engine metrics for debugging.
        """
        per_engine = request.query.get("per_engine", "").lower() == "true"
        if per_engine:
            results = await asyncio.gather(
                *(self._fetch_text(f"{u}/metrics") for u in self.shard_urls)
            )
            combined = "\n".join(r for r in results if r)
            return web.Response(
                text=combined,
                headers={
                    "Content-Type": "text/plain; version=0.0.4; charset=utf-8",
                },
            )
        # Default: JSON-based aggregation from /snapshot
        results = await asyncio.gather(
            *(self._fetch_json(f"{u}/snapshot") for u in self.shard_urls)
        )
        all_engines: list[dict] = []
        merged_counters = {
            "grpc_error_count": 0,
            "grpc_retry_count": 0,
            "grpc_cancel_forward_count": 0,
        }
        for r in results:
            if r and isinstance(r, dict):
                all_engines.extend(r.get("engines", []))
                for k in merged_counters:
                    merged_counters[k] += r.get("cluster_counters", {}).get(k, 0)
        metrics_text = generate_aggregated_prometheus_metrics(
            all_engines, merged_counters
        )
        return web.Response(
            text=metrics_text,
            headers={
                "Content-Type": "text/plain; version=0.0.4; charset=utf-8",
            },
        )

    async def _handle_engine_route(self, request: web.Request) -> web.Response:
        body = await request.json()
        engine_name = body.get("engine")
        if not engine_name:
            return web.json_response({"error": "missing 'engine' field"}, status=400)
        shard_url = self.engine_routes.get(engine_name)
        if not shard_url:
            return web.json_response(
                {"error": f"engine '{engine_name}' not found"}, status=404
            )
        try:
            async with self._session.post(
                f"{shard_url}{request.path}",
                json=body,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                data = await resp.json()
                return web.json_response(data, status=resp.status)
        except Exception as exc:
            return web.json_response({"error": f"upstream error: {exc}"}, status=502)

    async def aggregate_snapshot(self) -> dict:
        """Fetch and merge /snapshot from all shards (for snapshot_loop)."""
        results = await asyncio.gather(
            *(self._fetch_json(f"{u}/snapshot") for u in self.shard_urls)
        )
        all_engines: List[dict] = []
        for r in results:
            if r and isinstance(r, dict):
                all_engines.extend(r.get("engines", []))
        return {"engines": all_engines}


# ---------------------------------------------------------------------------
# Snapshot loop
# ---------------------------------------------------------------------------
async def snapshot_loop(proxy: ShardProxy, interval_s: float) -> None:
    if interval_s <= 0:
        return
    while True:
        await asyncio.sleep(interval_s)
        try:
            snapshot = await proxy.aggregate_snapshot()
        except Exception as exc:
            logger.warning("snapshot aggregation failed: %s", exc)
            continue
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
        total_running = sum(e["running"] for e in snapshot["engines"])
        total_accepted = sum(e["accepted"] for e in snapshot["engines"])
        total_completed = sum(e["completed"] for e in snapshot["engines"])
        total_cancelled = sum(e.get("cancelled_count", 0) for e in snapshot["engines"])
        print(
            f"[CLUSTER] engines={len(snapshot['engines'])} "
            f"total_running={total_running} total_accepted={total_accepted} "
            f"total_completed={total_completed} "
            f"total_cancelled={total_cancelled}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Subprocess cleanup
# ---------------------------------------------------------------------------
def terminate_shards(procs: List[subprocess.Popen]) -> None:
    """Send SIGTERM to all shards, then SIGKILL after 5s timeout."""
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)


def cleanup_partial_files(n_shards: int) -> None:
    for sid in range(n_shards):
        path = partial_file_path(sid)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    args = parse_args()

    # Start shard subprocesses.
    procs: List[subprocess.Popen] = []
    for sid in range(args.n_shards):
        cmd = build_shard_command(args, sid)
        logger.info("Starting shard %d: %s", sid, " ".join(cmd))
        proc = subprocess.Popen(cmd)
        procs.append(proc)

    try:
        # Wait for shards to become healthy.
        logger.info("Waiting for %d shards to become ready...", args.n_shards)
        ready = await wait_for_shards(
            args.host,
            args.base_grpc_port,
            args.n_shards,
            args.shard_ready_timeout,
            args.n_prefill,
            args.n_decode,
        )
        if not ready:
            raise RuntimeError("Shards did not become ready in time")
        logger.info("All shards ready")

        # Merge partial endpoint files.
        partials = await load_partial_endpoints(args.n_shards)
        engine_routes = merge_endpoints(args, partials)

        # Validate engine counts match expectations.
        all_prefill = [
            e for p in partials for e in p["engines"] if e["role"] == "prefill"
        ]
        all_decode = [
            e for p in partials for e in p["engines"] if e["role"] == "decode"
        ]
        if len(all_prefill) != args.n_prefill or len(all_decode) != args.n_decode:
            expected_pf = {f"prefill-{i}" for i in range(args.n_prefill)}
            actual_pf = {e["name"] for e in all_prefill}
            expected_dc = {f"decode-{i}" for i in range(args.n_decode)}
            actual_dc = {e["name"] for e in all_decode}
            missing_pf = sorted(expected_pf - actual_pf)
            missing_dc = sorted(expected_dc - actual_dc)
            logger.warning(
                "engine count mismatch! expected %d prefill + %d decode, "
                "got %d prefill + %d decode. missing prefill: %s, "
                "missing decode: %s",
                args.n_prefill,
                args.n_decode,
                len(all_prefill),
                len(all_decode),
                missing_pf if missing_pf else "none",
                missing_dc if missing_dc else "none",
            )

        shard_ports = [
            shard_http_port(args.base_grpc_port, sid, args.n_prefill, args.n_decode)
            for sid in range(args.n_shards)
        ]

        # Start HTTP proxy.
        proxy = ShardProxy(
            host=args.host,
            proxy_port=proxy_http_port(
                args.base_grpc_port, args.n_prefill, args.n_decode, args.n_shards
            ),
            shard_ports=shard_ports,
            engine_routes=engine_routes,
        )
        await proxy.start()

        p_port = proxy_http_port(
            args.base_grpc_port, args.n_prefill, args.n_decode, args.n_shards
        )
        print(
            f"shard launcher started: {args.n_shards} shards, "
            f"proxy on port {p_port}"
        )
        print(f"endpoint file: {args.endpoint_file}")
        print(f"flexlb env file: {args.env_file}")
        print("press Ctrl-C to stop")

        # Signal handling.
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)

        # Aggregated snapshot loop.
        snapshot_task = asyncio.create_task(
            snapshot_loop(proxy, args.snapshot_interval_s)
        )

        try:
            await stop_event.wait()
        finally:
            snapshot_task.cancel()
            await proxy.stop()

    finally:
        terminate_shards(procs)
        cleanup_partial_files(args.n_shards)


if __name__ == "__main__":
    asyncio.run(main())
