"""FlexLB stability monitor — concurrent polling of JVM, inflight, and mock-engine metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import signal
import sys
import time
from typing import Any, Dict

import aiohttp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flexlb-http-addr", default="127.0.0.1:7001")
    parser.add_argument("--management-port", type=int, default=7002)
    parser.add_argument("--mock-http-port", type=int, default=55150)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration-s", type=float, default=0.0)
    return parser.parse_args()


_HEAP_RE = re.compile(r'jvm_memory_used_bytes\{[^}]*area="heap"[^}]*\}\s+([0-9.eE+-]+)')
_GC_CNT_RE = re.compile(r"jvm_gc_pause_seconds_count(?:\{[^}]*\})?\s+([0-9.eE+-]+)")
_GC_SUM_RE = re.compile(r"jvm_gc_pause_seconds_sum(?:\{[^}]*\})?\s+([0-9.eE+-]+)")


def _err(exc: Exception) -> str:
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    if isinstance(exc, aiohttp.ContentTypeError):
        return "bad_content_type"
    if isinstance(exc, aiohttp.ClientConnectorError):
        return "connection_refused"
    if isinstance(exc, aiohttp.ClientResponseError):
        return f"http_{exc.status}"
    return type(exc).__name__


def _parse_prom(text: str) -> Dict[str, Any]:
    def _sum_vals(pattern: re.Pattern[str]) -> float:
        total = 0.0
        for m in pattern.finditer(text):
            try:
                total += float(m.group(1))
            except ValueError:
                pass
        return total

    heap = _sum_vals(_HEAP_RE)
    gc_sum = _sum_vals(_GC_SUM_RE)
    gc_cnt = int(_sum_vals(_GC_CNT_RE))
    return {
        "jvm_heap_used_mb": round(heap / (1024 * 1024), 1),
        "jvm_gc_pause_count": gc_cnt,
        "jvm_gc_pause_total_ms": round(gc_sum * 1000, 1),
    }


async def _probe_prom(session: aiohttp.ClientSession, base: str) -> str:
    for path in ("/prometheus", "/actuator/prometheus"):
        try:
            async with session.get(f"{base}{path}") as r:
                if r.status == 200:
                    print(f"[probe] prometheus path: {path}", file=sys.stderr)
                    return path
        except Exception:
            pass
    print("[probe] WARNING: prometheus not found, using /prometheus", file=sys.stderr)
    return "/prometheus"


async def _poll_jvm(
    session: aiohttp.ClientSession, base: str, prom_path: str
) -> Dict[str, Any]:
    try:
        async with session.get(f"{base}{prom_path}") as r:
            r.raise_for_status()
            return _parse_prom(await r.text())
    except Exception as exc:
        return {
            "jvm_heap_used_mb": None,
            "jvm_gc_pause_count": None,
            "jvm_gc_pause_total_ms": None,
            "jvm_error": _err(exc),
        }


async def _poll_inflight(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
    try:
        async with session.get(url) as r:
            r.raise_for_status()
            data = await r.json()
        return {
            "scheduler_inflight": data.get("scheduler_inflight", 0),
            "prefill_inflight": {
                ep.get("ip_port", "?"): ep.get("inflight_batches", 0)
                for ep in data.get("prefill_endpoints", [])
            },
            "decode_inflight": {
                ep.get("ip_port", "?"): ep.get("inflight_requests", 0)
                for ep in data.get("decode_endpoints", [])
            },
        }
    except Exception as exc:
        return {
            "scheduler_inflight": None,
            "prefill_inflight": None,
            "decode_inflight": None,
            "inflight_error": _err(exc),
        }


async def _poll_mock(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
    try:
        async with session.get(url) as r:
            r.raise_for_status()
            data = await r.json()
        return {
            "mock_engines": [
                {
                    "name": e.get("name", ""),
                    "running": e.get("running", 0),
                    "accepted": e.get("accepted", 0),
                    "completed": e.get("completed", 0),
                }
                for e in data.get("engines", [])
            ]
        }
    except Exception as exc:
        return {"mock_engines": None, "mock_error": _err(exc)}


async def main() -> None:
    args = parse_args()
    flexlb_host = args.flexlb_http_addr.rsplit(":", 1)[0]
    mgmt_base = f"http://{flexlb_host}:{args.management_port}"
    inflight_url = f"http://{args.flexlb_http_addr}/rtp_llm/inflight_status"
    mock_url = f"http://127.0.0.1:{args.mock_http_port}/snapshot"
    timeout = aiohttp.ClientTimeout(total=2.0)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    with open(args.output, "w", encoding="utf-8") as f:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            prom_path = await _probe_prom(session, mgmt_base)
            start = time.monotonic()
            while not stop.is_set():
                if args.duration_s > 0 and time.monotonic() - start >= args.duration_s:
                    break
                poll_start = time.monotonic()
                jvm, inflight, mock = await asyncio.gather(
                    _poll_jvm(session, mgmt_base, prom_path),
                    _poll_inflight(session, inflight_url),
                    _poll_mock(session, mock_url),
                )
                record: Dict[str, Any] = {
                    "ts": time.time(),
                    "elapsed_s": round(time.monotonic() - start, 1),
                    **jvm,
                    **inflight,
                    **mock,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                remaining = max(0.0, args.interval - (time.monotonic() - poll_start))
                if remaining > 0:
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        pass

    print(f"[monitor] wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
