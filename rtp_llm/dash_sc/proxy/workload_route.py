"""Optional selection from backend-owned, expiring workload snapshots."""
from __future__ import annotations

import asyncio
import logging
import math
import os
import random
import time
from dataclasses import dataclass

from rtp_llm.dash_sc.proxy.service_route import BackendAddr


@dataclass(frozen=True)
class RouteLease:
    addr: BackendAddr
    token: object


class WorkloadRouter:
    # These are advisory dispatch penalties within one proxy process, not
    # distributed reservations. Backend KV admission remains authoritative.
    def __init__(self, *, fetch=None, clock=time.monotonic, ttl=1.0,
                 dispatch_grace=3.0, dispatch_positions=1048576):
        self._fetch = fetch or self._fetch_http
        self._clock, self._ttl = clock, ttl
        self._dispatch_grace = dispatch_grace
        self._dispatch_positions = dispatch_positions
        self._cache = {}
        self._pending = {}
        self._refresh_lock = asyncio.Lock()
        self._session = None

    async def _fetch_http(self, addr):
        import aiohttp

        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=0.15),
                connector=aiohttp.TCPConnector(limit=32), trust_env=False)
        async with self._session.get(f"http://{addr.http_target}/scheduling",
                                     allow_redirects=False) as response:
            if response.status != 200:
                return None
            # Bound metadata reads even if discovery points at a bad endpoint.
            data = bytearray()
            async for chunk in response.content.iter_chunked(4096):
                data.extend(chunk)
                if len(data) > 16384:
                    return None
            import json
            return json.loads(data)

    @staticmethod
    def _valid(value):
        keys = ("replicas", "age_ms", "ingress_requests", "remaining_prefill_positions",
                "waiting_prefill_positions", "unknown_demand_requests", "decode_requests",
                "kv_growth_pages", "kv_free_pages", "kv_used_pages", "waiting_admission")
        return (isinstance(value, dict) and value.get("schema_version") == 1 and
                value.get("ready") is True and all(
                    type(value.get(key)) is int and 0 <= value[key] < 2**53 for key in keys) and
                value["replicas"] > 0 and value["age_ms"] < 3000 and
                value["kv_free_pages"] + value["kv_used_pages"] > 0 and
                value["waiting_prefill_positions"] <= value["remaining_prefill_positions"])

    async def _read(self, addr):
        try:
            value = await asyncio.wait_for(self._fetch(addr), timeout=0.2)
            return value if self._valid(value) else None
        except Exception:
            return None

    async def acquire(self, discovery):
        # Discovery is cached by its provider. Only probe a bounded candidate
        # sample per request; the lock coalesces concurrent refreshes.
        candidates = list(dict.fromkeys(discovery.candidates()))
        if not candidates:
            return None
        if len(candidates) > 32:
            candidates = random.sample(candidates, 32)
        async with self._refresh_lock:
            now = self._clock()
            wanted = set(candidates)
            self._cache = {addr: entry for addr, entry in self._cache.items() if addr in wanted}
            refresh = [addr for addr in candidates
                       if addr not in self._cache or now - self._cache[addr][0] >= self._ttl]
            values = await asyncio.gather(*(self._read(addr) for addr in refresh))
            for addr, value in zip(refresh, values):
                self._cache[addr] = (now, value)
            now = self._clock()
            self._pending = {token: entry for token, entry in self._pending.items()
                             if entry[0] in wanted and now - entry[1] < self._dispatch_grace}
            scored = []
            for addr in candidates:
                stamp, value = self._cache[addr]
                if value is None or value["age_ms"] + (now - stamp) * 1000 >= 3000:
                    continue
                pending = sum(entry[0] == addr for entry in self._pending.values())
                remaining = (value["remaining_prefill_positions"] +
                             (value["unknown_demand_requests"] + pending) * self._dispatch_positions)
                work = remaining + 4096 * max(value["ingress_requests"], value["decode_requests"])
                total = value["kv_free_pages"] + value["kv_used_pages"]
                headroom = max(0.05, (value["kv_free_pages"] - value["kv_growth_pages"]) / total)
                score = work / (value["replicas"] * headroom)
                if math.isfinite(score):
                    scored.append((score, addr))
            if not scored:
                logging.warning("[DashScGrpc] no fresh workload route: candidates=%d", len(candidates))
                return None
            best = min(score for score, _ in scored)
            addr = random.choice([addr for score, addr in scored if score == best])
            token = object()
            self._pending[token] = (addr, now)
            logging.info("[DashScGrpc] workload route: backend=%s score=%.0f candidates=%d",
                         addr.grpc_target, best, len(scored))
            return RouteLease(addr, token)

    def release(self, lease):
        if lease is not None:
            self._pending.pop(lease.token, None)

    async def close(self):
        if self._session is not None:
            await self._session.close()
            self._session = None
        self._cache.clear()
        self._pending.clear()


def create_workload_router_from_env():
    mode = os.environ.get("DASH_SC_GRPC_PROXY_LOAD_BALANCE", "random").strip()
    if mode == "random":
        return None
    if mode == "workload":
        return WorkloadRouter()
    raise ValueError(f"unsupported DASH_SC_GRPC_PROXY_LOAD_BALANCE: {mode!r}")
