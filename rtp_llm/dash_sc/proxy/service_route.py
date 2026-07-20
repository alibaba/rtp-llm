"""Address discovery implementations for dash_sc proxy service routes."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from rtp_llm.config.py_config_modules import DASH_SC_GRPC_SERVER_PORT_OFFSET
from rtp_llm.dash_sc.proxy.service_route_config import (
    SERVICE_ROUTE_TYPE_IP_PORT_LIST,
    SERVICE_ROUTE_TYPE_VIPSERVER,
    load_service_route_config_from_env,
    parse_ip_port_list,
)

VipHostResolver = Callable[[str], Optional[Any]]
MonotonicClock = Callable[[], float]

_VIPSERVER_FAILURE_CACHE_TTL_S = 1.0
_VIPSERVER_PREWARM_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class BackendAddr:
    ip: str
    http_port: int
    grpc_port: int

    @classmethod
    def from_host(cls, host: Any) -> "BackendAddr":
        return cls.from_http_port(str(host.ip), host.port)

    @classmethod
    def from_http_target(cls, target: str) -> "BackendAddr":
        ip, sep, port = target.rpartition(":")
        if not sep or not ip or not port:
            raise RuntimeError(f"invalid ip:port address: {target!r}")
        return cls.from_http_port(ip, port)

    @classmethod
    def from_http_port(cls, ip: str, http_port: Any) -> "BackendAddr":
        try:
            http_port = int(http_port)
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"invalid http port for {ip}: {http_port!r}") from e
        grpc_port = http_port + DASH_SC_GRPC_SERVER_PORT_OFFSET
        if http_port <= 0 or http_port > 65535 or grpc_port > 65535:
            raise RuntimeError(
                f"invalid http/grpc port for {ip}: "
                f"http_port={http_port}, grpc_port={grpc_port}"
            )
        return cls(ip=ip, http_port=http_port, grpc_port=grpc_port)

    @property
    def http_target(self) -> str:
        return f"{self.ip}:{self.http_port}"

    @property
    def grpc_target(self) -> str:
        return f"{self.ip}:{self.grpc_port}"


class IpPortListServiceDiscovery:
    def __init__(self, address: str):
        addrs = parse_ip_port_list(address)
        if not addrs:
            raise RuntimeError(f"{SERVICE_ROUTE_TYPE_IP_PORT_LIST} address is empty")
        self._addrs = tuple(BackendAddr.from_http_target(addr) for addr in addrs)

    def resolve(self) -> Optional[BackendAddr]:
        return self._addrs[random.randrange(len(self._addrs))]

    async def resolve_async(self) -> Optional[BackendAddr]:
        return self.resolve()

    async def prewarm(self) -> None:
        return

    async def close(self) -> None:
        return


class VipServerServiceDiscovery:
    """Dynamic address discovery backed by ``rtp_llm.vipserver.vip_client``.

    ``cached_resolver``, when supplied, must be a strictly memory-only lookup;
    full/blocking resolution is always isolated in ``asyncio.to_thread``.
    """

    def __init__(
        self,
        domain: str,
        *,
        resolver: Optional[VipHostResolver] = None,
        cached_resolver: Optional[VipHostResolver] = None,
        failure_cache_ttl_s: float = _VIPSERVER_FAILURE_CACHE_TTL_S,
        prewarm_timeout_s: float = _VIPSERVER_PREWARM_TIMEOUT_S,
        clock: MonotonicClock = time.monotonic,
    ):
        self._domain = domain.strip()
        if not self._domain:
            raise RuntimeError(f"{SERVICE_ROUTE_TYPE_VIPSERVER} address is empty")
        if resolver is None:
            self._resolver = _resolve_vipserver_host
            self._cached_resolver = (
                cached_resolver
                if cached_resolver is not None
                else _resolve_vipserver_host_cached
            )
        else:
            self._resolver = resolver
            self._cached_resolver = cached_resolver
        self._failure_cache_ttl_s = max(0.0, float(failure_cache_ttl_s))
        self._prewarm_timeout_s = max(0.0, float(prewarm_timeout_s))
        self._clock = clock
        self._failure_cache_until = 0.0
        self._resolve_task: Optional[asyncio.Task[Optional[BackendAddr]]] = None
        self._has_successful_resolution = False
        self._closed = False

    def _failure_is_cached(self) -> bool:
        return self._clock() < self._failure_cache_until

    def _cache_failure(self) -> None:
        self._has_successful_resolution = False
        self._failure_cache_until = self._clock() + self._failure_cache_ttl_s

    def _resolve_with(self, resolver: VipHostResolver) -> Optional[BackendAddr]:
        try:
            host = resolver(self._domain)
            if host is None:
                self._cache_failure()
                logging.warning(
                    "[DashScGrpc] vipserver resolved no host: domain=%s",
                    self._domain,
                )
                return None
            addr = BackendAddr.from_host(host)
            self._failure_cache_until = 0.0
            self._has_successful_resolution = True
            return addr
        except Exception as e:
            self._cache_failure()
            logging.warning(
                "[DashScGrpc] vipserver resolve failed: domain=%s error=%s",
                self._domain,
                e,
                exc_info=True,
            )
            return None

    def resolve(self) -> Optional[BackendAddr]:
        return self._resolve_with(self._resolver)

    def _resolve_cached(self) -> Optional[BackendAddr]:
        """Read the warm snapshot without turning a miss into a failure TTL."""
        assert self._cached_resolver is not None
        try:
            host = self._cached_resolver(self._domain)
            if host is None:
                self._has_successful_resolution = False
                return None
            return BackendAddr.from_host(host)
        except Exception as e:
            self._has_successful_resolution = False
            logging.warning(
                "[DashScGrpc] vipserver cached resolve failed: domain=%s error=%s",
                self._domain,
                e,
                exc_info=True,
            )
            return None

    def _clear_resolve_task(self, task: asyncio.Task[Optional[BackendAddr]]) -> None:
        if self._resolve_task is task:
            self._resolve_task = None

    async def resolve_async(self) -> Optional[BackendAddr]:
        """Resolve without running the synchronous VIP client on the aio loop.

        Concurrent cold requests share one worker task. ``shield`` prevents one
        cancelled RPC from cancelling that shared lookup; the VIP HTTP client
        supplies the actual connect/read timeout that bounds the worker.
        """
        if self._closed or self._failure_is_cached():
            return None

        # The built-in cached resolver is a strict memory-only lookup: it keeps
        # the hot path on-loop and preserves per-request random host selection
        # without paying one executor hop per RPC. Injected blocking resolvers
        # still run in a worker as the defensive fallback.
        if self._has_successful_resolution:
            if self._cached_resolver is not None:
                addr = self._resolve_cached()
                if addr is not None:
                    return addr
                # The snapshot disappeared between background refreshes. Fall
                # through to the shared full lookup instead of running network
                # I/O on-loop or returning a spurious failure.
            else:
                return await asyncio.to_thread(self.resolve)

        task = self._resolve_task
        if task is None or task.done():
            task = asyncio.create_task(asyncio.to_thread(self.resolve))
            self._resolve_task = task
            task.add_done_callback(self._clear_resolve_task)
        return await asyncio.shield(task)

    async def prewarm(self) -> None:
        """Best-effort domain warm-up before the gRPC server accepts traffic."""
        try:
            await asyncio.wait_for(
                self.resolve_async(), timeout=self._prewarm_timeout_s
            )
        except asyncio.TimeoutError:
            # ``resolve_async`` shields the shared worker, so startup can
            # continue while the bounded HTTP lookup finishes in the thread.
            logging.warning(
                "[DashScGrpc] vipserver prewarm timed out: domain=%s timeout_s=%s",
                self._domain,
                self._prewarm_timeout_s,
            )

    async def close(self) -> None:
        """Stop new lookups and wait for the bounded in-flight worker, if any."""
        self._closed = True
        task = self._resolve_task
        if task is not None and not task.done():
            await asyncio.shield(task)


def create_service_discovery_from_env():
    cfg = load_service_route_config_from_env()
    if cfg.type == SERVICE_ROUTE_TYPE_IP_PORT_LIST:
        return IpPortListServiceDiscovery(cfg.address)
    if cfg.type == SERVICE_ROUTE_TYPE_VIPSERVER:
        return VipServerServiceDiscovery(cfg.address)
    raise RuntimeError(f"unsupported service route type: {cfg.type!r}")


def _resolve_vipserver_host(domain: str) -> Optional[Any]:
    from rtp_llm.vipserver.vip_client import get_one_validate_host

    return get_one_validate_host(domain)


def _resolve_vipserver_host_cached(domain: str) -> Optional[Any]:
    from rtp_llm.vipserver.vip_client import get_one_validate_host_cached

    return get_one_validate_host_cached(domain)
