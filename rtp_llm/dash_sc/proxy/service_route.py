"""Address discovery implementations for dash_sc proxy service routes."""

from __future__ import annotations

import logging
import random
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

    def matches_any(self, targets: set[str]) -> bool:
        return self.http_target in targets or self.grpc_target in targets


class IpPortListServiceDiscovery:
    def __init__(self, address: str):
        addrs = parse_ip_port_list(address)
        if not addrs:
            raise RuntimeError(f"{SERVICE_ROUTE_TYPE_IP_PORT_LIST} address is empty")
        self._addrs = tuple(BackendAddr.from_http_target(addr) for addr in addrs)

    def resolve(
        self, excluded_targets: Optional[set[str]] = None
    ) -> Optional[BackendAddr]:
        excluded_targets = excluded_targets or set()
        candidates = [
            addr for addr in self._addrs if not addr.matches_any(excluded_targets)
        ]
        if not candidates:
            return None
        return candidates[random.randrange(len(candidates))]


class VipServerServiceDiscovery:
    """Dynamic address discovery backed by ``rtp_llm.vipserver.vip_client``."""

    def __init__(
        self,
        domain: str,
        *,
        resolver: Optional[VipHostResolver] = None,
    ):
        self._domain = domain.strip()
        if not self._domain:
            raise RuntimeError(f"{SERVICE_ROUTE_TYPE_VIPSERVER} address is empty")
        self._resolver = resolver or _resolve_vipserver_host

    def resolve(
        self, excluded_targets: Optional[set[str]] = None
    ) -> Optional[BackendAddr]:
        excluded_targets = excluded_targets or set()
        try:
            hosts = self._resolver(self._domain)
            if hosts is None:
                logging.warning(
                    "[DashScGrpc] vipserver resolved no host: domain=%s",
                    self._domain,
                )
                return None
            if not isinstance(hosts, (list, tuple)):
                hosts = [hosts]
            candidates = []
            for host in hosts:
                addr = BackendAddr.from_host(host)
                if not addr.matches_any(excluded_targets):
                    candidates.append(addr)
            if not candidates:
                logging.warning(
                    "[DashScGrpc] vipserver resolved only excluded hosts: "
                    "domain=%s excluded=%s",
                    self._domain,
                    sorted(excluded_targets),
                )
                return None
            return candidates[random.randrange(len(candidates))]
        except Exception as e:
            logging.warning(
                "[DashScGrpc] vipserver resolve failed: domain=%s error=%s",
                self._domain,
                e,
                exc_info=True,
            )
            return None


def create_service_discovery_from_env():
    cfg = load_service_route_config_from_env()
    if cfg.type == SERVICE_ROUTE_TYPE_IP_PORT_LIST:
        return IpPortListServiceDiscovery(cfg.address)
    if cfg.type == SERVICE_ROUTE_TYPE_VIPSERVER:
        return VipServerServiceDiscovery(cfg.address)
    raise RuntimeError(f"unsupported service route type: {cfg.type!r}")


def _resolve_vipserver_host(domain: str) -> Optional[Any]:
    from rtp_llm.vipserver.vip_client import get_host_list_by_domain

    return get_host_list_by_domain(domain)
