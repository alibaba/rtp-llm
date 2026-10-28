"""Resolve and format REAPI (CAS / executor) gRPC targets for logging and errors."""

from __future__ import annotations

import logging
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)
_FORCED_RESOLVE_SAMPLES = 3
_FORCED_RESOLVE_SLEEP_SECONDS = 0.05


def _split_host_port(grpc_uri: str) -> Tuple[str, int]:
    s = (grpc_uri or "").replace("grpc://", "").strip()
    if not s:
        return "unknown", 0
    if ":" in s:
        host, _, port_s = s.rpartition(":")
        try:
            return host, int(port_s)
        except ValueError:
            return s, 0
    return s, 0


def resolve_ipv4_addresses(host: str, port: int) -> List[str]:
    """Best-effort IPv4 resolution for a gRPC target (service discovery / DNS / literal IP)."""
    if not host or port <= 0:
        return []
    out: List[str] = []
    try:
        infos = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
    except OSError as exc:
        log.debug("getaddrinfo(%s:%s) failed: %s", host, port, exc)
        return []
    for _fam, _type, _proto, _canon, sockaddr in infos:
        ip = sockaddr[0]
        if ip not in out:
            out.append(ip)
    return out


def _is_ipv4_literal(host: str) -> bool:
    try:
        socket.inet_aton(host)
    except OSError:
        return False
    return host.count(".") == 3


@dataclass
class EndpointSpec:
    """A grpc://host:port endpoint that can be resolved by the remote framework."""

    uri: str
    host: str = field(init=False)
    port: int = field(init=False)
    is_literal_ip: bool = field(init=False)

    def __post_init__(self) -> None:
        self.host, self.port = _split_host_port(self.uri)
        self.is_literal_ip = _is_ipv4_literal(self.host)

    def uri_for_host(self, host: str) -> str:
        return f"grpc://{host}:{self.port}"


class ExecutorEndpointPool:
    """Resolve and rotate executor IPs inside pytest remote.

    Outer scripts pass hostnames. This pool owns service-discovery refresh and
    provides concrete grpc://ip:port endpoints to RemoteExecutor attempts.
    """

    def __init__(
        self,
        uri: str,
        *,
        fallback_uri: Optional[str] = None,
        refresh_seconds: int = 60,
    ):
        self.spec = EndpointSpec(uri)
        self.fallback_spec = EndpointSpec(fallback_uri) if fallback_uri else None
        self.refresh_seconds = max(0, int(refresh_seconds))
        self._lock = threading.RLock()
        self._hosts: List[str] = []
        self._active_spec = self.spec
        self._index = 0
        self._last_refresh = 0.0
        self.refresh(force=True)

    @property
    def source_uri(self) -> str:
        return self.spec.uri

    @property
    def active_source_uri(self) -> str:
        return self._active_spec.uri

    def refresh(self, *, force: bool = False) -> None:
        with self._lock:
            now = time.time()
            if (
                not force
                and self._hosts
                and self.refresh_seconds > 0
                and now - self._last_refresh < self.refresh_seconds
            ):
                return

            active_spec = self.spec
            if self.spec.is_literal_ip:
                resolved = [self.spec.host]
            else:
                resolved: List[str] = []
                sample_count = _FORCED_RESOLVE_SAMPLES if force else 1
                for sample_idx in range(sample_count):
                    for ip in resolve_ipv4_addresses(self.spec.host, self.spec.port):
                        if ip not in resolved:
                            resolved.append(ip)
                    if len(resolved) > 1 or sample_idx == sample_count - 1:
                        break
                    time.sleep(_FORCED_RESOLVE_SLEEP_SECONDS)
                if not resolved and self.fallback_spec is not None:
                    active_spec = self.fallback_spec
                    if active_spec.is_literal_ip:
                        resolved = [active_spec.host]
                    else:
                        for sample_idx in range(sample_count):
                            for ip in resolve_ipv4_addresses(active_spec.host, active_spec.port):
                                if ip not in resolved:
                                    resolved.append(ip)
                            if len(resolved) > 1 or sample_idx == sample_count - 1:
                                break
                            time.sleep(_FORCED_RESOLVE_SLEEP_SECONDS)
                    log.warning(
                        "[EXECUTOR_POOL] primary host %s:%d unresolved; "
                        "falling back to %s:%d",
                        self.spec.host,
                        self.spec.port,
                        active_spec.host,
                        active_spec.port,
                    )
                elif resolved and self.fallback_spec is not None:
                    fallback_hosts: List[str] = []
                    if self.fallback_spec.is_literal_ip:
                        fallback_hosts = [self.fallback_spec.host]
                    elif self.fallback_spec.port == active_spec.port:
                        for sample_idx in range(sample_count):
                            for ip in resolve_ipv4_addresses(
                                self.fallback_spec.host, self.fallback_spec.port
                            ):
                                if ip not in fallback_hosts:
                                    fallback_hosts.append(ip)
                            if len(fallback_hosts) > 1 or sample_idx == sample_count - 1:
                                break
                            time.sleep(_FORCED_RESOLVE_SLEEP_SECONDS)
                    for ip in fallback_hosts:
                        if ip not in resolved:
                            resolved.append(ip)
                if not resolved:
                    # Let gRPC attempt hostname resolution so local/dev setups still
                    # get a useful error when service discovery is unavailable.
                    resolved = [active_spec.host]

            if force and resolved and active_spec == self._active_spec:
                for host in self._hosts:
                    if host not in resolved:
                        resolved.append(host)

            if resolved != self._hosts or active_spec != self._active_spec:
                previous = self.current_endpoint() if self._hosts else None
                self._hosts = resolved
                self._active_spec = active_spec
                self._index = 0
                if previous:
                    for i, host in enumerate(self._hosts):
                        if self._active_spec.uri_for_host(host) == previous:
                            self._index = i
                            break
            self._last_refresh = now
            log.info(
                "[EXECUTOR_POOL] active_host=%s port=%d endpoints=[%s] source=%s primary=%s",
                self._active_spec.host,
                self._active_spec.port,
                ",".join(self._active_spec.uri_for_host(h) for h in self._hosts),
                self.active_source_uri,
                self.source_uri,
            )

    def current_endpoint(self) -> str:
        with self._lock:
            if not self._hosts:
                self.refresh(force=True)
            return self._active_spec.uri_for_host(self._hosts[self._index])

    def advance(self, *, refresh: bool = False) -> str:
        with self._lock:
            self.refresh(force=refresh)
            if len(self._hosts) > 1:
                self._index = (self._index + 1) % len(self._hosts)
            return self.current_endpoint()

    def endpoints(self) -> List[str]:
        self.refresh()
        return [self._active_spec.uri_for_host(h) for h in self._hosts]


def describe_reapi_endpoint(label: str, grpc_uri: str) -> str:
    """One-line description: label, host:port, and resolved IPv4 list."""
    host, port = _split_host_port(grpc_uri)
    if port <= 0:
        return f"{label}={grpc_uri!r}"
    ips = resolve_ipv4_addresses(host, port)
    ip_part = ",".join(ips) if ips else "unresolved"
    return f"{label}={host}:{port} ipv4=[{ip_part}]"


def combine_reapi_endpoints(cas_uri: str, executor_uri: str) -> str:
    return f"{describe_reapi_endpoint('cas', cas_uri)} | {describe_reapi_endpoint('executor', executor_uri)}"


_REMOTE_WORKER_IP_RE = re.compile(r"^>>>RTP_REMOTE_HOST_IP\s+(\S+)\s*$", re.MULTILINE)


def extract_remote_worker_ip(text: str) -> Optional[str]:
    """Parse first >>>RTP_REMOTE_HOST_IP line emitted by the remote worker shell."""
    if not text:
        return None
    m = _REMOTE_WORKER_IP_RE.search(text)
    return m.group(1) if m else None
