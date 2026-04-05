"""Resolve and format REAPI (CAS / executor) gRPC targets for logging and errors."""
from __future__ import annotations

import logging
import re
import socket
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)


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
    """Best-effort IPv4 resolution for a gRPC target (vipserver / DNS / literal IP)."""
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
