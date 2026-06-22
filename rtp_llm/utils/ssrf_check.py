import ipaddress
import logging
import socket
from typing import Dict
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


def _is_private_host(hostname: str) -> bool:
    """Return True if *hostname* resolves to a private/loopback/reserved/link-local address."""
    if not hostname:
        return True
    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        # Could not resolve: conservative behaviour is to block.
        return True
    for info in addr_info:
        ip_str = info[4][0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_reserved
            or ip_obj.is_link_local
        ):
            return True
    return False


def safe_request_get(url: str, headers: Dict[str, str], timeout: int = 10):
    """Fetch *url* with SSRF protection.

    Only http/https schemes are allowed and the resolved host must not be a
    private/internal address.  The function is intentionally restrictive and can
    be expanded later (e.g. explicit allow-lists) without breaking callers.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL scheme {parsed.scheme!r} is not allowed for safe download")
    if _is_private_host(parsed.hostname or ""):
        raise ValueError(
            f"URL host {parsed.hostname!r} resolves to a private/internal address "
            f"and is not allowed for safe download"
        )
    return requests.get(url, stream=True, headers=headers, timeout=timeout)
