import ipaddress
import logging
import socket
from typing import Dict
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

_MAX_REDIRECTS = 5


def _is_private_ip(ip_str: str) -> bool:
    """Return True if *ip_str* is a private/loopback/reserved/link-local address."""
    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return (
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_reserved
        or ip_obj.is_link_local
    )


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
        ip_str: str = info[4][0]
        if _is_private_ip(ip_str):
            return True
    return False


def _resolve_and_validate_host(hostname: str) -> str:
    """Resolve *hostname* to a validated IP and return it.

    If *hostname* is already an IP, validate it directly.  Raises ValueError if
    the host resolves to a private/internal address or cannot be resolved.
    """
    if not hostname:
        raise ValueError("URL host is empty")

    # If hostname is already an IP, validate it directly.
    try:
        ip_obj = ipaddress.ip_address(hostname)
        if _is_private_ip(hostname):
            raise ValueError(
                f"URL host {hostname!r} is a private/internal address "
                f"and is not allowed for safe download"
            )
        return str(ip_obj)
    except ValueError:
        pass

    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        raise ValueError(f"URL host {hostname!r} could not be resolved: {e}")

    for info in addr_info:
        ip_str: str = info[4][0]
        if not _is_private_ip(ip_str):
            return ip_str

    raise ValueError(
        f"URL host {hostname!r} resolves to a private/internal address "
        f"and is not allowed for safe download"
    )


def _validate_url(url: str) -> str:
    """Validate scheme and host of *url*.  Returns the URL if safe, else raises."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL scheme {parsed.scheme!r} is not allowed for safe download")
    if _is_private_host(parsed.hostname or ""):
        raise ValueError(
            f"URL host {parsed.hostname!r} resolves to a private/internal address "
            f"and is not allowed for safe download"
        )
    return url


class _SSRFAdapter(HTTPAdapter):
    """requests adapter that pins connections to a validated IP address.

    This closes the DNS-rebinding window: the hostname is resolved and its IP
    is validated before the connection is made, and the actual TCP connection
    uses that IP while the HTTP Host header / HTTPS SNI stay with the original
    host.
    """

    def get_connection_with_tls_context(
        self,
        request: requests.PreparedRequest,
        verify: object,
        proxies: object = None,
        cert: object = None,
    ):
        conn = super().get_connection_with_tls_context(request, verify, proxies, cert)
        # request.url has been rewritten to the validated IP by send(); keep
        # SNI / certificate hostname pinned to the original host name.
        original_host = getattr(request, "_ssrf_original_host", None)
        if original_host and urlparse(request.url).scheme == "https":
            conn.server_hostname = original_host
        return conn

    def send(
        self,
        request: requests.PreparedRequest,
        stream: bool = False,
        timeout: object = None,
        verify: object = True,
        cert: object = None,
        proxies: object = None,
    ):
        parsed = urlparse(request.url)
        original_host = parsed.hostname
        validated_ip = _resolve_and_validate_host(original_host or "")
        port = parsed.port
        if port is not None:
            new_netloc = f"[{validated_ip}]:{port}" if ":" in validated_ip else f"{validated_ip}:{port}"
        else:
            new_netloc = f"[{validated_ip}]" if ":" in validated_ip else validated_ip

        original_url = request.url
        request.url = parsed._replace(netloc=new_netloc).geturl()
        request.headers["Host"] = original_host
        request._ssrf_original_host = original_host
        try:
            return super().send(request, stream, timeout, verify, cert, proxies)
        finally:
            request.url = original_url


def safe_request_get(url: str, headers: Dict[str, str], timeout: int = 10):
    """Fetch *url* with SSRF protection.

    Only http/https schemes are allowed and the resolved host must not be a
    private/internal address.  Redirects are followed manually so that every
    intermediate Location is re-validated before the request is made, preventing
    SSRF via open-redirect or 3xx to internal hosts.  Relative Location headers
    are resolved against the previous request URL.

    Connections are pinned to the validated IP to close the DNS-rebinding window
    between URL validation and the actual TCP connect.
    """
    session = requests.Session()
    session.mount("http://", _SSRFAdapter())
    session.mount("https://", _SSRFAdapter())

    current_url = _validate_url(url)
    for _ in range(_MAX_REDIRECTS):
        response = session.get(
            current_url,
            stream=True,
            headers=headers,
            timeout=timeout,
            allow_redirects=False,
        )
        if response.is_redirect:
            location = response.headers.get("Location", "")
            if not location:
                break
            next_url = urljoin(current_url, location)
            current_url = _validate_url(next_url)
            continue
        return response
    raise ValueError(f"Exceeded maximum redirects ({_MAX_REDIRECTS}) for {url}")
