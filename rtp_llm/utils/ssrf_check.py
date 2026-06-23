import ipaddress
import logging
import socket
from typing import Any, Dict
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
    # Only the ip_address() parse error is caught here; private-IP validation
    # must raise ValueError that propagates to the caller.
    try:
        ip_obj = ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        if _is_private_ip(hostname):
            raise ValueError(
                f"URL host {hostname!r} is a private/internal address "
                f"and is not allowed for safe download"
            )
        return str(ip_obj)

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
    """requests adapter that pins TCP connections to a validated IP address.

    This closes the DNS-rebinding window: the hostname is resolved and its IP
    is validated before the connection is made, and the actual TCP connection
    uses that IP while the HTTP Host header / HTTPS SNI / certificate
    verification stay with the original host.

    The URL is NOT rewritten — instead the connection pool's ``host`` is
    overridden after creation so that urllib3 connects to the validated IP
    while TLS uses the original hostname for SNI and ``assert_hostname``.
    """

    @staticmethod
    def _pin_connection(conn: Any, validated_ip: str, original_host: str, scheme: str):
        """Override the pool's TCP target to *validated_ip* while keeping TLS
        hostname set to *original_host*."""
        conn.host = validated_ip
        if scheme == "https":
            conn.server_hostname = original_host
            conn.assert_hostname = original_host

    def get_connection_with_tls_context(
        self,
        request: requests.PreparedRequest,
        verify: object,
        proxies: object = None,
        cert: object = None,
    ):
        conn = super().get_connection_with_tls_context(request, verify, proxies, cert)
        validated_ip = getattr(request, "_ssrf_validated_ip", None)
        original_host = getattr(request, "_ssrf_original_host", None)
        if validated_ip and original_host:
            scheme = urlparse(request.url).scheme
            self._pin_connection(conn, validated_ip, original_host, scheme)
        return conn

    def get_connection(self, url: str, proxies: object = None):
        """Fallback for older requests versions (pre-2.32)."""
        conn = super().get_connection(url, proxies)
        parsed = urlparse(url)
        original_host = parsed.hostname
        if original_host:
            try:
                validated_ip = _resolve_and_validate_host(original_host)
                self._pin_connection(conn, validated_ip, original_host, parsed.scheme)
            except ValueError:
                pass
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
        original_host = str(parsed.hostname or "")
        validated_ip = _resolve_and_validate_host(original_host)
        # Store for get_connection_with_tls_context / get_connection.
        # Do NOT rewrite request.url — keep the original hostname so that
        # TLS SNI and certificate verification work correctly.
        request._ssrf_validated_ip = validated_ip
        request._ssrf_original_host = original_host
        return super().send(request, stream, timeout, verify, cert, proxies)


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
            response.close()
            if not location:
                break
            next_url = urljoin(current_url, location)
            current_url = _validate_url(next_url)
            continue
        return response
    if 'response' in locals() and response is not None:
        response.close()
    raise ValueError(f"Exceeded maximum redirects ({_MAX_REDIRECTS}) for {url}")
