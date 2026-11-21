"""Exact-host denylist for multimodal HTTP downloads."""

import json
import os
import re
import threading
from collections.abc import Iterable
from functools import lru_cache

import requests
from requests.utils import unquote_unreserved
from urllib3.util import parse_url

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException

_ENV = "MM_DOWNLOAD_DOMAIN_DENYLIST"
_REUSE_CONNECTIONS_ENV = "MM_VIT_HTTP_REUSE_CONNECTIONS"
_download_session = threading.local()
_STATELESS_HEADERS = frozenset({"accept", "accept-encoding", "user-agent"})
# Stop at the end of the authority, even for long signed URLs. Requests treats
# a backslash as a path separator when preparing an HTTP URL.
_AUTHORITY = re.compile(r"^https?://([^/?#\\]*)", re.IGNORECASE | re.ASCII)


def _hostname(host: str) -> str:
    # urllib3 has already handled IDNA. Requests also decodes unreserved escapes.
    return unquote_unreserved(host).lower().rstrip(".")


@lru_cache(maxsize=1)
def _parse_domains(raw: str) -> frozenset[str]:
    try:
        entries = json.loads(raw)
        if not isinstance(entries, list):
            raise ValueError
        hosts = set()
        for entry in entries:
            if not isinstance(entry, str):
                raise ValueError
            entry = entry.strip()
            parsed = parse_url(entry if "://" in entry else "http://" + entry)
            if (
                parsed.scheme not in ("http", "https")
                or not parsed.host
                or parsed.auth is not None
                or parsed.path not in (None, "", "/")
                or parsed.query is not None
                or parsed.fragment is not None
                or any(c.isspace() or c in r"\*%" for c in entry)
            ):
                raise ValueError
            host = _hostname(parsed.host)
            if not host:
                raise ValueError
            hosts.add(host)
        return frozenset(hosts)
    except ValueError as error:
        raise ValueError(
            f"{_ENV} expects a JSON list of hostnames or HTTP(S) origins, "
            'e.g. ["https://web.archive.org/"]'
        ) from error


def blocked_download_domains() -> frozenset[str]:
    return _parse_domains(os.environ.get(_ENV) or "[]")


def _check_url(url: str, hosts: frozenset[str]) -> None:
    match = _AUTHORITY.match(url.lstrip())
    if match is None:
        return
    try:
        host = _hostname(parse_url("http://" + match[1]).host or "")
    except ValueError as error:
        raise FtRuntimeException(
            ExceptionType.MM_WRONG_FORMAT_ERROR, "Invalid multimodal download URL"
        ) from error
    if host in hosts:
        raise FtRuntimeException(
            ExceptionType.MM_WRONG_FORMAT_ERROR, "Multimodal download domain is blocked"
        )


def check_download_urls(urls: Iterable[str]) -> None:
    hosts = blocked_download_domains()
    if hosts:
        for url in urls:
            _check_url(url, hosts)


class _DomainCheckedSession(requests.Session):
    def __init__(self, hosts: frozenset[str]):
        super().__init__()
        self._blocked_hosts = hosts

    def send(self, request, **kwargs):
        # Requests sends each resolved redirect through this same session.
        if self._blocked_hosts:
            _check_url(request.url, self._blocked_hosts)
        return super().send(request, **kwargs)


class _AnonymousDownloadAdapter(requests.adapters.HTTPAdapter):
    def send(self, request, **kwargs):
        proxy = requests.utils.select_proxy(request.url, kwargs.get("proxies"))
        if (
            any(
                name in request.headers
                for name in ("Authorization", "Proxy-Authorization", "Cookie")
            )
            or (proxy and "@" in proxy)
            or kwargs.get("verify") is not True
            or kwargs.get("cert") is not None
        ):
            # Redirects and .netrc can add credentials after request preparation.
            adapter = requests.adapters.HTTPAdapter()
            try:
                return adapter.send(request, **kwargs)
            finally:
                adapter.close()
        return super().send(request, **kwargs)


def _pooled_download(url, hosts, options):
    pid = os.getpid()
    entry = getattr(_download_session, "entry", None)
    # Preprocess workers must never use sockets inherited from a parent.
    if entry is None or entry[0] != pid or entry[1]._blocked_hosts != hosts:
        if entry is not None:
            entry[1].close()
        session = _DomainCheckedSession(hosts)
        session.mount("https://", _AnonymousDownloadAdapter())
        session.mount("http://", _AnonymousDownloadAdapter())
        _download_session.entry = (pid, session)
    else:
        session = entry[1]
    try:
        return session.get(url, **options)
    finally:
        # Cookies are available during redirects within this logical download.
        session.cookies.clear()


def domain_checked_get(url, **kwargs):
    hosts = blocked_download_domains()
    if (
        os.environ.get(_REUSE_CONNECTIONS_ENV) == "1"
        and kwargs.get("stream") is True
        and kwargs.keys() <= {"stream", "headers", "timeout", "proxies"}
        and all(
            isinstance(name, str) and name.lower() in _STATELESS_HEADERS
            for name in kwargs.get("headers") or ()
        )
    ):
        return _pooled_download(url, hosts, kwargs)
    if not hosts:
        return requests.get(url, **kwargs)
    with _DomainCheckedSession(hosts) as session:
        return session.get(url, **kwargs)
