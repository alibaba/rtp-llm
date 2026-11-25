"""Opt-in native HTTP transport for ViT preprocessors that use the URL loader."""

import hashlib
import json
import os
import threading
import time
from concurrent.futures import CancelledError, Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from contextvars import ContextVar
from importlib import import_module
from io import BytesIO
from urllib.parse import urlsplit
from urllib.request import getproxies

import requests

from rtp_llm.config.exceptions import ExceptionType
from rtp_llm.config.mm_download_domains import blocked_download_domains
from rtp_llm.multimodal.mm_error_messages import MMErr, raise_mm
from rtp_llm.multimodal.multimodal_util import (
    _check_download,
    _get_http_heads,
    _record_download_time,
    _validate_file_size,
    get_bytes_io_from_url,
)


_active_network = ContextVar("vit_native_media_network", default=None)


def uses_url_loader(preprocess_func):
    preprocess_func.uses_url_loader = True
    return preprocess_func


class UnsupportedMediaNetworkPolicy(RuntimeError):
    pass


class MediaNetwork:
    def __init__(self, config):
        from rtp_llm.cpp.media_network import _media_network as native

        self._native = native
        policy = native.Policy()
        policy.denied = blocked_download_domains()
        policy.ca = (
            os.getenv("REQUESTS_CA_BUNDLE")
            or os.getenv("CURL_CA_BUNDLE")
            or requests.certs.where()
        )
        if os.path.isdir(policy.ca):
            raise UnsupportedMediaNetworkPolicy(
                "the configured CA path is a directory"
            )
        self._download_proxy = self._configure_internal_policy(policy)
        self._pool = native.Network(
            max(1, config.mm_preprocess_max_workers),
            max(0, config.url_cache_item_num),
            policy,
        )
        self._lock = threading.Lock()
        self._closed = False
        self._pending = {}
        self._results_thread = threading.Thread(
            target=self._deliver, name="vit-network-results", daemon=True
        )
        try:
            self._results_thread.start()
        except BaseException:
            self._pool.close()
            raise

    @staticmethod
    def _configure_internal_policy(policy):
        ssrf_module = "internal_source.rtp_llm.utils.ssrf_check"
        try:
            ssrf_check = import_module(ssrf_module)
        except ModuleNotFoundError as error:
            if not error.name or not (
                error.name == ssrf_module or ssrf_module.startswith(error.name + ".")
            ):
                raise
            return None

        try:
            networks = ssrf_check.INTRANET_CIDRS
            download_proxy = import_module("internal_source.rtp_llm.utils.download_proxy")
        except (AttributeError, ModuleNotFoundError) as error:
            # An older internal loader still owns its URL policy. Do not switch
            # to a native path that cannot apply the same rules.
            raise UnsupportedMediaNetworkPolicy(
                "the installed internal downloader does not export its native policy"
            ) from error

        from rtp_llm.config.oss_download import get_oss_internal_region_allowlist

        policy.internal = True
        policy.set_networks(networks)
        policy.rewrite_oss = os.getenv("ENABLE_OSS_INTERNAL_REWRITE") != "0"
        policy.group_hosts = ssrf_check.OSS_PUBLIC_TO_GROUP_INTERNAL
        regions = get_oss_internal_region_allowlist()
        policy.region_list = regions is not None
        policy.regions = regions or set()
        return download_proxy

    @contextmanager
    def use(self, deadline):
        with self._lock:
            if self._closed:
                raise CancelledError("ViT media transport is closed")
        token = _active_network.set((self, deadline))
        try:
            yield
        finally:
            _active_network.reset(token)

    def _deliver(self):
        while (job := self._pool.next()) is not None:
            with self._lock:
                future = self._pending.pop(job)
            if future.set_running_or_notify_cancel():
                try:
                    result = job.result()
                except Exception as error:
                    future.set_exception(error)
                else:
                    future.set_result(result)
                    del result
            # Keep the last response from being held while waiting for new work.
            del future, job

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._pool.close()
        self._results_thread.join()

    def download(self, url, download_headers, max_file_size_kb, deadline):
        _check_download(deadline, None)
        request = self._native.Request()
        prepared = requests.PreparedRequest()
        prepared.prepare_url(url, None)
        url = prepared.url
        request.url = url
        headers = _get_http_heads(download_headers)
        request.headers = headers
        request.limit = max(0, max_file_size_kb or 0) * 1024
        request.connect_ms = request.idle_ms = 10_000
        request.budget_ms = max(1, int((deadline - time.monotonic()) * 1000))
        if self._download_proxy is not None:
            region = self._download_proxy.current_download_region.get()
            proxy = self._download_proxy.resolve_download_proxy(region)
            if proxy:
                request.proxy = proxy
                request.proxy_headers = self._download_proxy.DOWNLOAD_HEADERS
        request.cache_key = hashlib.sha256(
            json.dumps((url, headers, request.proxy), sort_keys=True).encode()
        ).hexdigest()

        with self._lock:
            if self._closed:
                raise CancelledError("ViT media transport is closed")
            job = self._pool.submit(request)
            future = Future()
            self._pending[job] = future

        try:
            return future.result(timeout=max(0, deadline - time.monotonic()))
        finally:
            if future.cancel():
                job.cancel()


def read_media_source(url, download_headers, max_file_size_kb):
    context = _active_network.get()
    if context is None or not url.startswith(("http://", "https://")):
        return get_bytes_io_from_url(url, download_headers, max_file_size_kb)

    network, deadline = context
    proxies = getproxies()
    try:
        authority = urlsplit(url).netloc
    except ValueError:
        raise_mm(MMErr.URL_INVALID, ExceptionType.MM_WRONG_FORMAT_ERROR)
    # Requests also reevaluates environment proxies after redirects. Keep that
    # behavior for environments that configure one, including NO_PROXY rules.
    if any(proxies.get(scheme) for scheme in ("http", "https", "all")) or (
        "@" in authority or authority.startswith("[")
    ):
        try:
            return get_bytes_io_from_url(
                url, download_headers, max_file_size_kb, deadline=deadline
            )
        except (TimeoutError, FutureTimeoutError):
            raise_mm(MMErr.DL_TIMEOUT, ExceptionType.MM_DOWNLOAD_FAILED)
    started = time.monotonic()
    cache_hit = False
    try:
        status, error, data, cache_hit, *_ = network.download(
            url, download_headers, max_file_size_kb, deadline
        )
        if error == "cancelled":
            raise CancelledError("ViT media download was cancelled")
        if error in ("timeout", "connect_timeout"):
            raise_mm(MMErr.DL_TIMEOUT, ExceptionType.MM_DOWNLOAD_FAILED)
        if error == "too_large":
            raise_mm(MMErr.FILE_TOO_LARGE)
        if error in ("blocked", "dns", "invalid_url", "invalid_header"):
            raise_mm(MMErr.URL_INVALID, ExceptionType.MM_WRONG_FORMAT_ERROR)
        if error or status != 200:
            raise_mm(MMErr.DL_FAILED, ExceptionType.MM_DOWNLOAD_FAILED)
        _check_download(deadline, None)
        _validate_file_size(len(data), max_file_size_kb)
        return BytesIO(data)
    except (TimeoutError, FutureTimeoutError):
        raise_mm(MMErr.DL_TIMEOUT, ExceptionType.MM_DOWNLOAD_FAILED)
    except (requests.exceptions.RequestException, ValueError):
        raise_mm(MMErr.URL_INVALID, ExceptionType.MM_WRONG_FORMAT_ERROR)
    finally:
        if not cache_hit:
            _record_download_time(started)
