"""Engine-owned native transfers. A single Python thread delivers completed bytes."""

import asyncio
import contextvars
import hashlib
import os
import threading
import time
from concurrent.futures import CancelledError, Future
from contextlib import contextmanager
from io import BytesIO

import requests

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.mm_download_domains import blocked_download_domains
from rtp_llm.multimodal.mm_error_messages import MMErr

current_network = contextvars.ContextVar("vit_media_network", default=None)


prepared_url_bytes = contextvars.ContextVar("vit_prepared_url_bytes", default=None)


def uses_url_loader(func):
    func.uses_url_loader = True
    return func


@contextmanager
def bind_url_sources(media, sources):
    token = prepared_url_bytes.set(
        {item.url: data for item, data in zip(media, sources) if data is not None}
    )
    try:
        yield
    finally:
        prepared_url_bytes.reset(token)


def accepts_source_bytes(func):
    func.accepts_source_bytes = True
    return func


class _TransferFuture(Future):
    def __init__(self, job):
        super().__init__()
        self.job = job

    def cancel(self):
        job = self.job
        cancelled = super().cancel()
        if cancelled and job is not None:
            job.cancel()
        return cancelled


class MediaNetwork:
    def __init__(self, config):
        from rtp_llm.cpp.media_network import _media_network as native

        self.native = native
        policy = native.Policy()
        policy.denied = blocked_download_domains()
        policy.ca = (
            os.getenv("REQUESTS_CA_BUNDLE")
            or os.getenv("CURL_CA_BUNDLE")
            or requests.certs.where()
        )
        self.proxy_for = None
        try:
            from internal_source.rtp_llm.utils import download_proxy, ssrf_check
        except ModuleNotFoundError as error:
            if not (
                error.name == "internal_source"
                or error.name.startswith("internal_source.")
            ):
                raise
        else:
            from rtp_llm.config.oss_download import get_oss_internal_region_allowlist

            policy.internal = True
            policy.set_networks(ssrf_check.INTRANET_CIDRS)
            policy.rewrite_oss = os.getenv("ENABLE_OSS_INTERNAL_REWRITE") != "0"
            policy.group_hosts = ssrf_check.OSS_PUBLIC_TO_GROUP_INTERNAL
            regions = get_oss_internal_region_allowlist()
            policy.region_list = regions is not None
            policy.regions = regions or set()
            self.region = download_proxy.current_download_region
            self.proxy_for = download_proxy.resolve_download_proxy
            self.proxy_headers = download_proxy.DOWNLOAD_HEADERS
            self.report_proxy = download_proxy._report_proxy_metric
        self.config = config
        workers = (
            config.vit_source_download_concurrency
            or config.mm_preprocess_max_workers
            or 1
        )
        ranges = config.mm_download_range_config
        if ranges["enable"]:
            policy.range_chunk_bytes = ranges["chunk_size_bytes"]
        self.pool = native.Network(workers, max(0, config.url_cache_item_num), policy)
        self.pending = {}
        self.lock = threading.Lock()
        self.thread = threading.Thread(
            target=self._deliver, name="vit-network-results", daemon=True
        )
        self.thread.start()

    def submit(self, req):
        with self.lock:
            job = self.pool.submit(req)
            future = _TransferFuture(job)
            self.pending[job] = future
        return future

    def _deliver(self):
        while (job := self.pool.next()) is not None:
            with self.lock:
                future = self.pending.pop(job)
            future.job = None
            if future.set_running_or_notify_cancel():
                try:
                    result = job.result()
                except Exception as error:
                    future.set_exception(error)
                else:
                    future.set_result(result)
                    del result
            # next() can block for the engine's lifetime after the last image.
            del future, job

    def close(self):
        self.pool.close()
        self.thread.join()

    def image(self, url, headers_text, max_bytes, deadline, region, *, v41=True):
        from rtp_llm.multimodal.multimodal_util import _get_http_heads
        from rtp_llm.multimodal.vit_metrics import VIT_PHASE_METRICS

        req = self.native.Request()
        if not url.isascii():
            prepared = requests.PreparedRequest()
            prepared.prepare_url(url, None)
            url = prepared.url
        req.url = url
        req.headers = _get_http_heads(headers_text)
        req.limit = max_bytes
        remaining = deadline - time.monotonic()
        req.budget_ms = max(0, int(1000 * (min(30.0, remaining) if v41 else remaining)))
        if v41:
            req.attempts = 3
        else:
            req.connect_ms = req.idle_ms = 10000
        req.cache_key = hashlib.sha256(
            headers_text.encode() + b"\0" + url.encode()
        ).hexdigest()
        if self.proxy_for is not None and (proxy := self.proxy_for(region)):
            req.proxy = proxy
            req.proxy_headers = self.proxy_headers
        via_proxy = bool(req.proxy)
        future = self.submit(req)
        if not via_proxy and not VIT_PHASE_METRICS:
            return future

        def report(done):
            if done.cancelled():
                return
            from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor

            result = done.result()
            status, error, body, hit, retries, counts = result[:6]
            joined = len(result) > 8 and result[8]
            proxy_attempts, fallbacks, connect_retries = counts
            if via_proxy and not hit and not joined:
                for _ in range(proxy_attempts):
                    self.report_proxy(AccMetrics.MM_DOWNLOAD_PROXY_QPS_METRIC, region)
                for _ in range(fallbacks):
                    self.report_proxy(
                        AccMetrics.MM_DOWNLOAD_PROXY_FALLBACK_QPS_METRIC, region
                    )
            if VIT_PHASE_METRICS:
                source = "cache" if hit else "load"
                if joined:
                    source = "shared"
                kmonitor.report(
                    AccMetrics.VIT_URL_SOURCE_QPS_METRIC, tags={"source": source}
                )
                if len(result) > 7:
                    kmonitor.report(
                        GaugeMetrics.VIT_HTTP_PHASE_RT_US_METRIC,
                        result[7] * 1000,
                        tags={"phase": "worker_queue", "source": source},
                    )
                if joined:
                    return
                for _ in range(connect_retries):
                    kmonitor.report(AccMetrics.VIT_HTTP_CONNECT_RETRY_QPS_METRIC)
                for _ in range(retries):
                    kmonitor.report(AccMetrics.VIT_HTTP_SOURCE_RETRY_QPS_METRIC)
                if not hit and status == 200 and not error:
                    kmonitor.report(GaugeMetrics.VIT_HTTP_BODY_BYTES_METRIC, len(body))

        future.add_done_callback(report)
        return future

    async def service_request(self, method, url, body, headers, timeout_s):
        req = self.native.Request()
        req.media = False
        req.method = method
        req.url = url
        req.headers = headers
        req.set_body(body)
        req.budget_ms = req.connect_ms = req.idle_ms = max(1, int(timeout_s * 1000))
        req.limit = 8 << 20
        status, error, data, *_ = await asyncio.wrap_future(self.submit(req))
        if error:
            raise OSError("Media service request failed: " + error)
        return status, data


def image_result(result, *, v41=True):
    status, error, data = result[:3]
    if error == "cancelled":
        raise CancelledError()
    if error == "too_large":
        raise FtRuntimeException(ExceptionType.INVALID_PARAMS, MMErr.FILE_TOO_LARGE)
    if error in ("blocked", "dns", "invalid_url", "invalid_header"):
        raise FtRuntimeException(ExceptionType.MM_WRONG_FORMAT_ERROR, MMErr.URL_INVALID)
    if error or status in (408, 429, 500, 502, 503, 504):
        raise FtRuntimeException(
            ExceptionType.MM_DOWNLOAD_FAILED,
            f"Temporary multimodal download failure: {error or status}",
        )
    if status != 200:
        raise FtRuntimeException(
            ExceptionType.INVALID_PARAMS if v41 else ExceptionType.MM_DOWNLOAD_FAILED,
            f"{MMErr.DL_FAILED}: HTTP {status}",
        )
    return data


def read_native_source(
    network, url, headers, limit, deadline, cancellation_event, *, v41
):
    from concurrent.futures import TimeoutError as FutureTimeout

    from rtp_llm.multimodal.multimodal_util import _check_download
    from rtp_llm.telemetry import vit_media

    deadline = deadline if deadline is not None else time.monotonic() + 30
    region = network.region.get() if network.proxy_for is not None else None
    future = network.image(url, headers, limit, deadline, region, v41=v41)
    try:
        while True:
            _check_download(deadline, cancellation_event)
            try:
                result = future.result(
                    timeout=min(0.1, max(0, deadline - time.monotonic()))
                )
                break
            except FutureTimeout:
                continue
        if result[3]:
            vit_media.download_cache_hit()
        return BytesIO(image_result(result, v41=v41))
    finally:
        future.cancel()


class SourceBatch:
    def __init__(self, net, item, *, v41=True):
        from rtp_llm.utils.base_model_datatypes import MMUrlType

        self.v41 = v41
        self.started = time.monotonic()
        self.finished = self.started
        self.network_ms = 0.0
        self.fetch_times_us = ()
        self.jobs = []
        deadline = self.started + item.remaining_timeout_s()
        try:
            for index, media in enumerate(item.mm_inputs):
                if media.url.startswith(("http://", "https://")):
                    limit = (
                        net.config.mm_video_max_file_size_kb
                        if media.mm_type == MMUrlType.VIDEO
                        else net.config.mm_image_max_file_size_kb
                    )
                    future = net.image(
                        media.url,
                        net.config.download_headers,
                        max(0, limit or 0) * 1024,
                        deadline,
                        item.region,
                        v41=v41,
                    )
                    self.jobs.append(future)
                    if index < len(getattr(item, "media_traces", ())):
                        trace = item.media_traces[index]
                        if trace is not None:
                            trace.begin("download")
                            future.add_done_callback(
                                lambda done, media_trace=trace: SourceBatch._finish_trace(
                                    done, media_trace, v41
                                )
                            )
                    ready = item.preprocess_ready_event
                    future.add_done_callback(lambda _, event=ready: event.set())
                else:
                    self.jobs.append(None)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _finish_trace(done, trace, v41):
        try:
            result = done.result()
            image_result(result, v41=v41)

        except BaseException as error:
            trace.finish_download(error)
        else:
            trace.finish_download(cached=result[3])

    def ready(self):
        for job in self.jobs:
            if job is not None and job.done():
                image_result(job.result(), v41=self.v41)
        return all(job is None or job.done() for job in self.jobs)

    def result(self):
        self.finished = time.monotonic()
        completed = [job.result() if job is not None else None for job in self.jobs]
        # The original native result ends at worker time (index 6). Newer
        # results also carry queue time at index 7, both in milliseconds.
        durations_ms = [sum(result[6:8]) if result else 0.0 for result in completed]
        self.fetch_times_us = tuple(ms * 1000 for ms in durations_ms)
        self.network_ms = max(
            (
                ms
                for ms, result in zip(durations_ms, completed)
                if result and not result[3]
            ),
            default=0.0,
        )
        return tuple(image_result(r, v41=self.v41) if r else None for r in completed)

    @property
    def elapsed_ms(self):
        return (self.finished - self.started) * 1000

    def close(self):
        for job in self.jobs:
            if job is not None:
                job.cancel()
