"""Remote media download observations reported as requests/s and bytes/s.

Set RTP_LLM_MEDIA_METRICS_HOSTS to at most 16 comma-separated host names
to split those final HTTP response hosts into individual series.
"""

import logging
import os
from threading import Lock
from urllib.parse import urlsplit


_HTTP_RESPONSES = "py_rtp_media_download_http_response_qps"
_PAYLOAD_BYTES = "py_rtp_media_download_payload_bytes_per_second"
_KNOWN_HOSTS = frozenset(
    host.strip().lower().rstrip(".")
    for host in os.getenv("RTP_LLM_MEDIA_METRICS_HOSTS", "").split(",")[:16]
)

_logger = logging.getLogger(__name__)
_lock = Lock()
_metrics = None


def use_worker_media_metrics() -> None:
    """Enable the existing reporter in a spawned preprocessing worker."""
    from rtp_llm.aios.kmonitor.python_client.kmonitor.report_worker import report_worker

    # This process imports the service reporter but never enters its serving state.
    report_worker.before_report = None


def _initialize_metrics():
    from rtp_llm.aios.kmonitor.python_client.kmonitor.kmonitor import KMonitor

    monitor = KMonitor()
    tags = {} if monitor.default_tags.get("cluster") else {"cluster": "unknown"}
    return {
        _HTTP_RESPONSES: monitor.register_acc_metric(
            _HTTP_RESPONSES, {**tags, "status": "unknown", "storage_domain": "unknown"}
        ),
        _PAYLOAD_BYTES: monitor.register_acc_metric(
            _PAYLOAD_BYTES,
            {**tags, "transport": "unknown", "storage_domain": "unknown"},
        ),
    }


def _get_metrics():
    global _metrics
    if _metrics is not None:
        return _metrics
    with _lock:
        if _metrics is None:
            try:
                _metrics = _initialize_metrics()
            except Exception:
                _logger.warning(
                    "Could not initialize media download metrics", exc_info=True
                )
                _metrics = {}
        return _metrics


def _record(name, domain, label, value) -> None:
    metric = _get_metrics().get(name)
    if metric is not None:
        label_name = "status" if name == _HTTP_RESPONSES else "transport"
        try:
            metric.report(value, {"storage_domain": domain, label_name: label})
        except Exception:
            # Monitoring must not change the result of a download.
            _logger.debug("Could not record media download metrics", exc_info=True)


def _storage_domain(url) -> str:
    if not isinstance(url, str):
        return "unknown"
    try:
        host = (urlsplit(url).hostname or "").rstrip(".")
    except ValueError:
        return "unknown"
    if not host:
        return "unknown"
    return host if host in _KNOWN_HOSTS else "other"


def record_http_response(response) -> None:
    """Count one final HTTP GET response, including unsuccessful statuses."""
    status_code = getattr(response, "status_code", None)
    status = "unknown"
    if isinstance(status_code, int) and 100 <= status_code <= 599:
        status = "503" if status_code == 503 else f"{status_code // 100}xx"
    _record(_HTTP_RESPONSES, _storage_domain(getattr(response, "url", None)), status, 1)


def record_http_payload(size: int, response) -> None:
    """Count bytes already consumed from a successful HTTP body."""
    _record(
        _PAYLOAD_BYTES, _storage_domain(getattr(response, "url", None)), "http", size
    )


def record_oss_payload(size: int) -> None:
    """Count bytes already consumed by the OSS SDK."""
    _record(_PAYLOAD_BYTES, "unknown", "oss", size)
