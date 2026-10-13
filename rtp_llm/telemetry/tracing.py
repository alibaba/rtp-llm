"""Process-level OTel trace runtime for the RTP-LLM Python frontend.

Design constraints:
- Degradable import: opentelemetry packages are NOT in requirements/lock yet;
  when missing, every API here becomes a safe no-op and telemetry is DISABLED.
- 唯一启动入口为 RTP_LLM_TRACE_CONFIG；校验失败只关闭 Trace。
- BSP uses bounded conservative defaults, fail-open on queue full; export
  failure never blocks inference.
- Explicit W3C TraceContext-only global propagator; Baggage not forwarded.
- Request-scoped state travels in an explicit ContextVar (RequestTraceState),
  not via implicit current-span magic across layers.
"""

import ipaddress
import logging
import os
import socket
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from rtp_llm.telemetry import attributes as trace_attrs
from rtp_llm.telemetry.config import (
    TraceConfig,
    load_trace_config,
    package_scope_version,
)

_LOGGER = logging.getLogger(__name__)

# SDK symbols are loaded only after JSON has enabled tracing, inside the
# environment-isolation context below.
OTEL_AVAILABLE = False
_OTEL_IMPORT_ERROR: Optional[BaseException] = None


def _load_otel_sdk() -> None:
    global OTEL_AVAILABLE, _OTEL_IMPORT_ERROR
    global otel_baggage, otel_context, propagate, trace, W3CBaggagePropagator
    global Resource, SpanLimits, TracerProvider, BatchSpanProcessor
    global ParentBased, TraceIdRatioBased, TraceContextTextMapPropagator
    try:
        from opentelemetry import baggage as otel_baggage
        from opentelemetry import context as otel_context
        from opentelemetry import propagate, trace
        from opentelemetry.baggage.propagation import W3CBaggagePropagator
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import SpanLimits, TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )
    except Exception as error:  # SDK import must never block inference startup
        OTEL_AVAILABLE = False
        _OTEL_IMPORT_ERROR = error
        raise
    OTEL_AVAILABLE = True
    _OTEL_IMPORT_ERROR = None


class TelemetryState(Enum):
    UNINITIALIZED = 0
    DISABLED = 1
    ACTIVE = 2
    INIT_FAILURE = 3
    SHUTDOWN = 4


_state_lock = threading.Lock()
_state: TelemetryState = TelemetryState.UNINITIALIZED
_provider = None  # TracerProvider when ACTIVE


_sdk_environment_lock = threading.Lock()
_active_scope_version = ""


@contextmanager
def _isolated_sdk_environment():
    """仅启动期屏蔽 SDK 的隐式 Trace 配置，异常时也恢复原环境。

    该锁只串行化本模块的构造；调用方必须在接收业务请求之前初始化。
    """
    with _sdk_environment_lock:
        keys = [
            key
            for key in os.environ
            if key.startswith(
                (
                    "OTEL_EXPORTER_OTLP",
                    "OTEL_BSP_",
                    "OTEL_SPAN_",
                    "OTEL_ATTRIBUTE_",
                    "OTEL_TRACES_",
                    "OTEL_PYTHON_",
                )
            )
            or key
            in ("OTEL_SDK_DISABLED", "OTEL_RESOURCE_ATTRIBUTES", "OTEL_SERVICE_NAME")
        ]
        saved = {key: os.environ.pop(key) for key in keys}
        try:
            yield
        finally:
            os.environ.update(saved)


def _endpoint_log_target(endpoint: str) -> str:
    """Returns a credential-free endpoint label suitable for logs."""
    try:
        parsed = urlsplit(endpoint)
        if not parsed.scheme or not parsed.hostname:
            return "configured"
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        port = f":{parsed.port}" if parsed.port is not None else ""
        return f"{parsed.scheme}://{host}{port}"
    except (TypeError, ValueError):
        return "configured"


def resolve_pod_ip() -> None:
    """启动器补全进程身份，不解析 Trace 配置、不回写 OTEL 环境变量。"""

    if os.environ.get("POD_IP"):
        return

    def valid_ip(value: str) -> bool:
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            return False
        return not address.is_loopback and not address.is_unspecified

    resolved_ip = os.environ.get("RequestedIP", "")
    if not valid_ip(resolved_ip):
        try:
            resolved_ip = socket.gethostbyname(socket.gethostname())
        except OSError as e:
            _LOGGER.warning("telemetry POD_IP DNS resolution failed: %s", e)
            return
    if not valid_ip(resolved_ip):
        return

    if not os.environ.get("POD_IP"):
        os.environ.pop("POD_IP", None)
        os.environ.setdefault("POD_IP", resolved_ip)


class _DiagnosticExporter:
    """Export-failure accounting wrapper around the OTLP exporter.

    The SDK's BatchSpanProcessor already warns when its queue overflows; this
    wrapper covers the other silent-loss channel — batches that fail on the
    wire — so a sudden trace gap on the platform can be attributed to export
    loss instead of guessing whether spans were never produced. Failures emit
    a rate-limited warning (one per 60s window) with cumulative counters, and
    shutdown logs a final summary when anything was lost.
    """

    _LOG_INTERVAL_S = 60.0

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self._total_batches = 0
        self._failed_batches = 0
        self._failed_spans = 0
        self._last_log_ts = 0.0

    def export(self, spans: Any) -> Any:
        from opentelemetry.sdk.trace.export import SpanExportResult

        try:
            result = self._inner.export(spans)
        except Exception as e:  # noqa: BLE001 - never break the BSP thread
            self._record(len(spans), failed=True, reason=type(e).__name__)
            return SpanExportResult.FAILURE
        self._record(
            len(spans),
            failed=result != SpanExportResult.SUCCESS,
            reason=str(result),
        )
        return result

    def _record(self, span_count: int, failed: bool, reason: str = "") -> None:
        with self._lock:
            self._total_batches += 1
            if not failed:
                return
            self._failed_batches += 1
            self._failed_spans += span_count
            now = time.monotonic()
            if now - self._last_log_ts < self._LOG_INTERVAL_S:
                return
            self._last_log_ts = now
            failed_batches, failed_spans, total = (
                self._failed_batches,
                self._failed_spans,
                self._total_batches,
            )
        _LOGGER.warning(
            "telemetry span export failing: %s (cumulative %d/%d batches failed,"
            " %d spans lost; next warning suppressed for %.0fs)",
            reason,
            failed_batches,
            total,
            failed_spans,
            self._LOG_INTERVAL_S,
        )

    def shutdown(self) -> None:
        with self._lock:
            failed_batches, failed_spans, total = (
                self._failed_batches,
                self._failed_spans,
                self._total_batches,
            )
        if failed_batches:
            _LOGGER.warning(
                "telemetry exporter shutdown: %d/%d batches failed, %d spans lost",
                failed_batches,
                total,
                failed_spans,
            )
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)


def init_telemetry(role: str, tp_rank: int = 0) -> bool:
    """Initialize the process-level tracer provider. Never raises (fail-open).

    Must be called after process spawn, once role/rank are known. Only
    tp_rank==0 enables span production.
    """
    global _state, _provider
    with _state_lock:
        if _state == TelemetryState.SHUTDOWN:
            _LOGGER.warning(
                "telemetry init requested after shutdown; reinitialization is not supported"
            )
            return False
        if _state == TelemetryState.ACTIVE:
            return True
        if _state != TelemetryState.UNINITIALIZED:
            return False
        config = load_trace_config(role, tp_rank)
        if not config.enabled:
            _state = TelemetryState.DISABLED
            return False
        exporter = None
        session = None
        try:
            with _isolated_sdk_environment():
                _load_otel_sdk()
                import ssl

                import requests
                from opentelemetry.exporter.otlp.proto.http import Compression
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )

                session = requests.Session()
                session.trust_env = False
                certificate = (
                    config.certificate or ssl.get_default_verify_paths().cafile or True
                )
                exporter = _DiagnosticExporter(
                    OTLPSpanExporter(
                        endpoint=config.endpoint,
                        headers=dict(config.headers),
                        certificate_file=certificate,
                        timeout=config.http_timeout_ms / 1000.0,
                        compression=Compression.NoCompression,
                        session=session,
                    )
                )
                return _init_with_exporter_locked(exporter, role, tp_rank, config)
        except Exception:  # noqa: BLE001 - 不回显可能包含凭证的 SDK 异常
            _state = TelemetryState.INIT_FAILURE
            if exporter is not None:
                try:
                    exporter.shutdown()
                except Exception:
                    pass
            if session is not None:
                session.close()
            _LOGGER.warning(
                "Trace 已关闭 role=%s field=sdk reason=initialization_failed", role
            )
            return False


def init_telemetry_for_test(
    exporter: Any,
    role: str = "test",
    tp_rank: int = 0,
    config: Optional[TraceConfig] = None,
) -> bool:
    """Test-only: initialize with an injected span exporter, bypassing env switch."""
    global _state
    with _state_lock:
        if _state == TelemetryState.SHUTDOWN:
            _LOGGER.warning(
                "telemetry test init requested after shutdown; reinitialization is not supported"
            )
            return False
        if _state == TelemetryState.ACTIVE:
            _LOGGER.warning(
                "telemetry init_for_test called while ACTIVE, call shutdown first"
            )
            return False
        try:
            with _isolated_sdk_environment():
                _load_otel_sdk()
                return _init_with_exporter_locked(
                    exporter, role, tp_rank, config or TraceConfig()
                )
        except Exception:  # noqa: BLE001 - do not expose config or SDK values
            _state = TelemetryState.INIT_FAILURE
            _LOGGER.warning(
                "Trace 已关闭 role=%s field=sdk reason=initialization_failed", role
            )
            return False


def _init_with_exporter_locked(
    exporter: Any, role: str, tp_rank: int, config: TraceConfig
) -> bool:
    """Init order: Resource -> Sampler -> BSP -> Provider -> global propagator."""
    global _state, _provider, _active_scope_version
    service_name = config.service_name or f"rtp_llm_{role}"
    # Resolved once so service.instance.id, host.name and host.ip can never
    # disagree about which host this process runs on.
    hostname = socket.gethostname()
    pid = os.getpid()
    resource_attributes = {
        "service.name": service_name,
        # "unknown" fallback mirrors the C++ runtime: the instance id must never
        # degrade into a bare "-<pid>".
        "service.instance.id": f"{hostname or 'unknown'}-{pid}",
        "process.pid": pid,
        "rtp_llm.role": role,
        # Fixed marker the platform's GenAI statistics match on. It names the
        # instrumentation contract being followed, not a library we link.
        "gen_ai.instrumentation.sdk.name": "loongsuite-genai-utils",
        # rtp_llm.tp_rank is intentionally NOT a resource attribute: the
        # rank0-only gate makes it constantly 0 on every exported span (zero
        # information). tp_rank stays an init_telemetry() gate parameter only.
    }
    # Aligned with the C++ runtime: host.ip carries "{hostname}-{pid}", not an IP.
    # The platform's per-instance request/error/latency panels key off host.ip,
    # and a pod IP is not process-unique -- frontend and backend share one pod --
    # so an IP-valued host.ip merges them into a single bucket. The real address
    # stays reported under rtp_llm.pod_ip. Neither host key is synthesized when
    # the hostname is unknown: "unknown-<pid>" would pollute exactly the
    # aggregation these fields exist to serve.
    if hostname:
        resource_attributes["host.name"] = hostname
        resource_attributes["host.ip"] = f"{hostname}-{pid}"
    pod_ip = os.environ.get("POD_IP", "")
    if pod_ip:
        resource_attributes["rtp_llm.pod_ip"] = pod_ip
    from opentelemetry.sdk.version import __version__ as sdk_version

    resource_attributes.update(
        {
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.version": sdk_version,
        }
    )
    resource = Resource(resource_attributes)
    sampler = ParentBased(TraceIdRatioBased(config.sampler_ratio))
    provider = TracerProvider(
        resource=resource,
        sampler=sampler,
        span_limits=SpanLimits(
            max_attributes=128,
            max_events=128,
            max_links=128,
            max_span_attributes=128,
            max_event_attributes=128,
            max_link_attributes=128,
        ),
        shutdown_on_exit=False,
    )
    processor = None
    try:
        processor = BatchSpanProcessor(
            exporter,
            max_queue_size=config.max_queue_size,
            schedule_delay_millis=config.schedule_delay_ms,
            max_export_batch_size=config.max_export_batch_size,
            export_timeout_millis=config.http_timeout_ms,
        )
        provider.add_span_processor(processor)
        propagate.set_global_textmap(TraceContextTextMapPropagator())
    except Exception:
        if processor is not None:
            processor.shutdown()
        provider.shutdown()
        raise
    # 使用自有 Provider，避免已安装的全局 Provider 接管手工 span。
    _provider = provider
    _active_scope_version = config.scope_version or package_scope_version()
    _state = TelemetryState.ACTIVE
    _LOGGER.info(
        "Trace 已启用 role=%s source=%s root_ratio=%s parent_based=true endpoint=%s",
        role,
        config.source,
        config.sampler_ratio,
        _endpoint_log_target(config.endpoint),
    )
    return True


def shutdown_telemetry(deadline_ms: int = 2000) -> bool:
    """Bounded shutdown on a helper thread; drops remaining spans on timeout."""
    global _state, _provider
    with _state_lock:
        if _state != TelemetryState.ACTIVE:
            if _state != TelemetryState.UNINITIALIZED:
                _state = TelemetryState.SHUTDOWN
            return True
        provider = _provider
        _provider = None
        _state = TelemetryState.SHUTDOWN

    done = threading.Event()

    def _shutdown_worker() -> None:
        try:
            provider.shutdown()
        except Exception:  # noqa: BLE001 - telemetry must never break shutdown
            pass
        finally:
            done.set()

    worker = threading.Thread(
        target=_shutdown_worker, name="otel-shutdown", daemon=True
    )
    worker.start()
    if not done.wait(deadline_ms / 1000.0):
        _LOGGER.warning(
            "telemetry shutdown exceeded deadline %d ms, remaining spans dropped",
            deadline_ms,
        )
        return False
    return True


def is_telemetry_active() -> bool:
    with _state_lock:
        return _state == TelemetryState.ACTIVE


def telemetry_state() -> TelemetryState:
    with _state_lock:
        return _state


def _scope_version() -> str:
    """仅返回启动时确定的 scope 版本，不再读取旧环境变量。"""
    return _active_scope_version


def get_tracer():
    """Returns a tracer when ACTIVE, otherwise None (callers must null-check)."""
    with _state_lock:
        if _state != TelemetryState.ACTIVE or _provider is None:
            return None
        return _provider.get_tracer("rtp_llm", _scope_version() or None)


# ---------------------------------------------------------------------------
# Carrier helpers (W3C traceparent/tracestate via the global propagator)
# ---------------------------------------------------------------------------


def extract_context_from_headers(headers: Any) -> Optional[Any]:
    """Extracts remote OTel context from HTTP headers (mapping-like).

    Invalid/missing headers yield an empty context; never raises.
    """
    if not is_telemetry_active():
        return None
    try:
        carrier: Dict[str, str] = {}
        for key in ("traceparent", "tracestate"):
            value = headers.get(key) if hasattr(headers, "get") else None
            if value:
                carrier[key] = value
        return propagate.extract(carrier)
    except Exception:  # noqa: BLE001 - fail-open
        return None


def select_valid_server_trace_carrier(
    body_headers: Any, metadata_headers: Any
) -> Tuple[Dict[str, str], str]:
    """Select one complete valid W3C carrier without mixing its sources."""
    if not OTEL_AVAILABLE:
        return {}, "none"
    for source, headers in (("body", body_headers), ("metadata", metadata_headers)):
        if not hasattr(headers, "get"):
            continue
        carrier = {
            key: str(value)
            for key in ("traceparent", "tracestate", "baggage")
            if (value := headers.get(key))
        }
        if "traceparent" not in carrier:
            continue
        try:
            context = TraceContextTextMapPropagator().extract(
                {
                    key: carrier[key]
                    for key in ("traceparent", "tracestate")
                    if key in carrier
                },
                context=otel_context.Context(),
            )
            if trace.get_current_span(context).get_span_context().is_valid:
                return carrier, source
        except Exception:  # noqa: BLE001 - malformed carriers are ignored
            continue
    return {}, "none"


def metadata_to_headers(metadata: Any) -> Dict[str, Any]:
    """Converts gRPC metadata to a lowercase, mapping-like carrier.

    Duplicate keys use the last value, matching ``dict(metadata)``. Malformed
    entries and undecodable byte keys are ignored so tracing cannot fail an RPC.
    Values are preserved because ``-bin`` metadata may legitimately be bytes.
    """
    headers: Dict[str, Any] = {}
    try:
        entries = metadata or ()
        for entry in entries:
            try:
                key, value = entry
                if key is None or value is None:
                    continue
                if isinstance(key, bytes):
                    key = key.decode("ascii")
                else:
                    key = str(key)
                headers[key.lower()] = value
            except Exception:  # noqa: BLE001 - malformed metadata is ignored
                continue
    except Exception:  # noqa: BLE001 - fail-open
        return {}
    return headers


# Bailian convention: consume `traffic.llm_sdk.*` baggage entries at the HTTP
# entry and write them (prefix stripped) onto the
# root SERVER span. This is entry-side CONSUMPTION only, and does not conflict
# with the rule that forbids FORWARDING baggage downstream (the global
# propagator stays TraceContext-only and gRPC metadata never carries baggage).
_BAGGAGE_ATTR_PREFIX = "traffic.llm_sdk."
_BAGGAGE_ALLOWED_ATTRIBUTES = frozenset({"scene"})
_BAGGAGE_MAX_ENTRIES = 16
_BAGGAGE_MAX_VALUE_LEN = 256


def _extract_llm_sdk_baggage(headers: Any) -> Dict[str, str]:
    """Parses explicitly allowed traffic.llm_sdk baggage entries locally.

    Uses a local W3CBaggagePropagator instance (NOT the global propagator) so
    baggage is never propagated implicitly. Hostile-header defenses: entry
    count cap + value length truncation. Never raises.
    """
    try:
        value = headers.get("baggage") if hasattr(headers, "get") else None
        if not value:
            return {}
        ctx = W3CBaggagePropagator().extract(
            {"baggage": value}, context=otel_context.Context()
        )
        result: Dict[str, str] = {}
        for key, entry in otel_baggage.get_all(ctx).items():
            if not key.startswith(_BAGGAGE_ATTR_PREFIX):
                continue
            stripped = key[len(_BAGGAGE_ATTR_PREFIX) :]
            if stripped not in _BAGGAGE_ALLOWED_ATTRIBUTES:
                continue
            result[stripped] = str(entry)[:_BAGGAGE_MAX_VALUE_LEN]
            if len(result) >= _BAGGAGE_MAX_ENTRIES:
                break
        return result
    except Exception:  # noqa: BLE001 - fail-open
        return {}


def inject_context_to_metadata(context: Optional[Any]) -> List[Tuple[str, str]]:
    """Injects the given context into gRPC metadata key/value pairs.

    Returns [] when telemetry is inactive so callers can pass it unconditionally.
    """
    if not is_telemetry_active() or context is None:
        return []
    try:
        carrier: Dict[str, str] = {}
        propagate.inject(carrier, context=context)
        return list(carrier.items())
    except Exception:  # noqa: BLE001 - fail-open
        return []


# ---------------------------------------------------------------------------
# Request-scoped trace state: explicit ContextVar, idempotent finish.
# ---------------------------------------------------------------------------


_REQUEST_ERROR_DESCRIPTIONS = {
    "Cancelled": "Request processing was cancelled",
    "FtRuntimeException": "Inference request failed",
    "TrafficLimit": "Request was rejected by traffic limits",
}
_CLIENT_ERROR_DESCRIPTIONS = {
    "Cancelled": "Client operation was cancelled",
    "RpcError": "Model RPC request failed",
    "TrafficLimit": "Request routing was rejected by traffic limits",
    "FlexlbBusinessRejected": "Master rejected the schedule request",
}


def _request_error_description(error_type: str) -> str:
    """Returns a predictable, non-sensitive root-span status description."""
    return _REQUEST_ERROR_DESCRIPTIONS.get(error_type, "Request processing failed")


def _client_error_description(error_type: str) -> str:
    """Returns a predictable, non-sensitive CLIENT-span status description."""
    return _CLIENT_ERROR_DESCRIPTIONS.get(error_type, "Client operation failed")


def _internal_error_description(error_type: str) -> str:
    """Returns a predictable description for in-process routing spans."""
    if error_type == "TrafficLimit":
        return "Request routing was rejected by traffic limits"
    return "Request routing failed"


class RequestTraceState:
    """Holds the spans of one HTTP request lifecycle.

    Owner rules (manual instrumentation, no ASGI middleware):
    - server_span: owned by the frontend request path; ended exactly once via
      finish() from the four streaming exits (success/cancel/error/finally).
    - Attributes must be written before finish(); writes after are dropped.
    """

    def __init__(
        self,
        server_span: Any = None,
        server_context: Any = None,
        request_start_ns: Optional[int] = None,
    ):
        self.server_span = server_span
        self._server_context = server_context
        self._request_start_ns = (
            request_start_ns if request_start_ns is not None else time.monotonic_ns()
        )
        self._first_visible_token_ns: Optional[int] = None
        self._visible_output_tokens = 0
        self._finished = False
        self._settled_ok: Optional[bool] = None
        self._renderer_completed = False
        self._lock = threading.Lock()

    @property
    def server_context(self) -> Optional[Any]:
        """Context containing the server span, parent for child CLIENT spans."""
        return self._server_context

    @property
    def settled_ok(self) -> Optional[bool]:
        """Root span outcome: None while unsettled, True/False once ended.

        A child span that can only settle during its own teardown reads this to
        tell "the request already succeeded, my generator is just being closed"
        from "the request itself was interrupted". Both cases surface as
        GeneratorExit/CancelledError inside the child, so the exception type
        alone cannot distinguish them, while the parent outcome can.
        """
        with self._lock:
            return self._settled_ok

    @property
    def renderer_completed(self) -> bool:
        """Whether the renderer deliberately completed the response stream."""
        with self._lock:
            return self._renderer_completed

    def mark_renderer_completed(self) -> None:
        """Publishes a normal renderer stop before its backend stream teardown."""
        with self._lock:
            self._renderer_completed = True

    def set_attribute(self, key: str, value: Any) -> None:
        try:
            with self._lock:
                if self.server_span is not None and not self._finished:
                    self.server_span.set_attribute(key, value)
        except Exception:  # noqa: BLE001 - fail-open
            pass

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Adds a span event stamped with the SDK wall-clock 'now'.

        Callers must invoke this AT the moment the event happens (e.g. the
        first_response_chunk event fires when the first frontend response
        object becomes available, before SSE serialization), not post-hoc like
        the aggregated Engine TTFT attribute. Dropped after finish(); fail-open.
        """
        try:
            with self._lock:
                if self.server_span is not None and not self._finished:
                    self.server_span.add_event(name, attributes=attributes)
        except Exception:  # noqa: BLE001 - fail-open
            pass

    def record_frontend_output_tokens(
        self, token_count: int, observed_time_ns: Optional[int] = None
    ) -> None:
        """Records caller-visible streaming tokens on the entry SERVER span.

        HTTP and Dash classify visible output at their protocol boundaries and
        call this immediately before yielding the response. Empty, role-only,
        finish-only, and internal control frames never reach this method.

        TPOT needs two distinct delivery instants to be an observation rather
        than an assumption. A single frame carrying N>1 tokens exposes no
        inter-token send boundary, so TPOT stays absent instead of being
        reported as 0.0, which a dashboard would read as instant decoding.
        """
        if (
            not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or token_count <= 0
        ):
            return
        try:
            observed_ns = (
                observed_time_ns
                if observed_time_ns is not None
                else time.monotonic_ns()
            )
            if not isinstance(observed_ns, int) or isinstance(observed_ns, bool):
                return
            with self._lock:
                if self.server_span is None or self._finished:
                    return
                if observed_ns < self._request_start_ns:
                    return
                if self._first_visible_token_ns is None:
                    self._first_visible_token_ns = observed_ns
                    ttft_ms = (observed_ns - self._request_start_ns) / 1e6
                    self.server_span.set_attribute(
                        trace_attrs.GEN_AI_TIME_TO_FIRST_TOKEN, ttft_ms
                    )
                elif observed_ns < self._first_visible_token_ns:
                    return

                self._visible_output_tokens += token_count
                if (
                    self._visible_output_tokens > 1
                    and observed_ns > self._first_visible_token_ns
                ):
                    tpot_ms = (
                        (observed_ns - self._first_visible_token_ns)
                        / 1e6
                        / (self._visible_output_tokens - 1)
                    )
                    self.server_span.set_attribute(
                        trace_attrs.RTP_LLM_FRONTEND_TIME_PER_OUTPUT_TOKEN_MS,
                        tpot_ms,
                    )
        except Exception:  # noqa: BLE001 - fail-open
            pass

    def finish(
        self, error: Optional[BaseException] = None, error_type: str = ""
    ) -> None:
        """Idempotent span end; safe from any exit path."""
        try:
            with self._lock:
                if self._finished:
                    return
                self._finished = True
                self._settled_ok = error is None and not error_type
                span = self.server_span
            if span is None:
                return
            if error is not None or error_type:
                resolved_error_type = error_type or type(error).__name__
                if OTEL_AVAILABLE:
                    span.set_status(
                        trace.StatusCode.ERROR,
                        _request_error_description(resolved_error_type),
                    )
                span.set_attribute(trace_attrs.ERROR_TYPE, resolved_error_type)
            elif OTEL_AVAILABLE:
                # Explicit OK keeps Python spans consistent with the C++ side
                # (GrpcStatusSpanGuard already sets kOk on success); Unset
                # renders as a blank status in the platform Details panel.
                span.set_status(trace.StatusCode.OK)
            span.end()
        except Exception:  # noqa: BLE001 - fail-open
            pass


# Explicit request-scoped carrier for the trace state (D3): set at the HTTP
# entry, read by the gRPC client layer. Not shared across requests because
# each request handler runs in its own contextvars snapshot chain.
CURRENT_TRACE_STATE: ContextVar[Optional[RequestTraceState]] = ContextVar(
    "rtp_llm_current_trace_state", default=None
)

# A short-lived child context used by in-process orchestration.  This lets a
# real outbound RPC started inside an INTERNAL span inherit that span without
# mutating the request root or relying on implicit thread-local propagation.
CURRENT_INTERNAL_CONTEXT: ContextVar[Optional[Any]] = ContextVar(
    "rtp_llm_current_internal_context", default=None
)


def reset_telemetry_for_test(deadline_ms: int = 2000) -> bool:
    """Reset process telemetry state for test isolation only."""
    if not shutdown_telemetry(deadline_ms):
        return False

    global _state, _provider
    with _state_lock:
        _provider = None
        _state = TelemetryState.UNINITIALIZED
    CURRENT_TRACE_STATE.set(None)
    CURRENT_INTERNAL_CONTEXT.set(None)
    return True


class ClientSpanHandle:
    """Idempotent finish wrapper for a child span."""

    def __init__(
        self,
        span: Any,
        error_description: Callable[[str], str] = _client_error_description,
        context_token: Any = None,
    ):
        self._span = span
        self._error_description = error_description
        self._context_token = context_token
        self._finished = False
        self._lock = threading.Lock()

    def set_attribute(self, key: str, value: Any) -> None:
        try:
            with self._lock:
                if not self._finished:
                    self._span.set_attribute(key, value)
        except Exception:  # noqa: BLE001 - fail-open
            pass

    def finish(
        self, error: Optional[BaseException] = None, error_type: str = ""
    ) -> None:
        try:
            with self._lock:
                if self._finished:
                    return
                self._finished = True
                context_token = self._context_token
                self._context_token = None
            if context_token is not None:
                try:
                    CURRENT_INTERNAL_CONTEXT.reset(context_token)
                except Exception:  # noqa: BLE001 - context cleanup is fail-open
                    pass
            if error is not None or error_type:
                resolved_error_type = error_type or type(error).__name__
                if OTEL_AVAILABLE:
                    self._span.set_status(
                        trace.StatusCode.ERROR,
                        self._error_description(resolved_error_type),
                    )
                self._span.set_attribute(trace_attrs.ERROR_TYPE, resolved_error_type)
            elif OTEL_AVAILABLE:
                # Mirror RequestTraceState.finish: explicit OK on success.
                self._span.set_status(trace.StatusCode.OK)
            self._span.end()
        except Exception:  # noqa: BLE001 - fail-open
            pass


def _parse_server_endpoint(target_address: Any) -> Dict[str, Any]:
    """Parses a gRPC host:port observation without changing connection behavior."""
    try:
        if not isinstance(target_address, str) or not target_address:
            return {}
        if target_address != target_address.strip() or "://" in target_address:
            return {}
        parsed = urlsplit("//" + target_address)
        if (
            parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            return {}
        address = parsed.hostname
        port = parsed.port
        if not address or port is None or not 1 <= port <= 65535:
            return {}
        if any(character.isspace() for character in address):
            return {}
        return {
            trace_attrs.SERVER_ADDRESS: address,
            trace_attrs.SERVER_PORT: port,
        }
    except (TypeError, ValueError):
        return {}


def start_client_span(
    span_name: str, target_address: Any = None
) -> Tuple[Optional[ClientSpanHandle], List[Tuple[str, str]]]:
    """Starts a gRPC CLIENT span as child of the current request's SERVER span.

    Returns (handle, metadata) where metadata carries W3C traceparent for gRPC.
    Scope guard: only produces a span when a RequestTraceState exists
    (chat completions entry), so batch/embedding routes stay untraced.
    Returns (None, []) whenever telemetry is inactive; never raises.
    """
    if not is_telemetry_active():
        return None, []
    try:
        state = CURRENT_TRACE_STATE.get()
        if state is None or state.server_context is None:
            return None, []
        tracer = get_tracer()
        if tracer is None:
            return None, []
        parent_context = CURRENT_INTERNAL_CONTEXT.get() or state.server_context
        span = tracer.start_span(
            span_name,
            context=parent_context,
            kind=trace.SpanKind.CLIENT,
            attributes=_parse_server_endpoint(target_address) or None,
        )
        # Deliberately NO rpc.system here: the platform re-classifies the span
        # as an RPC client call when rpc.system is present, which breaks the
        # top-bar Total tokens aggregation (it only aggregates plain client
        # spans; regression measured and verified). The C++ span factories keep
        # rpc.system for their grpc chips.
        metadata = inject_context_to_metadata(trace.set_span_in_context(span))
        return ClientSpanHandle(span), metadata
    except Exception:  # noqa: BLE001 - fail-open
        return None, []


def start_internal_span(span_name: str) -> Optional[ClientSpanHandle]:
    """Starts an INTERNAL span as child of the current request's SERVER span.

    For in-process orchestration stages (e.g. rtp_llm.master_route, which may
    resolve addrs without any outbound call when the request carries
    role_addrs); span kind is fixed at creation so CLIENT would be wrong for
    those paths. No traceparent metadata: nothing crosses a process boundary.
    Returns None whenever telemetry is inactive; never raises.
    """
    if not is_telemetry_active():
        return None
    try:
        state = CURRENT_TRACE_STATE.get()
        if state is None or state.server_context is None:
            return None
        tracer = get_tracer()
        if tracer is None:
            return None
        span = tracer.start_span(
            span_name,
            context=state.server_context,
            kind=trace.SpanKind.INTERNAL,
        )
        context_token = CURRENT_INTERNAL_CONTEXT.set(trace.set_span_in_context(span))
        return ClientSpanHandle(span, _internal_error_description, context_token)
    except Exception:  # noqa: BLE001 - fail-open
        return None


def start_server_span(
    span_name: str,
    headers: Any,
    initial_attributes: Optional[Dict[str, Any]] = None,
    start_time: Optional[int] = None,
    request_start_ns: Optional[int] = None,
) -> Optional[RequestTraceState]:
    """Starts an HTTP SERVER span with remote parent extracted from headers.

    Returns None when telemetry is inactive; otherwise a RequestTraceState
    (also published to CURRENT_TRACE_STATE for downstream layers).
    """
    if not is_telemetry_active():
        return None
    try:
        tracer = get_tracer()
        if tracer is None:
            return None
        remote_context = extract_context_from_headers(headers)
        monotonic_start_ns = (
            request_start_ns if request_start_ns is not None else time.monotonic_ns()
        )
        span = tracer.start_span(
            span_name,
            context=remote_context,
            kind=trace.SpanKind.SERVER,
            attributes=dict(initial_attributes) if initial_attributes else None,
            start_time=start_time,
        )
        for baggage_key, baggage_value in _extract_llm_sdk_baggage(headers).items():
            try:
                span.set_attribute(baggage_key, baggage_value)
            except Exception:  # noqa: BLE001 - fail-open
                pass
        server_context = trace.set_span_in_context(span)
        state = RequestTraceState(
            server_span=span,
            server_context=server_context,
            request_start_ns=monotonic_start_ns,
        )
        CURRENT_TRACE_STATE.set(state)
        return state
    except Exception:  # noqa: BLE001 - fail-open
        return None


# ---------------------------------------------------------------------------
# Response business attributes: the frontend collect path is the single point
# where the fully aggregated OpenAI response is available
# for BOTH streaming and non-streaming, and BOTH Fusion and PD topologies
# (the frontend gRPC client rebuilds the complete AuxInfo topology-agnostically,
# default aux_info=True). So request-level gen_ai.* metrics live on the root
# HTTP SERVER span here rather than on the per-hop C++ spans.
# ---------------------------------------------------------------------------


def _extract_finish_reasons(choices: Any) -> List[str]:
    reasons: List[str] = []
    try:
        for choice in choices or []:
            if isinstance(choice, dict):
                reason = choice.get("finish_reason")
            else:
                reason = getattr(choice, "finish_reason", None)
            if reason:
                reasons.append(
                    str(reason.value if isinstance(reason, Enum) else reason)
                )
    except Exception:  # noqa: BLE001 - fail-open
        return []
    return reasons


def _record_aux_attributes(state: "RequestTraceState", aux: Dict[str, Any]) -> None:
    # Engine TTFT/TPOT live on each generate_stream_call CLIENT span. The root
    # SERVER span keeps topology-independent request/cache/phase attributes;
    # streaming delivery latency is recorded in real time by the entry server.
    _record_phase_latency_attributes(state, aux)
    pd_sep = aux.get("pd_sep")
    if isinstance(pd_sep, bool):
        state.set_attribute(trace_attrs.RTP_LLM_PD_SEP, pd_sep)
    for key, attr in (
        ("reuse_len", trace_attrs.RTP_LLM_CACHE_TOTAL_REUSE_LEN),
        ("local_reuse_len", trace_attrs.RTP_LLM_CACHE_LOCAL_REUSE_LEN),
        ("remote_reuse_len", trace_attrs.RTP_LLM_CACHE_REMOTE_REUSE_LEN),
    ):
        value = aux.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            state.set_attribute(attr, value)


def _non_negative_ms(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return None


def _record_phase_latency_attributes(
    state: "RequestTraceState", aux: Dict[str, Any]
) -> None:
    """Derives prefill/decode phase latencies from AuxInfo.

    prefill = first_token_cost_time - wait_time; decode = cost_time - TTFT.
    The platform expects these two gen_ai.latency.* attributes in NANOSECONDS,
    unlike TTFT which stays in milliseconds.
    Skips silently on missing/nonsensical inputs — never writes bad values.
    """
    # In PD responses wait_time is Decode wait, so it cannot be subtracted
    # from frontend TTFT to recover Prefill time. Only Fusion's explicit false
    # value proves all three measurements came from the same local stream.
    if aux.get("pd_sep") is not False:
        return
    ttft_ms = _non_negative_ms(aux.get("first_token_cost_time"))
    wait_ms = _non_negative_ms(aux.get("wait_time"))
    cost_ms = _non_negative_ms(aux.get("cost_time"))
    if ttft_ms is None or ttft_ms <= 0:
        return
    if wait_ms is not None:
        prefill_ms = ttft_ms - wait_ms
        if prefill_ms > 0:
            state.set_attribute(
                trace_attrs.GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL,
                int(prefill_ms * 1e6),
            )
    if cost_ms is not None:
        decode_ms = cost_ms - ttft_ms
        if decode_ms > 0:
            state.set_attribute(
                trace_attrs.GEN_AI_LATENCY_TIME_IN_MODEL_DECODE,
                int(decode_ms * 1e6),
            )


def record_response_attributes(complete_response: Any) -> None:
    """Write request-level gen_ai.* business attributes onto the current SERVER span.

    Sourced from the fully aggregated OpenAI response (usage tokens, per-choice
    finish_reason, and the complete AuxInfo). Fail-open: never raises, no-op
    when telemetry is inactive or no request span exists.
    """
    if not is_telemetry_active():
        return
    state = CURRENT_TRACE_STATE.get()
    if state is None:
        return
    try:
        data = complete_response
        if data is None:
            return
        if not isinstance(data, dict) and hasattr(data, "model_dump"):
            data = data.model_dump(exclude_none=True)
        if not isinstance(data, dict):
            return

        finish_reasons = _extract_finish_reasons(data.get("choices"))
        if finish_reasons:
            # string[] per OTel GenAI SemConv; Python SDK supports sequence
            # attributes so no scalar fallback is needed.
            state.set_attribute(
                trace_attrs.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons
            )

        usage = data.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool):
                state.set_attribute(
                    trace_attrs.GEN_AI_USAGE_INPUT_TOKENS, prompt_tokens
                )
                # legacy alias: some platform views only read this older name
                state.set_attribute(
                    trace_attrs.GEN_AI_USAGE_PROMPT_TOKENS, prompt_tokens
                )
            if isinstance(completion_tokens, int) and not isinstance(
                completion_tokens, bool
            ):
                state.set_attribute(
                    trace_attrs.GEN_AI_USAGE_OUTPUT_TOKENS, completion_tokens
                )
                # legacy alias: some platform views only read this older name
                state.set_attribute(
                    trace_attrs.GEN_AI_USAGE_COMPLETION_TOKENS, completion_tokens
                )
            if isinstance(total_tokens, int) and not isinstance(total_tokens, bool):
                state.set_attribute(trace_attrs.GEN_AI_USAGE_TOTAL_TOKENS, total_tokens)

        aux = data.get("aux_info")
        if isinstance(aux, dict):
            _record_aux_attributes(state, aux)

        # LLM-view classification (see attributes.py): the platform trio plus
        # the OTel GenAI semconv classification pair — both written so the span
        # is recognized as an LLM call regardless of which rule set the platform
        # matches on. Only on the root SERVER span so the LLM view shows a
        # single model-call node per request.
        state.set_attribute(trace_attrs.GEN_AI_SPAN_KIND, "LLM")
        state.set_attribute(trace_attrs.LINGJI_FLAG, True)
        state.set_attribute(trace_attrs.ACS_ARMS_TENANT_SPAN_POLICY, "mask")
        state.set_attribute(trace_attrs.GEN_AI_OPERATION_NAME, "chat")
        state.set_attribute(trace_attrs.GEN_AI_SYSTEM, "rtp_llm")
    except Exception:  # noqa: BLE001 - fail-open
        pass
