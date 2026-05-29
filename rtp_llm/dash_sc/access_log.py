"""gRPC server interceptor that emits one access log line per RPC.

Hooks onto ``grpc.server(..., interceptors=[...])`` so both real and pure-forward
servicers get uniform coverage. Schema is a flat JSON line; fields always present
(null when N/A).

Two capture modes share the same interceptor + log file:

- **Struct mode** (default, real servicer): pulls request / response statistics
  and generation controls off the proto so the log is immediately human-readable.

- **Forward summary mode** (forward servicer, ``raw_mode=True``): the forwarder
  is a transparent proxy, so the useful production signal is request / response
  statistics plus proxy-stage timings, not a giant proto dump. The summary keeps
  one compact JSON record per RPC and does not retain request/response payloads.

File: ``<log_path>/dash_sc_grpc_access_r{rank}_s{server}.log`` (per-process via
``get_process_log_filename``).

Performance notes (what stays on the RPC worker thread):
- File I/O is async — ``AsyncRotatingFileHandler.emit`` just ``put_nowait`` on a
  bounded queue; a background worker thread does the real write. Queue full →
  drop, never block. So disk latency doesn't affect RPC latency.
- Per response frame: one pass over ``infer.outputs`` to extract token counts,
  finish markers, and backend aux counters. Token ids and raw proto payloads are
  intentionally not retained in the access log.
- At RPC end: ``orjson.dumps`` of one flat dict; logger.info then enqueues to the
  async handler queue in microseconds.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import grpc
import orjson

from rtp_llm.access_logger.log_utils import get_handler
from rtp_llm.dash_sc.proxy.access_record import (
    DASH_SC_GRPC_PROTOCOL,
    ForwardAccessRecord,
    format_access_log_ts,
    rpc_code,
    rpc_details,
)
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor

DASH_SC_GRPC_ACCESS_LOGGER_NAME = "dash_sc_grpc_access_logger"
DASH_SC_GRPC_ACCESS_LOG_FILENAME = "dash_sc_grpc_access.log"

# Separate query-log channel. The access log above is written at RPC completion
# (``_finalize``), which on long streaming RPCs can be minutes after the request
# actually arrived. For end-to-end link-latency debugging (forwarder arrival →
# frontend arrival) we need the arrival timestamp to hit the wire immediately,
# not after the response stream drains. Following the HTTP frontend convention
# (``rtp_llm/access_logger/access_logger.py``: ``access.log`` vs
# ``query_access.log``), the query log is a separate file so ``tail -f`` on
# arrivals isn't drowned by the heavier per-RPC completion record.
DASH_SC_GRPC_QUERY_LOGGER_NAME = "dash_sc_grpc_query_logger"
DASH_SC_GRPC_QUERY_LOG_FILENAME = "dash_sc_grpc_query.log"

# Protocol tag appended to every kmonitor report emitted from this interceptor, so
# dashboards/alerts can split the gRPC path from the HTTP frontend path that shares
# the same metric names (py_rtp_framework_qps / rt etc.).
_PROTOCOL_TAG = DASH_SC_GRPC_PROTOCOL

_STREAM_TYPE = {
    (False, False): "unary",
    (False, True): "server_stream",
    (True, False): "client_stream",
    (True, True): "bidi_stream",
}

# grpcio surfaces request-iterator exceptions under this exact ``details``
# string — matched verbatim rather than by regex so an unrelated error whose
# message happens to contain "iterating requests" doesn't silently get
# reclassified as a client fault.
_CLIENT_REQ_ITER_DETAILS = "Exception iterating requests!"

# Upstream correlation headers, by priority. Whichever header the client
# actually set wins; the key name is recorded alongside the value so
# operators can tell dashscope-serving-originated IDs apart from generic
# ``x-request-id`` / W3C trace context. ``traceparent`` matches the
# ``rtp_llm/flexlb`` Java convention (HttpHeaderNames.TRACE_PARENT) so
# HTTP and gRPC paths share a fallback.
_CORRELATION_METADATA_KEYS = (
    "x-dashscope-request-id",
    "x-request-id",
    "dashscope-request-id",
    "traceparent",
)


def _extract_correlation_id(context) -> tuple[Optional[str], Optional[str]]:
    """Pull an upstream correlation ID off ``context.invocation_metadata()``.

    Returns ``(header_key, header_value)`` for the first ``_CORRELATION_METADATA_KEYS``
    entry that is present and non-empty, or ``(None, None)``. Case-insensitive —
    gRPC normalises metadata names to lower-case but clients don't always.
    Called once per RPC before any body read, so it works even when
    ``req_count=0`` (peer closed before sending a frame).
    """
    try:
        md = context.invocation_metadata() or ()
    except Exception:
        return None, None
    lookup: dict[str, str] = {}
    for entry in md:
        try:
            k, v = entry
        except Exception:
            continue
        if k is None or v is None:
            continue
        lookup[str(k).lower()] = str(v)
    for key in _CORRELATION_METADATA_KEYS:
        v = lookup.get(key)
        if v:
            return key, v
    return None, None


def _classify_rpc_exception(
    exc: BaseException, record: ForwardAccessRecord
) -> tuple[str, Optional[str]]:
    """Map a server-side RPC exception to ``(status, status_detail)``.

    Precedence (narrowest first):
    1. ``GeneratorExit`` — gRPC framework closed our wrapper because the
       client handler terminated. Route to CANCELLED so benign disconnects
       don't pollute the error bucket.
    2. ``details == "Exception iterating requests!"`` — grpcio's fixed
       marker for a failed client request iterator; code is UNKNOWN but
       semantics are "client gone", so override to CANCELLED.
    3. Bare ``RpcError()`` (no ``code()``, no ``details()``) on a
       frame-less RPC — backend-frontend view of "peer closed before
       sending a frame". CANCELLED.
    4. ``RpcError`` with a real ``code()`` — use ``code.name`` directly
       (CANCELLED / UNAVAILABLE / DEADLINE_EXCEEDED / …). This is the
       forwarder's common path; before this existed, every
       ``_MultiThreadedRendezvous`` dumped into UNKNOWN.
    5. Everything else — UNKNOWN.
    """
    if isinstance(exc, GeneratorExit):
        return "CANCELLED", "client closed generator"
    if isinstance(exc, asyncio.CancelledError):
        return "CANCELLED", "peer cancelled (async)"
    if not isinstance(exc, grpc.RpcError):
        return f"UNKNOWN_{type(exc).__name__}", repr(exc)

    code = rpc_code(exc)
    details = rpc_details(exc)

    if details == _CLIENT_REQ_ITER_DETAILS:
        return "CANCELLED", "client request iterator failed"
    if code is None and not details and record.req_count == 0:
        return "CANCELLED", "peer closed before request arrived"
    if code is not None and code != grpc.StatusCode.OK:
        return code.name, details or code.name
    # Preserve the Python class name so ``error_code`` on Grafana is actionable
    # (``UNKNOWN_RuntimeError`` / ``UNKNOWN_ValueError`` / …) instead of a
    # single opaque ``UNKNOWN`` bucket — class names are bounded (few dozen at
    # most) so tag cardinality stays tight.
    return f"UNKNOWN_{type(exc).__name__}", repr(exc)


# Short-form classifiers for free-form backend ``error_message`` strings.
# Matched in order; first hit wins. Kept intentionally small — every entry
# becomes a permanent Grafana ``error_code`` tag bucket, so new entries need
# to correspond to a distinct operational signal.
_ERROR_MESSAGE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("BACKEND_EMPTY_OUTPUTS", "empty outputs_list"),
)


def _classify_error_message(msg: Optional[str]) -> str:
    """Map a backend ``error_message`` to a bounded ``error_code`` token.

    Primary path: the backend (``service.py`` ``iter_real_model_stream_infer``)
    formats exceptions as ``f"{type(e).__name__}: {e}"`` so the leading class
    name before ``":"`` is a stable, bounded categorizer — dozens of Python
    exception classes at most, no free-form cardinality blow-up. We extract
    that prefix and return ``BACKEND_<ClassName>``.

    Fallback path: bare phrases emitted without a type prefix (e.g.
    ``"empty outputs_list from backend"`` for the zero-chunk case) map through
    ``_ERROR_MESSAGE_PATTERNS``.

    Everything else collapses to ``BACKEND_INTERNAL`` — still more specific
    than the old blanket ``INTERNAL`` because the status itself now signals
    "backend error_message channel" vs "gRPC transport error".
    """
    if not msg:
        return "BACKEND_INTERNAL"
    head = msg.strip().split("\n", 1)[0]
    colon = head.find(":")
    if 0 < colon <= 48:
        prefix = head[:colon].strip()
        if prefix and all(c.isalnum() or c == "_" for c in prefix):
            return f"BACKEND_{prefix}"
    for code, needle in _ERROR_MESSAGE_PATTERNS:
        if needle in msg:
            return code
    return "BACKEND_INTERNAL"


def init_dash_sc_grpc_access_logger(
    log_path: str,
    backup_count: int,
    rank_id: Optional[int] = None,
    server_id: Optional[int] = None,
    async_mode: bool = True,
) -> None:
    """Configure the dedicated gRPC access logger. Safe to call multiple times."""
    logger = logging.getLogger(DASH_SC_GRPC_ACCESS_LOGGER_NAME)
    handler = get_handler(
        DASH_SC_GRPC_ACCESS_LOG_FILENAME,
        log_path,
        backup_count,
        rank_id,
        server_id,
        async_mode,
    )
    logger.handlers.clear()
    logger.parent = None
    logger.setLevel(logging.INFO)
    if handler is not None:
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)


def init_dash_sc_grpc_query_logger(
    log_path: str,
    backup_count: int,
    rank_id: Optional[int] = None,
    server_id: Optional[int] = None,
    async_mode: bool = True,
) -> None:
    """Configure the dedicated gRPC query (arrival) logger. Safe to call multiple times."""
    logger = logging.getLogger(DASH_SC_GRPC_QUERY_LOGGER_NAME)
    handler = get_handler(
        DASH_SC_GRPC_QUERY_LOG_FILENAME,
        log_path,
        backup_count,
        rank_id,
        server_id,
        async_mode,
    )
    logger.handlers.clear()
    logger.parent = None
    logger.setLevel(logging.INFO)
    if handler is not None:
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)


class DashScGrpcAccessLogInterceptor(grpc.ServerInterceptor):
    """Wrap every RPC with latency + content capture, emit one JSON line at completion.

    Also fans out the same lifecycle events to ``kmonitor`` so the gRPC path reports the
    same metric family (``py_rtp_framework_qps`` / ``py_rtp_framework_rt`` /
    ``py_rtp_response_first_token_rt`` / …) as the HTTP frontend in
    :class:`rtp_llm.frontend.frontend_server.FrontendServer`. Every report carries a
    ``protocol=grpc`` tag so dashboards can slice gRPC vs HTTP off the shared metric name,
    plus ``rank_id`` / ``server_id`` to mirror frontend tagging. Non-OK RPCs also emit
    ``error_code`` (the gRPC status code name, e.g. ``UNAVAILABLE``) so alert rules stay
    symmetric with the HTTP error path.

    ``raw_mode`` is the legacy flag used by the forward servicer. It now enables
    forward-summary capture: request/response/backend statistics only, with no raw
    proto payloads or token-id arrays. Both modes share this interceptor and the
    same log file, differentiated by the ``capture_mode`` field.
    """

    def __init__(
        self,
        rank_id: Optional[int] = None,
        server_id: Optional[int] = None,
        raw_mode: bool = False,
    ) -> None:
        self._logger = logging.getLogger(DASH_SC_GRPC_ACCESS_LOGGER_NAME)
        self._query_logger = logging.getLogger(DASH_SC_GRPC_QUERY_LOGGER_NAME)
        self._rank_id = rank_id
        self._server_id = server_id
        self._raw_mode = raw_mode
        self._base_tags: dict[str, str] = {
            "protocol": _PROTOCOL_TAG,
            "rank_id": str(rank_id) if rank_id is not None else "",
            "server_id": str(server_id) if server_id is not None else "",
        }

    def _tags(self, method: str, **extra: str) -> dict[str, str]:
        tags = dict(self._base_tags)
        tags["method"] = method
        for k, v in extra.items():
            if v is not None:
                tags[k] = v
        return tags

    def _report_request_arrived(self, access_record: ForwardAccessRecord) -> None:
        """Mirror ``FrontendServer.embedding`` entry-point QPS_METRIC."""
        kmonitor.report(AccMetrics.QPS_METRIC, 1, self._tags(access_record.method))

    def _report_chunk(
        self,
        access_record: ForwardAccessRecord,
        *,
        is_first: bool,
        now: Optional[float] = None,
    ) -> None:
        """Mirror ``_call_generate_with_report`` per-chunk metrics.

        ``is_first`` picks FIRST_TOKEN_RT vs ITER_RT/ITER_QPS — same split as the HTTP path.
        """
        if now is None:
            now = time.time()
        tags = self._tags(access_record.method)
        if is_first:
            ttfb_ms = (now - access_record.start_ts) * 1000.0
            kmonitor.report(GaugeMetrics.RESPONSE_FIRST_TOKEN_RT_METRIC, ttfb_ms, tags)
        else:
            last = (
                access_record.last_chunk_ts
                or access_record.first_resp_ts
                or access_record.start_ts
            )
            iter_rt_ms = (now - last) * 1000.0
            kmonitor.report(GaugeMetrics.RESPONSE_ITER_RT_METRIC, iter_rt_ms, tags)
        kmonitor.report(AccMetrics.ITER_QPS_METRIC, 1, tags)
        access_record.last_chunk_ts = now

    def _report_rpc_done(
        self,
        access_record: ForwardAccessRecord,
        *,
        status: str,
        status_detail: Optional[str],
    ) -> None:
        """Mirror ``_call_generate_with_report`` tail metrics + HTTP success/error QPS."""
        total_ms = (time.time() - access_record.start_ts) * 1000.0
        tags = self._tags(access_record.method)
        kmonitor.report(GaugeMetrics.LANTENCY_METRIC, total_ms, tags)
        kmonitor.report(
            GaugeMetrics.RESPONSE_ITERATE_COUNT, access_record.resp_count, tags
        )
        if access_record.input_len is not None:
            kmonitor.report(
                GaugeMetrics.INPUT_TOKEN_SIZE_METRIC,
                access_record.input_len,
                tags,
            )
        kmonitor.report(
            GaugeMetrics.OUTPUT_TOKEN_SIZE_METRIC, access_record.output_len, tags
        )
        if status == "OK":
            kmonitor.report(AccMetrics.SUCCESS_QPS_METRIC, 1, tags)
        elif status == "CANCELLED":
            kmonitor.report(AccMetrics.CANCEL_QPS_METRIC, 1, tags)
        else:
            kmonitor.report(
                AccMetrics.ERROR_QPS_METRIC,
                1,
                self._tags(access_record.method, error_code=status),
            )

    def intercept_service(self, continuation, handler_call_details):
        handler = continuation(handler_call_details)
        if handler is None:
            return handler

        method = handler_call_details.method
        stream_type = _STREAM_TYPE[
            (handler.request_streaming, handler.response_streaming)
        ]

        if handler.request_streaming and handler.response_streaming:
            return grpc.stream_stream_rpc_method_handler(
                self._wrap_stream_stream(handler.stream_stream, method, stream_type),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        if handler.request_streaming and not handler.response_streaming:
            return grpc.stream_unary_rpc_method_handler(
                self._wrap_stream_unary(handler.stream_unary, method, stream_type),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        if not handler.request_streaming and handler.response_streaming:
            return grpc.unary_stream_rpc_method_handler(
                self._wrap_unary_stream(handler.unary_stream, method, stream_type),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        return grpc.unary_unary_rpc_method_handler(
            self._wrap_unary_unary(handler.unary_unary, method, stream_type),
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )

    def _new_record(
        self, method: str, stream_type: str, context
    ) -> ForwardAccessRecord:
        try:
            peer = context.peer()
        except Exception:
            peer = ""
        up_key, up_val = _extract_correlation_id(context)
        access_record = ForwardAccessRecord(
            method=method,
            stream_type=stream_type,
            peer=peer,
            start_ts=time.time(),
            raw_mode=self._raw_mode,
            upstream_request_id=up_val,
            upstream_request_id_key=up_key,
        )
        access_record.attach_to_context(context)
        # Emit arrival QPS at RPC entry — once per RPC, symmetric with
        # ``_finalize``'s unconditional ``finally``. Previously this fired
        # from ``_capture_first_request`` (i.e. only after the client's first
        # request frame was read), so frame-less RPCs — peer closed before
        # sending, client-side iterator failure, immediate cancel — still
        # reached ``_finalize`` and reported success/error/cancel but were
        # missing from ``py_rtp_framework_qps``. That asymmetry made
        # ``success+error+cancel`` exceed ``framework_qps`` by the frame-less
        # share (~30% on the forwarder path) and broke parity between the
        # access log line count and the Grafana arrival curve.
        self._report_request_arrived(access_record)
        self._emit_query_log(access_record)
        return access_record

    def _on_response_chunk(self, access_record: ForwardAccessRecord, resp) -> None:
        """Unified per-chunk hook: count, capture content, emit first-token / iter metrics."""
        now = time.time()
        is_first = access_record.first_resp_ts is None
        access_record.mark_first_resp(now)
        access_record.resp_count += 1
        access_record.capture_response_chunk(resp, now=now)
        self._report_chunk(access_record, is_first=is_first, now=now)

    def _finalize(
        self, access_record: ForwardAccessRecord, context, exc: Optional[BaseException]
    ) -> None:
        try:
            code = context.code()
        except Exception:
            code = None
        if code is not None:
            try:
                access_record.context_code = code.name
            except Exception:
                access_record.context_code = str(code)
        try:
            access_record.context_active = bool(context.is_active())
        except Exception:
            access_record.context_active = None

        # Classification precedence (narrowest signal wins):
        # 1. Backend wrote a non-empty ``error_message`` frame — canonical
        #    protocol-level failure channel (predict_v2.proto), gRPC status
        #    stays OK. Before this branch existed these dropped into the
        #    success bucket, explaining the "success_qps 混失败" half of the
        #    Grafana mismatch.
        # 2. Inference completed (``finish_reason`` observed) and a teardown
        #    signal showed up afterwards (exception, or a non-OK status the
        #    server wrote at stream close). These are post-success events —
        #    client cancel / LBS drop / late ``grpc.RpcError`` — and must not
        #    poison the success counter. Explains the "error_qps 混成功" half.
        # 3. Falls through to the pre-existing exception / gRPC-code logic.
        if exc is not None:
            access_record.exc_type = type(exc).__name__
        if access_record.error_message:
            # Break backend errors out of the single ``INTERNAL`` bucket so
            # Grafana's ``error_code`` breakdown names the actual failure
            # (``BACKEND_RuntimeError`` / ``BACKEND_OutOfMemoryError`` /
            # ``BACKEND_EMPTY_OUTPUTS`` / …). Transport-level outcomes still
            # use their gRPC code names (``CANCELLED`` / ``UNAVAILABLE`` / …).
            access_record.status = _classify_error_message(access_record.error_message)
            access_record.status_detail = access_record.error_message
        elif exc is not None:
            if access_record.terminal_seen:
                access_record.status = "OK"
                access_record.status_detail = None
            else:
                access_record.status, access_record.status_detail = (
                    _classify_rpc_exception(exc, access_record)
                )
        elif code is None or code == grpc.StatusCode.OK:
            access_record.status = "OK"
            access_record.status_detail = None
        elif access_record.terminal_seen:
            access_record.status = "OK"
            access_record.status_detail = None
        else:
            access_record.status = code.name
            try:
                access_record.status_detail = context.details() or code.name
            except Exception:
                access_record.status_detail = code.name

        self._report_rpc_done(
            access_record,
            status=access_record.status,
            status_detail=access_record.status_detail,
        )

        end_ts = time.time()
        record = access_record.build_record(
            self._server_id, self._rank_id, end_ts=end_ts
        )
        try:
            # orjson emits bytes — decode once for the logging framework.
            self._logger.info(orjson.dumps(record).decode("utf-8"))
        except Exception as e:
            logging.warning("[DashScGrpc] access log emit failed: %s", e)

    def _emit_query_log(self, access_record: ForwardAccessRecord) -> None:
        """Write one JSON line to the query log — fires at handler entry.

        Called exactly once per RPC from :meth:`_new_record`, so the line
        hits disk as soon as gRPC dispatches the handler, before any inbound
        body read or backend work. That matches the HTTP frontend's
        ``query_access.log`` contract (``tail -f`` = see arrivals) and keeps
        the two tiers of this forwarder symmetric — without this, the inner
        ``DashScInferenceServicer`` (whose ``for req: yield from ...``
        pattern defers inbound-drain to after generation completes) would
        emit query lines only at RPC end, not at arrival.

        No proto-derived fields (``request_id`` / ``model_name`` /
        ``input_len`` / payload details) — those stay in the completion access log.
        Query log is an arrival breadcrumb only, so ``tail -f`` stays
        readable and the line shape does not depend on first-frame content.
        """
        record = {
            "ts": format_access_log_ts(access_record.start_ts),
            "arrival_ts_epoch_ms": int(access_record.start_ts * 1000),
            "server_id": self._server_id,
            "rank_id": self._rank_id,
            "method": access_record.method,
            "stream_type": access_record.stream_type,
            "peer": access_record.peer,
            "upstream_request_id": access_record.upstream_request_id,
            "upstream_request_id_key": access_record.upstream_request_id_key,
        }
        try:
            self._query_logger.info(orjson.dumps(record).decode("utf-8"))
        except Exception as e:
            logging.warning("[DashScGrpc] query log emit failed: %s", e)

    def _capture_first_request(
        self, access_record: ForwardAccessRecord, request
    ) -> None:
        """Capture first request content for the end-of-RPC access log."""
        access_record.capture_request(request)

    def _wrap_unary_unary(self, inner, method, stream_type):
        def behavior(request, context):
            access_record = self._new_record(method, stream_type, context)
            access_record.req_count = 1
            self._capture_first_request(access_record, request)
            access_record.mark_request_done("unary")
            exc: Optional[BaseException] = None
            try:
                resp = inner(request, context)
                self._on_response_chunk(access_record, resp)
                return resp
            except BaseException as e:
                exc = e
                raise
            finally:
                self._finalize(access_record, context, exc)

        return behavior

    def _wrap_unary_stream(self, inner, method, stream_type):
        def behavior(request, context):
            access_record = self._new_record(method, stream_type, context)
            access_record.req_count = 1
            self._capture_first_request(access_record, request)
            access_record.mark_request_done("unary")
            exc: Optional[BaseException] = None
            try:
                for resp in inner(request, context):
                    self._on_response_chunk(access_record, resp)
                    yield resp
            except BaseException as e:
                exc = e
                raise
            finally:
                self._finalize(access_record, context, exc)

        return behavior

    def _wrap_stream_unary(self, inner, method, stream_type):
        def behavior(request_iterator, context):
            access_record = self._new_record(method, stream_type, context)
            exc: Optional[BaseException] = None

            def counted_reqs():
                first = True
                status = "eof"
                try:
                    for req in request_iterator:
                        access_record.req_count += 1
                        if first:
                            self._capture_first_request(access_record, req)
                            first = False
                        yield req
                except BaseException:
                    status = "error"
                    raise
                finally:
                    access_record.mark_request_done(status)

            try:
                resp = inner(counted_reqs(), context)
                self._on_response_chunk(access_record, resp)
                return resp
            except BaseException as e:
                exc = e
                raise
            finally:
                self._finalize(access_record, context, exc)

        return behavior

    def _wrap_stream_stream(self, inner, method, stream_type):
        def behavior(request_iterator, context):
            access_record = self._new_record(method, stream_type, context)
            exc: Optional[BaseException] = None

            def counted_reqs():
                first = True
                status = "eof"
                try:
                    for req in request_iterator:
                        access_record.req_count += 1
                        if first:
                            self._capture_first_request(access_record, req)
                            first = False
                        yield req
                except BaseException:
                    status = "error"
                    raise
                finally:
                    access_record.mark_request_done(status)

            try:
                for resp in inner(counted_reqs(), context):
                    self._on_response_chunk(access_record, resp)
                    yield resp
            except BaseException as e:
                exc = e
                raise
            finally:
                self._finalize(access_record, context, exc)

        return behavior


class DashScGrpcAccessLogAioInterceptor(
    grpc.aio.ServerInterceptor, DashScGrpcAccessLogInterceptor
):
    """grpc.aio variant of the access-log interceptor.

    Reuses all non-wrap helpers from :class:`DashScGrpcAccessLogInterceptor`
    (``ForwardAccessRecord`` / ``_new_record`` / ``_finalize`` /
    ``_on_response_chunk`` / ``_emit_query_log`` / ``_classify_*`` /
    ``_report_*`` are all pure logic — no ``await`` inside — so the sync class
    is the correct mixin). Only ``intercept_service`` and the four
    ``_wrap_*_aio`` methods are redefined so the wrappers become ``async def``
    coroutines / ``async for``-driven generators.

    There is no thread pool under grpc.aio, so the pool-saturation gauge
    plumbing from the sync path is intentionally absent here.
    """

    async def intercept_service(self, continuation, handler_call_details):
        handler = await continuation(handler_call_details)
        if handler is None:
            return handler

        method = handler_call_details.method
        stream_type = _STREAM_TYPE[
            (handler.request_streaming, handler.response_streaming)
        ]

        if handler.request_streaming and handler.response_streaming:
            return grpc.stream_stream_rpc_method_handler(
                self._wrap_stream_stream_aio(
                    handler.stream_stream, method, stream_type
                ),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        if handler.request_streaming and not handler.response_streaming:
            return grpc.stream_unary_rpc_method_handler(
                self._wrap_stream_unary_aio(handler.stream_unary, method, stream_type),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        if not handler.request_streaming and handler.response_streaming:
            return grpc.unary_stream_rpc_method_handler(
                self._wrap_unary_stream_aio(handler.unary_stream, method, stream_type),
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        return grpc.unary_unary_rpc_method_handler(
            self._wrap_unary_unary_aio(handler.unary_unary, method, stream_type),
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )

    def _wrap_unary_unary_aio(self, inner, method, stream_type):
        async def behavior(request, context):
            access_record = self._new_record(method, stream_type, context)
            access_record.req_count = 1
            self._capture_first_request(access_record, request)
            access_record.mark_request_done("unary")
            exc: Optional[BaseException] = None
            try:
                resp = await inner(request, context)
                self._on_response_chunk(access_record, resp)
                return resp
            except BaseException as e:
                exc = e
                raise
            finally:
                self._finalize(access_record, context, exc)

        return behavior

    def _wrap_unary_stream_aio(self, inner, method, stream_type):
        async def behavior(request, context):
            access_record = self._new_record(method, stream_type, context)
            access_record.req_count = 1
            self._capture_first_request(access_record, request)
            access_record.mark_request_done("unary")
            exc: Optional[BaseException] = None
            try:
                async for resp in inner(request, context):
                    self._on_response_chunk(access_record, resp)
                    yield resp
            except BaseException as e:
                exc = e
                raise
            finally:
                self._finalize(access_record, context, exc)

        return behavior

    def _wrap_stream_unary_aio(self, inner, method, stream_type):
        async def behavior(request_iterator, context):
            access_record = self._new_record(method, stream_type, context)
            exc: Optional[BaseException] = None

            async def counted_reqs():
                first = True
                status = "eof"
                try:
                    async for req in request_iterator:
                        access_record.req_count += 1
                        if first:
                            self._capture_first_request(access_record, req)
                            first = False
                        yield req
                except BaseException:
                    status = "error"
                    raise
                finally:
                    access_record.mark_request_done(status)

            try:
                resp = await inner(counted_reqs(), context)
                self._on_response_chunk(access_record, resp)
                return resp
            except BaseException as e:
                exc = e
                raise
            finally:
                self._finalize(access_record, context, exc)

        return behavior

    def _wrap_stream_stream_aio(self, inner, method, stream_type):
        async def behavior(request_iterator, context):
            access_record = self._new_record(method, stream_type, context)
            exc: Optional[BaseException] = None

            async def counted_reqs():
                first = True
                status = "eof"
                try:
                    async for req in request_iterator:
                        access_record.req_count += 1
                        if first:
                            self._capture_first_request(access_record, req)
                            first = False
                        yield req
                except BaseException:
                    status = "error"
                    raise
                finally:
                    access_record.mark_request_done(status)

            try:
                async for resp in inner(counted_reqs(), context):
                    self._on_response_chunk(access_record, resp)
                    yield resp
            except BaseException as e:
                exc = e
                raise
            finally:
                self._finalize(access_record, context, exc)

        return behavior
