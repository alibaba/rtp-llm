"""gRPC server interceptor that emits one access log line per RPC.

Hooks onto ``grpc.server(..., interceptors=[...])`` so both real and pure-forward
servicers get uniform coverage. Schema is a flat JSON line; fields always present
(null when N/A). Content (input_ids / generated_ids / generate_config) is always
recorded in full — no env switches.

File: ``<log_path>/dash_sc_grpc_access_r{rank}_s{server}.log`` (per-process via
``get_process_log_filename``).

Performance notes (what stays on the RPC worker thread):
- File I/O is async — ``AsyncRotatingFileHandler.emit`` just ``put_nowait`` on a
  bounded queue; a background worker thread does the real write. Queue full →
  drop, never block. So disk latency doesn't affect RPC latency.
- Per chunk: one pass over ``infer.outputs`` (``_scan_response_outputs``) plus
  ``struct.unpack(f"<{n}i", raw)`` for the generated-ids delta. O(chunks × outputs).
- At RPC end: ``orjson.dumps`` of one flat dict (containing full input_ids and
  accumulated generated_ids). For 5k-token payloads this is ~100 µs; logger.info
  then enqueues to the async handler queue in microseconds.
- No orjson default hook needed — all record values are plain int/str/list/dict/None.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any, Optional

import grpc
import orjson

from rtp_llm.access_logger.log_utils import get_handler
from rtp_llm.dash_sc.codec import (
    parse_input_ids_from_request,
    parse_sampling_params,
    unpack_int_tensor_flat,
)
from rtp_llm.metrics import AccMetrics, GaugeMetrics, kmonitor

DASH_SC_GRPC_ACCESS_LOGGER_NAME = "dash_sc_grpc_access_logger"
DASH_SC_GRPC_ACCESS_LOG_FILENAME = "dash_sc_grpc_access.log"

# Protocol tag appended to every kmonitor report emitted from this interceptor, so
# dashboards/alerts can split the gRPC path from the HTTP frontend path that shares
# the same metric names (py_rtp_framework_qps / rt etc.).
_PROTOCOL_TAG = "grpc"

_STREAM_TYPE = {
    (False, False): "unary",
    (False, True): "server_stream",
    (True, False): "client_stream",
    (True, True): "bidi_stream",
}


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


def _declared_element_count(shape) -> int:
    """Return element count from tensor shape, or -1 when shape contains dynamic dims."""
    count = 1
    for d in shape:
        v = int(d)
        if v < 0:
            return -1
        count *= v
    return count


def _scan_response_outputs(
    infer_response,
) -> tuple[Optional[list[int]], Optional[int], Optional[int]]:
    """Single-pass scan: pulls generated_ids / finish_reason / prompt_cached_token_num in one loop.

    Respects the declared tensor ``shape`` — some producers append a 4-byte filler
    for empty ``generated_ids`` (shape=[1, 0]); without this check the accumulator
    would pick up spurious ``0`` tokens. See ``_append_generated_ids_output`` in
    ``codec.py``.
    """
    gen_ids: Optional[list[int]] = None
    finish_reason: Optional[int] = None
    cached: Optional[int] = None
    raw_contents = infer_response.raw_output_contents
    for i, out in enumerate(infer_response.outputs):
        if i >= len(raw_contents):
            continue
        raw = raw_contents[i]
        if not raw:
            continue
        declared = _declared_element_count(out.shape)
        if declared == 0:
            continue  # empty tensor — ignore filler bytes
        name = out.name
        if name == "generated_ids":
            ids = unpack_int_tensor_flat(out.datatype, raw)
            if ids is not None and declared > 0 and len(ids) > declared:
                ids = ids[:declared]  # trim over-sized raw to declared shape
            gen_ids = ids
        elif name == "finish_reason":
            ids = unpack_int_tensor_flat(out.datatype, raw)
            if ids:
                finish_reason = int(ids[0])
        elif name == "prompt_cached_token_num":
            ids = unpack_int_tensor_flat(out.datatype, raw)
            if ids:
                cached = int(ids[0])
    return gen_ids, finish_reason, cached


def _sampling_to_dict(sampling) -> dict[str, Any]:
    d = dataclasses.asdict(sampling)
    # stop_words_list is tuple[tuple[int,...],...]; dataclasses.asdict keeps tuples — convert for JSON
    swl = d.get("stop_words_list")
    if swl is not None:
        d["stop_words_list"] = [list(group) for group in swl]
    return d


@dataclasses.dataclass
class _RpcAggregate:
    method: str
    stream_type: str
    peer: str
    start_ts: float
    first_resp_ts: Optional[float] = None
    last_chunk_ts: Optional[float] = None
    req_count: int = 0
    resp_count: int = 0
    request_id: Optional[str] = None
    model_name: Optional[str] = None
    input_ids: Optional[list[int]] = None
    input_len: Optional[int] = None
    generate_config: Optional[dict[str, Any]] = None
    generated_ids: list[int] = dataclasses.field(default_factory=list)
    finish_reason: Optional[int] = None
    prompt_cached_token_num: Optional[int] = None
    status: str = "OK"
    status_detail: Optional[str] = None

    def capture_request(self, request) -> None:
        """Pull request_id / model_name / input_ids / generate_config from the first request message."""
        if request is None:
            return
        request_id = getattr(request, "id", None)
        if request_id:
            self.request_id = str(request_id)
        model_name = getattr(request, "model_name", None)
        if model_name:
            self.model_name = str(model_name)
        try:
            ids = parse_input_ids_from_request(request)
        except Exception:
            ids = None
        if ids is not None:
            self.input_ids = ids
            self.input_len = len(ids)
        try:
            sampling = parse_sampling_params(request)
            self.generate_config = _sampling_to_dict(sampling)
        except Exception:
            self.generate_config = None

    def capture_response_chunk(self, resp) -> None:
        """Extract generated_ids delta + optional aux fields from one streamed response message.

        Single-pass scan over ``infer.outputs`` — O(outputs) per chunk, not O(3·outputs).
        """
        if resp is None:
            return
        infer = getattr(resp, "infer_response", None)
        if infer is None:
            return
        delta, fr, cached = _scan_response_outputs(infer)
        if delta:
            self.generated_ids.extend(delta)
        if fr is not None:
            self.finish_reason = fr
        if cached is not None and self.prompt_cached_token_num is None:
            self.prompt_cached_token_num = cached

    def mark_first_resp(self) -> None:
        if self.first_resp_ts is None:
            self.first_resp_ts = time.time()

    def build_record(
        self, server_id: Optional[int], rank_id: Optional[int]
    ) -> dict[str, Any]:
        """Build the final log dict in display order. Called once per RPC in ``_finalize``."""
        end_ts = time.time()
        total_ms = (end_ts - self.start_ts) * 1000.0
        ttfb_ms = None
        if self.first_resp_ts is not None:
            ttfb_ms = (self.first_resp_ts - self.start_ts) * 1000.0
        local = time.localtime(end_ts)
        tz_off = -time.timezone if not local.tm_isdst else -time.altzone
        sign = "+" if tz_off >= 0 else "-"
        tz_off = abs(tz_off)
        tz_str = f"{sign}{tz_off // 3600:02d}:{(tz_off % 3600) // 60:02d}"
        ts = (
            time.strftime("%Y-%m-%dT%H:%M:%S", local)
            + f".{int((end_ts % 1) * 1000):03d}"
            + tz_str
        )
        return {
            "ts": ts,
            "ts_epoch_ms": int(end_ts * 1000),
            "server_id": server_id,
            "rank_id": rank_id,
            "method": self.method,
            "stream_type": self.stream_type,
            "peer": self.peer,
            "req_count": self.req_count,
            "resp_count": self.resp_count,
            "latency_total_ms": round(total_ms, 3),
            "latency_ttfb_ms": round(ttfb_ms, 3) if ttfb_ms is not None else None,
            "status": self.status,
            "status_detail": self.status_detail,
            "request_id": self.request_id,
            "model_name": self.model_name,
            "input_len": self.input_len,
            "input_ids": self.input_ids,
            "generate_config": self.generate_config,
            "output_len": len(self.generated_ids),
            "generated_ids": self.generated_ids if self.generated_ids else None,
            "finish_reason": self.finish_reason,
            "prompt_cached_token_num": self.prompt_cached_token_num,
        }


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
    """

    def __init__(
        self,
        rank_id: Optional[int] = None,
        server_id: Optional[int] = None,
    ) -> None:
        self._logger = logging.getLogger(DASH_SC_GRPC_ACCESS_LOGGER_NAME)
        self._rank_id = rank_id
        self._server_id = server_id
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

    def _report_request_arrived(self, agg: "_RpcAggregate") -> None:
        """Mirror ``FrontendServer.embedding`` entry-point QPS_METRIC."""
        kmonitor.report(AccMetrics.QPS_METRIC, 1, self._tags(agg.method))

    def _report_chunk(self, agg: "_RpcAggregate", *, is_first: bool) -> None:
        """Mirror ``_call_generate_with_report`` per-chunk metrics.

        ``is_first`` picks FIRST_TOKEN_RT vs ITER_RT/ITER_QPS — same split as the HTTP path.
        """
        now = time.time()
        tags = self._tags(agg.method)
        if is_first:
            ttfb_ms = (now - agg.start_ts) * 1000.0
            kmonitor.report(GaugeMetrics.RESPONSE_FIRST_TOKEN_RT_METRIC, ttfb_ms, tags)
        else:
            last = agg.last_chunk_ts or agg.first_resp_ts or agg.start_ts
            iter_rt_ms = (now - last) * 1000.0
            kmonitor.report(GaugeMetrics.RESPONSE_ITER_RT_METRIC, iter_rt_ms, tags)
        kmonitor.report(AccMetrics.ITER_QPS_METRIC, 1, tags)
        agg.last_chunk_ts = now

    def _report_rpc_done(
        self, agg: "_RpcAggregate", *, status: str, status_detail: Optional[str]
    ) -> None:
        """Mirror ``_call_generate_with_report`` tail metrics + HTTP success/error QPS."""
        total_ms = (time.time() - agg.start_ts) * 1000.0
        tags = self._tags(agg.method)
        kmonitor.report(GaugeMetrics.LANTENCY_METRIC, total_ms, tags)
        kmonitor.report(GaugeMetrics.RESPONSE_ITERATE_COUNT, agg.resp_count, tags)
        if agg.input_len is not None:
            kmonitor.report(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC, agg.input_len, tags)
        kmonitor.report(
            GaugeMetrics.OUTPUT_TOKEN_SIZE_METRIC, len(agg.generated_ids), tags
        )
        if status == "OK":
            kmonitor.report(AccMetrics.SUCCESS_QPS_METRIC, 1, tags)
        elif status == "CANCELLED":
            kmonitor.report(AccMetrics.CANCEL_QPS_METRIC, 1, tags)
        else:
            kmonitor.report(
                AccMetrics.ERROR_QPS_METRIC,
                1,
                self._tags(agg.method, error_code=status),
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

    def _new_aggregate(self, method: str, stream_type: str, context) -> _RpcAggregate:
        try:
            peer = context.peer()
        except Exception:
            peer = ""
        return _RpcAggregate(
            method=method,
            stream_type=stream_type,
            peer=peer,
            start_ts=time.time(),
        )

    def _on_response_chunk(self, agg: _RpcAggregate, resp) -> None:
        """Unified per-chunk hook: count, capture content, emit first-token / iter metrics."""
        is_first = agg.first_resp_ts is None
        agg.mark_first_resp()
        agg.resp_count += 1
        agg.capture_response_chunk(resp)
        self._report_chunk(agg, is_first=is_first)

    def _finalize(
        self, agg: _RpcAggregate, context, exc: Optional[BaseException]
    ) -> None:
        if exc is not None:
            agg.status = "UNKNOWN"
            agg.status_detail = repr(exc)
        else:
            try:
                code = context.code()
            except Exception:
                code = None
            if code is None or code == grpc.StatusCode.OK:
                agg.status = "OK"
                agg.status_detail = None
            else:
                agg.status = code.name
                try:
                    agg.status_detail = context.details() or code.name
                except Exception:
                    agg.status_detail = code.name

        self._report_rpc_done(agg, status=agg.status, status_detail=agg.status_detail)

        record = agg.build_record(self._server_id, self._rank_id)
        try:
            # orjson: ~5-10x faster than stdlib json.dumps on token-list payloads,
            # and it emits bytes — decode once for the logging framework.
            self._logger.info(orjson.dumps(record).decode("utf-8"))
        except Exception as e:
            logging.warning("[DashScGrpc] access log emit failed: %s", e)

    def _capture_first_request(self, agg: _RpcAggregate, request) -> None:
        """Capture request content and fire the arrival QPS metric (exactly once per RPC)."""
        agg.capture_request(request)
        self._report_request_arrived(agg)

    def _wrap_unary_unary(self, inner, method, stream_type):
        def behavior(request, context):
            agg = self._new_aggregate(method, stream_type, context)
            agg.req_count = 1
            self._capture_first_request(agg, request)
            exc: Optional[BaseException] = None
            try:
                resp = inner(request, context)
                self._on_response_chunk(agg, resp)
                return resp
            except BaseException as e:
                exc = e
                raise
            finally:
                self._finalize(agg, context, exc)

        return behavior

    def _wrap_unary_stream(self, inner, method, stream_type):
        def behavior(request, context):
            agg = self._new_aggregate(method, stream_type, context)
            agg.req_count = 1
            self._capture_first_request(agg, request)
            exc: Optional[BaseException] = None
            try:
                for resp in inner(request, context):
                    self._on_response_chunk(agg, resp)
                    yield resp
            except BaseException as e:
                exc = e
                raise
            finally:
                self._finalize(agg, context, exc)

        return behavior

    def _wrap_stream_unary(self, inner, method, stream_type):
        def behavior(request_iterator, context):
            agg = self._new_aggregate(method, stream_type, context)
            exc: Optional[BaseException] = None

            def counted_reqs():
                first = True
                for req in request_iterator:
                    agg.req_count += 1
                    if first:
                        self._capture_first_request(agg, req)
                        first = False
                    yield req

            try:
                resp = inner(counted_reqs(), context)
                self._on_response_chunk(agg, resp)
                return resp
            except BaseException as e:
                exc = e
                raise
            finally:
                self._finalize(agg, context, exc)

        return behavior

    def _wrap_stream_stream(self, inner, method, stream_type):
        def behavior(request_iterator, context):
            agg = self._new_aggregate(method, stream_type, context)
            exc: Optional[BaseException] = None

            def counted_reqs():
                first = True
                for req in request_iterator:
                    agg.req_count += 1
                    if first:
                        self._capture_first_request(agg, req)
                        first = False
                    yield req

            try:
                for resp in inner(counted_reqs(), context):
                    self._on_response_chunk(agg, resp)
                    yield resp
            except BaseException as e:
                exc = e
                raise
            finally:
                self._finalize(agg, context, exc)

        return behavior
