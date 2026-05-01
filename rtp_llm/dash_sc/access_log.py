"""gRPC server interceptor that emits one access log line per RPC.

Hooks onto ``grpc.server(..., interceptors=[...])`` so both real and pure-forward
servicers get uniform coverage. Schema is a flat JSON line; fields always present
(null when N/A).

Two capture modes share the same interceptor + log file:

- **Struct mode** (default, real servicer): pulls ``input_ids`` / ``generated_ids``
  / ``generate_config`` etc. off the proto so the log is immediately human-readable.

- **Raw mode** (forward servicer, ``raw_mode=True``): the forwarder is a
  transparent proxy, so by the time we need to debug a weird downstream response
  (``req_count=0`` from the real node, or mismatched generation) the only source
  of truth about what actually hit the wire is the full proto. In raw mode we
  dump each ``ModelInferRequest`` / ``ModelStreamInferResponse`` as a dict
  (``MessageToDict``) with ``raw_input_contents`` / ``raw_output_contents``
  decoded from base64 back to int/float/bool lists per Triton datatype — so
  eyeballing a log line is enough, no protobuf decode step. Content is capped
  per-tensor (``_RAW_MAX_DECODED_ELEMENTS``) and per-RPC (``_RAW_MAX_CHUNKS``)
  because LLM streams easily run multi-MB per RPC.

File: ``<log_path>/dash_sc_grpc_access_r{rank}_s{server}.log`` (per-process via
``get_process_log_filename``).

Performance notes (what stays on the RPC worker thread):
- File I/O is async — ``AsyncRotatingFileHandler.emit`` just ``put_nowait`` on a
  bounded queue; a background worker thread does the real write. Queue full →
  drop, never block. So disk latency doesn't affect RPC latency.
- Struct mode per chunk: one pass over ``infer.outputs`` (``_scan_response_outputs``)
  plus ``struct.unpack(f"<{n}i", raw)`` for the generated-ids delta.
- Raw mode per chunk: one ``MessageToDict`` + tensor base64→list rewrite. Heavier
  than struct mode, which is why it's opt-in (only on forward servicers).
- At RPC end: ``orjson.dumps`` of one flat dict; logger.info then enqueues to the
  async handler queue in microseconds.
"""

from __future__ import annotations

import base64
import dataclasses
import logging
import struct
import time
from typing import Any, Optional

import grpc
import orjson
from google.protobuf.json_format import MessageToDict

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

# Raw-mode caps. LLM streams can span hundreds of chunks and a single input_ids
# tensor can be tens of thousands of tokens; without these we'd dump multi-MB
# lines and defeat the "readable JSON" goal.
_RAW_MAX_CHUNKS = 512
_RAW_MAX_DECODED_ELEMENTS = 16384


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


def _decode_raw_tensor(datatype: str, raw: bytes) -> Optional[list]:
    """Bulk-unpack a little-endian tensor buffer based on Triton ``datatype`` name.

    Covers the datatype family actually observed in ``predict_v2.proto`` traffic
    (integer / float / bool). Returns ``None`` for unsupported types or misaligned
    buffers; the caller keeps the base64 blob plus a ``decoded_error`` note so
    nothing gets silently dropped.
    """
    if not raw:
        return []
    try:
        if datatype == "INT32":
            if len(raw) & 3:
                return None
            n = len(raw) >> 2
            return list(struct.unpack(f"<{n}i", raw))
        if datatype == "INT64":
            if len(raw) & 7:
                return None
            n = len(raw) >> 3
            return [int(x) for x in struct.unpack(f"<{n}q", raw)]
        if datatype == "UINT32":
            if len(raw) & 3:
                return None
            n = len(raw) >> 2
            return list(struct.unpack(f"<{n}I", raw))
        if datatype == "UINT64":
            if len(raw) & 7:
                return None
            n = len(raw) >> 3
            return [int(x) for x in struct.unpack(f"<{n}Q", raw)]
        if datatype == "FP32":
            if len(raw) & 3:
                return None
            n = len(raw) >> 2
            return list(struct.unpack(f"<{n}f", raw))
        if datatype == "FP64":
            if len(raw) & 7:
                return None
            n = len(raw) >> 3
            return list(struct.unpack(f"<{n}d", raw))
        if datatype == "INT8":
            return list(struct.unpack(f"<{len(raw)}b", raw))
        if datatype == "UINT8":
            return list(struct.unpack(f"<{len(raw)}B", raw))
        if datatype == "BOOL":
            return [bool(b) for b in raw]
    except struct.error:
        return None
    return None


def _rewrite_raw_contents(msg_dict: dict, tensors_key: str, raw_key: str) -> None:
    """Decode ``raw_*_contents`` (base64 after MessageToDict) onto the matching tensor entries.

    Positional correspondence: ``raw_input_contents[i]`` belongs to ``inputs[i]``. We
    attach ``decoded`` (or ``raw_b64`` + ``decoded_error``) to each tensor dict, then drop
    the top-level ``raw_*_contents`` key so the JSON stays flat.
    """
    tensors = msg_dict.get(tensors_key) or []
    raws_b64 = msg_dict.get(raw_key) or []
    if not raws_b64:
        return
    for i, b64 in enumerate(raws_b64):
        if i >= len(tensors):
            continue
        t = tensors[i]
        if not isinstance(t, dict):
            continue
        try:
            raw = base64.b64decode(b64) if isinstance(b64, str) else bytes(b64)
        except Exception as e:
            t["raw_b64"] = b64
            t["decoded_error"] = f"base64_decode_failed: {e!r}"
            continue
        t["size_bytes"] = len(raw)
        dt = t.get("datatype") or ""
        decoded = _decode_raw_tensor(dt, raw)
        if decoded is None:
            t["raw_b64"] = b64
            t["decoded_error"] = f"unsupported_or_misaligned_datatype={dt!r}"
        elif len(decoded) > _RAW_MAX_DECODED_ELEMENTS:
            t["decoded"] = decoded[:_RAW_MAX_DECODED_ELEMENTS]
            t["decoded_truncated"] = True
            t["decoded_total"] = len(decoded)
        else:
            t["decoded"] = decoded
    msg_dict.pop(raw_key, None)


def _proto_to_jsonable(msg: Any) -> dict:
    """Convert a proto to a flat dict with tensor payloads decoded. Never raises.

    On conversion failure returns ``{"_convert_error": repr(e)}`` so one malformed
    message cannot drop an entire RPC's log line.
    """
    try:
        d = MessageToDict(
            msg,
            preserving_proto_field_name=True,
            including_default_value_fields=False,
            use_integers_for_enums=False,
        )
    except Exception as e:
        return {"_convert_error": repr(e)}
    _rewrite_raw_contents(d, "inputs", "raw_input_contents")
    infer = d.get("infer_response")
    if isinstance(infer, dict):
        _rewrite_raw_contents(infer, "outputs", "raw_output_contents")
    return d


@dataclasses.dataclass
class _RpcAggregate:
    method: str
    stream_type: str
    peer: str
    start_ts: float
    raw_mode: bool = False
    first_resp_ts: Optional[float] = None
    last_chunk_ts: Optional[float] = None
    req_count: int = 0
    resp_count: int = 0
    request_id: Optional[str] = None
    model_name: Optional[str] = None
    # Struct-mode content (real frontend path).
    input_ids: Optional[list[int]] = None
    input_len: Optional[int] = None
    generate_config: Optional[dict[str, Any]] = None
    generated_ids: list[int] = dataclasses.field(default_factory=list)
    finish_reason: Optional[int] = None
    prompt_cached_token_num: Optional[int] = None
    # Raw-mode content (forward path): full proto dumps of every request / response
    # message, with ``raw_*_contents`` decoded back from base64. Kept capped so an
    # extreme stream cannot blow up a single log line.
    raw_requests: list[dict] = dataclasses.field(default_factory=list)
    raw_responses: list[dict] = dataclasses.field(default_factory=list)
    raw_requests_truncated: bool = False
    raw_responses_truncated: bool = False
    status: str = "OK"
    status_detail: Optional[str] = None
    # Exception-path diagnostics. ``repr(grpc.RpcError())`` collapses to the
    # string "RpcError()" and drops the concrete subclass + grpc status code,
    # which makes triage of peer-cancel vs real error impossible from the log
    # alone. These three fields preserve what context and the interpreter know.
    exc_type: Optional[str] = None
    context_code: Optional[str] = None
    context_active: Optional[bool] = None
    # Forward-path diagnostics (populated by :class:`PureForwardServicer` via
    # ``context._dash_sc_access_agg``). Consolidates signal that otherwise lives
    # in the forwarder's separate debug log — specifically the ones needed to
    # distinguish "downstream never produced anything" from "downstream produced
    # a token but the buffer swallowed it" on a ``resp_count=0`` line without
    # cross-grepping a second file.
    #
    # ``downstream_addr``: which backend this RPC was routed to (random pick
    #     across ``DASH_SC_GRPC_FORWARD_ADDR``). Without this an operator has
    #     no way to correlate ``resp_count=0`` clusters with a sick backend.
    # ``downstream_resp_count``: how many frames the forwarder's stub actually
    #     read back from the backend. On ``resp_count=0`` lines the gap between
    #     this (>0) and ``resp_count`` (0) localises the fault to the local
    #     ``_buffered_iter`` / stream-wrap path; matching zeros localise it to
    #     the backend or the LBS below it.
    # ``buffered_stage``: final state of ``_buffered_iter`` when the RPC ended.
    #     Values: ``waiting_first`` (stuck on ``next(it)`` #1, no frame seen),
    #     ``waiting_second`` (token 1 buffered, stuck on ``next(it)`` #2),
    #     ``flushed_first`` / ``flushed_both`` (normal flush paths),
    #     ``flushed_first_on_exception`` (exception after buffering, buffered
    #     frame still made it to the wire), ``dropped_buffered_on_exception``
    #     (client gone too — buffered frame was lost).
    downstream_addr: Optional[str] = None
    downstream_resp_count: int = 0
    buffered_stage: Optional[str] = None

    def capture_request(self, request) -> None:
        """Capture a request message.

        Both modes cheaply extract ``id`` / ``model_name`` for correlation. Struct
        mode additionally parses ``input_ids`` / sampling params; raw mode appends
        the full proto dict up to the cap.
        """
        if request is None:
            return
        request_id = getattr(request, "id", None)
        if request_id and self.request_id is None:
            self.request_id = str(request_id)
        model_name = getattr(request, "model_name", None)
        if model_name and self.model_name is None:
            self.model_name = str(model_name)
        if self.raw_mode:
            if len(self.raw_requests) < _RAW_MAX_CHUNKS:
                self.raw_requests.append(_proto_to_jsonable(request))
            else:
                self.raw_requests_truncated = True
            return
        # Struct-mode parsed content — only on the first request message.
        if self.input_ids is None:
            try:
                ids = parse_input_ids_from_request(request)
            except Exception:
                ids = None
            if ids is not None:
                self.input_ids = ids
                self.input_len = len(ids)
        if self.generate_config is None:
            try:
                sampling = parse_sampling_params(request)
                self.generate_config = _sampling_to_dict(sampling)
            except Exception:
                self.generate_config = None

    def capture_response_chunk(self, resp) -> None:
        """Extract per-chunk content from one streamed response message.

        Struct mode: single-pass scan over ``infer.outputs`` — O(outputs) per chunk.
        Raw mode: one ``MessageToDict`` + tensor rewrite; capped by ``_RAW_MAX_CHUNKS``.
        """
        if resp is None:
            return
        if self.raw_mode:
            if len(self.raw_responses) < _RAW_MAX_CHUNKS:
                self.raw_responses.append(_proto_to_jsonable(resp))
            else:
                self.raw_responses_truncated = True
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
        """Build the final log dict. Called once per RPC in ``_finalize``.

        The record carries both struct and raw fields — in practice only one set
        is populated per RPC (the other is null / empty). One flat shape keeps the
        downstream jq-grep workflow trivial regardless of which servicer emitted it.
        """
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
        record: dict[str, Any] = {
            "ts": ts,
            "ts_epoch_ms": int(end_ts * 1000),
            "server_id": server_id,
            "rank_id": rank_id,
            "method": self.method,
            "stream_type": self.stream_type,
            "capture_mode": "raw" if self.raw_mode else "struct",
            "peer": self.peer,
            "req_count": self.req_count,
            "resp_count": self.resp_count,
            "latency_total_ms": round(total_ms, 3),
            "latency_ttfb_ms": round(ttfb_ms, 3) if ttfb_ms is not None else None,
            "status": self.status,
            "status_detail": self.status_detail,
            "exc_type": self.exc_type,
            "context_code": self.context_code,
            "context_active": self.context_active,
            "downstream_addr": self.downstream_addr,
            "downstream_resp_count": self.downstream_resp_count,
            "buffered_stage": self.buffered_stage,
            "request_id": self.request_id,
            "model_name": self.model_name,
        }
        if self.raw_mode:
            record["raw_requests"] = self.raw_requests
            record["raw_responses"] = self.raw_responses
            record["raw_requests_truncated"] = self.raw_requests_truncated
            record["raw_responses_truncated"] = self.raw_responses_truncated
        else:
            record["input_len"] = self.input_len
            record["input_ids"] = self.input_ids
            record["generate_config"] = self.generate_config
            record["output_len"] = len(self.generated_ids)
            record["generated_ids"] = self.generated_ids if self.generated_ids else None
            record["finish_reason"] = self.finish_reason
            record["prompt_cached_token_num"] = self.prompt_cached_token_num
        return record


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

    ``raw_mode`` flips the content capture strategy: struct (parsed fields) for the real
    servicer, raw (full proto dumps with decoded tensors) for the forward servicer. Both
    modes share this interceptor and the same log file — one line per RPC, one schema to
    grep through, differentiated by the ``capture_mode`` field.
    """

    def __init__(
        self,
        rank_id: Optional[int] = None,
        server_id: Optional[int] = None,
        raw_mode: bool = False,
    ) -> None:
        self._logger = logging.getLogger(DASH_SC_GRPC_ACCESS_LOGGER_NAME)
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
        agg = _RpcAggregate(
            method=method,
            stream_type=stream_type,
            peer=peer,
            start_ts=time.time(),
            raw_mode=self._raw_mode,
        )
        # Make the aggregate reachable from downstream servicer code via the
        # gRPC context object. :class:`PureForwardServicer` reads this back to
        # write ``downstream_addr`` / ``downstream_resp_count`` / ``buffered_stage``
        # directly, so one access-log line is self-contained — no side log to
        # cross-reference when debugging a ``resp_count=0`` event.
        try:
            context._dash_sc_access_agg = agg
        except Exception:
            pass
        return agg

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
        try:
            code = context.code()
        except Exception:
            code = None
        if code is not None:
            try:
                agg.context_code = code.name
            except Exception:
                agg.context_code = str(code)
        try:
            agg.context_active = bool(context.is_active())
        except Exception:
            agg.context_active = None

        if exc is not None:
            agg.exc_type = type(exc).__name__
            agg.status = "UNKNOWN"
            agg.status_detail = repr(exc)
        elif code is None or code == grpc.StatusCode.OK:
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
                    else:
                        # Raw mode needs every request message; struct mode already
                        # captured everything it needs from the first request.
                        if agg.raw_mode:
                            agg.capture_request(req)
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
                    else:
                        if agg.raw_mode:
                            agg.capture_request(req)
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
