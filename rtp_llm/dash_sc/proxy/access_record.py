"""Forward/proxy access-log record and frame statistics.

This module intentionally has no logger, interceptor, or kmonitor dependencies.
It owns the pure data path for dash-sc access logs:

- request-derived fields such as request id, model name, input length, and
  generation controls;
- response-frame statistics such as output token count, finish markers, and
  prompt/backend counters;
- proxy/backend diagnostics and derived latency fields;
- final flat JSON schema construction.

Keeping this separate from ``access_log.py`` leaves the interceptor responsible
only for lifecycle orchestration: when to capture, report metrics, and emit log
lines.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Optional

from rtp_llm.dash_sc.codec import (
    parse_ds_header_attributes,
    parse_input_ids_from_request,
    parse_other_params,
    parse_sampling_params,
    unpack_int_tensor_flat,
)
from rtp_llm.dash_sc.repetition_monitor import (
    RequestRepetitionMonitor,
    RequestRepetitionMonitorConfig,
)

DASH_SC_GRPC_PROTOCOL = "grpc"

_FINISH_REASON_NOT_FINISHED = 2


def rpc_code(exc: BaseException) -> Optional[Any]:
    try:
        return exc.code()
    except Exception:
        return None


def rpc_details(exc: BaseException) -> Optional[str]:
    try:
        return exc.details()
    except Exception:
        return None


def epoch_ms(ts: Optional[float]) -> Optional[int]:
    return int(ts * 1000) if ts is not None else None


def duration_ms(start: Optional[float], end: Optional[float]) -> Optional[float]:
    if start is None or end is None:
        return None
    return round((end - start) * 1000.0, 3)


def format_access_log_ts(ts: float) -> str:
    """Local ISO-8601 timestamp with millisecond precision and numeric TZ offset."""
    local = time.localtime(ts)
    tz_off = -time.timezone if not local.tm_isdst else -time.altzone
    sign = "+" if tz_off >= 0 else "-"
    tz_off = abs(tz_off)
    tz_str = f"{sign}{tz_off // 3600:02d}:{(tz_off % 3600) // 60:02d}"
    return (
        time.strftime("%Y-%m-%dT%H:%M:%S", local)
        + f".{int((ts % 1) * 1000):03d}"
        + tz_str
    )


def _declared_element_count(shape) -> int:
    """Return element count from tensor shape, or -1 when shape contains dynamic dims."""
    count = 1
    for d in shape:
        v = int(d)
        if v < 0:
            return -1
        count *= v
    return count


@dataclasses.dataclass(frozen=True)
class ForwardFrameStats:
    """Statistics extracted from one forwarded response frame."""

    generated_token_len: int = 0
    finish_reason: Optional[int] = None
    finished: Optional[bool] = None
    prompt_token_num: Optional[int] = None
    prompt_cached_token_num: Optional[int] = None

    @property
    def is_terminal(self) -> bool:
        if self.finished is True:
            return True
        return (
            self.finish_reason is not None
            and self.finish_reason != _FINISH_REASON_NOT_FINISHED
        )


def extract_forward_frame_stats(infer_response) -> ForwardFrameStats:
    """Single-pass extraction of response-frame statistics for access logs.

    Respects the declared tensor ``shape`` because some producers append a
    4-byte filler for empty ``generated_ids`` (shape=[1, 0]). Without this check
    the accumulator would pick up a spurious output token.
    """
    generated_token_len = 0
    finish_reason: Optional[int] = None
    finished: Optional[bool] = None
    prompt_token_num: Optional[int] = None
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
            continue
        name = out.name
        if name == "generated_ids":
            if declared > 0:
                generated_token_len = declared
            elif out.datatype == "INT32" and len(raw) >= 4:
                generated_token_len = len(raw) // 4
        elif name == "finish_reason":
            ids = unpack_int_tensor_flat(out.datatype, raw)
            if ids:
                finish_reason = int(ids[0])
        elif name == "finished":
            if out.datatype == "BOOL":
                finished = bool(raw[0])
            else:
                ids = unpack_int_tensor_flat(out.datatype, raw)
                if ids:
                    finished = bool(ids[0])
        elif name == "prompt_token_num":
            ids = unpack_int_tensor_flat(out.datatype, raw)
            if ids:
                prompt_token_num = int(ids[0])
        elif name == "prompt_cached_token_num":
            ids = unpack_int_tensor_flat(out.datatype, raw)
            if ids:
                cached = int(ids[0])
    return ForwardFrameStats(
        generated_token_len=generated_token_len,
        finish_reason=finish_reason,
        finished=finished,
        prompt_token_num=prompt_token_num,
        prompt_cached_token_num=cached,
    )


def _sampling_to_dict(sampling) -> dict[str, Any]:
    d = dataclasses.asdict(sampling)
    # stop_words_list is tuple[tuple[int,...],...]; convert for stable JSON.
    swl = d.get("stop_words_list")
    if swl is not None:
        d["stop_words_list"] = [list(group) for group in swl]
    return d


_CONTEXT_ATTR = "_dash_sc_forward_access_record"


@dataclasses.dataclass
class ForwardAccessRecord:
    """Mutable per-RPC access record for the dash-sc forward path."""

    method: str
    stream_type: str
    peer: str
    start_ts: float
    raw_mode: bool = False
    first_request_ts: Optional[float] = None
    request_end_ts: Optional[float] = None
    request_read_status: Optional[str] = None
    first_resp_ts: Optional[float] = None
    last_chunk_ts: Optional[float] = None
    first_token_ts: Optional[float] = None
    last_token_ts: Optional[float] = None
    terminal_ts: Optional[float] = None
    req_count: int = 0
    resp_count: int = 0
    request_id: Optional[str] = None
    model_name: Optional[str] = None
    # Upstream correlation ID captured from request metadata. Independent of
    # the proto ``id`` field, which becomes ``request_id``.
    upstream_request_id: Optional[str] = None
    upstream_request_id_key: Optional[str] = None
    # Parsed request / response statistics.
    input_len: Optional[int] = None
    generate_config: Optional[dict[str, Any]] = None
    output_len: int = 0
    first_token_frame_len: int = 0
    token_frame_count: int = 0
    empty_frame_count: int = 0
    finished_only_frame_count: int = 0
    multi_token_frame_count: int = 0
    max_tokens_per_frame: int = 0
    backend_input_len: Optional[int] = None
    finish_reason: Optional[int] = None
    finished: Optional[bool] = None
    terminal_seen: bool = False
    prompt_cached_token_num: Optional[int] = None
    # Protocol-level backend error channel (predict_v2.proto: empty means
    # success). Kept here so a successful gRPC status can still be classified as
    # a backend failure when the payload says so.
    error_message: Optional[str] = None
    # Optional structured backend code propagated by the real dash-sc servicer.
    # When present it keeps framework ERROR_QPS aligned with HTTP frontend tags
    # such as "8400_MASTER_NO_AVAILABLE_WORKER"; otherwise we fall back to the
    # bounded protocol/transport buckets below.
    backend_error_code: Optional[str] = None
    status: str = "OK"
    status_detail: Optional[str] = None
    # Exception-path diagnostics.
    exc_type: Optional[str] = None
    context_code: Optional[str] = None
    context_active: Optional[bool] = None
    # Proxy/backend diagnostics populated by DashScProxyServicer through the
    # context helpers in ``rtp_llm.dash_sc.proxy.context``.
    backend_addr: Optional[str] = None
    backend_call_start_ts: Optional[float] = None
    backend_first_resp_ts: Optional[float] = None
    backend_done_ts: Optional[float] = None
    backend_terminal_ts: Optional[float] = None
    backend_exc_type: Optional[str] = None
    backend_rpc_code: Optional[str] = None
    backend_rpc_detail: Optional[str] = None
    backend_resp_count: int = 0
    buffered_stage: Optional[str] = None
    _pending_backend_stats: list = dataclasses.field(default_factory=list)
    # Tool-call loop observability. The monitor owns detection and the flat
    # access-log fields; this record only feeds it request markers + the
    # generated token stream and reads back the result.
    repetition_monitor_config: RequestRepetitionMonitorConfig = dataclasses.field(
        default_factory=RequestRepetitionMonitorConfig, repr=False
    )
    _repetition_monitor: RequestRepetitionMonitor = dataclasses.field(
        init=False, repr=False
    )
    # Transient per-RPC ``generated_ids`` byte buffer, populated only when a
    # request declares tool-call markers. Frames carry token deltas, so raw bytes
    # are concatenated here (a C-level memcpy) and decoded exactly once at
    # end-of-RPC — never a per-frame Python unpack on the streaming hot path.
    # Marker-less requests stay ``None``, keeping the no-raw-payload contract.
    _tool_loop_raw: Optional[bytearray] = dataclasses.field(default=None, repr=False)
    _tool_loop_datatype: Optional[str] = dataclasses.field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._repetition_monitor = RequestRepetitionMonitor(
            raw_mode=self.raw_mode,
            monitor_config=self.repetition_monitor_config,
        )
        # Markers are a service-level constant, so whether we collect the token
        # stream is known at construction time. Only allocate the buffer when
        # detection will actually run, keeping the marker-less path zero-cost.
        if self._repetition_monitor.is_active():
            self._tool_loop_raw = bytearray()

    def check_repetition(self) -> None:
        ids = ()
        if self._tool_loop_raw:
            ids = unpack_int_tensor_flat(
                self._tool_loop_datatype, bytes(self._tool_loop_raw)
            )
        self._repetition_monitor.check_generated_ids(ids)

    @property
    def repetition_monitor(self) -> RequestRepetitionMonitor:
        """The request-scoped monitor — the single source of truth for tool-call
        loop detection results, log fields, and kmonitor reports.

        Exposed read-only so the interceptor can ask the monitor for its metric
        report plan directly instead of routing through a passthrough here. This
        record stays free of any kmonitor dependency: it only hands out the
        collaborator it already owns; the monitor does the metric work.
        """
        return self._repetition_monitor

    def capture_request(self, request) -> None:
        """Capture stable request statistics without retaining raw payloads."""
        if request is None:
            return
        if self.first_request_ts is None:
            self.first_request_ts = time.time()
        request_id = getattr(request, "id", None)
        if request_id and self.request_id is None:
            self.request_id = str(request_id)
        model_name = getattr(request, "model_name", None)
        if model_name and self.model_name is None:
            self.model_name = str(model_name)
        parsed_ids = None
        if self.input_len is None:
            try:
                parsed_ids = parse_input_ids_from_request(request)
            except Exception:
                parsed_ids = None
            if parsed_ids is not None:
                self.input_len = len(parsed_ids)
                # Feed the request input to the monitor (not the log) so the
                # detector can anchor tool-call spans; honors the no-raw-payload
                # contract because these ids never reach the access record.
                if self._tool_loop_raw is not None:
                    self._repetition_monitor.set_input_ids(parsed_ids)
        if self.generate_config is None:
            try:
                ds_attrs = parse_ds_header_attributes(request)
                sampling = parse_sampling_params(request, ds_attrs)
                self.generate_config = _sampling_to_dict(sampling)
                other = parse_other_params(request, ds_attrs)
                self.generate_config["enable_thinking"] = other.enable_thinking
                self.generate_config["max_new_think_tokens"] = (
                    other.max_new_think_tokens
                )
                self.generate_config["reasoning_effort"] = other.reasoning_effort
                self.generate_config["timeout_ms"] = other.timeout_ms
                self.generate_config["traffic_reject_priority"] = (
                    other.traffic_reject_priority
                )
            except Exception:
                self.generate_config = None

    def mark_request_done(self, status: str) -> None:
        self.request_end_ts = time.time()
        self.request_read_status = status

    def _absorb_shared_fields(self, stats: ForwardFrameStats) -> None:
        """Write-once absorption of fields originating from the backend."""
        if stats.prompt_token_num is not None and self.backend_input_len is None:
            self.backend_input_len = stats.prompt_token_num
        if (
            stats.prompt_cached_token_num is not None
            and self.prompt_cached_token_num is None
        ):
            self.prompt_cached_token_num = stats.prompt_cached_token_num
        if stats.finish_reason is not None:
            self.finish_reason = stats.finish_reason
        if stats.finished is not None:
            self.finished = stats.finished

    def _accumulate_client_frame(self, stats: ForwardFrameStats, now: float) -> None:
        """Update client-visible token counters and timing."""
        self._absorb_shared_fields(stats)
        delta_len = stats.generated_token_len
        if delta_len:
            self.output_len += delta_len
            self.token_frame_count += 1
            self.max_tokens_per_frame = max(self.max_tokens_per_frame, delta_len)
            if delta_len > 1:
                self.multi_token_frame_count += 1
            if self.first_token_ts is None:
                self.first_token_ts = now
                self.first_token_frame_len = delta_len
            self.last_token_ts = now
        else:
            self.empty_frame_count += 1
        if stats.is_terminal and not self.terminal_seen:
            self.terminal_seen = True
            self.terminal_ts = now
            if delta_len == 0:
                self.finished_only_frame_count += 1

    def _accumulate_backend_frame(self, stats: ForwardFrameStats, now: float) -> None:
        """Update backend-side terminal timestamp."""
        self._absorb_shared_fields(stats)
        if stats.is_terminal and self.backend_terminal_ts is None:
            self.backend_terminal_ts = now

    def _capture_error_message(self, resp, now: float) -> None:
        err_msg = getattr(resp, "error_message", None)
        if err_msg and not self.error_message:
            self.error_message = str(err_msg)
            if not self.terminal_seen:
                self.terminal_seen = True
                self.terminal_ts = now

    def _accumulate_tool_loop_tokens(self, resp) -> None:
        """Concatenate this frame's ``generated_ids`` bytes into the buffer.

        Frames carry token deltas; raw bytes are appended (a cheap memcpy) and
        decoded once at end-of-RPC, never per frame. Only invoked when a request
        declared tool-call markers, so nothing is retained otherwise.
        """
        infer = getattr(resp, "infer_response", None)
        if infer is None:
            return
        raw_contents = infer.raw_output_contents
        for i, out in enumerate(infer.outputs):
            if out.name != "generated_ids":
                continue
            if i < len(raw_contents):
                raw = raw_contents[i]
                if raw and _declared_element_count(out.shape) != 0:
                    self._tool_loop_raw += raw
                    if self._tool_loop_datatype is None:
                        self._tool_loop_datatype = out.datatype
            break

    def capture_response_chunk(self, resp, *, now: Optional[float] = None) -> None:
        """Extract per-chunk content from one streamed response message.

        If ``capture_backend_response_chunk`` already scanned this frame, the
        cached stats are reused to avoid a second proto scan.
        """
        if resp is None:
            return
        if now is None:
            now = time.time()
        self._capture_error_message(resp, now)
        if self._tool_loop_raw is not None:
            self._accumulate_tool_loop_tokens(resp)
        if self._pending_backend_stats:
            stats = self._pending_backend_stats.pop(0)
            self._accumulate_client_frame(stats, now)
            return
        infer = getattr(resp, "infer_response", None)
        if infer is not None:
            self._accumulate_client_frame(extract_forward_frame_stats(infer), now)

    def mark_backend_call_start(self, addr: str) -> None:
        self.backend_addr = addr
        if self.backend_call_start_ts is None:
            self.backend_call_start_ts = time.time()

    def capture_backend_response_chunk(self, resp) -> Optional[ForwardFrameStats]:
        """Capture backend frame. Returns stats for reuse in capture_response_chunk."""
        now = time.time()
        self.backend_resp_count += 1
        if self.backend_first_resp_ts is None:
            self.backend_first_resp_ts = now
        err_msg = getattr(resp, "error_message", None)
        if err_msg and not self.error_message:
            self.error_message = str(err_msg)
        infer = getattr(resp, "infer_response", None)
        if infer is None:
            return None
        stats = extract_forward_frame_stats(infer)
        self._accumulate_backend_frame(stats, now)
        self._pending_backend_stats.append(stats)
        return stats

    def mark_backend_error(self, exc: BaseException) -> None:
        self.backend_exc_type = type(exc).__name__
        code = rpc_code(exc)
        if code is not None:
            try:
                self.backend_rpc_code = code.name
            except Exception:
                self.backend_rpc_code = str(code)
        self.backend_rpc_detail = rpc_details(exc) or repr(exc)

    def mark_backend_done(self) -> None:
        if self.backend_done_ts is None:
            self.backend_done_ts = time.time()

    def mark_first_resp(self, now: Optional[float] = None) -> None:
        if self.first_resp_ts is None:
            self.first_resp_ts = time.time() if now is None else now

    def build_record(
        self,
        server_id: Optional[int],
        rank_id: Optional[int],
        *,
        end_ts: Optional[float] = None,
    ) -> dict[str, Any]:
        """Build the final access-log dict. Called once per RPC."""
        if end_ts is None:
            end_ts = time.time()
        total_ms = (end_ts - self.start_ts) * 1000.0
        ttfb_ms = None
        if self.first_resp_ts is not None:
            ttfb_ms = (self.first_resp_ts - self.start_ts) * 1000.0
        ttft_ms = duration_ms(self.start_ts, self.first_token_ts)
        tpot_ms = None
        tokens_after_first_frame = self.output_len - self.first_token_frame_len
        if tokens_after_first_frame > 0:
            span_ms = duration_ms(self.first_token_ts, self.last_token_ts)
            if span_ms is not None:
                tpot_ms = round(span_ms / tokens_after_first_frame, 3)
        backend_ttfb_ms = duration_ms(
            self.backend_call_start_ts, self.backend_first_resp_ts
        )
        backend_total_ms = duration_ms(self.backend_call_start_ts, self.backend_done_ts)
        forward_buffer_wait_ms = duration_ms(
            self.backend_first_resp_ts, self.first_resp_ts
        )
        finish_to_close_ms = duration_ms(self.terminal_ts, end_ts)
        self.check_repetition()
        record = {
            "schema_version": 1,
            "log_type": "access",
            "event": "rpc_completed",
            "component": "dash_sc_grpc",
            "component_role": "forwarder" if self.raw_mode else "frontend",
            "protocol": DASH_SC_GRPC_PROTOCOL,
            "ts": format_access_log_ts(end_ts),
            "ts_epoch_ms": int(end_ts * 1000),
            "server_id": server_id,
            "rank_id": rank_id,
            "method": self.method,
            "stream_type": self.stream_type,
            "capture_mode": "forward_summary" if self.raw_mode else "struct",
            "forward_log_version": 1 if self.raw_mode else None,
            "peer": self.peer,
            "req_count": self.req_count,
            "resp_count": self.resp_count,
            "latency_total_ms": round(total_ms, 3),
            "latency_ttfb_ms": round(ttfb_ms, 3) if ttfb_ms is not None else None,
            "latency_ttft_ms": ttft_ms,
            "latency_tpot_ms": tpot_ms,
            "finish_to_close_ms": finish_to_close_ms,
            "status": self.status,
            "status_detail": self.status_detail,
            "error_message": self.error_message,
            "backend_error_code": self.backend_error_code,
            "exc_type": self.exc_type,
            "context_code": self.context_code,
            "context_active": self.context_active,
            "backend_addr": self.backend_addr,
            "backend_resp_count": self.backend_resp_count,
            "backend_call_start_ts_epoch_ms": epoch_ms(self.backend_call_start_ts),
            "backend_first_resp_ts_epoch_ms": epoch_ms(self.backend_first_resp_ts),
            "backend_done_ts_epoch_ms": epoch_ms(self.backend_done_ts),
            "backend_latency_ttfb_ms": backend_ttfb_ms,
            "backend_latency_total_ms": backend_total_ms,
            "backend_exc_type": self.backend_exc_type,
            "backend_rpc_code": self.backend_rpc_code,
            "backend_rpc_detail": self.backend_rpc_detail,
            "forward_buffer_wait_ms": forward_buffer_wait_ms,
            "buffered_stage": self.buffered_stage,
            "request_enter_ts_epoch_ms": epoch_ms(self.start_ts),
            "first_request_ts_epoch_ms": epoch_ms(self.first_request_ts),
            "request_end_ts_epoch_ms": epoch_ms(self.request_end_ts),
            "request_read_status": self.request_read_status,
            "first_response_ts_epoch_ms": epoch_ms(self.first_resp_ts),
            "first_token_ts_epoch_ms": epoch_ms(self.first_token_ts),
            "finished_ts_epoch_ms": epoch_ms(self.terminal_ts),
            "stream_close_ts_epoch_ms": epoch_ms(end_ts),
            "request_id": self.request_id,
            "model_name": self.model_name,
            "upstream_request_id": self.upstream_request_id,
            "upstream_request_id_key": self.upstream_request_id_key,
            "input_token_len": self.input_len,
            "backend_input_token_len": self.backend_input_len,
            "output_token_len": self.output_len,
            "finish_reason": self.finish_reason,
            "finished": self.finished,
            "terminal_seen": self.terminal_seen,
            "prompt_cached_token_num": self.prompt_cached_token_num,
            "token_frame_count": self.token_frame_count,
            "empty_frame_count": self.empty_frame_count,
            "finished_only_frame_count": self.finished_only_frame_count,
            "multi_token_frame_count": self.multi_token_frame_count,
            "max_tokens_per_frame": self.max_tokens_per_frame,
            "generate_config": self.generate_config,
        }
        record.update(self._repetition_monitor.record_fields())
        return record

    def attach_to_context(self, context) -> bool:
        try:
            setattr(context, _CONTEXT_ATTR, self)
            return True
        except Exception:
            return False

    @staticmethod
    def from_context(context) -> Optional["ForwardAccessRecord"]:
        try:
            return getattr(context, _CONTEXT_ATTR, None)
        except Exception:
            return None
