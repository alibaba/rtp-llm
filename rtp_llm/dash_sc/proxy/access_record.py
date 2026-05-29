"""Forward access summary model for the dash_sc proxy path.

This module owns proxy-specific statistics. It intentionally records counts,
timestamps, scalar status, and backend lifecycle only. It never stores raw
protobufs, prompt token ids, generated token ids, logprobs, or embeddings.
"""

from __future__ import annotations

import dataclasses
import struct
import time
from typing import Any, Optional

import grpc

DASH_SC_GRPC_PROTOCOL = "grpc"
FORWARD_ACCESS_SCHEMA_VERSION = 1
FORWARD_ACCESS_LOG_VERSION = 2

_INT_WIDTH = {
    "INT8": 1,
    "UINT8": 1,
    "BOOL": 1,
    "INT32": 4,
    "UINT32": 4,
    "INT64": 8,
    "UINT64": 8,
}

_SAFE_INT_INPUTS = (
    "max_new_tokens",
    "num_return_sequences",
    "top_k",
    "min_new_tokens",
    "seed",
    "random_seed",
    "max_new_think_tokens",
    "max_thinking_tokens",
    "timeout_ms",
    "max_rpc_timeout_ms",
    "traffic_reject_priority",
)
_SAFE_FLOAT_INPUTS = (
    "top_p",
    "temperature",
    "repetition_penalty",
    "frequency_penalty",
    "presence_penalty",
)
_SAFE_BOOL_INPUTS = (
    "return_input_ids",
    "enable_thinking",
)


def format_access_log_ts(ts: float) -> str:
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


def epoch_ms(ts: Optional[float]) -> Optional[int]:
    return int(ts * 1000) if ts is not None else None


def elapsed_ms(start: Optional[float], end: Optional[float]) -> Optional[float]:
    if start is None or end is None:
        return None
    return round((end - start) * 1000.0, 3)


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


def _safe_detail(value: Optional[Any], limit: int = 512) -> Optional[str]:
    if value is None:
        return None
    text = str(value).replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _declared_element_count(shape) -> Optional[int]:
    count = 1
    saw_dim = False
    for dim in shape:
        saw_dim = True
        value = int(dim)
        if value < 0:
            return None
        count *= value
    return count if saw_dim else None


def _raw_element_count(datatype: str, raw: Optional[bytes]) -> Optional[int]:
    if raw is None:
        return None
    width = _INT_WIDTH.get(datatype)
    if not width or len(raw) % width:
        return None
    return len(raw) // width


def _tensor_element_count(tensor, raw: Optional[bytes]) -> Optional[int]:
    declared = _declared_element_count(getattr(tensor, "shape", ()))
    if declared is not None:
        return declared
    return _raw_element_count(getattr(tensor, "datatype", ""), raw)


def _find_input_raw(request, name: str):
    for i, inp in enumerate(getattr(request, "inputs", ())):
        if getattr(inp, "name", None) != name:
            continue
        raw = None
        raw_contents = getattr(request, "raw_input_contents", ())
        if i < len(raw_contents):
            raw = raw_contents[i]
        return inp, raw
    return None, None


def _first_int(datatype: str, raw: Optional[bytes]) -> Optional[int]:
    if raw is None:
        return None
    try:
        if datatype == "BOOL" and len(raw) >= 1:
            return 1 if raw[0] else 0
        if datatype == "INT32" and len(raw) >= 4:
            return int(struct.unpack_from("<i", raw, 0)[0])
        if datatype == "INT64" and len(raw) >= 8:
            return int(struct.unpack_from("<q", raw, 0)[0])
        if datatype == "UINT32" and len(raw) >= 4:
            return int(struct.unpack_from("<I", raw, 0)[0])
        if datatype == "UINT64" and len(raw) >= 8:
            return int(struct.unpack_from("<Q", raw, 0)[0])
        if datatype == "INT8" and len(raw) >= 1:
            return int(struct.unpack_from("<b", raw, 0)[0])
        if datatype == "UINT8" and len(raw) >= 1:
            return int(raw[0])
    except struct.error:
        return None
    return None


def _first_float(datatype: str, raw: Optional[bytes]) -> Optional[float]:
    if raw is None:
        return None
    try:
        if datatype == "FP32" and len(raw) >= 4:
            return float(struct.unpack_from("<f", raw, 0)[0])
        if datatype == "FP64" and len(raw) >= 8:
            return float(struct.unpack_from("<d", raw, 0)[0])
    except struct.error:
        return None
    v = _first_int(datatype, raw)
    return float(v) if v is not None else None


def _read_input_scalar(request, name: str) -> Optional[int | float | bool]:
    inp, raw = _find_input_raw(request, name)
    if inp is None or raw is None:
        return None
    datatype = getattr(inp, "datatype", "")
    if name in _SAFE_FLOAT_INPUTS:
        return _first_float(datatype, raw)
    if name in _SAFE_BOOL_INPUTS:
        v = _first_int(datatype, raw)
        if v is None:
            fv = _first_float(datatype, raw)
            return None if fv is None else bool(fv)
        return bool(v)
    return _first_int(datatype, raw)


def _read_parameter_scalar(request, name: str) -> Optional[int | float | bool | str]:
    try:
        param = request.parameters[name]
    except Exception:
        return None
    for field, convert in (
        ("int64_param", int),
        ("uint64_param", int),
        ("bool_param", bool),
        ("string_param", str),
        ("double_param", float),
    ):
        try:
            if param.HasField(field):
                return convert(getattr(param, field))
        except Exception:
            continue
    return None


def extract_generate_config_summary(request) -> Optional[dict[str, Any]]:
    config: dict[str, Any] = {}
    for name in _SAFE_INT_INPUTS + _SAFE_FLOAT_INPUTS + _SAFE_BOOL_INPUTS:
        value = _read_input_scalar(request, name)
        if value is not None:
            config[name] = value
    if "top_k" not in config:
        value = _read_parameter_scalar(request, "top_k")
        if value is not None:
            config["top_k"] = value

    inp, raw = _find_input_raw(request, "stop_words_list")
    if inp is not None:
        shape = [int(x) for x in getattr(inp, "shape", ())]
        token_count = _tensor_element_count(inp, raw) or 0
        group_count = shape[0] if len(shape) >= 2 else (1 if token_count else 0)
        config["stop_words_group_count"] = max(0, group_count)
        config["stop_words_token_count"] = max(0, token_count)

    return config or None


def extract_input_token_len(request) -> Optional[int]:
    inp, raw = _find_input_raw(request, "input_ids")
    if inp is None:
        return None
    count = _tensor_element_count(inp, raw)
    return int(count) if count is not None else None


@dataclasses.dataclass
class ForwardFrameStats:
    output_token_len: int = 0
    finish_reason: Optional[int] = None
    finished: Optional[bool] = None
    prompt_token_num: Optional[int] = None
    prompt_cached_token_num: Optional[int] = None
    error_message: Optional[str] = None

    @property
    def terminal_seen(self) -> bool:
        if self.finished is True:
            return True
        return self.finish_reason is not None and self.finish_reason != 2


def extract_forward_frame_stats(resp) -> ForwardFrameStats:
    stats = ForwardFrameStats(
        error_message=_safe_detail(getattr(resp, "error_message", None))
    )
    infer = getattr(resp, "infer_response", None)
    if infer is None:
        return stats
    raw_contents = getattr(infer, "raw_output_contents", ())
    for i, out in enumerate(getattr(infer, "outputs", ())):
        raw = raw_contents[i] if i < len(raw_contents) else None
        name = getattr(out, "name", "")
        datatype = getattr(out, "datatype", "")
        if name == "generated_ids":
            count = _tensor_element_count(out, raw)
            if count is not None:
                stats.output_token_len += max(0, int(count))
        elif name == "finish_reason":
            stats.finish_reason = _first_int(datatype, raw)
        elif name == "finished":
            v = _first_int(datatype, raw)
            if v is not None:
                stats.finished = bool(v)
        elif name == "prompt_token_num":
            stats.prompt_token_num = _first_int(datatype, raw)
        elif name == "prompt_cached_token_num":
            stats.prompt_cached_token_num = _first_int(datatype, raw)
    return stats


@dataclasses.dataclass
class ForwardAccessRecord:
    method: str
    stream_type: str
    peer: str
    start_ts: float
    upstream_request_id: Optional[str] = None
    upstream_request_id_key: Optional[str] = None

    first_request_ts: Optional[float] = None
    request_end_ts: Optional[float] = None
    request_read_status: Optional[str] = None
    first_resp_ts: Optional[float] = None
    first_token_ts: Optional[float] = None
    last_token_ts: Optional[float] = None
    finished_ts: Optional[float] = None
    stream_close_ts: Optional[float] = None
    last_chunk_ts: Optional[float] = None

    req_count: int = 0
    resp_count: int = 0
    request_id: Optional[str] = None
    model_name: Optional[str] = None
    input_token_len: Optional[int] = None
    generate_config: Optional[dict[str, Any]] = None

    output_token_len: int = 0
    finish_reason: Optional[int] = None
    finished: Optional[bool] = None
    terminal_seen: bool = False
    prompt_cached_token_num: Optional[int] = None
    token_frame_count: int = 0
    empty_frame_count: int = 0
    finished_only_frame_count: int = 0
    multi_token_frame_count: int = 0
    max_tokens_per_frame: int = 0
    error_message: Optional[str] = None

    backend_addr: Optional[str] = None
    backend_addr_index: Optional[int] = None
    backend_resp_count: int = 0
    backend_call_start_ts: Optional[float] = None
    backend_first_resp_ts: Optional[float] = None
    backend_terminal_ts: Optional[float] = None
    backend_done_ts: Optional[float] = None
    backend_error_ts: Optional[float] = None
    backend_exc_type: Optional[str] = None
    backend_rpc_code: Optional[str] = None
    backend_rpc_detail: Optional[str] = None
    backend_input_token_len: Optional[int] = None
    backend_prompt_cached_token_num: Optional[int] = None

    buffer_wait_start_ts: Optional[float] = None
    buffer_flush_ts: Optional[float] = None
    buffered_stage: Optional[str] = None

    status: str = "OK"
    status_detail: Optional[str] = None
    exc_type: Optional[str] = None
    context_code: Optional[str] = None
    context_active: Optional[bool] = None
    close_reason: Optional[str] = None

    @property
    def input_len(self) -> Optional[int]:
        # MetricReporter still reads this common aggregate attribute; the
        # emitted forward log uses the explicit input_token_len field only.
        return self.input_token_len

    @property
    def output_len(self) -> int:
        # MetricReporter still reads this common aggregate attribute; the
        # emitted forward log uses the explicit output_token_len field only.
        return self.output_token_len

    def capture_request(self, request) -> None:
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
        if self.input_token_len is None:
            self.input_token_len = extract_input_token_len(request)
        if self.generate_config is None:
            self.generate_config = extract_generate_config_summary(request)

    def mark_request_end(self, status: str) -> None:
        if self.request_end_ts is None:
            self.request_end_ts = time.time()
            self.request_read_status = status

    def mark_first_resp(self) -> None:
        if self.first_resp_ts is None:
            self.first_resp_ts = time.time()

    def capture_response_chunk(self, resp) -> None:
        stats = extract_forward_frame_stats(resp)
        now = time.time()
        if stats.error_message and not self.error_message:
            self.error_message = stats.error_message
        if stats.output_token_len > 0:
            if self.first_token_ts is None:
                self.first_token_ts = now
            self.last_token_ts = now
            self.token_frame_count += 1
            if stats.output_token_len > 1:
                self.multi_token_frame_count += 1
            self.max_tokens_per_frame = max(
                self.max_tokens_per_frame, stats.output_token_len
            )
        else:
            self.empty_frame_count += 1
        self.output_token_len += stats.output_token_len
        if stats.finish_reason is not None:
            self.finish_reason = stats.finish_reason
        if stats.finished is not None:
            self.finished = stats.finished
        if stats.prompt_cached_token_num is not None:
            self.prompt_cached_token_num = stats.prompt_cached_token_num
        if stats.terminal_seen:
            self.terminal_seen = True
            if stats.output_token_len == 0:
                self.finished_only_frame_count += 1
            if self.finished_ts is None:
                self.finished_ts = now

    def mark_backend_call_start(self, addr: str, addr_index: int) -> None:
        self.backend_addr = addr
        self.backend_addr_index = addr_index
        if self.backend_call_start_ts is None:
            self.backend_call_start_ts = time.time()

    def capture_backend_response_chunk(self, resp) -> None:
        self.backend_resp_count += 1
        now = time.time()
        if self.backend_first_resp_ts is None:
            self.backend_first_resp_ts = now
        stats = extract_forward_frame_stats(resp)
        if stats.prompt_token_num is not None:
            self.backend_input_token_len = stats.prompt_token_num
        if stats.prompt_cached_token_num is not None:
            self.backend_prompt_cached_token_num = stats.prompt_cached_token_num
        if stats.terminal_seen and self.backend_terminal_ts is None:
            self.backend_terminal_ts = now

    def mark_backend_error(self, exc: BaseException) -> None:
        self.backend_error_ts = time.time()
        self.backend_exc_type = type(exc).__name__
        code = rpc_code(exc)
        if code is not None:
            self.backend_rpc_code = getattr(code, "name", str(code))
        self.backend_rpc_detail = _safe_detail(rpc_details(exc))

    def mark_backend_done(self) -> None:
        if self.backend_done_ts is None:
            self.backend_done_ts = time.time()

    def mark_buffer_wait_start(self) -> None:
        if self.buffer_wait_start_ts is None:
            self.buffer_wait_start_ts = time.time()

    def mark_buffer_stage(self, stage: str) -> None:
        self.buffered_stage = stage

    def mark_buffer_flushed(self, stage: str) -> None:
        self.buffered_stage = stage
        if self.buffer_flush_ts is None:
            self.buffer_flush_ts = time.time()

    def mark_stream_close(self, context, exc: Optional[BaseException]) -> None:
        self.stream_close_ts = time.time()
        if exc is not None:
            self.exc_type = type(exc).__name__
        try:
            code = context.code()
        except Exception:
            code = None
        if code is not None:
            self.context_code = getattr(code, "name", str(code))
        try:
            self.context_active = bool(context.is_active())
        except Exception:
            self.context_active = None
        self.close_reason = self._derive_close_reason(exc)

    def _derive_close_reason(self, exc: Optional[BaseException]) -> Optional[str]:
        if self.error_message:
            return "backend_protocol_error"
        exc_type = type(exc).__name__
        if exc_type in ("CancelledError", "GeneratorExit"):
            return (
                "client_cancel_after_terminal"
                if self.terminal_seen
                else "client_cancel_before_terminal"
            )
        if exc_type == "AbortError":
            return "proxy_context_abort"
        if self.backend_exc_type or self.backend_rpc_code:
            return "backend_transport_fail"
        if exc is None:
            if self.context_code and self.context_code != "OK":
                if self.context_code == "CANCELLED":
                    return (
                        "client_cancel_after_terminal"
                        if self.terminal_seen
                        else "client_cancel_before_terminal"
                    )
                return "grpc_context_non_ok"
            return "eof"
        return "proxy_internal_error"

    def backend_aux_info(self) -> Optional[dict[str, int]]:
        aux: dict[str, int] = {}
        if self.backend_input_token_len is not None:
            aux["prompt_token_num"] = self.backend_input_token_len
        if self.backend_prompt_cached_token_num is not None:
            aux["prompt_cached_token_num"] = self.backend_prompt_cached_token_num
        return aux or None

    def build_record(
        self, server_id: Optional[int], rank_id: Optional[int]
    ) -> dict[str, Any]:
        end_ts = self.stream_close_ts or time.time()
        tpot_ms = None
        if (
            self.first_token_ts is not None
            and self.last_token_ts is not None
            and self.output_token_len > 1
        ):
            tpot_ms = (
                (self.last_token_ts - self.first_token_ts)
                * 1000.0
                / float(self.output_token_len - 1)
            )
        record: dict[str, Any] = {
            "schema_version": FORWARD_ACCESS_SCHEMA_VERSION,
            "log_type": "access",
            "event": "rpc_completed",
            "component": "dash_sc_grpc",
            "component_role": "forwarder",
            "protocol": DASH_SC_GRPC_PROTOCOL,
            "ts": format_access_log_ts(end_ts),
            "ts_epoch_ms": epoch_ms(end_ts),
            "server_id": server_id,
            "rank_id": rank_id,
            "method": self.method,
            "stream_type": self.stream_type,
            "capture_mode": "forward_summary",
            "legacy_capture_mode": "raw",
            "forward_log_version": FORWARD_ACCESS_LOG_VERSION,
            "peer": self.peer,
            "req_count": self.req_count,
            "resp_count": self.resp_count,
            "latency_total_ms": elapsed_ms(self.start_ts, end_ts),
            "latency_ttfb_ms": elapsed_ms(self.start_ts, self.first_resp_ts),
            "latency_ttft_ms": elapsed_ms(self.start_ts, self.first_token_ts),
            "latency_tpot_ms": round(tpot_ms, 3) if tpot_ms is not None else None,
            "finish_to_close_ms": elapsed_ms(self.finished_ts, end_ts),
            "status": self.status,
            "status_detail": _safe_detail(self.status_detail),
            "close_reason": self.close_reason,
            "error_message": _safe_detail(self.error_message),
            "exc_type": self.exc_type,
            "context_code": self.context_code,
            "context_active": self.context_active,
            "request_enter_ts_epoch_ms": epoch_ms(self.start_ts),
            "first_request_ts_epoch_ms": epoch_ms(self.first_request_ts),
            "request_end_ts_epoch_ms": epoch_ms(self.request_end_ts),
            "request_read_status": self.request_read_status,
            "first_response_ts_epoch_ms": epoch_ms(self.first_resp_ts),
            "first_token_ts_epoch_ms": epoch_ms(self.first_token_ts),
            "last_token_ts_epoch_ms": epoch_ms(self.last_token_ts),
            "finished_ts_epoch_ms": epoch_ms(self.finished_ts),
            "stream_close_ts_epoch_ms": epoch_ms(end_ts),
            "request_id": self.request_id,
            "model_name": self.model_name,
            "upstream_request_id": self.upstream_request_id,
            "upstream_request_id_key": self.upstream_request_id_key,
            "input_len": None,
            "input_ids": None,
            "input_token_len": self.input_token_len,
            "backend_input_token_len": self.backend_input_token_len,
            "backend_aux_info": self.backend_aux_info(),
            "output_token_len": self.output_token_len,
            "output_len": None,
            "generated_ids": None,
            "raw_requests": None,
            "raw_responses": None,
            "raw_requests_truncated": None,
            "raw_responses_truncated": None,
            "finish_reason": self.finish_reason,
            "finished": self.finished,
            "terminal_seen": self.terminal_seen,
            "prompt_cached_token_num": self.prompt_cached_token_num,
            "backend_prompt_cached_token_num": self.backend_prompt_cached_token_num,
            "token_frame_count": self.token_frame_count,
            "empty_frame_count": self.empty_frame_count,
            "finished_only_frame_count": self.finished_only_frame_count,
            "multi_token_frame_count": self.multi_token_frame_count,
            "max_tokens_per_frame": self.max_tokens_per_frame,
            "generate_config": self.generate_config,
            "backend_addr": self.backend_addr,
            "backend_addr_index": self.backend_addr_index,
            "backend_resp_count": self.backend_resp_count,
            "backend_call_start_ts_epoch_ms": epoch_ms(self.backend_call_start_ts),
            "backend_first_resp_ts_epoch_ms": epoch_ms(self.backend_first_resp_ts),
            "backend_terminal_ts_epoch_ms": epoch_ms(self.backend_terminal_ts),
            "backend_done_ts_epoch_ms": epoch_ms(self.backend_done_ts),
            "backend_error_ts_epoch_ms": epoch_ms(self.backend_error_ts),
            "backend_latency_ttfb_ms": elapsed_ms(
                self.backend_call_start_ts, self.backend_first_resp_ts
            ),
            "backend_latency_total_ms": elapsed_ms(
                self.backend_call_start_ts, self.backend_done_ts
            ),
            "backend_exc_type": self.backend_exc_type,
            "backend_rpc_code": self.backend_rpc_code,
            "backend_rpc_detail": self.backend_rpc_detail,
            "forward_buffer_wait_ms": elapsed_ms(
                self.buffer_wait_start_ts, self.buffer_flush_ts
            ),
            "buffered_stage": self.buffered_stage,
        }
        return record
