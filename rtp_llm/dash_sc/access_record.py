"""dash-sc gRPC access-log record and frame statistics.

Shared data model for both servicers (frontend ``inference`` + transparent
``proxy``). It lives at the ``dash_sc`` root — not under ``proxy/`` — because
the proxy is just one of its two consumers; ``access_log.py`` / ``grpc_metrics``
also depend on it.

This module intentionally has no logger or kmonitor dependencies. It owns the
pure data path for dash-sc access logs:

- record construction from a gRPC context (peer, upstream correlation id);
- request-derived fields such as request id, model name, input length, and
  generation controls;
- response-frame statistics such as output token count, finish markers, and
  prompt/backend counters;
- proxy/backend diagnostics and derived latency fields;
- final flat JSON schema construction (``build_record``).

Status *classification* (how an exception / code / error_message maps to a
status token) lives in :mod:`rtp_llm.dash_sc.status`; ``resolve_status`` here
only gathers the context inputs and writes the result onto its own fields.

The two servicers (``proxy`` / ``inference``) own the RPC lifecycle inline in
their own ``ModelStreamInfer``; this record is the data they share. Logging
lives in ``access_log.py`` (emit only) and metrics in ``grpc_metrics.py``
(kmonitor fan-out), both consuming a fully-populated record.
"""

from __future__ import annotations

import dataclasses
import json
import re
import time
from typing import Any, Optional

import grpc

from rtp_llm.config.generate_config import GenerateConfig, RoleAddr
from rtp_llm.dash_sc.codec import (
    parse_ds_header_attributes,
    parse_input_ids_from_request,
    parse_other_params,
    parse_sampling_params,
)
from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.dash_sc.repetition_monitor import (
    RequestRepetitionMonitor,
    RequestRepetitionMonitorConfig,
)
from rtp_llm.dash_sc.status import (
    classify_error_message,
    classify_rpc_exception,
    rpc_code,
    rpc_details,
)

DASH_SC_GRPC_PROTOCOL = "grpc"

_MAX_CONTROL_STRING_LEN = 4096
_MAX_CONTROL_DEPTH = 8
_REDACTED_CONTROL_VALUE = "<redacted>"
_SENSITIVE_KEY_PARTS = {
    "auth",
    "authentication",
    "authorization",
    "bearer",
    "cookie",
    "credential",
    "credentials",
    "jwt",
    "password",
    "passwd",
    "private",
    "session",
    "sig",
    "secret",
    "signature",
    "sts",
}
_SENSITIVE_TOKEN_CONTEXT_PARTS = {
    "access",
    "auth",
    "authentication",
    "authorization",
    "bearer",
    "credential",
    "credentials",
    "jwt",
    "refresh",
    "security",
    "session",
    "sts",
}
_SENSITIVE_TOKEN_EXACT_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "authorization_token",
    "id_token",
    "refresh_token",
    "session_token",
    "token",
}
_TOKEN_ID_EXACT_KEYS = {"eos_token_id", "token_id"}


def _normalized_control_key(key: Any) -> str:
    key_str = str(key)
    key_str = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key_str)
    key_str = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", key_str)
    return re.sub(r"[^a-z0-9]+", "_", key_str.lower()).strip("_")


def _is_sensitive_control_key(key: Any) -> bool:
    normalized = _normalized_control_key(key)
    if not normalized or normalized in _TOKEN_ID_EXACT_KEYS:
        return False
    if normalized in _SENSITIVE_TOKEN_EXACT_KEYS:
        return True
    if "api_key" in normalized or "apikey" in normalized:
        return True
    if "access_key" in normalized or "accesskey" in normalized:
        return True
    parts = set(normalized.split("_"))
    if parts & _SENSITIVE_KEY_PARTS:
        return True
    if "token" in parts:
        if parts & _SENSITIVE_TOKEN_CONTEXT_PARTS:
            return True
        return not (
            normalized.endswith("_token_id") or normalized.endswith("_token_ids")
        )
    return False


def to_optional_int(value: Any) -> Optional[int]:
    """Coerce a server-id/rank-id value to ``Optional[int]`` for log JSON.

    The snowflake ``server_id`` reaches the servicer as a string; access-log
    fields expect ``Optional[int]``. Returns ``None`` for ``None``/empty/
    unparseable input rather than raising, so a malformed id degrades the log
    field to null instead of failing the RPC.
    """
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# Upstream correlation headers, by priority. Whichever header the client
# actually set wins; the key name is recorded alongside the value so operators
# can tell dashscope-serving-originated IDs apart from generic ``x-request-id``
# / W3C trace context. ``traceparent`` matches the ``rtp_llm/flexlb`` Java
# convention (HttpHeaderNames.TRACE_PARENT) so HTTP and gRPC paths share a
# fallback.
_CORRELATION_METADATA_KEYS = (
    "x-dashscope-request-id",
    "x-request-id",
    "dashscope-request-id",
    "traceparent",
)


def _extract_correlation_id(context) -> tuple[Optional[str], Optional[str]]:
    """Pull an upstream correlation ID off ``context.invocation_metadata()``.

    Returns ``(header_key, header_value)`` for the first ``_CORRELATION_METADATA_KEYS``
    entry that is present and non-empty, or ``(None, None)``. Case-insensitive --
    gRPC normalises metadata names to lower-case but clients don't always.
    Read once at record creation, before any body read, so it works even when
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


def _extract_metadata_controls(context) -> Optional[list[dict[str, Any]]]:
    try:
        md = context.invocation_metadata() or ()
    except Exception:
        return None
    controls: list[dict[str, Any]] = []
    for entry in md:
        try:
            k, v = entry
        except Exception:
            continue
        if k is None:
            continue
        key = str(k)
        if key.endswith("-bin"):
            value = (
                _REDACTED_CONTROL_VALUE
                if _is_sensitive_control_key(key)
                else {"omitted": "binary_metadata"}
            )
            try:
                if isinstance(value, dict):
                    value["byte_len"] = len(v)
            except Exception:
                pass
        else:
            value = _control_value(v, key=key)
        controls.append({"key": key, "value": value})
    return controls or None


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


def _control_value(value: Any, *, key: Any = None, depth: int = 0) -> Any:
    """Return a JSON-safe control-plane value without large binary payloads."""
    if key is not None and _is_sensitive_control_key(key):
        return _REDACTED_CONTROL_VALUE
    if depth >= _MAX_CONTROL_DEPTH:
        return "<max_depth>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, bytes):
        return {"byte_len": len(value), "omitted": "bytes"}
    if isinstance(value, str):
        if len(value) > _MAX_CONTROL_STRING_LEN:
            return {
                "value": value[:_MAX_CONTROL_STRING_LEN],
                "truncated": True,
                "original_len": len(value),
            }
        return value
    if isinstance(value, dict):
        return {
            str(k): _control_value(v, key=k, depth=depth + 1) for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_control_value(v, depth=depth + 1) for v in value]
    return str(value)


def _proto_parameter_value(key: Any, param) -> Any:
    try:
        if param.HasField("string_param"):
            value = param.string_param
            if _is_sensitive_control_key(key):
                return _REDACTED_CONTROL_VALUE
            try:
                return _control_value(json.loads(value), key=key)
            except Exception:
                return _control_value(value, key=key)
        if param.HasField("int64_param"):
            return _control_value(int(param.int64_param), key=key)
        if param.HasField("bool_param"):
            return _control_value(bool(param.bool_param), key=key)
        if param.HasField("double_param"):
            return _control_value(float(param.double_param), key=key)
    except Exception:
        return _control_value(str(param), key=key)
    return _control_value(str(param), key=key)


def _parse_ds_header_attributes_for_log(request) -> dict[str, Any]:
    """Parse ds_header_attributes for logging, preserving upstream key spelling."""
    try:
        p = request.parameters.get("ds_header_attributes", None)
    except Exception:
        p = None
    if p is None:
        return {}
    try:
        if not p.HasField("string_param") or not p.string_param:
            return {}
        attrs = json.loads(p.string_param)
    except Exception:
        return {}
    return attrs if isinstance(attrs, dict) else {}


def _sampling_to_dict(sampling) -> dict[str, Any]:
    d = dataclasses.asdict(sampling)
    # stop_words_list is tuple[tuple[int,...],...]; convert for stable JSON.
    swl = d.get("stop_words_list")
    if swl is not None:
        d["stop_words_list"] = [list(group) for group in swl]
    return d


_CONTEXT_ATTR = "_dash_sc_forward_access_record"


@dataclasses.dataclass
class GrpcAccessRecord:
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
    request_controls: Optional[dict[str, Any]] = None
    input_len: Optional[int] = None
    # Full request input token ids. Only the frontend struct path calls the
    # structured request writer; the forwarder never decodes payloads.
    input_ids: Optional[list[int]] = None
    # Full generated token ids. Only the frontend struct path appends ids; the
    # forwarder does not call the generated-id writer.
    generated_ids: list[int] = dataclasses.field(default_factory=list)
    generate_config: Optional[dict[str, Any]] = None
    generate_config_role_addrs: Optional[dict[str, list[dict[str, Any]]]] = None
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
    # Structured backend code populated by the real dash-sc inference servicer.
    # When present it is a more precise error bucket than free-form
    # ``error_message`` classification.
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
    backend_exc_type: Optional[str] = None
    backend_rpc_code: Optional[str] = None
    backend_rpc_detail: Optional[str] = None
    backend_resp_count: int = 0
    buffered_stage: Optional[str] = None
    # Tool-call loop observability. The monitor owns detection and the flat
    # access-log fields; this record only feeds it request markers + the
    # generated token stream and reads back the result.
    repetition_monitor_config: RequestRepetitionMonitorConfig = dataclasses.field(
        default_factory=RequestRepetitionMonitorConfig, repr=False
    )
    _repetition_monitor: RequestRepetitionMonitor = dataclasses.field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._repetition_monitor = RequestRepetitionMonitor(
            raw_mode=self.raw_mode,
            monitor_config=self.repetition_monitor_config,
        )

    @classmethod
    def create(
        cls,
        context,
        method: str,
        stream_type: str,
        *,
        raw_mode: bool,
        repetition_monitor_config: Optional[RequestRepetitionMonitorConfig] = None,
    ) -> "GrpcAccessRecord":
        """Build a record from a gRPC context and attach it to that context.

        Called once at the top of a servicer's ``ModelStreamInfer`` — before any
        inbound frame is read — so peer / correlation id are captured even on a
        frame-less RPC (peer closed before sending). This is the data class's own
        factory, not a lifecycle orchestrator: it allocates and seeds the record,
        nothing else.
        """
        try:
            peer = context.peer()
        except Exception:
            peer = ""
        up_key, up_val = _extract_correlation_id(context)
        metadata_controls = _extract_metadata_controls(context)
        record = cls(
            method=method,
            stream_type=stream_type,
            peer=peer,
            start_ts=time.time(),
            raw_mode=raw_mode,
            upstream_request_id=up_val,
            upstream_request_id_key=up_key,
            request_controls=(
                {"metadata": metadata_controls} if metadata_controls else None
            ),
            repetition_monitor_config=(
                repetition_monitor_config or RequestRepetitionMonitorConfig()
            ),
        )
        record.attach_to_context(context)
        return record

    def record_generated_ids(self, token_ids) -> None:
        """Append one frame's generated token ids on the frontend struct path.

        The frontend servicer already holds the token-id list it streams to the
        client (it built the proto frame from it), so it hands that list here
        directly — no decode back out of the wire proto. The forwarder never
        calls this, keeping its no-raw-payload contract.
        """
        if token_ids:
            self.generated_ids.extend(token_ids)

    def check_repetition(self) -> None:
        self._repetition_monitor.check_generated_ids(self.generated_ids or ())

    def record_generate_config_role_addrs(
        self, generate_config: GenerateConfig, *, phase: str = "phase1"
    ) -> None:
        self.record_role_addrs(generate_config.role_addrs, phase=phase)

    def record_role_addrs(
        self, role_addrs: list[RoleAddr], *, phase: str = "phase1"
    ) -> None:
        if not role_addrs:
            return
        if self.generate_config_role_addrs is None:
            self.generate_config_role_addrs = {}
        phase_key = str(phase)
        if phase_key in self.generate_config_role_addrs:
            return
        self.generate_config_role_addrs[phase_key] = [
            role_addr.model_dump(mode="json") for role_addr in role_addrs
        ]

    @property
    def repetition_monitor(self) -> RequestRepetitionMonitor:
        """The request-scoped monitor — the single source of truth for tool-call
        loop detection results, log fields, and kmonitor reports.

        Exposed read-only so ``grpc_metrics`` can ask the monitor for its metric
        report plan directly instead of routing through a passthrough here. This
        record stays free of any kmonitor dependency: it only hands out the
        collaborator it already owns; the monitor does the metric work.
        """
        return self._repetition_monitor

    def record_request_frame(self, request) -> None:
        """Capture transport-level request metadata."""
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
        controls = self.request_controls
        if controls is None:
            controls = {}
            self.request_controls = controls
        if "parameters" not in controls and "ds_header_attributes" not in controls:
            controls.update(self._build_request_controls(request))

    def _build_request_controls(self, request) -> dict[str, Any]:
        parameters: dict[str, Any] = {}
        for key, param in getattr(request, "parameters", {}).items():
            if str(key) == "ds_header_attributes":
                continue
            parameters[str(key)] = _proto_parameter_value(key, param)
        ds_attrs = _parse_ds_header_attributes_for_log(request)
        return {
            "parameters": parameters,
            "ds_header_attributes": _control_value(ds_attrs),
        }

    def capture_structured_request(
        self, request, *, input_ids=None, sampling=None, other=None
    ) -> None:
        """Capture structured request statistics.

        The frontend servicer has already parsed ``input_ids`` / ``sampling`` /
        ``other`` for inference, so it hands them in and the record reuses them
        rather than decoding the same request proto a second time (the input_ids
        tensor is large for long context). Direct tests may omit parsed values;
        in that case this method falls back to parsing.
        """
        if request is None:
            return
        self.record_request_frame(request)
        if self.input_len is None:
            if input_ids is None:
                try:
                    input_ids = parse_input_ids_from_request(request)
                except Exception:
                    input_ids = None
            if input_ids is not None:
                self.input_len = len(input_ids)
                self.input_ids = input_ids
                self._repetition_monitor.set_input_ids(input_ids)
        if self.generate_config is None:
            try:
                if sampling is None:
                    ds_attrs = parse_ds_header_attributes(request)
                    sampling = parse_sampling_params(request, ds_attrs)
                    other = parse_other_params(request, ds_attrs)
                self.generate_config = _sampling_to_dict(sampling)
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

    def capture_response_chunk(
        self,
        resp: predict_v2_pb2.ModelStreamInferResponse,
        *,
        now: Optional[float] = None,
    ) -> None:
        """Capture transport-level response metadata."""
        if resp is None:
            return
        if now is None:
            now = time.time()
        # Protocol-level error channel (predict_v2.proto: empty means success).
        if resp.error_message and not self.error_message:
            self.error_message = resp.error_message
        if resp.error_message:
            self.mark_terminal(now=now)
            return
        params = resp.infer_response.parameters
        error_no_param = params.get("error_no")
        if error_no_param is None or not error_no_param.HasField("int64_param"):
            return
        error_no = int(error_no_param.int64_param)
        if error_no == 0:
            return

        self.status = f"DASH_ERROR_{error_no}"
        self.mark_terminal(now=now)

    def mark_terminal(self, *, now: Optional[float] = None) -> None:
        if self.terminal_seen:
            return
        self.terminal_seen = True
        self.terminal_ts = time.time() if now is None else now

    def mark_backend_call_start(self, addr: str) -> None:
        self.backend_addr = addr
        if self.backend_call_start_ts is None:
            self.backend_call_start_ts = time.time()

    def capture_backend_response_chunk(
        self, resp: predict_v2_pb2.ModelStreamInferResponse
    ) -> None:
        """Record backend-side receive timing and the protocol error channel.

        Frame *content* — token counts, finish markers, prompt sizes — belongs
        to the structured response layer. The transparent proxy keeps backend
        diagnostics lightweight and does not scan token payloads here.
        """
        now = time.time()
        self.backend_resp_count += 1
        if self.backend_first_resp_ts is None:
            self.backend_first_resp_ts = now
        if resp.error_message and not self.error_message:
            self.error_message = resp.error_message

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

    def record_response_chunk(
        self,
        resp: predict_v2_pb2.ModelStreamInferResponse,
        *,
        now: Optional[float] = None,
    ) -> tuple[bool, float]:
        """Mark + count one outbound response frame at transport level.

        Returns ``(is_first, now)`` so the caller can drive ``grpc_metrics``
        without recomputing the timestamp. Pure data mutation — no metrics, no
        logging. Both servicers call this once per yielded frame; structured
        token accounting is an explicit second layer.
        """
        if now is None:
            now = time.time()
        is_first = self.first_resp_ts is None
        self.mark_first_resp(now)
        self.resp_count += 1
        self.capture_response_chunk(resp, now=now)
        return is_first, now

    def resolve_status(self, context, exc: Optional[BaseException]) -> float:
        """Classify the final RPC outcome and return the end timestamp.

        Reads the gRPC ``context`` status + the handler exception (if any) and
        writes ``status`` / ``status_detail`` / ``exc_type`` / ``context_code`` /
        ``context_active`` on this record. Pure classification — no logging, no
        metrics. The caller emits the log line (``access_log.emit_access_log``)
        and reports role-specific tail metrics afterwards.

        Classification precedence (narrowest signal wins):
        1. Backend wrote a non-empty ``error_message`` frame — canonical
           protocol-level failure channel (predict_v2.proto); gRPC status stays
           OK. Without this these dropped into the success bucket.
        2. Inference completed (``terminal_seen``) and a teardown signal showed
           up afterwards (exception, or a non-OK close status). These are
           post-success events — client cancel / LBS drop / late RpcError — and
           must not poison the success counter.
        3. Falls through to the exception / gRPC-code logic.
        """
        try:
            code = context.code()
        except Exception:
            code = None
        if code is not None:
            try:
                self.context_code = code.name
            except Exception:
                self.context_code = str(code)
        try:
            self.context_active = bool(context.is_active())
        except Exception:
            self.context_active = None

        if exc is not None:
            self.exc_type = type(exc).__name__
        if self.error_message:
            # Break backend errors out of the single ``INTERNAL`` bucket so
            # Grafana's ``error_code`` breakdown names the actual failure
            # (``BACKEND_RuntimeError`` / ``BACKEND_EMPTY_OUTPUTS`` / …).
            # Transport-level outcomes still use their gRPC code names.
            self.status = self.backend_error_code or classify_error_message(
                self.error_message
            )
            self.status_detail = self.error_message
        elif exc is not None:
            if self.terminal_seen:
                if self.status == "OK":
                    self.status_detail = None
            else:
                self.status, self.status_detail = classify_rpc_exception(
                    exc, req_count=self.req_count
                )
        elif code is None or code == grpc.StatusCode.OK:
            if self.status == "OK":
                self.status_detail = None
        elif self.terminal_seen:
            if self.status == "OK":
                self.status_detail = None
        else:
            self.status = code.name
            try:
                self.status_detail = context.details() or code.name
            except Exception:
                self.status_detail = code.name
        return time.time()

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
        backend_ttfb_ms = duration_ms(
            self.backend_call_start_ts, self.backend_first_resp_ts
        )
        backend_total_ms = duration_ms(self.backend_call_start_ts, self.backend_done_ts)
        forward_buffer_wait_ms = duration_ms(
            self.backend_first_resp_ts, self.first_resp_ts
        )
        finish_to_close_ms = duration_ms(self.terminal_ts, end_ts)
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
            "peer": self.peer,
            "req_count": self.req_count,
            "resp_count": self.resp_count,
            "latency_total_ms": round(total_ms, 3),
            "latency_ttfb_ms": round(ttfb_ms, 3) if ttfb_ms is not None else None,
            "finish_to_close_ms": finish_to_close_ms,
            "status": self.status,
            "status_detail": self.status_detail,
            "error_message": self.error_message,
            "backend_error_code": self.backend_error_code,
            "exc_type": self.exc_type,
            "context_code": self.context_code,
            "context_active": self.context_active,
            "request_enter_ts_epoch_ms": epoch_ms(self.start_ts),
            "first_request_ts_epoch_ms": epoch_ms(self.first_request_ts),
            "request_end_ts_epoch_ms": epoch_ms(self.request_end_ts),
            "request_read_status": self.request_read_status,
            "first_response_ts_epoch_ms": epoch_ms(self.first_resp_ts),
            "finished_ts_epoch_ms": epoch_ms(self.terminal_ts),
            "stream_close_ts_epoch_ms": epoch_ms(end_ts),
            "request_id": self.request_id,
            "model_name": self.model_name,
            "upstream_request_id": self.upstream_request_id,
            "upstream_request_id_key": self.upstream_request_id_key,
            "request_controls": self.request_controls,
        }

        if self.raw_mode:
            record.update(
                {
                    "forward_log_version": 1,
                    "backend_addr": self.backend_addr,
                    "backend_resp_count": self.backend_resp_count,
                    "backend_call_start_ts_epoch_ms": epoch_ms(
                        self.backend_call_start_ts
                    ),
                    "backend_first_resp_ts_epoch_ms": epoch_ms(
                        self.backend_first_resp_ts
                    ),
                    "backend_done_ts_epoch_ms": epoch_ms(self.backend_done_ts),
                    "backend_latency_ttfb_ms": backend_ttfb_ms,
                    "backend_latency_total_ms": backend_total_ms,
                    "backend_exc_type": self.backend_exc_type,
                    "backend_rpc_code": self.backend_rpc_code,
                    "backend_rpc_detail": self.backend_rpc_detail,
                    "forward_buffer_wait_ms": forward_buffer_wait_ms,
                    "buffered_stage": self.buffered_stage,
                }
            )
            return record

        ttft_ms = duration_ms(self.start_ts, self.first_token_ts)
        tpot_ms = None
        tokens_after_first_frame = self.output_len - self.first_token_frame_len
        if tokens_after_first_frame > 0:
            span_ms = duration_ms(self.first_token_ts, self.last_token_ts)
            if span_ms is not None:
                tpot_ms = round(span_ms / tokens_after_first_frame, 3)
        self.check_repetition()
        record.update(
            {
                "latency_ttft_ms": ttft_ms,
                "latency_tpot_ms": tpot_ms,
                "first_token_ts_epoch_ms": epoch_ms(self.first_token_ts),
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
                "input_ids": self.input_ids,
                "generated_ids": self.generated_ids or None,
            }
        )
        if self.generate_config_role_addrs is not None:
            record["generate_config_role_addrs"] = self.generate_config_role_addrs
        record.update(self._repetition_monitor.record_fields())
        return record

    def attach_to_context(self, context) -> bool:
        try:
            setattr(context, _CONTEXT_ATTR, self)
            return True
        except Exception:
            return False

    @staticmethod
    def from_context(context) -> Optional["GrpcAccessRecord"]:
        try:
            return getattr(context, _CONTEXT_ATTR, None)
        except Exception:
            return None
