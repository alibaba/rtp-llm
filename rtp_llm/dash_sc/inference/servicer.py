"""DashSc gRPC servicer (aio) + real-inference bridge.

* :class:`DashScInferenceServicer` implements ``ModelStreamInfer`` (predict_v2.proto wire)
  as a ``grpc.aio``-native async generator.
* :func:`iter_real_model_stream_infer` awaits ``backend_visitor.enqueue`` and forwards the
  async stream chunk-by-chunk. No sync→async bridge — the whole path runs on one asyncio
  event loop (the one :class:`~rtp_llm.dash_sc.app.DashScApp` spins up).

Cancel propagation is now implicit: a gRPC peer RESET_STREAM raises ``asyncio.CancelledError``
inside the ``async for`` in the handler, which unwinds through the ``await
backend_visitor.enqueue`` / ``async for go in stream`` frames and cancels the backend
coroutine automatically.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator, Callable, Iterable, Optional, Protocol

import torch

from rtp_llm.config.exceptions import (
    AdmissionRejectReason,
    ExceptionCategory,
    ExceptionType,
    FtRuntimeException,
)
from rtp_llm.config.generate_config import (
    GenerateConfig,
    ThinkingMode,
    thinking_mode_from_value,
)
from rtp_llm.config.response_format import normalize_think_tag
from rtp_llm.config.response_format_compiler import (
    ReasoningFormat,
    restore_final_constraint,
)
from rtp_llm.dash_sc.access_log import emit_access_log, emit_query_log
from rtp_llm.dash_sc.access_record import (
    GrpcAccessRecord,
    extract_body_trace_headers,
    extract_span_external_request_id,
    to_optional_int,
)
from rtp_llm.dash_sc.codec import (
    DASH_ERROR_ABORT,
    DASH_ERROR_ADMISSION_OVERLOADED,
    DASH_ERROR_AUTO_TPM_PREEMPTED,
    DASH_ERROR_BAD_REQUEST,
    DASH_ERROR_CAPACITY,
    DASH_ERROR_INTERNAL,
    DASH_ERROR_INVALID_OUTPUT,
    DASH_ERROR_RESOURCE_EXHAUSTED,
    DASH_ERROR_TIMEOUT,
    DASH_ERROR_TOO_LONG,
    DASH_ERROR_UNSUPPORTED,
    DashErrorSpec,
    DashScInputIdsError,
    DashScParameterError,
    DashScRequestControls,
    LLMFinishReason,
    SamplingParams,
    StreamResponseBuilder,
    _lookup_ds_request_control,
    _token_ids_list_from_generate_output,
    build_dash_error_response,
    parse_dash_sc_grpc_request,
    parse_ds_header_attributes,
    parse_multimodal_parts_from_request,
    prepend_to_generated_ids_tensor,
)
from rtp_llm.dash_sc.grpc_metrics import (
    report_arrival_priority,
    report_chunk,
    report_frontend_rpc_done,
)
from rtp_llm.dash_sc.inference.grammar_validator import (
    GrammarCheckUnavailable,
    GrammarCompilationError,
    GrammarValidator,
)
from rtp_llm.dash_sc.proto import predict_v2_pb2, predict_v2_pb2_grpc
from rtp_llm.dash_sc.repetition_monitor import RequestRepetitionMonitorConfig
from rtp_llm.frontend.request_id_generator import generate_request_id
from rtp_llm.metrics import AccMetrics, kmonitor
from rtp_llm.server.request_headers import (
    extract_correlation_request_id,
    extract_request_headers,
    extract_trace_id,
)
from rtp_llm.telemetry import CURRENT_TRACE_STATE, start_server_span
from rtp_llm.telemetry.tracing import (
    metadata_to_headers,
    select_valid_server_trace_carrier,
)
from rtp_llm.utils.base_model_datatypes import (
    GenerateInput,
    GenerateOutputs,
    RequestInfo,
)
from rtp_llm.utils.util import AtomicCounter

if TYPE_CHECKING:
    from rtp_llm.config.py_config_modules import GenerateEnvConfig
    from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer

# Phase-2 dash_sc_request_id (response infer.id) suffix; keeps client able to tell
# the two response halves apart. NOT applied to the dashscope-side trace_id, which
# must stay identical across phases for end-to-end log search to work.
_PHASE2_SUFFIX = "-2"
# Body inserted between think_start_tag and think_end_tag to form the "empty think"
# block that becomes phase-2 prompt body. DSV4 protocol convention: a single LF.
_EMPTY_THINK_BODY = "\n"
# DSV4: token id == 1 signals "stop thinking immediately" mid-stream. Mirrors the
# default in ``GenerateEnvConfig.think_terminate_token_id`` (single source of truth
# for production; this constant exists so unit tests don't have to repeat it).
_DEFAULT_TERMINATE_TOKEN_ID = 1
# Model types whose dash_sc protocol uses the empty-think second pass.
_EMPTY_THINK_PHASE2_MODEL_TYPES = {"deepseek_v4"}
_INT32_MAX = 2_147_483_647
_PARTIAL_RESPONSE_METADATA = (("x-dashscope-partialresponse", "true"),)
GrpcMetadata = Iterable[tuple[object, object]]
_DASH_RPC_METHOD = "GRPCInferenceService/ModelStreamInfer"
_DASH_SERVER_SPAN_NAME = "dash_sc.ModelStreamInfer"
_DASH_SERVER_ATTRIBUTES = {
    "gen_ai.span.kind": "LLM",
    "gen_ai.operation.name": "chat",
    "gen_ai.system": "rtp_llm",
    "rpc.system": "grpc",
    "rpc.method": _DASH_RPC_METHOD,
}


def _build_mm_inputs_from_request(
    request: predict_v2_pb2.ModelInferRequest,
) -> list:
    """Convert DashSc message parts to the engine's generic multimodal inputs."""
    parts = parse_multimodal_parts_from_request(request)
    if not parts:
        return []

    from rtp_llm.ops import MMPreprocessConfig, MultimodalInput

    return [
        MultimodalInput(
            part.url,
            part.mm_type,
            torch.empty(0),
            MMPreprocessConfig(
                min_pixels=part.min_pixels,
                max_pixels=part.max_pixels,
                fps=part.fps,
                min_frames=part.min_frames,
                max_frames=part.max_frames,
            ),
        )
        for part in parts
    ]


def _request_parameter_string(request, name: str) -> str:
    if name not in request.parameters:
        return ""
    parameter = request.parameters[name]
    if not parameter.HasField("string_param"):
        return ""
    return str(parameter.string_param or "")


def _exception_metric_code(error_code: int | ExceptionType) -> str:
    code = int(error_code)
    try:
        return f"{code}_{ExceptionType.from_value(code)}"
    except ValueError:
        return str(code)


def _set_access_backend_error_code(
    access_agg: GrpcAccessRecord | None, e: BaseException
) -> None:
    if access_agg is None:
        return
    # Engine exceptions carry the last known aux_info; keep whatever the chunk loop
    # already recorded (``overwrite=False``) so a mid-stream failure still logs the
    # real token accounting.
    access_agg.record_aux_info(getattr(e, "aux_info", None), overwrite=False)
    if not isinstance(e, FtRuntimeException):
        return
    access_agg.backend_error_code = _exception_metric_code(int(e.exception_type))


def _dash_error_spec_for_ft_exception(exc: FtRuntimeException) -> DashErrorSpec:
    return _dash_error_mapping_for_ft_exception(exc).error_spec


@dataclass(frozen=True)
class _DashFtErrorMapping:
    error_spec: DashErrorSpec
    public_message: Optional[str] = None
    protocol_error: bool = False
    priority_attribution_unavailable: bool = False


@dataclass(frozen=True)
class _AutoTpmPublicContract:
    allowed_reasons: frozenset[AdmissionRejectReason]
    without_qos: _DashFtErrorMapping
    low_qos: _DashFtErrorMapping
    high_qos: _DashFtErrorMapping


_SERVICE_UNAVAILABLE_MAPPING = _DashFtErrorMapping(
    DASH_ERROR_CAPACITY,
    "Service unavailable.",
)
_PRIORITY_ATTRIBUTION_UNAVAILABLE_MAPPING = _DashFtErrorMapping(
    DASH_ERROR_CAPACITY,
    "Service unavailable.",
    priority_attribution_unavailable=True,
)
_INVALID_AUTO_TPM_PROTOCOL_MAPPING = _DashFtErrorMapping(
    DASH_ERROR_CAPACITY,
    "Service unavailable.",
    protocol_error=True,
)
_PREEMPTED_WITH_QOS_MAPPING = _DashFtErrorMapping(
    DASH_ERROR_AUTO_TPM_PREEMPTED,
    "Too many requests.",
)
_LOW_QOS_REJECTION_MAPPING = _DashFtErrorMapping(
    DASH_ERROR_ADMISSION_OVERLOADED,
    "Too many requests.",
)
_HIGH_QOS_REJECTION_MAPPING = _DashFtErrorMapping(
    DASH_ERROR_RESOURCE_EXHAUSTED,
    "Too many requests.",
)

# This is the single public-status contract for typed Auto-TPM outcomes.  Each
# row owns both protocol validation and all three externally visible QoS cases.
_AUTO_TPM_PUBLIC_CONTRACT = {
    ExceptionType.PRIORITY_PREEMPTED: _AutoTpmPublicContract(
        allowed_reasons=frozenset((AdmissionRejectReason.UNSPECIFIED,)),
        without_qos=_SERVICE_UNAVAILABLE_MAPPING,
        low_qos=_PREEMPTED_WITH_QOS_MAPPING,
        high_qos=_PREEMPTED_WITH_QOS_MAPPING,
    ),
    ExceptionType.PRIORITY_ADMISSION_REJECTED: _AutoTpmPublicContract(
        allowed_reasons=frozenset(
            (
                AdmissionRejectReason.HIGHER_PRIORITY_AHEAD,
                AdmissionRejectReason.SAME_PRIORITY_AHEAD,
            )
        ),
        without_qos=_SERVICE_UNAVAILABLE_MAPPING,
        low_qos=_LOW_QOS_REJECTION_MAPPING,
        high_qos=_HIGH_QOS_REJECTION_MAPPING,
    ),
    ExceptionType.RESOURCE_EXHAUSTED: _AutoTpmPublicContract(
        allowed_reasons=frozenset((AdmissionRejectReason.RESOURCE_EXHAUSTED,)),
        without_qos=_SERVICE_UNAVAILABLE_MAPPING,
        low_qos=_LOW_QOS_REJECTION_MAPPING,
        high_qos=_HIGH_QOS_REJECTION_MAPPING,
    ),
    ExceptionType.ADMISSION_UNAVAILABLE: _AutoTpmPublicContract(
        allowed_reasons=frozenset((AdmissionRejectReason.UNSPECIFIED,)),
        without_qos=_PRIORITY_ATTRIBUTION_UNAVAILABLE_MAPPING,
        low_qos=_PRIORITY_ATTRIBUTION_UNAVAILABLE_MAPPING,
        high_qos=_PRIORITY_ATTRIBUTION_UNAVAILABLE_MAPPING,
    ),
}


def _auto_tpm_public_mapping(
    exception_type: ExceptionType,
    raw_reason: Any,
    qos_level: Optional[int],
) -> Optional[_DashFtErrorMapping]:
    contract = _AUTO_TPM_PUBLIC_CONTRACT.get(exception_type)
    if contract is None:
        return None

    try:
        reason = AdmissionRejectReason(raw_reason)
    except (TypeError, ValueError):
        return _INVALID_AUTO_TPM_PROTOCOL_MAPPING
    if reason not in contract.allowed_reasons:
        return _INVALID_AUTO_TPM_PROTOCOL_MAPPING

    if qos_level is None:
        return contract.without_qos
    return contract.low_qos if qos_level < 50 else contract.high_qos


def _dash_error_mapping_for_ft_exception(
    exc: FtRuntimeException,
    qos_level: Optional[int] = None,
) -> _DashFtErrorMapping:
    """Map an internal failure to the public Dash contract.

    Scheduler diagnostics in ``str(exc)`` are deliberately excluded.  For an
    admission rejection, an explicitly supplied DashScope QoS header selects
    the public high/low-priority 429 contract.  Without a valid explicit QoS
    header, admission failures retain the historical 503 contract: a default
    scheduling priority is not evidence that the caller opted into QoS-tiered
    throttling.
    """

    exception_type = exc.exception_type
    raw_reason = getattr(
        exc,
        "admission_reject_reason",
        AdmissionRejectReason.UNSPECIFIED,
    )
    typed_mapping = _auto_tpm_public_mapping(
        exception_type,
        raw_reason,
        qos_level,
    )
    if typed_mapping is not None:
        return typed_mapping

    return _DashFtErrorMapping(
        _DASH_ERROR_SPEC_BY_EXCEPTION_CATEGORY[exception_type.category]
    )


def _parse_valid_qos_level(value: Any) -> Optional[int]:
    qos_level = to_optional_int(value)
    if qos_level is None or not 1 <= qos_level <= 100:
        return None
    return qos_level


def _request_qos_level(
    request_controls: DashScRequestControls,
    invocation_metadata: Optional[Any],
) -> Optional[int]:
    """Return valid metadata QoS, otherwise valid parsed request-header QoS."""
    metadata_value = _headers_from_invocation_metadata(invocation_metadata).get(
        "x-dashscope-inner-qos-level"
    )
    metadata_qos = _parse_valid_qos_level(metadata_value)
    if metadata_qos is not None:
        return metadata_qos
    request_headers = {
        str(key).lower(): value
        for key, value in request_controls.request_headers.items()
    }
    return _parse_valid_qos_level(request_headers.get("x-dashscope-inner-qos-level"))


_DASH_ERROR_SPEC_BY_EXCEPTION_CATEGORY = {
    ExceptionCategory.BAD_REQUEST: DASH_ERROR_BAD_REQUEST,
    ExceptionCategory.TOO_LONG: DASH_ERROR_TOO_LONG,
    ExceptionCategory.UNSUPPORTED: DASH_ERROR_UNSUPPORTED,
    ExceptionCategory.CAPACITY: DASH_ERROR_CAPACITY,
    ExceptionCategory.TIMEOUT: DASH_ERROR_TIMEOUT,
    ExceptionCategory.INVALID_OUTPUT: DASH_ERROR_INVALID_OUTPUT,
    ExceptionCategory.CANCELLED: DASH_ERROR_ABORT,
    ExceptionCategory.INTERNAL: DASH_ERROR_INTERNAL,
}


def stream_log_tag(
    *, request_id_numeric: int, trace_id: str, phase: Optional[int] = None
) -> str:
    """Align with C++ ``GenerateStream::streamLogTag()`` for log correlation.

    ``phase`` is appended only when set, so phase-1 logs stay byte-identical to the
    pre-refactor format and grep patterns keep working.
    """
    base = f"request_id={request_id_numeric} trace_id={trace_id}"
    return f"{base} phase={phase}" if phase is not None else base


def _headers_from_invocation_metadata(
    invocation_metadata: Optional[GrpcMetadata],
) -> dict[str, str]:
    return extract_request_headers(metadata_to_headers(invocation_metadata))


def _finish_server_trace(
    trace_state, record: GrpcAccessRecord, exc: Optional[BaseException]
) -> None:
    if trace_state is None:
        return
    try:
        if record.status == "OK":
            trace_state.finish()
        else:
            error_type = "Cancelled" if record.status == "CANCELLED" else record.status
            trace_state.finish(error=exc, error_type=error_type)
    finally:
        if CURRENT_TRACE_STATE.get() is trace_state:
            CURRENT_TRACE_STATE.set(None)


class _InitialMetadataSender(Protocol):
    def send_initial_metadata(
        self, metadata: tuple[tuple[str, str], ...]
    ) -> object: ...


async def _send_partial_response_metadata(context: _InitialMetadataSender) -> None:
    result = context.send_initial_metadata(_PARTIAL_RESPONSE_METADATA)
    if inspect.isawaitable(result):
        await result


class _TextEncoder(Protocol):
    def encode(self, text: str, *args: object, **kwargs: object) -> list[int]: ...


def _positive_int(value: object) -> Optional[int]:
    try:
        size = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return size if size > 0 else None


def _tokenizer_size(tokenizer: BaseTokenizer) -> Optional[int]:
    try:
        size = _positive_int(len(tokenizer))
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        size = None
    if size is not None:
        return size
    try:
        vocab_size = tokenizer.vocab_size
    except (AttributeError, NotImplementedError):
        return None
    return _positive_int(vocab_size)


def _tokenizer_eos_token_id(tokenizer: BaseTokenizer | None) -> Optional[int]:
    if tokenizer is None:
        return None
    try:
        eos_token_id = tokenizer.eos_token_id
    except (AttributeError, NotImplementedError):
        return None
    if eos_token_id is None:
        return None
    try:
        return int(eos_token_id)
    except (TypeError, ValueError, OverflowError):
        return None


def _derive_max_token_id(tokenizer: BaseTokenizer | None) -> Optional[int]:
    if tokenizer is None:
        return None
    size = _tokenizer_size(tokenizer)
    return size - 1 if size is not None else None


def _hf_tokenizer(tokenizer: BaseTokenizer | None) -> _TextEncoder | None:
    if tokenizer is None:
        return None
    return getattr(tokenizer, "tokenizer", tokenizer)


class _BackendVisitor(Protocol):
    async def enqueue(self, input: GenerateInput) -> AsyncIterator[GenerateOutputs]: ...


def _decode_env_tag(value: str) -> str:
    """Normalize literal newlines from a configured think tag.

    No literal default here — ``GenerateEnvConfig`` is the single source of truth
    for tag defaults. Empty value returns "".
    """
    return normalize_think_tag(value or "")


def _encode_tag(tokenizer: BaseTokenizer | None, text: str) -> list[int]:
    hf_tok = _hf_tokenizer(tokenizer)
    if hf_tok is None or not text:
        return []
    return list(hf_tok.encode(text, add_special_tokens=False))


def _normalized_model_type(model_type: Optional[str]) -> str:
    return str(model_type or "").replace("-", "_").lower()


def _uses_dash_sc_empty_think_phase2(model_type: Optional[str]) -> bool:
    return _normalized_model_type(model_type) in _EMPTY_THINK_PHASE2_MODEL_TYPES


def _matched_echo_prefix_ids(
    input_ids_list: list[int], echo_prefix_ids: Optional[list[int]]
) -> list[int]:
    """Return the exact think-BOS ids already present at the input tail."""
    prefix_ids = list(echo_prefix_ids) if echo_prefix_ids else []
    if not prefix_ids:
        return []
    if input_ids_list[-len(prefix_ids) :] == prefix_ids:
        return prefix_ids
    # Some tokenizers collapse the think-BOS prefix to its first token.
    if len(prefix_ids) > 1 and input_ids_list[-1:] == prefix_ids[:1]:
        return prefix_ids[:1]
    return []


def _dash_sc_reasoning_format(
    generate_env_config: GenerateEnvConfig,
    *,
    prompt_end_with_think: bool,
) -> ReasoningFormat:
    """Build the Dash-SC-local reasoning grammar envelope.

    Dash SC receives tokenized input, so it owns the decision whether the
    grammar must generate the think begin tag. The OpenAI path makes the same
    decision after rendering its prompt.
    """
    base_format = ReasoningFormat.from_generate_env_config(generate_env_config)
    tag_begin = ""
    if not prompt_end_with_think:
        tag_begin = _decode_env_tag(generate_env_config.think_start_tag)
        if not tag_begin:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "think_start_tag is required when thinking is enabled and "
                "input_ids do not end with the think begin tokens",
            )
    return ReasoningFormat(
        tag_begin=tag_begin,
        tag_end=base_format.tag_end,
        suffix=base_format.suffix,
        no_think_excludes=base_format.no_think_excludes,
    )


@dataclass(frozen=True)
class _ThinkRuntime:
    """Init-time-resolved think/dashllm snapshot read by every request.

    Built once in ``DashScInferenceServicer.__init__`` (or by a caller via
    :func:`build_think_runtime`) so the hot path skips repeated
    ``tokenizer.encode`` / model_type comparisons / vocab lookups. All fields are
    immutable ``tuple``/``int``/``bool`` so the same instance is safely shared
    across concurrent requests.

    Fields:
      ``bos_tokens``         encode(think_start_tag), e.g. ``<think>\\n``
      ``eos_tokens``         encode(think_end_tag),   e.g. ``</think>\\n\\n``
      ``empty_tokens``       encode(start + body + end); used as phase-2 prompt body
      ``close_token_id``     first id of ``eos_tokens`` (the ``</think>`` token);
                             ``None`` when ``eos_tokens`` is empty
      ``terminate_token_id`` token id that signals "stop thinking immediately" mid-
                             stream (DSV4: 1). ``None`` disables the token-terminate
                             branch (the regular ``</think>`` path keeps working).
      ``phase2_enabled``     model type is in ``_EMPTY_THINK_PHASE2_MODEL_TYPES``
                             and ``empty_tokens`` is available. ``terminate_token_id``
                             is a separate request-path gate for actually entering
                             phase-2.
      ``eos_token_id``       tokenizer.eos_token_id; written to dashllm
                             ``stop_token_id`` response param
      ``max_token_id``       ``len(tokenizer) - 1``; written to dashllm
                             ``max_token_id`` response param
    """

    bos_tokens: tuple[int, ...] = ()
    eos_tokens: tuple[int, ...] = ()
    empty_tokens: tuple[int, ...] = ()
    close_token_id: Optional[int] = None
    terminate_token_id: Optional[int] = None
    phase2_enabled: bool = False
    eos_token_id: Optional[int] = None
    max_token_id: Optional[int] = None


def build_think_runtime(
    tokenizer: BaseTokenizer | None,
    generate_env_config: GenerateEnvConfig | None,
    model_type: Optional[str],
    *,
    terminate_token_id: Optional[int] = _DEFAULT_TERMINATE_TOKEN_ID,
    eos_token_id: Optional[int] = None,
    max_token_id: Optional[int] = None,
) -> _ThinkRuntime:
    """Pre-compute the per-startup think/dashllm snapshot.

    ``terminate_token_id`` defaults to ``_DEFAULT_TERMINATE_TOKEN_ID`` (1, the
    DSV4 convention). Pass ``None`` or any value ``<= 0`` to disable the
    token-terminate branch entirely.

    ``eos_token_id`` / ``max_token_id`` of ``None`` fall back to tokenizer-derived
    values; explicit values from the caller win.

    Returns a safe-empty runtime (``phase2_enabled=False``, all token tuples
    empty) when ``tokenizer`` or ``generate_env_config`` is ``None``, matching
    the "missing tokenizer" fallback shape the previous derive helpers produced.
    """
    eos_tid = (
        eos_token_id if eos_token_id is not None else _tokenizer_eos_token_id(tokenizer)
    )
    max_tid = (
        max_token_id if max_token_id is not None else _derive_max_token_id(tokenizer)
    )
    term_id = (
        int(terminate_token_id)
        if terminate_token_id is not None and int(terminate_token_id) > 0
        else None
    )
    if tokenizer is None or generate_env_config is None:
        return _ThinkRuntime(
            terminate_token_id=term_id,
            eos_token_id=eos_tid,
            max_token_id=max_tid,
        )
    think_start_tag = _decode_env_tag(generate_env_config.think_start_tag)
    think_end_tag = _decode_env_tag(generate_env_config.think_end_tag)
    bos_tokens = tuple(_encode_tag(tokenizer, think_start_tag))
    eos_tokens = tuple(_encode_tag(tokenizer, think_end_tag))
    empty_tokens = tuple(
        _encode_tag(tokenizer, think_start_tag + _EMPTY_THINK_BODY + think_end_tag)
    )
    close_token_id = int(eos_tokens[0]) if eos_tokens else None
    phase2_enabled = _uses_dash_sc_empty_think_phase2(model_type) and bool(empty_tokens)
    return _ThinkRuntime(
        bos_tokens=bos_tokens,
        eos_tokens=eos_tokens,
        empty_tokens=empty_tokens,
        close_token_id=close_token_id,
        terminate_token_id=term_id,
        phase2_enabled=phase2_enabled,
        eos_token_id=eos_tid,
        max_token_id=max_tid,
    )


def _build_empty_think_phase2_input_ids(
    input_ids_list: list[int],
    matched_bos_ids: list[int],
    empty_think_tokens: list[int],
) -> list[int]:
    base = list(input_ids_list)
    if matched_bos_ids and base[-len(matched_bos_ids) :] == matched_bos_ids:
        base = base[: -len(matched_bos_ids)]
    return base + list(empty_think_tokens)


def _make_generate_input(
    *,
    request_id: int,
    input_ids_list: list[int],
    generate_config: GenerateConfig,
    invocation_metadata: Optional[GrpcMetadata],
    request_headers: Optional[dict[str, str]] = None,
    mm_inputs: Optional[list] = None,
    input_ids_tensor: Optional[torch.Tensor] = None,
) -> GenerateInput:
    headers = dict(request_headers or {})
    headers.update(_headers_from_invocation_metadata(invocation_metadata))
    trace_id = str(generate_config.trace_id or extract_trace_id(headers) or "")
    # ``input_ids_tensor`` is the zero-copy INT32 view the codec built straight off the
    # request payload (see ``ParsedInputIds``). Long-context dash requests otherwise pay
    # a full list->tensor materialization here.
    token_ids = (
        input_ids_tensor
        if input_ids_tensor is not None
        else torch.tensor(input_ids_list, dtype=torch.int)
    )
    return GenerateInput(
        request_id=request_id,
        token_ids=token_ids,
        mm_inputs=list(mm_inputs) if mm_inputs else [],
        generate_config=generate_config,
        headers=headers,
        request_info=RequestInfo(
            trace_id=trace_id,
            request_id=extract_correlation_request_id(headers) or trace_id,
            source_role="dash",
        ),
    )


async def _close_async_stream_if_possible(stream: object, tag: str) -> None:
    try:
        close = stream.aclose
    except AttributeError:
        return
    if not callable(close):
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception as e:
        logging.warning("[DashScGrpc] [%s] phase-1 stream close failed: %s", tag, e)


def _phase2_max_new_tokens_for_completion_alias(
    sampling: SamplingParams,
    generate_think_token_num: Optional[int],
) -> int:
    max_new_tokens = int(sampling.max_new_tokens)
    if (
        sampling.max_total_tokens is not None
        and sampling.max_total_tokens > 0
        and generate_think_token_num is not None
    ):
        max_new_tokens = min(
            max_new_tokens,
            max(0, int(sampling.max_total_tokens) - int(generate_think_token_num)),
        )
    return max_new_tokens


def _clone_generate_config(generate_config: GenerateConfig) -> GenerateConfig:
    cloned_config = generate_config.model_copy(deep=True)
    # Phase-2 must re-enter routing; copied role_addrs would bypass FlexLB master.
    cloned_config.role_addrs = []
    return cloned_config


def _apply_dash_sc_controls_to_generate_config(
    generate_config: GenerateConfig,
    sampling: SamplingParams,
    request_controls: DashScRequestControls,
    runtime: _ThinkRuntime,
    default_thinking_mode: ThinkingMode,
) -> None:
    """Apply DashSC request controls over the deployment THINK_MODE default."""
    request_max_think = sampling.max_new_think_tokens
    if request_max_think is None:
        request_max_think = request_controls.max_new_think_tokens
    if request_max_think is not None:
        max_think = int(request_max_think)
        generate_config.max_thinking_tokens = _INT32_MAX if max_think < 0 else max_think
    if request_max_think == 0 or request_controls.enable_thinking is False:
        thinking_mode = ThinkingMode.DISABLED
    elif request_controls.enable_thinking is True or request_max_think is not None:
        thinking_mode = ThinkingMode.ENABLED
    else:
        thinking_mode = default_thinking_mode
    implicit_adaptive = (
        request_controls.enable_thinking is None
        and request_max_think is None
        and thinking_mode == ThinkingMode.ADAPTIVE
    )
    if (
        thinking_mode == ThinkingMode.ADAPTIVE
        and implicit_adaptive
        and (
            generate_config.has_num_beams() or generate_config.num_return_sequences > 1
        )
    ):
        logging.warning(
            "DashSC implicit adaptive thinking is disabled for multi-sequence generation"
        )
        thinking_mode = ThinkingMode.DISABLED

    generate_config.thinking_mode = thinking_mode
    generate_config.in_think_mode = False
    if thinking_mode == ThinkingMode.DISABLED:
        generate_config.max_thinking_tokens = 0
    elif thinking_mode == ThinkingMode.ENABLED and (
        generate_config.end_think_token_ids or runtime.eos_tokens
    ):
        generate_config.in_think_mode = True
        if not generate_config.end_think_token_ids:
            generate_config.end_think_token_ids = list(runtime.eos_tokens)
    elif thinking_mode == ThinkingMode.ENABLED:
        # Preserve the legacy DashSC gate: fixed thinking requires a runtime
        # end boundary. A request budget alone cannot make that state usable.
        generate_config.thinking_mode = ThinkingMode.DISABLED
    if request_controls.timeout_ms is not None:
        # Subtract a margin so the engine times out BEFORE the upstream gateway sends
        # RST_STREAM. This makes the timeout surface as a normal
        # finish_reason=STOP_TIMEOUT response (200) rather than gRPC CANCELLED (5xx).
        margin_ms = max(2000, min(5000, int(int(request_controls.timeout_ms) * 0.15)))
        engine_timeout_ms = max(5000, int(request_controls.timeout_ms) - margin_ms)
        generate_config.timeout_ms = engine_timeout_ms
        generate_config.ttft_timeout_ms = engine_timeout_ms
    if request_controls.traffic_reject_priority is not None:
        generate_config.traffic_reject_priority = int(
            request_controls.traffic_reject_priority
        )
    # Auto-TPM QoS priority from x-dashscope-inner-qos-level. Mirrors
    # openai_endpoint.py which sets qos_priority from the HTTP header so
    # it survives IPC to the dash_sc enqueue loop where
    # GenerateInput.headers may be absent. Do NOT confuse with
    # traffic_reject_priority (x-ds-request-priority) above.
    qos_level = request_controls.request_headers.get("x-dashscope-inner-qos-level")
    if qos_level is not None:
        try:
            generate_config.qos_priority = int(str(qos_level).strip())
        except (TypeError, ValueError):
            pass
    if request_controls.reasoning_effort is not None:
        kwargs = dict(generate_config.chat_template_kwargs or {})
        kwargs["reasoning_effort"] = request_controls.reasoning_effort
        generate_config.chat_template_kwargs = kwargs


# ----------------------------------------------------------------------------
# Real inference bridge: async backend enqueue -> aio gRPC async generator
# ----------------------------------------------------------------------------


async def iter_real_model_stream_infer(
    request: predict_v2_pb2.ModelInferRequest,
    input_ids_list: list[int],
    sampling: SamplingParams,
    request_controls: DashScRequestControls,
    backend_visitor: _BackendVisitor,
    *,
    rtp_llm_request_id: int,
    echo_prefix_ids: Optional[list[int]] = None,
    extra_stop_word_ids: Optional[list[list[int]]] = None,
    invocation_metadata: Optional[GrpcMetadata] = None,
    tokenizer: BaseTokenizer | None = None,
    generate_env_config: GenerateEnvConfig | None = None,
    think_runtime: Optional[_ThinkRuntime] = None,
    phase2_request_id_factory: Optional[Callable[[], int]] = None,
    access_agg: GrpcAccessRecord | None = None,
    yield_access_stats: bool = False,
    mm_inputs: Optional[list] = None,
    input_ids_tensor: Optional[torch.Tensor] = None,
) -> AsyncIterator[predict_v2_pb2.ModelStreamInferResponse]:
    """Run enqueue on ``backend_visitor`` and yield one proto per chunk as the backend streams.

    ``rtp_llm_request_id`` is the int64 used for ``GenerateInput.request_id`` and log tags;
    the upstream servicer generates it via ``generate_request_id`` (same snowflake scheme as
    the HTTP path). ``request.id`` (string) is preserved as the trace id.

    ``echo_prefix_ids`` is the auto-derived "thinking prefill" token id sequence. When
    non-empty and ``input_ids_list`` ends with it, the first non-empty ``generated_ids``
    chunk gets ``echo_prefix_ids`` prepended so downstream consumers that rely on the
    prefill-echo contract (dashllm-style) see the expected first token.

    ``extra_stop_word_ids`` is the per-startup snapshot of model-specific stop tokens
    (renderer-injected extras + env-supplied) the dash-sc path otherwise misses because
    upstream pre-tokenization bypasses the OpenAI endpoint. Contract: it MUST be
    pre-deduped (``_derive_stop_word_ids_list`` does this at startup) and is treated
    as read-only — the hot path shares inner-list references rather than copying.

    ``think_runtime`` is the init-time-resolved think/dashllm snapshot
    (:class:`_ThinkRuntime`). Caller (servicer) builds it once via
    :func:`build_think_runtime` so the hot path skips repeated tokenizer.encode /
    model_type comparisons. ``None`` means "no think state" (phase-2 disabled, all
    dashllm limit params null).

    Hot-path layout: dashscope-serving doesn't ship ``stop_words_list`` per request,
    so 99% of calls hit the fast branch (empty ``existing``) and skip the dedup set
    + tuple hashing entirely. The slow branch only fires when a caller explicitly
    sets ``stop_words_list`` on the request.
    """
    trace_str = str(request.id)
    tag = stream_log_tag(request_id_numeric=rtp_llm_request_id, trace_id=trace_str)
    runtime = think_runtime if think_runtime is not None else _ThinkRuntime()
    logging.debug(
        "[DashScGrpc] [%s] real infer start: model_name=%s input_len=%s sampling=%s",
        tag,
        request.model_name,
        len(input_ids_list),
        sampling,
    )
    matched_echo_ids = _matched_echo_prefix_ids(input_ids_list, echo_prefix_ids)
    should_echo = bool(matched_echo_ids)
    echoed = False
    stream: object | None = None
    phase2_stream: object | None = None
    try:
        generate_config = sampling.to_generate_config(request_controls=request_controls)
        generate_config.trace_id = trace_str
        default_thinking_mode = (
            thinking_mode_from_value(generate_env_config.think_mode)
            if generate_env_config is not None
            else ThinkingMode.DISABLED
        )
        begin_think_tokens = list(runtime.bos_tokens or tuple(echo_prefix_ids or ()))
        _apply_dash_sc_controls_to_generate_config(
            generate_config,
            sampling,
            request_controls,
            runtime,
            default_thinking_mode,
        )
        configured_thinking_mode = generate_config.thinking_mode
        matched_think_bos_ids = matched_echo_ids or _matched_echo_prefix_ids(
            input_ids_list, begin_think_tokens
        )
        if configured_thinking_mode == ThinkingMode.ADAPTIVE and matched_think_bos_ids:
            generate_config.thinking_mode = ThinkingMode.ENABLED
            generate_config.in_think_mode = True
            configured_thinking_mode = ThinkingMode.ENABLED
        # Boundary IDs are request metadata as well as processor inputs. Keep
        # them available even when this request disables thinking.
        if begin_think_tokens:
            generate_config.begin_think_token_ids = begin_think_tokens
        reasoning_format = None
        if generate_env_config is not None and configured_thinking_mode in (
            ThinkingMode.ENABLED,
            ThinkingMode.ADAPTIVE,
        ):
            reasoning_format = _dash_sc_reasoning_format(
                generate_env_config,
                prompt_end_with_think=bool(matched_think_bos_ids),
            )
        if extra_stop_word_ids:
            existing = generate_config.stop_words_list
            if existing:
                # Slow path: request carries its own stops; dedup against them.
                seen = {tuple(w) for w in existing}
                for w in extra_stop_word_ids:
                    t = tuple(w)
                    if t not in seen:
                        existing.append(w)
                        seen.add(t)
            else:
                # Fast path: shallow-copy the startup snapshot. Outer copy keeps
                # any future engine-side mutation request-local; inner lists are
                # shared (snapshot is read-only by contract).
                generate_config.stop_words_list = list(extra_stop_word_ids)
        if generate_env_config is None:
            final_constraint = generate_config.finalize_response_format()
        else:
            final_constraint = generate_config.add_thinking_params(
                _hf_tokenizer(tokenizer),
                generate_env_config,
                enable_thinking=(
                    None
                    if configured_thinking_mode == ThinkingMode.ADAPTIVE
                    else generate_config.in_think_mode
                ),
                reasoning_format=reasoning_format,
            )
        if runtime.eos_tokens and not generate_config.end_think_token_ids:
            generate_config.end_think_token_ids = list(runtime.eos_tokens)
        # All these are pre-resolved at servicer init via ``build_think_runtime``;
        # reading them here is O(1) and avoids per-request tokenizer.encode.
        eos_id = runtime.eos_token_id
        max_id = runtime.max_token_id
        term_id = runtime.terminate_token_id
        think_close_token_id = runtime.close_token_id
        max_new_tokens = int(generate_config.max_new_tokens or 0)
        # ``runtime.phase2_enabled`` is the init-time gate (model_type + empty_tokens
        # availability). ``in_think_mode`` is per-request — ``add_thinking_params``
        # sets it from generate_config and a request can override it.
        phase2_enabled = runtime.phase2_enabled and (
            configured_thinking_mode == ThinkingMode.ENABLED
        )
        adaptive_phase2_pending = runtime.phase2_enabled and (
            configured_thinking_mode == ThinkingMode.ADAPTIVE
        )
        cumulative_sent_ids: list[int] = []
        generate_think_token_num: Optional[int] = None
        generate_input = _make_generate_input(
            request_id=rtp_llm_request_id,
            input_ids_list=input_ids_list,
            generate_config=generate_config,
            invocation_metadata=invocation_metadata,
            request_headers=request_controls.request_headers,
            mm_inputs=mm_inputs,
            input_ids_tensor=input_ids_tensor,
        )
        is_streaming = bool(generate_config.is_streaming)
        logging.debug("[DashScGrpc] [%s] generate_input: %s", tag, generate_input)
        # Every streaming frame repeats the same tensor descriptors, request identity and
        # generation limits. The builder materializes that protobuf template once and
        # patches only the per-frame values (see ``StreamResponseBuilder``).
        response_builder = StreamResponseBuilder(
            dash_sc_request_id=request.id,
            model_name=request.model_name,
            request_log_tag=tag,
            request_input_ids=input_ids_list,
            return_input_ids=request_controls.return_input_ids,
            is_streaming=is_streaming,
            generate_config=generate_config,
            eos_token_id=eos_id,
            max_token_id=max_id,
        )
        chunk_idx = 0
        phase2_needed = False
        # One-shot guard: ``phase2_triggered`` flips True the instant we
        # commit to phase-2 (before the ``await backend_visitor.enqueue``
        # below). It pins the invariant "at most ONE phase-2 enqueue per
        # ModelStreamInfer request" — even if some future refactor lets the
        # term-token detection or natural-finish fall-through re-fire,
        # ``phase2_triggered`` blocks the second entry. Tracking only one
        # boolean keeps the guard cheap on the hot path.
        phase2_triggered = False
        stream = await backend_visitor.enqueue(generate_input)
        async for go in stream:
            chunk_idx += 1
            logging.debug("[DashScGrpc] [%s] real infer chunk %s", tag, chunk_idx)
            if not go.generate_outputs:
                raise ValueError("empty generate_outputs in backend chunk")
            out_py = go.generate_outputs[0]
            generated_ids = _token_ids_list_from_generate_output(out_py)
            if adaptive_phase2_pending and generated_ids:
                phase2_enabled = bool(
                    generate_config.begin_think_token_ids
                    and generated_ids[0] == generate_config.begin_think_token_ids[0]
                )
                adaptive_phase2_pending = False
            aux_info = out_py.aux_info
            prompt_token_num = (
                int(aux_info.input_len) if aux_info is not None else len(input_ids_list)
            )
            prompt_cached_token_num = (
                int(aux_info.reuse_len) if aux_info is not None else 0
            )
            if access_agg is not None:
                access_agg.record_aux_info(aux_info)
                if aux_info is not None and aux_info.role_addrs:
                    # model_rpc_client copies the final submitted role_addrs here.
                    access_agg.record_role_addrs(aux_info.role_addrs, phase="phase1")
            if not generated_ids and not out_py.finished:
                response = response_builder.build(go)
                stats = (
                    0,
                    False,
                    LLMFinishReason.STREAMING,
                    prompt_token_num,
                    prompt_cached_token_num,
                    (),
                )
                yield (response, stats) if yield_access_stats else response
                continue
            ids_for_accounting = generated_ids
            if should_echo and not echoed and generated_ids:
                ids_for_accounting = matched_echo_ids + generated_ids
            close_offset: Optional[int] = None
            term_offset: Optional[int] = None
            if generate_think_token_num is None:
                if think_close_token_id is not None:
                    for offset, token_id in enumerate(ids_for_accounting):
                        if token_id == think_close_token_id:
                            close_offset = offset
                            break
                if (
                    phase2_enabled
                    and not phase2_triggered
                    and term_id is not None
                    and term_id in generated_ids
                ):
                    term_in_generated = generated_ids.index(term_id)
                    term_offset = term_in_generated
                    if should_echo and not echoed and generated_ids:
                        term_offset += len(matched_echo_ids)
                if close_offset is not None and (
                    term_offset is None or close_offset < term_offset
                ):
                    generate_think_token_num = (
                        len(cumulative_sent_ids) + close_offset + 1
                    )
                    # Natural ``</think>`` close keeps the stream single-phase
                    # (DashLLM-aligned). Phase-2 is exclusively triggered by
                    # the terminate-token-id (DSV4 token 1) path below — see
                    # the comment block near ``phase2_triggered`` init.
            if (
                phase2_enabled
                and not phase2_triggered
                and term_id is not None
                and generate_think_token_num is None
                and generated_ids
                and term_id in generated_ids
            ):
                generated_ids = generated_ids[: generated_ids.index(term_id)]
                ids_for_accounting = generated_ids
                if should_echo and not echoed and generated_ids:
                    ids_for_accounting = matched_echo_ids + generated_ids
                generate_think_token_num = len(cumulative_sent_ids) + len(
                    ids_for_accounting
                )
                will_do_phase2 = True
                if sampling.max_new_tokens_from_completion_alias:
                    will_do_phase2 = (
                        _phase2_max_new_tokens_for_completion_alias(
                            sampling, generate_think_token_num
                        )
                        > 0
                    )
                cumulative_sent_ids.extend(ids_for_accounting)
                # Yield thinking content (always intermediate)
                if generated_ids:
                    response = response_builder.build(
                        go,
                        generate_think_token_num=generate_think_token_num,
                        stream_finished=False,
                        token_ids=generated_ids,
                    )
                    if should_echo and not echoed:
                        if prepend_to_generated_ids_tensor(
                            response.infer_response, matched_echo_ids
                        ):
                            echoed = True
                    stats = (
                        len(ids_for_accounting),
                        False,
                        LLMFinishReason.STREAMING,
                        prompt_token_num,
                        prompt_cached_token_num,
                        ids_for_accounting,
                    )
                    yield (response, stats) if yield_access_stats else response
                # Yield </think> close tokens
                if runtime.eos_tokens:
                    eos_response = response_builder.build(
                        go,
                        generate_think_token_num=generate_think_token_num,
                        finish_reason_override=(
                            LLMFinishReason.LENGTH if not will_do_phase2 else None
                        ),
                        stream_finished=not will_do_phase2,
                        token_ids=list(runtime.eos_tokens),
                    )
                    eos_finished = not will_do_phase2
                    eos_finish_reason = (
                        LLMFinishReason.LENGTH
                        if eos_finished
                        else LLMFinishReason.STREAMING
                    )
                    stats = (
                        len(runtime.eos_tokens),
                        eos_finished,
                        eos_finish_reason,
                        prompt_token_num,
                        prompt_cached_token_num,
                        runtime.eos_tokens,
                    )
                    yield (
                        (eos_response, stats) if yield_access_stats else eos_response
                    )
                phase2_needed = will_do_phase2
                break
            cumulative_sent_ids.extend(ids_for_accounting)
            finish_reason_override = None
            if (
                out_py.finished
                and max_new_tokens > 0
                and len(cumulative_sent_ids) >= max_new_tokens
            ):
                finish_reason_override = LLMFinishReason.LENGTH
            response = response_builder.build(
                go,
                generate_think_token_num=generate_think_token_num,
                finish_reason_override=finish_reason_override,
            )
            if should_echo and not echoed and generated_ids:
                if prepend_to_generated_ids_tensor(
                    response.infer_response, matched_echo_ids
                ):
                    echoed = True
            response_finished = bool(out_py.finished)
            response_finish_reason = (
                finish_reason_override
                if finish_reason_override is not None
                else (
                    LLMFinishReason.STOP
                    if response_finished
                    else LLMFinishReason.STREAMING
                )
            )
            stats = (
                len(ids_for_accounting),
                response_finished,
                response_finish_reason,
                prompt_token_num,
                prompt_cached_token_num,
                ids_for_accounting,
            )
            yield (response, stats) if yield_access_stats else response
            if phase2_needed:
                break
        if chunk_idx:
            logging.debug(
                "[DashScGrpc] [%s] real infer done: output_chunks=%s",
                tag,
                chunk_idx,
            )
        if chunk_idx == 0:
            logging.warning("[DashScGrpc] [%s] empty outputs_list", tag)
            error_spec = DASH_ERROR_INTERNAL
            response = build_dash_error_response(
                str(request.id),
                request.model_name,
                error_spec=error_spec,
                status_message="empty outputs_list from backend",
            )
            stats = (0, True, error_spec.finish_reason, len(input_ids_list), 0, ())
            yield (response, stats) if yield_access_stats else response
            return
        # No implicit natural-finish phase-2 trigger here. DashLLM-aligned
        # policy: phase-2 is exclusively initiated by terminate_token_id
        # (DSV4 token 1) in the think phase. If phase-1 reaches stream end
        # without ever emitting close or term token, treat the whole stream
        # as reasoning content — do NOT silently restart with empty-think.
        if phase2_needed:
            await _close_async_stream_if_possible(stream, tag)
        if phase2_needed and not phase2_triggered:
            # One-shot pin BEFORE any await so a future / unexpected re-entry
            # cannot double-fire phase-2. Set before metric report so even an
            # accidental re-entry by the metric call would still be guarded.
            phase2_triggered = True
            # Phase-2 entry metric — operators alarm on spikes (think-abort
            # rate). Wrapped in try/except so metric failure never breaks the
            # response stream.
            try:
                kmonitor.report(
                    AccMetrics.DASH_SC_DSV4_PHASE2_QPS_METRIC,
                    1,
                    {
                        "protocol": "dash_sc_grpc",
                        "model": str(request.model_name or "unknown"),
                    },
                )
            except Exception as metric_err:
                logging.warning(
                    "[DashScGrpc] [%s] phase-2 metric report failed: %s",
                    tag,
                    metric_err,
                )
        if phase2_needed:
            phase2_config = _clone_generate_config(generate_config)
            phase2_config.in_think_mode = False
            phase2_config.thinking_mode = ThinkingMode.DISABLED
            phase2_config.max_thinking_tokens = 0
            restore_final_constraint(phase2_config, final_constraint)
            if sampling.max_new_tokens_from_completion_alias:
                phase2_config.max_new_tokens = (
                    _phase2_max_new_tokens_for_completion_alias(
                        sampling, generate_think_token_num
                    )
                )
            # trace_id stays equal across phases so the dashscope log search
            # aggregates both halves under a single trace; phase distinction is
            # carried by request_log_tag (phase=2) and by the ``-2`` suffix on
            # the response infer.id (client-facing).
            phase2_config.trace_id = trace_str
            phase2_input_ids = _build_empty_think_phase2_input_ids(
                input_ids_list, matched_think_bos_ids, list(runtime.empty_tokens)
            )
            phase2_request_id = (
                phase2_request_id_factory()
                if phase2_request_id_factory is not None
                else rtp_llm_request_id
            )
            phase2_tag = stream_log_tag(
                request_id_numeric=phase2_request_id, trace_id=trace_str, phase=2
            )
            phase2_generate_input = _make_generate_input(
                request_id=phase2_request_id,
                input_ids_list=phase2_input_ids,
                generate_config=phase2_config,
                invocation_metadata=invocation_metadata,
                request_headers=request_controls.request_headers,
                mm_inputs=mm_inputs,
            )
            logging.debug(
                "[DashScGrpc] [%s] phase-2 generate_input: %s",
                phase2_tag,
                phase2_generate_input,
            )
            phase2_stream = await backend_visitor.enqueue(phase2_generate_input)
            phase2_response_builder = StreamResponseBuilder(
                dash_sc_request_id=f"{request.id}{_PHASE2_SUFFIX}",
                model_name=request.model_name,
                request_log_tag=phase2_tag,
                request_input_ids=phase2_input_ids,
                return_input_ids=request_controls.return_input_ids,
                is_streaming=is_streaming,
                generate_config=phase2_config,
                eos_token_id=eos_id,
                max_token_id=max_id,
            )
            phase2_cumulative_sent_ids: list[int] = []

            def _build_phase2_response(
                resp_go: GenerateOutputs,
            ) -> tuple[
                predict_v2_pb2.ModelStreamInferResponse,
                tuple[int, bool, int, int, int, list[int]],
            ]:
                resp_out = resp_go.generate_outputs[0]
                resp_ids = _token_ids_list_from_generate_output(resp_out)
                phase2_cumulative_sent_ids.extend(resp_ids)
                phase2_max_new_tokens = int(phase2_config.max_new_tokens or 0)
                finish_reason_override = None
                if (
                    resp_out.finished
                    and phase2_max_new_tokens > 0
                    and len(phase2_cumulative_sent_ids) >= phase2_max_new_tokens
                ):
                    finish_reason_override = LLMFinishReason.LENGTH
                response_finished = bool(resp_out.finished)
                response_finish_reason = (
                    finish_reason_override
                    if finish_reason_override is not None
                    else (
                        LLMFinishReason.STOP
                        if response_finished
                        else LLMFinishReason.STREAMING
                    )
                )
                aux_info = resp_out.aux_info
                prompt_token_num = (
                    int(aux_info.input_len)
                    if aux_info is not None
                    else len(phase2_input_ids)
                )
                prompt_cached_token_num = (
                    int(aux_info.reuse_len) if aux_info is not None else 0
                )
                if access_agg is not None:
                    access_agg.record_aux_info(aux_info)
                    if aux_info is not None and aux_info.role_addrs:
                        # model_rpc_client copies the final submitted role_addrs here.
                        access_agg.record_role_addrs(
                            aux_info.role_addrs, phase="phase2"
                        )
                response = phase2_response_builder.build(
                    resp_go,
                    generate_think_token_num=generate_think_token_num,
                    finish_reason_override=finish_reason_override,
                )
                stats = (
                    len(resp_ids),
                    response_finished,
                    response_finish_reason,
                    prompt_token_num,
                    prompt_cached_token_num,
                    resp_ids,
                )
                return response, stats

            async for go in phase2_stream:
                if not go.generate_outputs:
                    raise ValueError("empty generate_outputs in phase-2 backend chunk")
                out_py = go.generate_outputs[0]
                generated_ids = _token_ids_list_from_generate_output(out_py)
                if not generated_ids and not out_py.finished:
                    continue
                resp, stats = _build_phase2_response(go)
                yield (resp, stats) if yield_access_stats else resp

    except FtRuntimeException as e:
        _set_access_backend_error_code(access_agg, e)
        error_mapping = _dash_error_mapping_for_ft_exception(
            e,
            qos_level=_request_qos_level(request_controls, invocation_metadata),
        )
        error_spec = error_mapping.error_spec
        status_message = error_mapping.public_message or str(e)
        if error_mapping.protocol_error:
            logging.error(
                "[DashScGrpc] [%s] invalid admission code/reason pair: "
                "code=%s reason=%s",
                tag,
                int(e.exception_type),
                getattr(e, "admission_reject_reason", None),
            )
        if error_spec.status_code == 500:
            logging.exception("[DashScGrpc] [%s] engine error: %s", tag, e)
        elif error_spec.status_code == 499:
            logging.info("[DashScGrpc] [%s] engine cancelled: %s", tag, e)
        else:
            logging.warning("[DashScGrpc] [%s] engine rejected request: %s", tag, e)
        response = build_dash_error_response(
            str(request.id),
            request.model_name,
            error_spec=error_spec,
            status_message=status_message,
        )
        stats = (0, True, error_spec.finish_reason, len(input_ids_list), 0, ())
        yield (response, stats) if yield_access_stats else response
    except Exception as e:
        # Non-Ft failures (route RPC errors, transport aborts) still carry the
        # ``aux_info`` that ``BackendRPCServerVisitor.enqueue`` attaches, and it is
        # the only token accounting / pd_sep diagnostic the access log will ever
        # see for this request.
        _set_access_backend_error_code(access_agg, e)
        logging.exception("[DashScGrpc] [%s] enqueue failed: %s", tag, e)
        error_spec = DASH_ERROR_INTERNAL
        fallback_status_message = f"{type(e).__name__}: {e}"
        response = build_dash_error_response(
            str(request.id),
            request.model_name,
            error_spec=error_spec,
            status_message=fallback_status_message,
        )
        stats = (0, True, error_spec.finish_reason, len(input_ids_list), 0, ())
        yield (response, stats) if yield_access_stats else response
    finally:
        if phase2_stream is not None:
            await _close_async_stream_if_possible(phase2_stream, tag)
        if stream is not None:
            await _close_async_stream_if_possible(stream, tag)


# ----------------------------------------------------------------------------
# gRPC servicer (ModelStreamInfer entry)
# ----------------------------------------------------------------------------


class DashScInferenceServicer(predict_v2_pb2_grpc.GRPCInferenceServiceServicer):
    """ModelStreamInfer bridge to ``backend_visitor.enqueue``.

    ``ip`` / ``port`` / ``server_id`` derive the snowflake-style ``GenerateInput.request_id``
    via :func:`generate_request_id` — same scheme as the HTTP path in ``FrontendServer``, so
    the backend sees a single request_id generation policy. ``port`` should be the dash_sc
    gRPC listening port. The per-servicer sequence counter is intentionally independent of
    ``FrontendServer._global_controller``.
    """

    def __init__(
        self,
        backend_visitor: _BackendVisitor,
        *,
        ip: str = "",
        port: int = 0,
        server_id: str = "",
        echo_prefix_ids: Optional[list[int]] = None,
        extra_stop_word_ids: Optional[list[list[int]]] = None,
        tokenizer: BaseTokenizer | None = None,
        generate_env_config: GenerateEnvConfig | None = None,
        think_runtime: Optional[_ThinkRuntime] = None,
        rank_id: Optional[int] = None,
        repetition_monitor_config: Optional[RequestRepetitionMonitorConfig] = None,
        grammar_validator: Optional[GrammarValidator] = None,
    ):
        if backend_visitor is None:
            raise ValueError("backend_visitor is required for DashScInferenceServicer")
        self._backend_visitor = backend_visitor
        self._ip = ip
        self._port = port
        # Raw snowflake string seed for ``generate_request_id`` (request_id
        # generation needs the original string, not the log int below).
        self._snowflake_server_id = server_id
        self._echo_prefix_ids = list(echo_prefix_ids) if echo_prefix_ids else []
        self._extra_stop_word_ids = (
            [list(w) for w in extra_stop_word_ids] if extra_stop_word_ids else []
        )
        self._tokenizer = tokenizer
        self._generate_env_config = generate_env_config
        # Empty runtime is a safe default — phase-2 disabled, all dashllm limit
        # params null. Production callers (``DashScApp``) pre-build via
        # ``build_think_runtime`` so the per-request hot path is allocation-free.
        self._think_runtime = (
            think_runtime if think_runtime is not None else _ThinkRuntime()
        )
        self._seq_counter = AtomicCounter()
        set_request_id_factory = getattr(
            self._backend_visitor, "set_request_id_factory", None
        )
        if set_request_id_factory is not None:
            set_request_id_factory(self._next_rtp_llm_request_id)
        # Access-log identity is injected at construction. The two ids are the
        # only state the log + metric projections need; ``server_id`` arrives as
        # the snowflake string, coerced to ``Optional[int]`` once here. The
        # kmonitor tag dict is memoized per (rank, server) in ``grpc_metrics``,
        # so the per-chunk hot path never re-stringifies them. The repetition
        # monitor config lives only on this inference path, not the transparent
        # proxy.
        self._rank_id = rank_id
        self._server_id = to_optional_int(server_id)
        self._rep_cfg = repetition_monitor_config or RequestRepetitionMonitorConfig()
        # Optional admission-time grammar check. ``None`` keeps the legacy behaviour
        # (invalid grammars surface as an engine-side error mid-stream).
        self._grammar_validator = grammar_validator

    async def _validate_request_grammar(
        self, sampling: SamplingParams, request_id: str
    ) -> Optional[tuple[DashErrorSpec, str]]:
        """Trial-compile the current branch's grammar fields before enqueue.

        A structured-output request whose grammar cannot compile must fail as a 400 at
        admission; letting it reach the engine turns it into a mid-stream abort that the
        client sees as a 5xx (and, under MTP, can escalate into an executor abort).
        """
        validator = self._grammar_validator
        if validator is None:
            return None

        try:
            if sampling.structural_tag is not None:
                ok = await asyncio.to_thread(
                    validator.validate_structural_tag,
                    sampling.structural_tag,
                    request_id,
                )
                field_name = "tool_call_structural_tag"
            elif sampling.response_format is not None:
                ok = await asyncio.to_thread(
                    validator.validate_response_format,
                    sampling.response_format,
                    request_id,
                )
                field_name = "response_format"
            elif sampling.json_format:
                ok = await asyncio.to_thread(
                    validator.validate_json,
                    {"type": "object"},
                    request_id,
                )
                field_name = "json_format"
            else:
                return None
        except GrammarCompilationError as e:
            return DASH_ERROR_BAD_REQUEST, str(e)
        except GrammarCheckUnavailable as e:
            return (
                DASH_ERROR_BAD_REQUEST,
                f"grammar validation or compilation failed: {e}",
            )

        if ok:
            return None
        return (
            DASH_ERROR_BAD_REQUEST,
            f"invalid {field_name}: grammar validation or compilation failed",
        )

    def _record_and_report_chunk(
        self,
        record: GrpcAccessRecord,
        resp,
        *,
        delta_len: Optional[int] = None,
        finished: Optional[bool] = None,
        finish_reason: Optional[int] = None,
        prompt_token_num: Optional[int] = None,
        prompt_cached_token_num: Optional[int] = None,
    ) -> None:
        """Capture the frame and fan out per-chunk metrics (records, no log)."""
        is_first, now = record.record_response_chunk(resp)
        if prompt_token_num is not None and record.backend_input_len is None:
            record.backend_input_len = prompt_token_num
        if (
            prompt_cached_token_num is not None
            and record.prompt_cached_token_num is None
        ):
            record.prompt_cached_token_num = prompt_cached_token_num
        if finish_reason is not None:
            record.finish_reason = finish_reason
        if finished is not None:
            record.finished = finished
        if delta_len is not None:
            if delta_len:
                record.output_len += delta_len
                record.token_frame_count += 1
                record.max_tokens_per_frame = max(
                    record.max_tokens_per_frame, delta_len
                )
                if delta_len > 1:
                    record.multi_token_frame_count += 1
                if record.first_token_ts is None:
                    record.first_token_ts = now
                    record.first_token_frame_len = delta_len
                record.last_token_ts = now
            else:
                record.empty_frame_count += 1
        is_terminal = finished is True or (
            finish_reason is not None and finish_reason != LLMFinishReason.STREAMING
        )
        if is_terminal and not record.terminal_seen:
            record.terminal_seen = True
            record.terminal_ts = now
            if not delta_len:
                record.finished_only_frame_count += 1
        report_chunk(
            record,
            rank_id=self._rank_id,
            server_id=self._server_id,
            is_first=is_first,
            now=now,
        )

    async def close(self) -> None:
        """Hook for teardown; currently holds no resources (backend_visitor is owned by
        the caller, sequence counter is in-memory). Kept so future handles can be flushed
        here without changing the call-site in ``DashScGrpcServer.stop``.
        """

    def _next_rtp_llm_request_id(self) -> int:
        sequence = self._seq_counter.increment() % 4096  # 12 bits
        return generate_request_id(
            self._ip, self._port, self._snowflake_server_id, sequence
        )

    async def ModelStreamInfer(self, request_iterator, context):
        request_start_time = time.time_ns()
        request_start_ns = time.monotonic_ns()
        try:
            invocation_metadata = context.invocation_metadata()
        except Exception:
            invocation_metadata = ()
        metadata_headers = metadata_to_headers(invocation_metadata)
        # Self-managed access-log lifecycle (the shared interceptor is gone).
        # Create/arrival/query go first — before any inbound frame — so a
        # frame-less RPC (peer closed before sending) still reports arrival and
        # produces an access line via the ``finally`` below.
        # The SERVER span is delayed until the first frame reveals body-carried
        # trace context. The finally block idempotently creates a metadata/root
        # span for frame-less and pre-parse failure paths, then always ends it.
        record = GrpcAccessRecord.create(
            context,
            "ModelStreamInfer",
            "bidi_stream",
            raw_mode=False,
            repetition_monitor_config=self._rep_cfg,
        )
        trace_state = None
        current_rtp_llm_request_id: Optional[int] = None
        current_external_request_id = ""

        def _ensure_span(body_headers: Optional[dict[str, str]] = None):
            nonlocal trace_state
            if trace_state is not None:
                return trace_state
            headers, source = select_valid_server_trace_carrier(
                body_headers or {}, metadata_headers
            )
            trace_state = start_server_span(
                _DASH_SERVER_SPAN_NAME,
                headers,
                _DASH_SERVER_ATTRIBUTES,
                start_time=request_start_time,
                request_start_ns=request_start_ns,
            )
            if trace_state is not None:
                trace_state.set_attribute("rtp_llm.trace_context_source", source)
                if current_rtp_llm_request_id is not None:
                    trace_state.set_attribute(
                        "request_id", str(current_rtp_llm_request_id)
                    )
                    trace_state.set_attribute(
                        "rtp_llm.request_id", current_rtp_llm_request_id
                    )
                if current_external_request_id:
                    trace_state.set_attribute(
                        "rtp_llm.external_request_id", current_external_request_id
                    )
            return trace_state

        exc: Optional[BaseException] = None
        try:
            emit_query_log(record, rank_id=self._rank_id, server_id=self._server_id)
            partial_metadata_sent = False
            first_request = True
            async for request in request_iterator:
                record.req_count += 1
                rtp_llm_request_id = self._next_rtp_llm_request_id()
                current_rtp_llm_request_id = rtp_llm_request_id
                current_external_request_id = extract_span_external_request_id(
                    invocation_metadata, request
                )
                logging.debug(
                    "[DashScGrpc] ModelInferRequest: id=%s model_name=%s",
                    request.id,
                    request.model_name,
                )
                body_headers = extract_body_trace_headers(request)
                _ensure_span(body_headers)
                try:
                    parsed_input_ids, sampling, request_controls = (
                        parse_dash_sc_grpc_request(request)
                    )
                    mm_inputs = _build_mm_inputs_from_request(request)
                    traceparent_new = _lookup_ds_request_control(
                        parse_ds_header_attributes(request), "traceparent_new"
                    )
                    if (
                        traceparent_new
                        and body_headers.get("traceparent")
                        and str(traceparent_new) != body_headers["traceparent"]
                    ):
                        logging.warning(
                            "[DashScGrpc] body traceparent differs from traceparent_new"
                        )
                except (DashScParameterError, DashScInputIdsError) as e:
                    if first_request:
                        record.record_request_frame(request)
                        record.mark_request_done("eof")
                        first_request = False
                    error_spec = (
                        DASH_ERROR_BAD_REQUEST
                        if isinstance(e, DashScParameterError)
                        else DASH_ERROR_INTERNAL
                    )
                    resp = build_dash_error_response(
                        str(request.id),
                        request.model_name,
                        error_spec=error_spec,
                        status_message=str(e),
                    )
                    self._record_and_report_chunk(
                        record,
                        resp,
                        delta_len=0,
                        finished=True,
                        finish_reason=error_spec.finish_reason,
                    )
                    yield resp
                    return
                if parsed_input_ids is None:
                    if first_request:
                        record.record_request_frame(request)
                        record.mark_request_done("eof")
                        first_request = False
                    error_spec = DASH_ERROR_BAD_REQUEST
                    resp = build_dash_error_response(
                        str(request.id),
                        request.model_name,
                        error_spec=error_spec,
                        status_message="input_ids not found or raw_input_contents mismatch",
                    )
                    self._record_and_report_chunk(
                        record,
                        resp,
                        delta_len=0,
                        finished=True,
                        finish_reason=error_spec.finish_reason,
                    )
                    yield resp
                    return
                input_ids_list = parsed_input_ids.values
                if first_request:
                    # Hand the record the payload we just parsed so it does not
                    # decode the same request proto again (the input_ids tensor
                    # is large for long context).
                    record.capture_structured_request(
                        request,
                        input_ids=input_ids_list,
                        sampling=sampling,
                        request_controls=request_controls,
                    )
                    record.mark_request_done("eof")
                    # Priority-tagged twin of the entry-point arrival: deferred
                    # to here because the qos priority only becomes known once
                    # the first frame is parsed. RPCs that never get here are
                    # back-filled with priority="0" by the done metrics in the
                    # ``finally`` below.
                    report_arrival_priority(
                        record, rank_id=self._rank_id, server_id=self._server_id
                    )
                    first_request = False
                if (
                    not partial_metadata_sent
                    and request_controls is not None
                    and request_controls.timeout_ms is not None
                ):
                    await _send_partial_response_metadata(context)
                    partial_metadata_sent = True

                if sampling is not None and sampling.max_new_tokens <= 0:
                    param_name = (
                        "max_completion_tokens"
                        if sampling.max_new_tokens_from_completion_alias
                        else "max_new_tokens"
                    )
                    error_spec = DASH_ERROR_BAD_REQUEST
                    resp = build_dash_error_response(
                        str(request.id),
                        request.model_name,
                        error_spec=error_spec,
                        status_message=f"invalid {param_name}: {sampling.max_new_tokens}; must be greater than 0",
                    )
                    self._record_and_report_chunk(
                        record,
                        resp,
                        delta_len=0,
                        finished=True,
                        finish_reason=error_spec.finish_reason,
                    )
                    yield resp
                    return

                invalid_grammar = await self._validate_request_grammar(
                    sampling, str(request.id)
                )
                if invalid_grammar is not None:
                    error_spec, status_message = invalid_grammar
                    resp = build_dash_error_response(
                        str(request.id),
                        request.model_name,
                        error_spec=error_spec,
                        status_message=status_message,
                    )
                    self._record_and_report_chunk(
                        record,
                        resp,
                        delta_len=0,
                        finished=True,
                        finish_reason=error_spec.finish_reason,
                    )
                    yield resp
                    return

                response_iter = iter_real_model_stream_infer(
                    request,
                    input_ids_list,
                    sampling,
                    request_controls,
                    self._backend_visitor,
                    rtp_llm_request_id=rtp_llm_request_id,
                    echo_prefix_ids=self._echo_prefix_ids,
                    extra_stop_word_ids=self._extra_stop_word_ids,
                    invocation_metadata=invocation_metadata,
                    tokenizer=self._tokenizer,
                    generate_env_config=self._generate_env_config,
                    think_runtime=self._think_runtime,
                    phase2_request_id_factory=self._next_rtp_llm_request_id,
                    access_agg=record,
                    input_ids_tensor=parsed_input_ids.tensor,
                    yield_access_stats=True,
                    mm_inputs=mm_inputs,
                )
                try:
                    async for resp, stats in response_iter:
                        (
                            delta_len,
                            finished,
                            finish_reason,
                            prompt_token_num,
                            prompt_cached_token_num,
                            generated_ids_for_log,
                        ) = stats
                        record.record_generated_ids(generated_ids_for_log)
                        self._record_and_report_chunk(
                            record,
                            resp,
                            delta_len=delta_len,
                            finished=finished,
                            finish_reason=finish_reason,
                            prompt_token_num=prompt_token_num,
                            prompt_cached_token_num=prompt_cached_token_num,
                        )
                        if trace_state is not None and generated_ids_for_log:
                            trace_state.record_frontend_output_tokens(
                                len(generated_ids_for_log)
                            )
                        yield resp
                finally:
                    await response_iter.aclose()
                return
            if first_request:
                record.mark_request_done("eof")
        except BaseException as e:
            exc = e
            # Keep whatever the chunk loop already recorded; only fill in when the
            # exception itself is the sole aux_info carrier.
            record.record_aux_info(getattr(e, "aux_info", None), overwrite=False)
            raise
        finally:
            _ensure_span()
            end_ts = record.resolve_status(context, exc)
            try:
                # Log first, metrics second — a kmonitor hiccup must never delay or
                # drop the access record (user-mandated ordering).
                emit_access_log(
                    record,
                    rank_id=self._rank_id,
                    server_id=self._server_id,
                    end_ts=end_ts,
                )
                report_frontend_rpc_done(
                    record,
                    rank_id=self._rank_id,
                    server_id=self._server_id,
                    status=record.status,
                )
            finally:
                _finish_server_trace(trace_state, record, exc)
