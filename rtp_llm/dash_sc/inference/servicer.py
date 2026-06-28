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

import inspect
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Iterator, Optional

import torch

from rtp_llm.config.exceptions import (
    ExceptionCategory,
    ExceptionType,
    FtRuntimeException,
)
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.dash_sc.access_log import emit_access_log, emit_query_log
from rtp_llm.dash_sc.access_record import GrpcAccessRecord, to_optional_int
from rtp_llm.dash_sc.codec import (
    DASH_ERROR_ABORT,
    DASH_ERROR_BAD_REQUEST,
    DASH_ERROR_CAPACITY,
    DASH_ERROR_INTERNAL,
    DASH_ERROR_INVALID_OUTPUT,
    DASH_ERROR_TIMEOUT,
    DASH_ERROR_TOO_LONG,
    DASH_ERROR_UNSUPPORTED,
    DashErrorSpec,
    DashScParameterError,
    LLMFinishReason,
    OtherParams,
    SamplingParams,
    _token_ids_list_from_generate_output,
    build_dash_error_response,
    build_stream_response_from_generate_outputs,
    iter_fake_model_stream_infer,
    parse_dash_sc_grpc_request,
    prepend_to_generated_ids_tensor,
)
from rtp_llm.dash_sc.grpc_metrics import (
    report_arrival,
    report_chunk,
    report_frontend_rpc_done,
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
from rtp_llm.utils.base_model_datatypes import GenerateInput, RequestInfo
from rtp_llm.utils.deepseekv4_constants import DSML_PREFIX, DSML_TOOL_CALLS_MARKER
from rtp_llm.utils.util import AtomicCounter

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
_INT32_MAX = 2_147_483_647
_PARTIAL_RESPONSE_METADATA = (("x-dashscope-partialresponse", "true"),)


def _exception_metric_code(error_code: Any) -> str:
    code = int(error_code)
    try:
        return f"{code}_{ExceptionType.from_value(code)}"
    except ValueError:
        return str(code)


def _set_access_backend_error_code(access_agg: Any, e: BaseException) -> None:
    if access_agg is None:
        return
    if not isinstance(e, FtRuntimeException):
        return
    access_agg.backend_error_code = _exception_metric_code(int(e.exception_type))


def _dash_error_spec_for_ft_exception(exc: FtRuntimeException) -> DashErrorSpec:
    return _DASH_ERROR_SPEC_BY_EXCEPTION_CATEGORY[exc.exception_type.category]


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
    invocation_metadata: Optional[Any],
) -> dict[str, str]:
    metadata_headers = {
        str(key).lower(): value
        for key, value in invocation_metadata or ()
        if key is not None and value is not None
    }
    return extract_request_headers(metadata_headers)


async def _send_partial_response_metadata(context: Any) -> None:
    sender = getattr(context, "send_initial_metadata", None)
    if sender is None:
        return
    result = sender(_PARTIAL_RESPONSE_METADATA)
    if inspect.isawaitable(result):
        await result


def _optional_int_attr(obj: Any, attr: str) -> Optional[int]:
    try:
        value = getattr(obj, attr, None)
    except Exception:
        return None
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _derive_max_token_id(tokenizer: Any) -> Optional[int]:
    if tokenizer is None:
        return None
    try:
        size = len(tokenizer)
    except Exception:
        size = _optional_int_attr(tokenizer, "vocab_size") or 0
    return size - 1 if size > 0 else None


def _hf_tokenizer(tokenizer: Any) -> Any:
    return getattr(tokenizer, "tokenizer", tokenizer)


def _decode_env_tag(generate_env_config: Any, attr: str) -> str:
    """Read ``attr`` from generate_env_config and unescape literal ``\\n`` etc.

    No literal default here — ``GenerateEnvConfig`` is the single source of truth
    for tag defaults. Missing attribute or empty value returns "".
    """
    value = getattr(generate_env_config, attr, "") or ""
    return str(value).encode("utf-8").decode("unicode_escape")


def _encode_tag(tokenizer: Any, text: str) -> list[int]:
    hf_tok = _hf_tokenizer(tokenizer)
    if hf_tok is None or not text:
        return []
    return list(hf_tok.encode(text, add_special_tokens=False))


def _decode_token_ids(tokenizer: Any, ids: list[int]) -> str:
    hf_tok = _hf_tokenizer(tokenizer)
    if hf_tok is None or not ids:
        return ""
    decode = getattr(hf_tok, "decode", None)
    if decode is None:
        return ""
    # ``clean_up_tokenization_spaces`` MUST stay off: the DSML marker detection
    # maps decoded *character* offsets back to token offsets via
    # ``_token_offset_for_decoded_char``, which assumes ``decode(ids[:k])`` is a
    # length-monotonic prefix of ``decode(ids)``. Space cleanup rewrites spacing
    # non-locally and breaks that invariant, shifting the reasoning boundary.
    try:
        return str(decode(list(ids), clean_up_tokenization_spaces=False))
    except TypeError:
        # Some tokenizer shims don't accept the kwarg; fall back to plain decode.
        try:
            return str(decode(list(ids)))
        except Exception:
            return ""
    except Exception:
        return ""


def _is_deepseek_v4(model_type: Optional[str]) -> bool:
    normalized = str(model_type or "").replace("-", "_").lower()
    return normalized == "deepseek_v4" or normalized.startswith("deepseek_v4_")


def _matched_echo_prefix_ids(
    input_ids_list: list[int], echo_prefix_ids: Optional[list[int]]
) -> list[int]:
    """Return the exact think-BOS ids already present at the input tail."""
    prefix_ids = list(echo_prefix_ids) if echo_prefix_ids else []
    if not prefix_ids:
        return []
    n = len(prefix_ids)
    if len(input_ids_list) >= n and list(input_ids_list[-n:]) == prefix_ids:
        return prefix_ids
    if n > 1 and input_ids_list and input_ids_list[-1] == prefix_ids[0]:
        return [prefix_ids[0]]
    return []


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
      ``phase2_enabled``     ``is_dsv4 and bool(empty_tokens)``. ``terminate_token_id``
                             is intentionally *not* part of this gate — even with the
                             token-terminate branch off, dsv4 still needs the
                             phase-2-on-close machinery.
      ``tool_calls_marker``  text marker that implicitly ends reasoning when DSV4
                             starts tool-call markup before emitting ``</think>``.
                             This is intentionally text-level rather than
                             token-level: DSV4 tokenizers may merge ``<`` with the
                             previous character and ``>`` with the following one.
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
    tool_calls_marker: str = ""
    eos_token_id: Optional[int] = None
    max_token_id: Optional[int] = None


def build_think_runtime(
    tokenizer: Any,
    generate_env_config: Any,
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
        eos_token_id
        if eos_token_id is not None
        else _optional_int_attr(tokenizer, "eos_token_id")
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
    think_start_tag = _decode_env_tag(generate_env_config, "think_start_tag")
    think_end_tag = _decode_env_tag(generate_env_config, "think_end_tag")
    bos_tokens = tuple(_encode_tag(tokenizer, think_start_tag))
    eos_tokens = tuple(_encode_tag(tokenizer, think_end_tag))
    empty_tokens = tuple(
        _encode_tag(tokenizer, think_start_tag + _EMPTY_THINK_BODY + think_end_tag)
    )
    close_token_id = int(eos_tokens[0]) if eos_tokens else None
    is_dsv4 = _is_deepseek_v4(model_type)
    phase2_enabled = is_dsv4 and bool(empty_tokens)
    tool_calls_marker = DSML_TOOL_CALLS_MARKER if is_dsv4 else ""
    return _ThinkRuntime(
        bos_tokens=bos_tokens,
        eos_tokens=eos_tokens,
        empty_tokens=empty_tokens,
        close_token_id=close_token_id,
        terminate_token_id=term_id,
        phase2_enabled=phase2_enabled,
        tool_calls_marker=tool_calls_marker,
        eos_token_id=eos_tid,
        max_token_id=max_tid,
    )


def _phase2_input_ids_for_deepseek_v4(
    input_ids_list: list[int],
    matched_bos_ids: list[int],
    empty_think_tokens: list[int],
) -> list[int]:
    base = list(input_ids_list)
    if matched_bos_ids and base[-len(matched_bos_ids) :] == matched_bos_ids:
        base = base[: -len(matched_bos_ids)]
    return base + list(empty_think_tokens)


def _strip_trailing_eos(
    generated_ids: list[int], eos_seq: tuple[int, ...]
) -> list[int]:
    """Drop a single trailing ``eos_seq`` match from ``generated_ids``.

    Phase-2 sometimes ends its answer with a structural ``</think>\\n\\n``
    closing tag mirroring the empty-think prompt body — that artifact must not
    leak into the dashscope-side ``content`` field.
    """
    n = len(eos_seq)
    if n == 0 or len(generated_ids) < n:
        return generated_ids
    if list(generated_ids[-n:]) == list(eos_seq):
        return list(generated_ids[:-n])
    return generated_ids


def _split_on_first_close(
    generated_ids: list[int],
    close_token_id: Optional[int],
    eos_seq: tuple[int, ...],
) -> tuple[Optional[int], list[int]]:
    """Find the first ``close_token_id`` and return ``(idx, post_close_ids)``.

    The post-close suffix has the rest of ``eos_seq`` consumed if it appears
    immediately after the close token, so a multi-token ``</think>\\n\\n`` is
    treated as a single boundary. Returns ``(None, generated_ids)`` if the
    close token is not present.
    """
    if close_token_id is None:
        return None, list(generated_ids)
    for i, tid in enumerate(generated_ids):
        if tid == close_token_id:
            tail_start = i + 1
            if len(eos_seq) > 1:
                rest = list(eos_seq[1:])
                if list(generated_ids[tail_start : tail_start + len(rest)]) == rest:
                    tail_start += len(rest)
            return i, list(generated_ids[tail_start:])
    return None, list(generated_ids)


def _longest_suffix_prefix_text_len(text: str, prefix: str) -> int:
    max_len = min(len(text), max(0, len(prefix) - 1))
    for n in range(max_len, 0, -1):
        if text[-n:] == prefix[:n]:
            return n
    return 0


def _token_offset_for_decoded_char(
    ids: list[int], tokenizer: Any, char_offset: int
) -> int:
    if char_offset <= 0:
        return 0
    if not ids:
        return 0
    decoded_len_cache: dict[int, int] = {0: 0}

    def decoded_prefix_len(token_count: int) -> int:
        if token_count not in decoded_len_cache:
            decoded_len_cache[token_count] = len(
                _decode_token_ids(tokenizer, ids[:token_count])
            )
        return decoded_len_cache[token_count]

    total_len = decoded_prefix_len(len(ids))
    if char_offset >= total_len:
        return len(ids)

    lo, hi = 1, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if decoded_prefix_len(mid) >= char_offset:
            hi = mid
        else:
            lo = mid + 1
    boundary = lo
    if decoded_prefix_len(boundary) == char_offset:
        return boundary
    return max(0, boundary - 1)


def _find_dsv4_tool_call_marker_offset(
    ids: list[int], marker: str, tokenizer: Any, text: Optional[str] = None
) -> Optional[int]:
    if not ids or not marker:
        return None
    try:
        if text is None:
            text = _decode_token_ids(tokenizer, ids)
        marker_char_offset = text.find(marker)
        if marker_char_offset < 0:
            return None
        return _token_offset_for_decoded_char(ids, tokenizer, marker_char_offset)
    except Exception:
        return None


def _dsv4_tool_call_marker_pending_token_len(
    ids: list[int], marker: str, tokenizer: Any, text: Optional[str] = None
) -> int:
    if not ids or not marker:
        return 0
    try:
        if text is None:
            text = _decode_token_ids(tokenizer, ids)
        hold_chars = _longest_suffix_prefix_text_len(text, marker)
        if hold_chars <= 0:
            return 0
        suffix_char_offset = len(text) - hold_chars
        suffix_token_offset = _token_offset_for_decoded_char(
            ids, tokenizer, suffix_char_offset
        )
        return max(0, len(ids) - suffix_token_offset)
    except Exception:
        return 0


def _first_token_offset(ids: list[int], token_id: Optional[int]) -> Optional[int]:
    if token_id is None:
        return None
    try:
        return ids.index(token_id)
    except ValueError:
        return None


def _earliest_boundary(
    *items: tuple[str, Optional[int]],
) -> tuple[Optional[str], Optional[int]]:
    candidates = [(name, offset) for name, offset in items if offset is not None]
    if not candidates:
        return None, None
    return min(candidates, key=lambda item: item[1])


def _is_dsv4_dsml_begin(text: Any) -> bool:
    """True if ``text`` opens a DSV4 DSML grammar.

    Matches the ``<｜DSML｜`` prefix rather than only ``<｜DSML｜tool_calls>`` so
    grammars expressed via the inner ``invoke`` trigger (which legitimately
    follows the tool-call marker) are recognized too. Non-DSV4 grammars
    (``json_object``/``regex``/other tool formats like ``<tool_call>``) never
    start with ``<｜DSML｜``, so this stays a precise DSV4 signal.
    """
    return isinstance(text, str) and text.startswith(DSML_PREFIX)


def _format_marks_dsv4_tool_call(fmt: Any) -> bool:
    if not isinstance(fmt, dict):
        return False
    fmt_type = fmt.get("type")
    if fmt_type == "tag":
        return _is_dsv4_dsml_begin(fmt.get("begin"))
    if fmt_type == "sequence":
        elements = fmt.get("elements")
        if not isinstance(elements, list) or not elements:
            return False
        first = elements[0]
        return (
            isinstance(first, dict)
            and first.get("type") == "const_string"
            and _is_dsv4_dsml_begin(first.get("value"))
        )
    if fmt_type == "triggered_tags":
        triggers = fmt.get("triggers")
        if isinstance(triggers, list) and any(_is_dsv4_dsml_begin(t) for t in triggers):
            return True
        tags = fmt.get("tags")
        if isinstance(tags, list) and any(
            isinstance(tag, dict) and _is_dsv4_dsml_begin(tag.get("begin"))
            for tag in tags
        ):
            return True
    return False


def _legacy_structural_tag_marks_dsv4_tool_call(value: dict) -> bool:
    """Detect DSV4 tool-call grammar in the legacy ``structures``/``triggers``
    schema (the non-``format`` shape that ``validate_structural_tag_shape``
    still accepts)."""
    triggers = value.get("triggers")
    if isinstance(triggers, list) and any(_is_dsv4_dsml_begin(t) for t in triggers):
        return True
    structures = value.get("structures")
    if isinstance(structures, list):
        for st in structures:
            if isinstance(st, dict) and _is_dsv4_dsml_begin(st.get("begin")):
                return True
    return False


def _has_dsv4_tool_call_structural_tag(generate_config: Any) -> bool:
    value = getattr(generate_config, "structural_tag", None)
    if not value:
        return False
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return False
    if not isinstance(value, dict):
        return False
    fmt = value.get("format")
    if isinstance(fmt, dict):
        return _format_marks_dsv4_tool_call(fmt)
    return _legacy_structural_tag_marks_dsv4_tool_call(value)


def _make_generate_input(
    *,
    request_id: int,
    input_ids_list: list[int],
    generate_config: Any,
    invocation_metadata: Optional[Any],
    request_headers: Optional[dict[str, str]] = None,
) -> GenerateInput:
    headers = dict(request_headers or {})
    headers.update(_headers_from_invocation_metadata(invocation_metadata))
    trace_id = str(
        getattr(generate_config, "trace_id", "") or extract_trace_id(headers) or ""
    )
    return GenerateInput(
        request_id=request_id,
        token_ids=torch.tensor(input_ids_list, dtype=torch.int),
        mm_inputs=[],
        generate_config=generate_config,
        headers=headers,
        request_info=RequestInfo(
            trace_id=trace_id,
            request_id=extract_correlation_request_id(headers) or trace_id,
            source_role="dash",
        ),
    )


async def _close_async_stream_if_possible(stream: Any, tag: str) -> None:
    close = getattr(stream, "aclose", None)
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


def _apply_request_overrides(
    generate_config: Any,
    sampling: SamplingParams,
    other: OtherParams,
    runtime: _ThinkRuntime,
) -> None:
    """Apply dash-sc request-level controls after env defaults.

    ``GenerateConfig.add_thinking_params`` seeds the config from process-level
    environment. DashScope-serving still sends per-request thinking, timeout,
    and priority controls; those explicit controls must win before enqueue.
    """
    request_max_think = sampling.max_new_think_tokens
    if request_max_think is None:
        request_max_think = other.max_new_think_tokens
    if request_max_think is not None:
        max_think = int(request_max_think)
        generate_config.max_thinking_tokens = _INT32_MAX if max_think < 0 else max_think
    # Only the selected budget disables thinking; ``max_think_length`` may
    # intentionally override a zero ``max_new_think_tokens`` alias.
    disable_by_budget = request_max_think == 0
    # DashLLM/sglang-compatible Dash requests stay in chat mode unless the
    # caller explicitly asks for thinking or a request-scoped think budget.
    disable_by_default = other.enable_thinking is None and request_max_think is None
    if other.enable_thinking is False or disable_by_budget or disable_by_default:
        generate_config.in_think_mode = False
        generate_config.max_thinking_tokens = 0
        if hasattr(generate_config, "thinking"):
            generate_config.thinking = False
    elif (other.enable_thinking is True or request_max_think is not None) and (
        getattr(generate_config, "end_think_token_ids", None) or runtime.eos_tokens
    ):
        generate_config.in_think_mode = True
        if not getattr(generate_config, "end_think_token_ids", None):
            generate_config.end_think_token_ids = list(runtime.eos_tokens)
        if hasattr(generate_config, "thinking"):
            generate_config.thinking = True
    if other.timeout_ms is not None:
        # Subtract a margin so the engine times out BEFORE the upstream gateway
        # sends RST_STREAM. This ensures the timeout surfaces as a normal
        # finish_reason=STOP_TIMEOUT response (200) rather than gRPC CANCELLED (5xx).
        margin_ms = max(2000, min(5000, int(other.timeout_ms * 0.15)))
        engine_timeout_ms = max(5000, int(other.timeout_ms) - margin_ms)
        generate_config.timeout_ms = engine_timeout_ms
        generate_config.ttft_timeout_ms = engine_timeout_ms
    if other.traffic_reject_priority is not None:
        generate_config.traffic_reject_priority = int(other.traffic_reject_priority)
    if other.reasoning_effort is not None:
        kwargs = dict(getattr(generate_config, "chat_template_kwargs", None) or {})
        kwargs["reasoning_effort"] = other.reasoning_effort
        generate_config.chat_template_kwargs = kwargs


# ----------------------------------------------------------------------------
# Real inference bridge: async backend enqueue -> aio gRPC async generator
# ----------------------------------------------------------------------------


async def iter_real_model_stream_infer(
    request,
    input_ids_list: list[int],
    sampling: SamplingParams,
    other: OtherParams,
    backend_visitor: Any,
    *,
    rtp_llm_request_id: int,
    echo_prefix_ids: Optional[list[int]] = None,
    extra_stop_word_ids: Optional[list[list[int]]] = None,
    invocation_metadata: Optional[Any] = None,
    tokenizer: Any = None,
    generate_env_config: Any = None,
    think_runtime: Optional[_ThinkRuntime] = None,
    phase2_request_id_factory: Optional[Callable[[], int]] = None,
    access_agg: Any = None,
    yield_access_stats: bool = False,
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
    try:
        generate_config = sampling.to_generate_config(other=other)
        generate_config.trace_id = trace_str
        if generate_env_config is not None:
            try:
                hf_tok = _hf_tokenizer(tokenizer)
                if (
                    hf_tok is None
                    and getattr(generate_env_config, "think_end_token_id", -1) == -1
                ):
                    logging.warning(
                        "[DashScGrpc] [%s] skip add_thinking_params: tokenizer missing",
                        tag,
                    )
                else:
                    generate_config.add_thinking_params(hf_tok, generate_env_config)
            except Exception as e:
                logging.warning(
                    "[DashScGrpc] [%s] add_thinking_params failed: %s", tag, e
                )
        begin_think_tokens = list(runtime.bos_tokens or tuple(echo_prefix_ids or ()))
        if begin_think_tokens and hasattr(generate_config, "begin_think_token_ids"):
            generate_config.begin_think_token_ids = begin_think_tokens
        if runtime.eos_tokens and not getattr(
            generate_config, "end_think_token_ids", None
        ):
            generate_config.end_think_token_ids = list(runtime.eos_tokens)
        _apply_request_overrides(generate_config, sampling, other, runtime)
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
        # All these are pre-resolved at servicer init via ``build_think_runtime``;
        # reading them here is O(1) and avoids per-request tokenizer.encode.
        eos_id = runtime.eos_token_id
        max_id = runtime.max_token_id
        term_id = runtime.terminate_token_id
        think_close_token_id = runtime.close_token_id
        max_new_tokens = int(getattr(generate_config, "max_new_tokens", 0) or 0)
        matched_think_bos_ids = matched_echo_ids or _matched_echo_prefix_ids(
            input_ids_list, list(runtime.bos_tokens)
        )
        # ``runtime.phase2_enabled`` is the init-time gate (model_type + empty_tokens
        # availability). ``in_think_mode`` is per-request — ``add_thinking_params``
        # sets it from generate_config and a request can override it.
        in_think_mode = bool(getattr(generate_config, "in_think_mode", False))
        phase2_enabled = runtime.phase2_enabled and in_think_mode
        cumulative_sent_ids: list[int] = []
        generate_think_token_num: Optional[int] = None
        tool_call_marker = runtime.tool_calls_marker
        # The DSML tool-call marker only implies an *implicit reasoning boundary*,
        # so it is only meaningful while thinking. With ``in_think_mode=False``
        # there is no reasoning to split: the tool call is generated directly
        # (grammar applied by the backend via ``structural_tag``), and fabricating
        # a ``generate_think_token_num`` here would wrongly tag leading content as
        # reasoning. Gate it on ``in_think_mode`` to mirror ``phase2_enabled``.
        tool_call_marker_active = bool(tool_call_marker) and in_think_mode
        pending_tool_call_marker_ids: list[int] = []
        generate_input = _make_generate_input(
            request_id=rtp_llm_request_id,
            input_ids_list=input_ids_list,
            generate_config=generate_config,
            invocation_metadata=invocation_metadata,
            request_headers=other.request_headers,
        )
        is_streaming = bool(getattr(generate_config, "is_streaming", True))
        logging.debug("[DashScGrpc] [%s] generate_input: %s", tag, generate_input)
        request_shape = list(request.inputs[0].shape) if request.inputs else None
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
            aux_info = getattr(out_py, "aux_info", None)
            prompt_token_num = (
                int(aux_info.input_len) if aux_info is not None else len(input_ids_list)
            )
            prompt_cached_token_num = (
                int(aux_info.reuse_len) if aux_info is not None else 0
            )
            if access_agg is not None and aux_info is not None and aux_info.role_addrs:
                # model_rpc_client copies the final submitted role_addrs here.
                access_agg.record_role_addrs(aux_info.role_addrs, phase="phase1")
            if not generated_ids and not out_py.finished:
                response = build_stream_response_from_generate_outputs(
                    dash_sc_request_id=request.id,
                    model_name=request.model_name,
                    go=go,
                    request_log_tag=tag,
                    request_input_ids=input_ids_list,
                    return_input_ids=other.return_input_ids,
                    is_streaming=is_streaming,
                    generate_config=generate_config,
                    eos_token_id=eos_id,
                    max_token_id=max_id,
                    _request_shape=request_shape,
                )
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
            force_phase2_boundary: Optional[int] = None
            if generate_think_token_num is None and tool_call_marker_active:
                combined_ids = pending_tool_call_marker_ids + generated_ids
                # Decode the small window once and reuse it for both the
                # full-marker search and the partial-prefix hold check, so the
                # reasoning hot path pays at most one decode per chunk.
                combined_text = _decode_token_ids(tokenizer, combined_ids)
                marker_offset = _find_dsv4_tool_call_marker_offset(
                    combined_ids, tool_call_marker, tokenizer, text=combined_text
                )
                close_candidate = _first_token_offset(
                    combined_ids, think_close_token_id
                )
                term_candidate = (
                    _first_token_offset(combined_ids, term_id)
                    if phase2_enabled and not phase2_triggered
                    else None
                )
                boundary_kind, boundary_offset = _earliest_boundary(
                    ("close", close_candidate),
                    ("term", term_candidate),
                    ("tool_call_marker", marker_offset),
                )
                if boundary_kind == "tool_call_marker" and boundary_offset is not None:
                    pending_tool_call_marker_ids = []
                    generated_ids = combined_ids
                    tool_call_marker_active = False
                    if phase2_enabled and _has_dsv4_tool_call_structural_tag(
                        generate_config
                    ):
                        force_phase2_boundary = boundary_offset
                        logging.info(
                            "[DashScGrpc] [%s] DSV4 tool-call marker ended thinking; "
                            "switch to phase-2 structural_tag grammar",
                            tag,
                        )
                    else:
                        echo_len = (
                            len(matched_echo_ids)
                            if should_echo and not echoed and generated_ids
                            else 0
                        )
                        generate_think_token_num = (
                            len(cumulative_sent_ids) + echo_len + boundary_offset
                        )
                        logging.info(
                            "[DashScGrpc] [%s] DSV4 tool-call marker ended thinking; "
                            "continue same stream for downstream tool parser",
                            tag,
                        )
                elif boundary_kind is not None:
                    pending_tool_call_marker_ids = []
                    generated_ids = combined_ids
                else:
                    hold_len = (
                        0
                        if out_py.finished
                        else _dsv4_tool_call_marker_pending_token_len(
                            combined_ids,
                            tool_call_marker,
                            tokenizer,
                            text=combined_text,
                        )
                    )
                    if hold_len:
                        generated_ids = combined_ids[:-hold_len]
                        pending_tool_call_marker_ids = combined_ids[-hold_len:]
                    else:
                        generated_ids = combined_ids
                        pending_tool_call_marker_ids = []
                    if not generated_ids and not out_py.finished:
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
                    # (DashLLM-aligned). Phase-2 is reserved for explicit
                    # think-abort boundaries: DSV4 token 1, or a DSV4 tool-call
                    # marker when a matching structural_tag grammar is active.
            if (
                phase2_enabled
                and not phase2_triggered
                and generate_think_token_num is None
                and generated_ids
                and (
                    force_phase2_boundary is not None
                    or (term_id is not None and term_id in generated_ids)
                )
            ):
                phase2_cut_idx = (
                    force_phase2_boundary
                    if force_phase2_boundary is not None
                    else generated_ids.index(term_id)
                )
                generated_ids = generated_ids[:phase2_cut_idx]
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
                    response = build_stream_response_from_generate_outputs(
                        dash_sc_request_id=request.id,
                        model_name=request.model_name,
                        go=go,
                        request_log_tag=tag,
                        request_input_ids=input_ids_list,
                        return_input_ids=other.return_input_ids,
                        is_streaming=is_streaming,
                        generate_config=generate_config,
                        eos_token_id=eos_id,
                        max_token_id=max_id,
                        generate_think_token_num=generate_think_token_num,
                        _request_shape=request_shape,
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
                    eos_response = build_stream_response_from_generate_outputs(
                        dash_sc_request_id=request.id,
                        model_name=request.model_name,
                        go=go,
                        request_log_tag=tag,
                        request_input_ids=input_ids_list,
                        return_input_ids=other.return_input_ids,
                        is_streaming=is_streaming,
                        generate_config=generate_config,
                        eos_token_id=eos_id,
                        max_token_id=max_id,
                        generate_think_token_num=generate_think_token_num,
                        finish_reason_override=(
                            LLMFinishReason.LENGTH if not will_do_phase2 else None
                        ),
                        _request_shape=request_shape,
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
            response = build_stream_response_from_generate_outputs(
                dash_sc_request_id=request.id,
                model_name=request.model_name,
                go=go,
                request_log_tag=tag,
                request_input_ids=input_ids_list,
                return_input_ids=other.return_input_ids,
                is_streaming=is_streaming,
                generate_config=generate_config,
                eos_token_id=eos_id,
                max_token_id=max_id,
                generate_think_token_num=generate_think_token_num,
                finish_reason_override=finish_reason_override,
                _request_shape=request_shape,
                token_ids=generated_ids,
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
        # policy: phase-2 is initiated only by explicit think-abort boundaries
        # (DSV4 token 1, or tool-call marker + matching structural_tag grammar).
        # If phase-1 reaches stream end without such a boundary, do NOT silently
        # restart with empty-think.
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
            if hasattr(phase2_config, "thinking"):
                phase2_config.thinking = False
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
            phase2_input_ids = _phase2_input_ids_for_deepseek_v4(
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
                request_headers=other.request_headers,
            )
            logging.debug(
                "[DashScGrpc] [%s] phase-2 generate_input: %s",
                phase2_tag,
                phase2_generate_input,
            )
            phase2_stream = await backend_visitor.enqueue(phase2_generate_input)
            phase2_cumulative_sent_ids: list[int] = []

            def _build_phase2_response(
                resp_go: Any,
            ) -> predict_v2_pb2.ModelStreamInferResponse:
                resp_out = resp_go.generate_outputs[0]
                resp_ids = _token_ids_list_from_generate_output(resp_out)
                phase2_cumulative_sent_ids.extend(resp_ids)
                phase2_max_new_tokens = int(
                    getattr(phase2_config, "max_new_tokens", 0) or 0
                )
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
                aux_info = getattr(resp_out, "aux_info", None)
                prompt_token_num = (
                    int(aux_info.input_len)
                    if aux_info is not None
                    else len(phase2_input_ids)
                )
                prompt_cached_token_num = (
                    int(aux_info.reuse_len) if aux_info is not None else 0
                )
                if (
                    access_agg is not None
                    and aux_info is not None
                    and aux_info.role_addrs
                ):
                    # model_rpc_client copies the final submitted role_addrs here.
                    access_agg.record_role_addrs(aux_info.role_addrs, phase="phase2")
                response = build_stream_response_from_generate_outputs(
                    dash_sc_request_id=f"{request.id}{_PHASE2_SUFFIX}",
                    model_name=request.model_name,
                    go=resp_go,
                    request_log_tag=phase2_tag,
                    request_input_ids=phase2_input_ids,
                    return_input_ids=other.return_input_ids,
                    is_streaming=is_streaming,
                    generate_config=phase2_config,
                    eos_token_id=eos_id,
                    max_token_id=max_id,
                    generate_think_token_num=generate_think_token_num,
                    finish_reason_override=finish_reason_override,
                    _request_shape=request_shape,
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

            # Phase-2 sanitization. The phase-2 prompt ends with
            # ``<think>\n</think>\n\n``; the model occasionally interprets that
            # as "think + close again" instead of "content only from here".
            # Two failure modes observed in MRCR:
            #
            #   Case A (leading thinking + answer):
            #     phase-2 emits ``[reasoning..., </think>, answer...]``. Pre-
            #     close tokens are accidental reasoning and must NOT reach
            #     ``content``. Discard them, drop the close + eos rest, emit
            #     only post-close.
            #
            #   Case B (clean answer + trailing eos artifact):
            #     phase-2 emits ``[answer..., </think>\n\n]`` and then EOSes.
            #     Pre-close tokens ARE the real content. Keep them, drop only
            #     the trailing close + eos rest.
            #
            # The two cases are distinguished by whether tokens follow the
            # close: post-close non-empty → Case A; post-close empty AND chunk
            # finished → Case B; otherwise ambiguous (close split across
            # chunks) → default to Case A so the next chunk's content streams
            # cleanly. Pre-close chunks are buffered in ``phase2_pending``
            # until classification completes.
            phase2_pending: list[Any] = []
            phase2_seen_close = False

            def _flush_phase2_pending() -> (
                Iterator[predict_v2_pb2.ModelStreamInferResponse]
            ):
                """Yield buffered chunks, stripping a trailing eos artifact
                from whichever chunk carries the finish flag."""
                for buf_go in phase2_pending:
                    buf_out = buf_go.generate_outputs[0]
                    if buf_out.finished and runtime.eos_tokens:
                        buf_ids = _token_ids_list_from_generate_output(buf_out)
                        cleaned = _strip_trailing_eos(buf_ids, runtime.eos_tokens)
                        if cleaned != buf_ids:
                            buf_out.output_ids = torch.tensor(
                                cleaned, dtype=torch.int32
                            )
                    resp, stats = _build_phase2_response(buf_go)
                    yield (resp, stats) if yield_access_stats else resp

            async for go in phase2_stream:
                if not go.generate_outputs:
                    raise ValueError("empty generate_outputs in phase-2 backend chunk")
                out_py = go.generate_outputs[0]
                generated_ids = _token_ids_list_from_generate_output(out_py)
                if not generated_ids and not out_py.finished:
                    continue

                if phase2_seen_close:
                    # Past the boundary: trailing-eos cleanup on the final
                    # chunk, otherwise pass through.
                    if out_py.finished and runtime.eos_tokens:
                        cleaned = _strip_trailing_eos(generated_ids, runtime.eos_tokens)
                        if cleaned != generated_ids:
                            generated_ids = cleaned
                            out_py.output_ids = torch.tensor(
                                generated_ids, dtype=torch.int32
                            )
                    if generated_ids or out_py.finished:
                        resp, stats = _build_phase2_response(go)
                        yield (resp, stats) if yield_access_stats else resp
                    continue

                close_idx, post_close = _split_on_first_close(
                    generated_ids, think_close_token_id, runtime.eos_tokens
                )
                if close_idx is None:
                    phase2_pending.append(go)
                    if out_py.finished:
                        # No close ever — buffered chunks are all content.
                        for item in _flush_phase2_pending():
                            yield item
                        phase2_pending = []
                    continue

                if post_close:
                    # Case A: discard pending + emit post-close.
                    phase2_pending = []
                    phase2_seen_close = True
                    if out_py.finished and runtime.eos_tokens:
                        post_close = _strip_trailing_eos(post_close, runtime.eos_tokens)
                    out_py.output_ids = torch.tensor(post_close, dtype=torch.int32)
                    if post_close or out_py.finished:
                        resp, stats = _build_phase2_response(go)
                        yield (resp, stats) if yield_access_stats else resp
                elif out_py.finished:
                    # Case B: pre-close is real content; keep it, drop close.
                    pre_close = list(generated_ids[:close_idx])
                    out_py.output_ids = torch.tensor(pre_close, dtype=torch.int32)
                    phase2_pending.append(go)
                    for item in _flush_phase2_pending():
                        yield item
                    phase2_pending = []
                    phase2_seen_close = True
                else:
                    # Ambiguous: close split across chunks. Default to Case A
                    # (drop pending + this chunk's pre-close); next chunk's
                    # content will stream as content normally.
                    phase2_pending = []
                    phase2_seen_close = True
    except FtRuntimeException as e:
        _set_access_backend_error_code(access_agg, e)
        error_spec = _dash_error_spec_for_ft_exception(e)
        status_message = str(e)
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
        logging.exception("[DashScGrpc] [%s] enqueue failed: %s", tag, e)
        error_spec = DASH_ERROR_INTERNAL
        response = build_dash_error_response(
            str(request.id),
            request.model_name,
            error_spec=error_spec,
            status_message=f"{type(e).__name__}: {e}",
        )
        stats = (0, True, error_spec.finish_reason, len(input_ids_list), 0, ())
        yield (response, stats) if yield_access_stats else response


# ----------------------------------------------------------------------------
# gRPC servicer (ModelStreamInfer entry)
# ----------------------------------------------------------------------------


class DashScInferenceServicer(predict_v2_pb2_grpc.GRPCInferenceServiceServicer):
    """ModelStreamInfer: fake mode (mock) or real mode (``backend_visitor.enqueue``).

    ``ip`` / ``port`` / ``server_id`` derive the snowflake-style ``GenerateInput.request_id``
    via :func:`generate_request_id` — same scheme as the HTTP path in ``FrontendServer``, so
    the backend sees a single request_id generation policy. ``port`` should be the dash_sc
    gRPC listening port. The per-servicer sequence counter is intentionally independent of
    ``FrontendServer._global_controller``.
    """

    def __init__(
        self,
        backend_visitor=None,
        *,
        ip: str = "",
        port: int = 0,
        server_id: str = "",
        echo_prefix_ids: Optional[list[int]] = None,
        extra_stop_word_ids: Optional[list[list[int]]] = None,
        tokenizer: Any = None,
        generate_env_config: Any = None,
        think_runtime: Optional[_ThinkRuntime] = None,
        rank_id: Optional[int] = None,
        repetition_monitor_config: Optional[RequestRepetitionMonitorConfig] = None,
    ):
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
        # Access-log identity, injected at construction (``DashScApp`` /
        # ``__main__`` own the rank/server identity). The two ids are the only
        # state the log + metric projections need; ``server_id`` arrives as the
        # snowflake string, coerced to ``Optional[int]`` once here. The kmonitor
        # tag dict is memoized per (rank, server) in ``grpc_metrics``, so the
        # per-chunk hot path never re-stringifies them. The repetition monitor
        # config lives only on this inference path, not the transparent proxy.
        self._rank_id = rank_id
        self._server_id = to_optional_int(server_id)
        self._rep_cfg = repetition_monitor_config or RequestRepetitionMonitorConfig()

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
        # Self-managed access-log lifecycle (the shared interceptor is gone).
        # Create/arrival/query go first — before any inbound frame — so a
        # frame-less RPC (peer closed before sending) still reports arrival and
        # produces an access line via the ``finally`` below.
        record = GrpcAccessRecord.create(
            context,
            "ModelStreamInfer",
            "bidi_stream",
            raw_mode=False,
            repetition_monitor_config=self._rep_cfg,
        )
        emit_query_log(record, rank_id=self._rank_id, server_id=self._server_id)
        report_arrival(rank_id=self._rank_id, server_id=self._server_id)
        exc: Optional[BaseException] = None
        try:
            try:
                invocation_metadata = context.invocation_metadata()
            except Exception:
                invocation_metadata = ()
            partial_metadata_sent = False
            first_request = True
            async for request in request_iterator:
                record.req_count += 1
                logging.debug(
                    "[DashScGrpc] ModelInferRequest: id=%s model_name=%s",
                    request.id,
                    request.model_name,
                )
                try:
                    input_ids_list, sampling, other = parse_dash_sc_grpc_request(
                        request
                    )
                except DashScParameterError as e:
                    if first_request:
                        record.record_request_frame(request)
                        record.mark_request_done("eof")
                        first_request = False
                    error_spec = DASH_ERROR_BAD_REQUEST
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
                if input_ids_list is None:
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
                if first_request:
                    # Hand the record the payload we just parsed so it does not
                    # decode the same request proto again (the input_ids tensor
                    # is large for long context).
                    record.capture_structured_request(
                        request,
                        input_ids=input_ids_list,
                        sampling=sampling,
                        other=other,
                    )
                    record.mark_request_done("eof")
                    first_request = False
                if (
                    not partial_metadata_sent
                    and other is not None
                    and other.timeout_ms is not None
                ):
                    await _send_partial_response_metadata(context)
                    partial_metadata_sent = True

                if sampling is not None and sampling.max_new_tokens <= 0:
                    param_name = (
                        "max_completion_tokens"
                        if getattr(
                            sampling, "max_new_tokens_from_completion_alias", False
                        )
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

                if self._backend_visitor is None:
                    fake_generated_ids = [x + 100 for x in input_ids_list]
                    for resp in iter_fake_model_stream_infer(
                        request, input_ids_list, sampling.top_k
                    ):
                        record.record_generated_ids(fake_generated_ids)
                        self._record_and_report_chunk(
                            record,
                            resp,
                            delta_len=len(fake_generated_ids),
                            finished=True,
                            finish_reason=LLMFinishReason.STOP,
                        )
                        yield resp
                    return
                else:
                    async for resp, stats in iter_real_model_stream_infer(
                        request,
                        input_ids_list,
                        sampling,
                        other,
                        self._backend_visitor,
                        rtp_llm_request_id=self._next_rtp_llm_request_id(),
                        echo_prefix_ids=self._echo_prefix_ids,
                        extra_stop_word_ids=self._extra_stop_word_ids,
                        invocation_metadata=invocation_metadata,
                        tokenizer=self._tokenizer,
                        generate_env_config=self._generate_env_config,
                        think_runtime=self._think_runtime,
                        phase2_request_id_factory=self._next_rtp_llm_request_id,
                        access_agg=record,
                        yield_access_stats=True,
                    ):
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
                        yield resp
                    return
            if first_request:
                record.mark_request_done("eof")
        except BaseException as e:
            exc = e
            raise
        finally:
            end_ts = record.resolve_status(context, exc)
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
