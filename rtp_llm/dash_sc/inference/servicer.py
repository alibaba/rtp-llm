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

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.dash_sc.codec import (
    FINISH_REASON_ABORT,
    FINISH_REASON_LENGTH,
    FINISH_REASON_STOP_ENGINE_PARAM,
    FINISH_REASON_STOP_TIMEOUT,
    DashScParameterError,
    OtherParams,
    SamplingParams,
    _token_ids_list_from_generate_output,
    build_finish_reason_done_response,
    build_parameter_error_response,
    build_stream_response_from_generate_outputs,
    iter_fake_model_stream_infer,
    parse_dash_sc_grpc_request,
    prepend_to_generated_ids_tensor,
)
from rtp_llm.dash_sc.proto import predict_v2_pb2, predict_v2_pb2_grpc
from rtp_llm.dash_sc.proxy.access_record import ForwardAccessRecord
from rtp_llm.frontend.request_id_generator import generate_request_id
from rtp_llm.metrics import AccMetrics, kmonitor
from rtp_llm.server.request_headers import extract_request_headers
from rtp_llm.utils.base_model_datatypes import GenerateInput
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
_DEBUG_SCORE_TOKEN_IDS_PARAM = "dash_sc_debug_score_token_ids"
_DEBUG_SCORE_LABEL_PARAM = "dash_sc_debug_score_label"
_DEBUG_SCORE_TOKEN_LABELS = {
    128821: "<think>",
    128822: "</think>",
}
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
    raw_code = getattr(e, "rtp_error_code", None)
    if raw_code is None and isinstance(e, FtRuntimeException):
        raw_code = int(e.exception_type)
    if raw_code is None:
        return
    try:
        access_agg.backend_error_code = _exception_metric_code(raw_code)
    except (TypeError, ValueError):
        access_agg.backend_error_code = str(raw_code)


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


def _is_deepseek_v4(model_type: Optional[str]) -> bool:
    return str(model_type or "").replace("-", "_").lower() == "deepseek_v4"


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
    phase2_enabled = _is_deepseek_v4(model_type) and bool(empty_tokens)
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
    return GenerateInput(
        request_id=request_id,
        token_ids=torch.tensor(input_ids_list, dtype=torch.int),
        mm_inputs=[],
        generate_config=generate_config,
        headers=headers,
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


def _clone_generate_config(generate_config: Any) -> Any:
    try:
        cloned_config = generate_config.model_copy(deep=True)
    except AttributeError:
        cloned_config = generate_config.copy(deep=True)
    if hasattr(cloned_config, "role_addrs"):
        # Phase-2 must re-enter routing; copied role_addrs would bypass FlexLB master.
        cloned_config.role_addrs = []
    if hasattr(cloned_config, "original_role_addrs"):
        cloned_config.original_role_addrs = []
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


def _debug_score_token_ids_from_request(request: Any) -> list[int]:
    if _DEBUG_SCORE_TOKEN_IDS_PARAM not in request.parameters:
        return []
    param = request.parameters[_DEBUG_SCORE_TOKEN_IDS_PARAM]
    values: Any = []
    if param.HasField("int64_param"):
        values = [param.int64_param]
    elif param.HasField("string_param") and param.string_param:
        try:
            values = json.loads(param.string_param)
        except (TypeError, ValueError, json.JSONDecodeError):
            values = [param.string_param]
    elif param.HasField("bool_param"):
        return []
    if not isinstance(values, list):
        values = [values]
    result: list[int] = []
    for value in values:
        try:
            token_id = int(value)
        except (TypeError, ValueError):
            continue
        if token_id >= 0 and token_id not in result:
            result.append(token_id)
    return result


def _debug_score_label_from_request(request: Any) -> str:
    if _DEBUG_SCORE_LABEL_PARAM not in request.parameters:
        return ""
    param = request.parameters[_DEBUG_SCORE_LABEL_PARAM]
    if param.HasField("string_param"):
        return str(param.string_param)
    if param.HasField("int64_param"):
        return str(param.int64_param)
    if param.HasField("bool_param"):
        return str(param.bool_param)
    return ""


def _log_debug_token_scores(
    *,
    tag: str,
    case_label: str,
    chunk_idx: int,
    generated_ids: list[int],
    logits: Any,
    token_ids: list[int],
) -> None:
    if not token_ids or logits is None or not generated_ids:
        return
    if not isinstance(logits, torch.Tensor) or logits.numel() == 0:
        return
    rows = logits.detach()
    if rows.dim() == 1:
        rows = rows.unsqueeze(0)
    elif rows.dim() == 3 and rows.size(0) == 1:
        rows = rows.squeeze(0)
    elif rows.dim() != 2:
        logging.info(
            "[DashScDebugTokenScore] [%s] case=%s chunk=%s skip logits_dim=%s",
            tag,
            case_label,
            chunk_idx,
            rows.dim(),
        )
        return
    rows = rows.float().cpu()
    vocab_size = rows.size(-1)
    for step, sampled_token in enumerate(generated_ids):
        row = rows[min(step, rows.size(0) - 1)]
        probs = torch.softmax(row, dim=-1)
        for token_id in token_ids:
            label = _DEBUG_SCORE_TOKEN_LABELS.get(token_id, str(token_id))
            if token_id >= vocab_size:
                logging.info(
                    "[DashScDebugTokenScore] [%s] case=%s chunk=%s step=%s "
                    "sampled_token=%s token=%s token_id=%s out_of_vocab=%s",
                    tag,
                    case_label,
                    chunk_idx,
                    step,
                    sampled_token,
                    label,
                    token_id,
                    vocab_size,
                )
                continue
            logging.info(
                "[DashScDebugTokenScore] [%s] case=%s chunk=%s step=%s "
                "sampled_token=%s token=%s token_id=%s logit=%.8g prob=%.8g",
                tag,
                case_label,
                chunk_idx,
                step,
                sampled_token,
                label,
                token_id,
                float(row[token_id].item()),
                float(probs[token_id].item()),
            )


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
        debug_score_token_ids = _debug_score_token_ids_from_request(request)
        debug_score_label = _debug_score_label_from_request(request)
        if debug_score_token_ids:
            generate_config.return_logits = True
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
        phase2_enabled = runtime.phase2_enabled and bool(
            getattr(generate_config, "in_think_mode", False)
        )
        cumulative_sent_ids: list[int] = []
        generate_think_token_num: Optional[int] = None
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
            _log_debug_token_scores(
                tag=tag,
                case_label=debug_score_label,
                chunk_idx=chunk_idx,
                generated_ids=generated_ids,
                logits=getattr(out_py, "logits", None),
                token_ids=debug_score_token_ids,
            )
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
                yield response
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
                    yield response
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
                            FINISH_REASON_LENGTH if not will_do_phase2 else None
                        ),
                        _request_shape=request_shape,
                        stream_finished=not will_do_phase2,
                        token_ids=list(runtime.eos_tokens),
                    )
                    yield eos_response
                phase2_needed = will_do_phase2
                break
            cumulative_sent_ids.extend(ids_for_accounting)
            finish_reason_override = None
            if (
                out_py.finished
                and max_new_tokens > 0
                and len(cumulative_sent_ids) >= max_new_tokens
            ):
                finish_reason_override = FINISH_REASON_LENGTH
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
            )
            if should_echo and not echoed and generated_ids:
                if prepend_to_generated_ids_tensor(
                    response.infer_response, matched_echo_ids
                ):
                    echoed = True
            yield response
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
            yield predict_v2_pb2.ModelStreamInferResponse(
                error_message="empty outputs_list from backend",
            )
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
                    finish_reason_override = FINISH_REASON_LENGTH
                return build_stream_response_from_generate_outputs(
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
                    yield _build_phase2_response(buf_go)

            phase2_chunk_idx = 0
            async for go in phase2_stream:
                phase2_chunk_idx += 1
                if not go.generate_outputs:
                    raise ValueError("empty generate_outputs in phase-2 backend chunk")
                out_py = go.generate_outputs[0]
                generated_ids = _token_ids_list_from_generate_output(out_py)
                _log_debug_token_scores(
                    tag=phase2_tag,
                    case_label=debug_score_label,
                    chunk_idx=phase2_chunk_idx,
                    generated_ids=generated_ids,
                    logits=getattr(out_py, "logits", None),
                    token_ids=debug_score_token_ids,
                )
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
                        yield _build_phase2_response(go)
                    continue

                close_idx, post_close = _split_on_first_close(
                    generated_ids, think_close_token_id, runtime.eos_tokens
                )
                if close_idx is None:
                    phase2_pending.append(go)
                    if out_py.finished:
                        # No close ever — buffered chunks are all content.
                        for resp in _flush_phase2_pending():
                            yield resp
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
                        yield _build_phase2_response(go)
                elif out_py.finished:
                    # Case B: pre-close is real content; keep it, drop close.
                    pre_close = list(generated_ids[:close_idx])
                    out_py.output_ids = torch.tensor(pre_close, dtype=torch.int32)
                    phase2_pending.append(go)
                    for resp in _flush_phase2_pending():
                        yield resp
                    phase2_pending = []
                    phase2_seen_close = True
                else:
                    # Ambiguous: close split across chunks. Default to Case A
                    # (drop pending + this chunk's pre-close); next chunk's
                    # content will stream as content normally.
                    phase2_pending = []
                    phase2_seen_close = True
    except FtRuntimeException as e:
        if e.exception_type == ExceptionType.GENERATE_TIMEOUT:
            logging.warning("[DashScGrpc] [%s] generate timeout: %s", tag, e)
            yield build_finish_reason_done_response(
                str(request.id),
                request.model_name,
                FINISH_REASON_STOP_TIMEOUT,
            )
        elif e.exception_type in (
            ExceptionType.ERROR_INPUT_FORMAT_ERROR,
            ExceptionType.NO_PROMPT_ERROR,
            ExceptionType.EMPTY_PROMPT_ERROR,
            ExceptionType.INVALID_PARAMS,
        ):
            logging.warning("[DashScGrpc] [%s] parameter error: %s", tag, e)
            yield build_finish_reason_done_response(
                str(request.id),
                request.model_name,
                FINISH_REASON_STOP_ENGINE_PARAM,
            )
        elif e.exception_type == ExceptionType.CANCELLED_ERROR:
            logging.info("[DashScGrpc] [%s] engine cancelled: %s", tag, e)
            yield build_finish_reason_done_response(
                str(request.id),
                request.model_name,
                FINISH_REASON_ABORT,
            )
        else:
            _set_access_backend_error_code(access_agg, e)
            logging.exception("[DashScGrpc] [%s] engine error: %s", tag, e)
            yield predict_v2_pb2.ModelStreamInferResponse(
                error_message=f"{type(e).__name__}: {e}"
            )
    except Exception as e:
        logging.exception("[DashScGrpc] [%s] enqueue failed: %s", tag, e)
        # Prefix with exception class name so access-log _classify_error_message
        # maps it to a bounded error_code tag (e.g. BACKEND_RuntimeError).
        yield predict_v2_pb2.ModelStreamInferResponse(
            error_message=f"{type(e).__name__}: {e}"
        )


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
    ):
        self._backend_visitor = backend_visitor
        self._ip = ip
        self._port = port
        self._server_id = server_id
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

    async def close(self) -> None:
        """Hook for teardown; currently holds no resources (backend_visitor is owned by
        the caller, sequence counter is in-memory). Kept so future handles can be flushed
        here without changing the call-site in ``DashScGrpcServer.stop``.
        """

    def _next_rtp_llm_request_id(self) -> int:
        sequence = self._seq_counter.increment() % 4096  # 12 bits
        return generate_request_id(self._ip, self._port, self._server_id, sequence)

    async def ModelStreamInfer(self, request_iterator, context):
        try:
            invocation_metadata = context.invocation_metadata()
        except Exception:
            invocation_metadata = ()
        partial_metadata_sent = False
        async for request in request_iterator:
            logging.debug(
                "[DashScGrpc] ModelInferRequest: id=%s model_name=%s",
                request.id,
                request.model_name,
            )
            try:
                input_ids_list, sampling, other = parse_dash_sc_grpc_request(request)
            except DashScParameterError as e:
                yield build_parameter_error_response(str(request.id), str(e))
                return
            if input_ids_list is None:
                yield build_parameter_error_response(
                    str(request.id),
                    "input_ids not found or raw_input_contents mismatch",
                )
                return
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
                    if getattr(sampling, "max_new_tokens_from_completion_alias", False)
                    else "max_new_tokens"
                )
                yield build_parameter_error_response(
                    str(request.id),
                    f"invalid {param_name}: {sampling.max_new_tokens}; must be greater than 0",
                )
                return

            if self._backend_visitor is None:
                for resp in iter_fake_model_stream_infer(
                    request, input_ids_list, sampling.top_k
                ):
                    yield resp
                return
            else:
                async for resp in iter_real_model_stream_infer(
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
                    access_agg=ForwardAccessRecord.from_context(context),
                ):
                    yield resp
                return
