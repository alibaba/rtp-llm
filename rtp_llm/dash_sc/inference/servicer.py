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

import logging
from typing import Any, AsyncIterator, Callable, Optional

import torch

from rtp_llm.dash_sc.codec import (
    OtherParams,
    SamplingParams,
    _token_ids_list_from_generate_output,
    build_stream_response_from_generate_outputs,
    iter_fake_model_stream_infer,
    parse_dash_sc_grpc_request,
    prepend_to_generated_ids_tensor,
)
from rtp_llm.dash_sc.proto import predict_v2_pb2, predict_v2_pb2_grpc
from rtp_llm.frontend.request_id_generator import generate_request_id
from rtp_llm.server.request_headers import extract_request_headers
from rtp_llm.utils.base_model_datatypes import GenerateInput
from rtp_llm.utils.util import AtomicCounter


def stream_log_tag(*, request_id_numeric: int, trace_id: str) -> str:
    """Align with C++ ``GenerateStream::streamLogTag()`` for log correlation."""
    return f"request_id={request_id_numeric} trace_id={trace_id}"


def _headers_from_invocation_metadata(
    invocation_metadata: Optional[Any],
) -> dict[str, str]:
    metadata_headers = {
        str(key).lower(): value
        for key, value in invocation_metadata or ()
        if key is not None and value is not None
    }
    return extract_request_headers(metadata_headers)


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


def _decode_env_tag(generate_env_config: Any, attr: str, default: str = "") -> str:
    value = getattr(generate_env_config, attr, default) or default
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


def _derive_think_close_token_id(tokenizer: Any, generate_config: Any) -> Optional[int]:
    if tokenizer is not None:
        try:
            ids = _encode_tag(tokenizer, "</think>")
            if ids:
                return int(ids[-1])
        except Exception:
            pass
    ids = list(getattr(generate_config, "end_think_token_ids", []) or [])
    return int(ids[0]) if ids else None


class _ThinkTokenSet:
    def __init__(
        self,
        *,
        bos_tokens: list[int],
        eos_tokens: list[int],
        empty_tokens: list[int],
    ):
        self.bos_tokens = bos_tokens
        self.eos_tokens = eos_tokens
        self.empty_tokens = empty_tokens


def _derive_think_token_set(tokenizer: Any, generate_env_config: Any) -> _ThinkTokenSet:
    if tokenizer is None or generate_env_config is None:
        return _ThinkTokenSet(bos_tokens=[], eos_tokens=[], empty_tokens=[])
    think_start_tag = _decode_env_tag(
        generate_env_config, "think_start_tag", "<think>\n"
    )
    think_end_tag = _decode_env_tag(
        generate_env_config, "think_end_tag", "</think>\n\n"
    )
    bos_tokens = _encode_tag(tokenizer, think_start_tag)
    eos_tokens = _encode_tag(tokenizer, think_end_tag)
    empty_tokens = _encode_tag(tokenizer, think_start_tag + "\n" + think_end_tag)
    return _ThinkTokenSet(
        bos_tokens=bos_tokens,
        eos_tokens=eos_tokens,
        empty_tokens=empty_tokens,
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


def _make_generate_input(
    *,
    request_id: int,
    input_ids_list: list[int],
    generate_config: Any,
    invocation_metadata: Optional[Any],
) -> GenerateInput:
    return GenerateInput(
        request_id=request_id,
        token_ids=torch.tensor(input_ids_list, dtype=torch.int),
        mm_inputs=[],
        generate_config=generate_config,
        headers=_headers_from_invocation_metadata(invocation_metadata),
    )


def _clone_generate_config(generate_config: Any) -> Any:
    try:
        return generate_config.model_copy(deep=True)
    except AttributeError:
        return generate_config.copy(deep=True)


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
    model_type: Optional[str] = None,
    eos_token_id: Optional[int] = None,
    max_token_id: Optional[int] = None,
    phase2_request_id_factory: Optional[Callable[[], int]] = None,
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

    Hot-path layout: dashscope-serving doesn't ship ``stop_words_list`` per request,
    so 99% of calls hit the fast branch (empty ``existing``) and skip the dedup set
    + tuple hashing entirely. The slow branch only fires when a caller explicitly
    sets ``stop_words_list`` on the request.
    """
    trace_str = str(request.id)
    tag = stream_log_tag(request_id_numeric=rtp_llm_request_id, trace_id=trace_str)
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
        eos_id = (
            eos_token_id
            if eos_token_id is not None
            else _optional_int_attr(tokenizer, "eos_token_id")
        )
        max_id = (
            max_token_id
            if max_token_id is not None
            else _derive_max_token_id(tokenizer)
        )
        think_tokens = _derive_think_token_set(tokenizer, generate_env_config)
        matched_think_bos_ids = matched_echo_ids or _matched_echo_prefix_ids(
            input_ids_list, think_tokens.bos_tokens
        )
        is_dsv4 = _is_deepseek_v4(model_type)
        phase2_enabled = (
            is_dsv4
            and bool(getattr(generate_config, "in_think_mode", False))
            and bool(think_tokens.empty_tokens)
        )
        think_close_token_id = _derive_think_close_token_id(tokenizer, generate_config)
        cumulative_sent_ids: list[int] = []
        generate_think_token_num: Optional[int] = None
        generate_input = _make_generate_input(
            request_id=rtp_llm_request_id,
            input_ids_list=input_ids_list,
            generate_config=generate_config,
            invocation_metadata=invocation_metadata,
        )
        is_streaming = bool(getattr(generate_config, "is_streaming", True))
        logging.debug("[DashScGrpc] [%s] generate_input: %s", tag, generate_input)
        request_shape = list(request.inputs[0].shape) if request.inputs else None
        chunk_idx = 0
        phase2_needed = False
        stream = await backend_visitor.enqueue(generate_input)
        async for go in stream:
            chunk_idx += 1
            logging.debug("[DashScGrpc] [%s] real infer chunk %s", tag, chunk_idx)
            if not go.generate_outputs:
                raise ValueError("empty generate_outputs in backend chunk")
            out_py = go.generate_outputs[0]
            generated_ids = _token_ids_list_from_generate_output(out_py)
            if not generated_ids and not out_py.finished:
                logging.debug(
                    "[DashScGrpc] [%s] skip empty streaming chunk %s", tag, chunk_idx
                )
                continue
            ids_for_accounting = generated_ids
            if should_echo and not echoed and generated_ids:
                ids_for_accounting = matched_echo_ids + generated_ids
            close_offset: Optional[int] = None
            token1_offset: Optional[int] = None
            if generate_think_token_num is None:
                if think_close_token_id is not None:
                    for offset, token_id in enumerate(ids_for_accounting):
                        if token_id == think_close_token_id:
                            close_offset = offset
                            break
                if phase2_enabled and 1 in generated_ids:
                    token1_in_generated = generated_ids.index(1)
                    token1_offset = token1_in_generated
                    if should_echo and not echoed and generated_ids:
                        token1_offset += len(matched_echo_ids)
                if close_offset is not None and (
                    token1_offset is None or close_offset < token1_offset
                ):
                    generate_think_token_num = (
                        len(cumulative_sent_ids) + close_offset + 1
                    )
                    phase2_needed = phase2_enabled
            if (
                phase2_enabled
                and generate_think_token_num is None
                and generated_ids
                and 1 in generated_ids
            ):
                generated_ids = generated_ids[: generated_ids.index(1)]
                out_py.output_ids = torch.tensor(generated_ids, dtype=torch.int32)
                out_py.finished = False
                ids_for_accounting = generated_ids
                if should_echo and not echoed and generated_ids:
                    ids_for_accounting = matched_echo_ids + generated_ids
                generate_think_token_num = (
                    len(cumulative_sent_ids)
                    + len(ids_for_accounting)
                    + (1 if think_tokens.eos_tokens else 0)
                )
                cumulative_sent_ids.extend(ids_for_accounting)
                eos_go = go
                response = build_stream_response_from_generate_outputs(
                    dash_sc_request_id=request.id,
                    model_name=request.model_name,
                    go=eos_go,
                    request_log_tag=tag,
                    request_input_ids=input_ids_list,
                    return_input_ids=other.return_input_ids,
                    is_streaming=is_streaming,
                    generate_config=generate_config,
                    eos_token_id=eos_id,
                    max_token_id=max_id,
                    generate_think_token_num=generate_think_token_num,
                    _request_shape=request_shape,
                )
                if should_echo and not echoed and generated_ids:
                    if prepend_to_generated_ids_tensor(
                        response.infer_response, matched_echo_ids
                    ):
                        echoed = True
                if generated_ids:
                    yield response
                if think_tokens.eos_tokens:
                    out_py.output_ids = torch.tensor(
                        think_tokens.eos_tokens, dtype=torch.int32
                    )
                    out_py.finished = True
                    eos_response = build_stream_response_from_generate_outputs(
                        dash_sc_request_id=request.id,
                        model_name=request.model_name,
                        go=eos_go,
                        request_log_tag=tag,
                        request_input_ids=input_ids_list,
                        return_input_ids=other.return_input_ids,
                        is_streaming=is_streaming,
                        generate_config=generate_config,
                        eos_token_id=eos_id,
                        max_token_id=max_id,
                        generate_think_token_num=generate_think_token_num,
                        _request_shape=request_shape,
                    )
                    yield eos_response
                phase2_needed = True
                break
            cumulative_sent_ids.extend(ids_for_accounting)
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
        if phase2_needed:
            phase2_config = _clone_generate_config(generate_config)
            phase2_config.in_think_mode = False
            if hasattr(phase2_config, "thinking"):
                phase2_config.thinking = False
            phase2_config.trace_id = f"{trace_str}-2"
            phase2_input_ids = _phase2_input_ids_for_deepseek_v4(
                input_ids_list, matched_think_bos_ids, think_tokens.empty_tokens
            )
            phase2_request_id = (
                phase2_request_id_factory()
                if phase2_request_id_factory is not None
                else rtp_llm_request_id
            )
            phase2_generate_input = _make_generate_input(
                request_id=phase2_request_id,
                input_ids_list=phase2_input_ids,
                generate_config=phase2_config,
                invocation_metadata=invocation_metadata,
            )
            logging.debug(
                "[DashScGrpc] [%s] phase-2 generate_input: %s",
                tag,
                phase2_generate_input,
            )
            phase2_stream = await backend_visitor.enqueue(phase2_generate_input)
            async for go in phase2_stream:
                if not go.generate_outputs:
                    raise ValueError("empty generate_outputs in phase-2 backend chunk")
                out_py = go.generate_outputs[0]
                generated_ids = _token_ids_list_from_generate_output(out_py)
                if not generated_ids and not out_py.finished:
                    continue
                response = build_stream_response_from_generate_outputs(
                    dash_sc_request_id=f"{request.id}-2",
                    model_name=request.model_name,
                    go=go,
                    request_log_tag=tag,
                    request_input_ids=phase2_input_ids,
                    return_input_ids=other.return_input_ids,
                    is_streaming=is_streaming,
                    generate_config=phase2_config,
                    eos_token_id=eos_id,
                    max_token_id=max_id,
                    generate_think_token_num=generate_think_token_num,
                    _request_shape=request_shape,
                )
                yield response
    except Exception as e:
        logging.exception("[DashScGrpc] [%s] enqueue failed: %s", tag, e)
        # Prefix with the exception class name so the access-log interceptor's
        # ``_classify_error_message`` can map it to a bounded ``error_code``
        # tag (e.g. ``BACKEND_RuntimeError``). Without the prefix every
        # backend failure collapses into a single ``BACKEND_INTERNAL`` bucket
        # on Grafana and operators can't tell OOM from concurrency-limit from
        # shape-mismatch without reading individual log lines.
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
        model_type: Optional[str] = None,
        eos_token_id: Optional[int] = None,
        max_token_id: Optional[int] = None,
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
        self._model_type = model_type
        self._eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else _optional_int_attr(tokenizer, "eos_token_id")
        )
        self._max_token_id = (
            max_token_id
            if max_token_id is not None
            else _derive_max_token_id(tokenizer)
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
        async for request in request_iterator:
            logging.debug(
                "[DashScGrpc] ModelInferRequest: id=%s model_name=%s",
                request.id,
                request.model_name,
            )
            input_ids_list, sampling, other = parse_dash_sc_grpc_request(request)
            if input_ids_list is None:
                yield predict_v2_pb2.ModelStreamInferResponse(
                    error_message="input_ids not found or raw_input_contents mismatch"
                )
                return

            if self._backend_visitor is None:
                # ``iter_fake_model_stream_infer`` is a sync generator; iterating it
                # inside an ``async def`` is valid — it yields synchronously without
                # awaiting. Fake mode is test-only, no concurrency concern.
                for resp in iter_fake_model_stream_infer(
                    request, input_ids_list, sampling.top_k
                ):
                    yield resp
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
                    model_type=self._model_type,
                    eos_token_id=self._eos_token_id,
                    max_token_id=self._max_token_id,
                    phase2_request_id_factory=self._next_rtp_llm_request_id,
                ):
                    yield resp
