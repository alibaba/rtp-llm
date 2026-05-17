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
import logging
from typing import Any, AsyncIterator, Callable, Optional

import torch

from rtp_llm.dash_sc.codec import (
    OtherParams,
    SamplingParams,
    build_stream_response_from_generate_outputs,
    iter_fake_model_stream_infer,
    parse_dash_sc_grpc_request,
    prepend_to_generated_ids_tensor,
)
from rtp_llm.dash_sc.proto import predict_v2_pb2, predict_v2_pb2_grpc
from rtp_llm.dash_sc.think import (
    DashScThinkConfig,
    DEFAULT_MAX_THINKING_TOKENS,
    INT32_MAX,
    plan_dash_sc_thinking,
)
from rtp_llm.frontend.request_id_generator import generate_request_id
from rtp_llm.server.request_headers import extract_request_headers
from rtp_llm.utils.base_model_datatypes import GenerateInput, GenerateOutputs
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


def _merge_extra_stop_word_ids(generate_config: Any, extra_stop_word_ids: Any) -> None:
    if not extra_stop_word_ids:
        return
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
        # Fast path: shallow-copy the startup snapshot. Outer copy keeps any
        # future engine-side mutation request-local; inner lists are shared
        # (snapshot is read-only by contract).
        generate_config.stop_words_list = list(extra_stop_word_ids)


def _remove_stop_word_ids(generate_config: Any, stop_word_ids: list[list[int]]) -> None:
    if not stop_word_ids or not generate_config.stop_words_list:
        return
    removed = {tuple(ids) for ids in stop_word_ids if ids}
    if not removed:
        return
    generate_config.stop_words_list = [
        ids for ids in generate_config.stop_words_list if tuple(ids) not in removed
    ]


def _adjust_aux_prompt_metrics(go: GenerateOutputs, prompt_delta: int) -> None:
    if prompt_delta <= 0 or not go.generate_outputs:
        return
    aux = getattr(go.generate_outputs[0], "aux_info", None)
    if aux is None:
        return
    if getattr(aux, "input_len", 0) > prompt_delta:
        aux.input_len -= prompt_delta
    if getattr(aux, "reuse_len", 0) > prompt_delta:
        aux.reuse_len -= prompt_delta


def _build_generate_input(
    *,
    request_id: int,
    token_ids: list[int],
    generate_config: Any,
    headers: dict[str, str],
) -> GenerateInput:
    kwargs = dict(
        request_id=request_id,
        token_ids=torch.tensor(token_ids, dtype=torch.int),
        mm_inputs=[],
        generate_config=generate_config,
    )
    if "headers" in inspect.signature(GenerateInput).parameters:
        kwargs["headers"] = headers
    return GenerateInput(**kwargs)


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
    think_config: Optional[DashScThinkConfig] = None,
    extra_stop_word_ids: Optional[list[list[int]]] = None,
    invocation_metadata: Optional[Any] = None,
    request_id_factory: Optional[Callable[[], int]] = None,
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
    headers = _headers_from_invocation_metadata(invocation_metadata)
    think_plan = plan_dash_sc_thinking(
        input_ids_list,
        think_config=think_config,
        enable_thinking=other.enable_thinking,
        max_new_think_tokens=sampling.max_new_think_tokens,
        default_max_thinking_tokens=DEFAULT_MAX_THINKING_TOKENS,
    )

    has_usable_think_config = think_config is not None and think_config.usable
    prefix_ids = [] if has_usable_think_config else list(echo_prefix_ids or [])
    n = len(prefix_ids)
    should_echo = (
        n > 0 and len(input_ids_list) >= n and list(input_ids_list[-n:]) == prefix_ids
    )
    echoed = False

    def make_generate_config(*, in_think_mode: bool):
        generate_config = sampling.to_generate_config(other=other)
        generate_config.trace_id = trace_str
        if in_think_mode and think_config is not None:
            generate_config.in_think_mode = True
            generate_config.max_thinking_tokens = think_plan.max_thinking_tokens
            generate_config.end_think_token_ids = list(think_config.eos_tokens)
            generate_config.max_new_tokens = min(
                INT32_MAX,
                generate_config.max_new_tokens
                + max(0, think_plan.max_thinking_tokens)
                + len(generate_config.end_think_token_ids),
            )
            if think_config.is_deepseek_v4:
                generate_config.abort_think_token_ids = [
                    int(think_config.dsv4_abort_token_id)
                ]
        else:
            generate_config.in_think_mode = False
        _merge_extra_stop_word_ids(generate_config, extra_stop_word_ids)
        if in_think_mode and think_config is not None:
            _remove_stop_word_ids(
                generate_config,
                [
                    list(think_config.end_think_token_ids),
                    list(think_config.eos_tokens),
                ],
            )
        return generate_config

    try:
        request_shape = list(request.inputs[0].shape) if request.inputs else None
        chunk_idx = 0

        if think_plan.thinking and think_config is not None:
            logging.debug(
                "[DashScGrpc] [%s] thinking enabled: reason=%s input_len=%s budget=%s",
                tag,
                think_plan.reason,
                len(think_plan.input_ids),
                think_plan.max_thinking_tokens,
            )
            generate_config = make_generate_config(in_think_mode=True)
            generate_input = _build_generate_input(
                request_id=rtp_llm_request_id,
                token_ids=think_plan.input_ids,
                generate_config=generate_config,
                headers=headers,
            )
            is_streaming = bool(getattr(generate_config, "is_streaming", True))
            logging.debug("[DashScGrpc] [%s] generate_input: %s", tag, generate_input)
            prompt_metric_delta = (
                think_plan.prompt_append_len + think_plan.prompt_metric_excluded_len
            )
            stream = await backend_visitor.enqueue(generate_input)
            async for go in stream:
                chunk_idx += 1
                logging.debug("[DashScGrpc] [%s] real infer chunk %s", tag, chunk_idx)
                _adjust_aux_prompt_metrics(go, prompt_metric_delta)
                response = build_stream_response_from_generate_outputs(
                    dash_sc_request_id=request.id,
                    model_name=request.model_name,
                    go=go,
                    request_log_tag=tag,
                    request_input_ids=input_ids_list,
                    return_input_ids=other.return_input_ids,
                    is_streaming=is_streaming,
                    _request_shape=request_shape,
                )
                if think_plan.echo_prefix_ids and not echoed:
                    if prepend_to_generated_ids_tensor(
                        response.infer_response, think_plan.echo_prefix_ids
                    ):
                        echoed = True
                yield response

            if chunk_idx == 0:
                logging.warning("[DashScGrpc] [%s] empty outputs_list", tag)
                yield predict_v2_pb2.ModelStreamInferResponse(
                    error_message="empty outputs_list from backend",
                )
                return

            logging.debug(
                "[DashScGrpc] [%s] real infer done: output_chunks=%s",
                tag,
                chunk_idx,
            )
            return

        generate_config = make_generate_config(in_think_mode=False)
        generate_input = _build_generate_input(
            request_id=rtp_llm_request_id,
            token_ids=think_plan.input_ids,
            generate_config=generate_config,
            headers=headers,
        )
        is_streaming = bool(getattr(generate_config, "is_streaming", True))
        logging.debug("[DashScGrpc] [%s] generate_input: %s", tag, generate_input)
        stream = await backend_visitor.enqueue(generate_input)
        async for go in stream:
            chunk_idx += 1
            logging.debug("[DashScGrpc] [%s] real infer chunk %s", tag, chunk_idx)
            response = build_stream_response_from_generate_outputs(
                dash_sc_request_id=request.id,
                model_name=request.model_name,
                go=go,
                request_log_tag=tag,
                request_input_ids=input_ids_list,
                return_input_ids=other.return_input_ids,
                is_streaming=is_streaming,
                _request_shape=request_shape,
            )
            if should_echo and not echoed:
                if prepend_to_generated_ids_tensor(response.infer_response, prefix_ids):
                    echoed = True
            yield response
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
        think_config: Optional[DashScThinkConfig] = None,
        extra_stop_word_ids: Optional[list[list[int]]] = None,
    ):
        self._backend_visitor = backend_visitor
        self._ip = ip
        self._port = port
        self._server_id = server_id
        self._echo_prefix_ids = list(echo_prefix_ids) if echo_prefix_ids else []
        self._think_config = think_config
        self._extra_stop_word_ids = (
            [list(w) for w in extra_stop_word_ids] if extra_stop_word_ids else []
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
                    think_config=self._think_config,
                    extra_stop_word_ids=self._extra_stop_word_ids,
                    invocation_metadata=invocation_metadata,
                    request_id_factory=self._next_rtp_llm_request_id,
                ):
                    yield resp
