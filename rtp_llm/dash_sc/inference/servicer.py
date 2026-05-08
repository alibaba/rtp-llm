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
import socket
from typing import Any, AsyncIterator, Optional

import torch

from rtp_llm.dash_sc.codec import (
    OtherParams,
    SamplingParams,
    build_stream_response_from_generate_outputs,
    iter_fake_model_stream_infer,
    parse_dash_sc_grpc_request,
    prepend_to_generated_ids_tensor,
)
from rtp_llm.dash_sc.dashscope_compat import (
    DashScopeResponseEchoArgs,
    apply_dashscope_extras_to_generate_config,
    parse_dashscope_request_extras,
)
from rtp_llm.dash_sc.proto import predict_v2_pb2, predict_v2_pb2_grpc
from rtp_llm.frontend.request_id_generator import generate_request_id
from rtp_llm.utils.base_model_datatypes import GenerateInput
from rtp_llm.utils.util import AtomicCounter


def stream_log_tag(*, request_id_numeric: int, trace_id: str) -> str:
    """Align with C++ ``GenerateStream::streamLogTag()`` for log correlation."""
    return f"request_id={request_id_numeric} trace_id={trace_id}"


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
    instance_ip: Optional[str] = None,
    hostname: Optional[str] = None,
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

    ``invocation_metadata`` / ``instance_ip`` / ``hostname`` are threaded through
    from the servicer so the dashscope-compat layer can: (1) read the
    ``x-ds-max-matched-token-num`` header to gate ``prompt_cached_token_num``
    suppression, and (2) echo the worker identity (``model_instance_ip`` /
    ``model_hostname``) on every response chunk — same fields dashllm emits at
    ``dashllm/worker/processors/model.py:832-834``.
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
    prefix_ids = list(echo_prefix_ids) if echo_prefix_ids else []
    n = len(prefix_ids)
    should_echo = (
        n > 0 and len(input_ids_list) >= n and list(input_ids_list[-n:]) == prefix_ids
    )
    echoed = False
    try:
        # Parse dashscope-specific extras (parameters JSON + invocation_metadata)
        # BEFORE building GenerateConfig so apply step can override defaults
        # without being clobbered by `to_generate_config`.
        extras = parse_dashscope_request_extras(request, invocation_metadata)

        generate_config = sampling.to_generate_config(other=other)
        generate_config.trace_id = trace_str
        # Apply dashllm-style overrides (stop / stop_token_ids / max_thinking_tokens
        # / return_all_probs / return_incremental / chat_id; unsupported fields
        # logged and dropped here).
        apply_dashscope_extras_to_generate_config(
            generate_config, extras, request_log_tag=tag
        )
        echo_args = DashScopeResponseEchoArgs(
            suppress_cached_token_num=extras.suppress_cached_token_num,
            model_name=request.model_name or None,
            instance_ip=instance_ip,
            hostname=hostname,
            scheduler_request_id=extras.scheduler_request_id,
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
        token_ids = torch.tensor(input_ids_list, dtype=torch.int)
        generate_input = GenerateInput(
            request_id=rtp_llm_request_id,
            token_ids=token_ids,
            mm_inputs=[],
            generate_config=generate_config,
        )
        is_streaming = bool(getattr(generate_config, "is_streaming", True))
        logging.debug("[DashScGrpc] [%s] generate_input: %s", tag, generate_input)
        request_shape = list(request.inputs[0].shape) if request.inputs else None
        chunk_idx = 0
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
                dashscope_response_echo=echo_args,
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
        extra_stop_word_ids: Optional[list[list[int]]] = None,
    ):
        self._backend_visitor = backend_visitor
        self._ip = ip
        self._port = port
        self._server_id = server_id
        self._echo_prefix_ids = list(echo_prefix_ids) if echo_prefix_ids else []
        self._extra_stop_word_ids = (
            [list(w) for w in extra_stop_word_ids] if extra_stop_word_ids else []
        )
        self._seq_counter = AtomicCounter()
        # Cache hostname once at startup. ``socket.gethostname`` is a syscall
        # but contention-free; calling it per-request would still be cheap, but
        # caching keeps the response builder allocation-free in steady state.
        # ``instance_ip`` reuses the bound listening IP from ``server_config``;
        # avoids the ``socket.gethostbyname`` DNS roundtrip that
        # ``dashllm/utils.get_cen_ip`` does (which can stall on broken DNS).
        try:
            self._hostname = socket.gethostname()
        except Exception:
            self._hostname = ""
        self._instance_ip = ip or ""

    async def close(self) -> None:
        """Hook for teardown; currently holds no resources (backend_visitor is owned by
        the caller, sequence counter is in-memory). Kept so future handles can be flushed
        here without changing the call-site in ``DashScGrpcServer.stop``.
        """

    def _next_rtp_llm_request_id(self) -> int:
        sequence = self._seq_counter.increment() % 4096  # 12 bits
        return generate_request_id(self._ip, self._port, self._server_id, sequence)

    async def ModelStreamInfer(self, request_iterator, context):
        # Snapshot once per RPC: ``invocation_metadata()`` returns the
        # client-sent headers (case-insensitive lookup performed downstream by
        # ``dashscope_compat._extract_metadata_value``). Capturing here means
        # every request frame on the same RPC sees the same headers, matching
        # gRPC semantics (metadata is per-RPC, not per-frame).
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
                    instance_ip=self._instance_ip,
                    hostname=self._hostname,
                ):
                    yield resp
