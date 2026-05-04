"""DashSc gRPC servicer + real-inference bridge + enqueue asyncio loop.

Consolidates servicer, real-inference bridge, and enqueue asyncio loop in one module:

* :class:`DashScGrpcInferenceServicer` implements ``ModelStreamInfer`` (predict_v2.proto wire).
* :func:`iter_real_model_stream_infer` wraps ``backend_visitor.enqueue`` for the servicer.
* :func:`_iter_enqueue_sync` polls the async backend stream and honors client-side cancel.
* :func:`resolve_loop_for_enqueue` picks the loop set by :class:`DashScApp` (fallback: a
  dedicated process-level asyncio loop).

Cancel propagation: the servicer binds ``grpc.ServicerContext`` into the enqueue pump via
``functools.partial``; ``_iter_enqueue_sync`` registers ``fut.cancel`` on
``context.add_callback`` and polls ``context.is_active()`` between chunks so a peer RESET_STREAM
/ deadline promptly cancels the backend coroutine instead of waiting for it to flush.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Optional

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
from rtp_llm.frontend.request_id_generator import generate_request_id
from rtp_llm.utils.base_model_datatypes import GenerateInput, GenerateOutputs
from rtp_llm.utils.util import AtomicCounter

# ----------------------------------------------------------------------------
# Enqueue asyncio loop (set by DashScApp; fallback = dedicated process-level loop)
# ----------------------------------------------------------------------------


# Set by DashScApp startup; dedicated asyncio loop for backend enqueue coroutines.
_enqueue_loop: Optional[asyncio.AbstractEventLoop] = None

# Fallback when ``_enqueue_loop`` is unset (tests / standalone fake without DashScApp).
_async_loop: Optional[asyncio.AbstractEventLoop] = None
_async_loop_thread: Optional[threading.Thread] = None
_async_loop_lock = threading.Lock()


def set_dash_sc_grpc_enqueue_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _enqueue_loop
    _enqueue_loop = loop
    logging.info("[DashScGrpc] enqueue_event_loop set (DashScApp dedicated loop)")


def get_dash_sc_grpc_enqueue_event_loop() -> Optional[asyncio.AbstractEventLoop]:
    return _enqueue_loop


def _get_async_loop() -> asyncio.AbstractEventLoop:
    """Dedicated event loop in its own thread when the DashScApp loop is not wired."""
    global _async_loop, _async_loop_thread
    with _async_loop_lock:
        if _async_loop is not None and _async_loop.is_running():
            return _async_loop
        _async_loop = asyncio.new_event_loop()

        def _run_loop():
            asyncio.set_event_loop(_async_loop)
            _async_loop.run_forever()

        _async_loop_thread = threading.Thread(target=_run_loop, daemon=True)
        _async_loop_thread.start()
        deadline = time.monotonic() + 5.0
        while not _async_loop.is_running() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not _async_loop.is_running():
            raise RuntimeError("Async loop failed to start for enqueue")
    return _async_loop


def resolve_loop_for_enqueue() -> asyncio.AbstractEventLoop:
    """Loop on which ``visitor.enqueue`` runs: DashScApp loop if set, else process fallback."""
    if _enqueue_loop is not None and _enqueue_loop.is_running():
        return _enqueue_loop
    return _get_async_loop()


# ----------------------------------------------------------------------------
# Real inference bridge: async backend enqueue -> gRPC worker-thread iterator
# ----------------------------------------------------------------------------


def stream_log_tag(*, request_id_numeric: int, trace_id: str) -> str:
    """Align with C++ ``GenerateStream::streamLogTag()`` for log correlation."""
    return f"request_id={request_id_numeric} trace_id={trace_id}"


# Sentinel marking end of chunked stream in ``_iter_enqueue_sync``'s handoff queue.
_ITER_ENQUEUE_DONE = object()

# Poll interval for ``out_q.get`` — bounded so peer-cancel is observed within this window
# even while the backend is mid-token. 100ms is roughly one TPOT budget and adds negligible
# CPU overhead per in-flight request.
_ITER_ENQUEUE_POLL_INTERVAL_S = 0.1


def _context_is_active(context) -> bool:
    """Safe ``context.is_active()``: tolerates ``None`` and mock objects without the method."""
    if context is None:
        return True
    try:
        return bool(context.is_active())
    except Exception:
        # MagicMock without an explicit return_value answers truthy; tests that
        # do not care about cancel stay unaffected.
        return True


def _register_cancel_callback(context, cancel_fn: Callable[[], None]) -> None:
    """Register ``cancel_fn`` on ``context.add_callback`` (no-op if unavailable)."""
    if context is None:
        return
    try:
        context.add_callback(cancel_fn)
    except Exception:
        # Some mock contexts (unit tests) don't implement add_callback; skip silently.
        pass


def _iter_enqueue_sync(visitor, generate_input, *, context=None):
    """Yield enqueue stream chunks from the gRPC worker thread as they arrive.

    Runs ``visitor.enqueue`` on the loop from :func:`resolve_loop_for_enqueue` (DashScApp
    dedicated loop, or a process-level fallback). Polls the handoff queue with
    :data:`_ITER_ENQUEUE_POLL_INTERVAL_S` so client cancel / deadline reaches the backend
    promptly; the ``context.add_callback`` hook additionally cancels the pump coroutine so
    ``visitor.enqueue`` unwinds instead of buffering.
    """
    out_q: queue.Queue = queue.Queue()
    pump_error: list[BaseException] = []

    async def pump():
        try:
            stream = await visitor.enqueue(generate_input)
            async for x in stream:
                out_q.put(x)
        except BaseException as e:
            pump_error.append(e)
        finally:
            out_q.put(_ITER_ENQUEUE_DONE)

    loop = resolve_loop_for_enqueue()
    fut = asyncio.run_coroutine_threadsafe(pump(), loop)

    def _cancel_pump() -> None:
        if not fut.done():
            fut.cancel()

    _register_cancel_callback(context, _cancel_pump)
    try:
        while True:
            if not _context_is_active(context):
                _cancel_pump()
                break
            try:
                item = out_q.get(timeout=_ITER_ENQUEUE_POLL_INTERVAL_S)
            except queue.Empty:
                continue
            if item is _ITER_ENQUEUE_DONE:
                break
            yield item
        if pump_error:
            raise pump_error[0]
        # Future may have been cancelled (peer cancel) — don't block on result in that case.
        if not fut.cancelled():
            try:
                fut.result(timeout=0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
    finally:
        if not fut.done():
            fut.cancel()


def iter_real_model_stream_infer(
    request,
    input_ids_list: list[int],
    sampling: SamplingParams,
    other: OtherParams,
    backend_visitor: Any,
    *,
    rtp_llm_request_id: int,
    run_enqueue_sync: Callable[[Any, GenerateInput], Iterable[GenerateOutputs]],
    echo_prefix_ids: Optional[list[int]] = None,
) -> Iterator[predict_v2_pb2.ModelStreamInferResponse]:
    """Run enqueue on ``backend_visitor`` and yield one proto per chunk as the backend streams.

    ``rtp_llm_request_id`` is the int64 used for ``GenerateInput.request_id`` and log tags;
    the upstream servicer generates it via ``generate_request_id`` (same snowflake scheme as
    the HTTP path). ``request.id`` (string) is preserved as the trace id.

    ``run_enqueue_sync`` is invoked with ``(visitor, generate_input)`` — the servicer pre-binds
    the gRPC ``ServicerContext`` via ``functools.partial`` so cancellation reaches the pump
    without changing the callable's positional signature (keeps test injection simple).

    ``echo_prefix_ids`` is the auto-derived "thinking prefill" token id sequence. When
    non-empty and ``input_ids_list`` ends with it, the first non-empty ``generated_ids``
    chunk gets ``echo_prefix_ids`` prepended so downstream consumers that rely on the
    prefill-echo contract (dashllm-style) see the expected first token.
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
        generate_config = sampling.to_generate_config(other=other)
        generate_config.trace_id = trace_str
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
        for go in run_enqueue_sync(backend_visitor, generate_input):
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


class DashScGrpcInferenceServicer(predict_v2_pb2_grpc.GRPCInferenceServiceServicer):
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
    ):
        self._backend_visitor = backend_visitor
        self._ip = ip
        self._port = port
        self._server_id = server_id
        self._echo_prefix_ids = list(echo_prefix_ids) if echo_prefix_ids else []
        self._seq_counter = AtomicCounter()

    def close(self) -> None:
        """Hook for teardown; currently holds no resources (backend_visitor is owned by
        the caller, sequence counter is in-memory). Kept so future async/file handles can
        be flushed here without changing the call-site in ``DashScGrpcServer.stop``.
        """

    def _next_rtp_llm_request_id(self) -> int:
        sequence = self._seq_counter.increment() % 4096  # 12 bits
        return generate_request_id(self._ip, self._port, self._server_id, sequence)

    def ModelStreamInfer(self, request_iterator, context):
        for request in request_iterator:
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
                yield from iter_fake_model_stream_infer(
                    request, input_ids_list, sampling.top_k
                )
            else:
                # Bind context into the enqueue pump so peer cancel / deadline reaches
                # ``visitor.enqueue``; keeps the (visitor, generate_input) positional
                # signature so tests can inject a 2-arg fake.
                run_enqueue_sync = functools.partial(
                    _iter_enqueue_sync, context=context
                )
                yield from iter_real_model_stream_infer(
                    request,
                    input_ids_list,
                    sampling,
                    other,
                    self._backend_visitor,
                    rtp_llm_request_id=self._next_rtp_llm_request_id(),
                    run_enqueue_sync=run_enqueue_sync,
                    echo_prefix_ids=self._echo_prefix_ids,
                )
