"""
Bailian gRPC server: GRPCInferenceService / ModelStreamInfer (wire: predict_v2.proto).
- Fake mode (backend_visitor=None): mock logic, input_ids + 100 -> generated_ids.
- Non-fake mode (backend_visitor set): use backend_rpc_server_visitor.enqueue() for real inference.
Started from FrontendApp via start_bailian_grpc_server_in_thread (daemon thread holds wait_for_termination; caller blocks until server.start() succeeds).
Standalone fake: python -m rtp_llm.bailian.bailian_grpc_server [--port PORT]
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent import futures

import grpc

from rtp_llm.bailian.bailian_grpc_request import parse_bailian_grpc_request
from rtp_llm.bailian.bailian_grpc_response_fake import iter_fake_model_stream_infer
from rtp_llm.bailian.bailian_grpc_response_real import iter_real_model_stream_infer
from rtp_llm.bailian.proto import predict_v2_pb2, predict_v2_pb2_grpc


def _resolve_bailian_grpc_config(bailian_grpc_config):
    if bailian_grpc_config is not None:
        return bailian_grpc_config
    from rtp_llm.ops import BailianGrpcConfig

    return BailianGrpcConfig()


def bailian_grpc_server_channel_options(bailian_grpc_config) -> list[tuple[str, int]]:
    """``grpc.server(..., options=...)`` from ``BailianGrpcConfig.get_server_config()``."""
    cfg = _resolve_bailian_grpc_config(bailian_grpc_config)
    return sorted((str(k), int(v)) for k, v in cfg.get_server_config().items())


def bailian_grpc_server_max_server_workers(bailian_grpc_config) -> int:
    cfg = _resolve_bailian_grpc_config(bailian_grpc_config)
    n = int(cfg.max_server_workers)
    assert n > 0, "bailian_grpc_config.max_server_workers must be > 0"
    return n


# Uvicorn / app main loop: set in frontend_app startup so async enqueue + route_ips run there.
_enqueue_loop: asyncio.AbstractEventLoop | None = None

# Fallback when _enqueue_loop is unset (e.g. tests / standalone fake without FrontendApp).
_async_loop: asyncio.AbstractEventLoop | None = None
_async_loop_thread: threading.Thread | None = None
_async_loop_lock = threading.Lock()


def set_bailian_grpc_enqueue_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _enqueue_loop
    _enqueue_loop = loop
    logging.info("[BailianGrpc] enqueue_event_loop set (uvicorn main loop)")


def get_bailian_grpc_enqueue_event_loop() -> asyncio.AbstractEventLoop | None:
    return _enqueue_loop


def _get_async_loop():
    """Get or create the dedicated event loop (runs in its own thread)."""
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
        # Wait until loop is actually running (timeout 5s)
        deadline = time.monotonic() + 5.0
        while not _async_loop.is_running() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not _async_loop.is_running():
            raise RuntimeError("Async loop failed to start for enqueue")
    return _async_loop


def _run_enqueue_sync(visitor, generate_input):
    """Drain async ``visitor.enqueue`` from the gRPC worker thread.

    Runs the coroutine on ``_enqueue_loop`` (uvicorn main loop) when set, otherwise on a
    dedicated loop thread (``_get_async_loop``).
    """

    async def collect():
        out = []
        stream = await visitor.enqueue(generate_input)
        async for x in stream:
            out.append(x)
        return out

    loop = None
    if _enqueue_loop is not None and _enqueue_loop.is_running():
        loop = _enqueue_loop
    if loop is None:
        loop = _get_async_loop()
    return asyncio.run_coroutine_threadsafe(collect(), loop).result()


class BailianGrpcInferenceServicer(predict_v2_pb2_grpc.GRPCInferenceServiceServicer):
    """ModelStreamInfer: fake mode (mock) or real mode (backend_visitor.enqueue)."""

    def __init__(self, backend_visitor=None):
        self._backend_visitor = backend_visitor

    def ModelStreamInfer(self, request_iterator, context):
        for request in request_iterator:
            logging.debug(
                "[BailianGrpc] ModelInferRequest: id=%s model_name=%s",
                request.id,
                request.model_name,
            )
            input_ids_list, sampling, other = parse_bailian_grpc_request(request)
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
                yield from iter_real_model_stream_infer(
                    request,
                    input_ids_list,
                    sampling,
                    other,
                    self._backend_visitor,
                    run_enqueue_sync=_run_enqueue_sync,
                )


_bailian_grpc_server = None
_bailian_grpc_server_thread = None

# Max time for grpc.Server.start() + bind before start_bailian_grpc_server_in_thread returns.
_DEFAULT_BAILIAN_GRPC_STARTUP_TIMEOUT_S = 30.0


def start_bailian_grpc_server(
    port: int,
    backend_visitor=None,
    bailian_grpc_config=None,
) -> grpc.Server:
    """Create and start Bailian gRPC server. backend_visitor=None -> fake (mock); else use enqueue."""
    global _bailian_grpc_server
    if _bailian_grpc_server is not None:
        logging.warning("[BailianGrpc] server already started")
        return _bailian_grpc_server
    cfg = _resolve_bailian_grpc_config(bailian_grpc_config)
    opts = bailian_grpc_server_channel_options(cfg)
    pool_size = bailian_grpc_server_max_server_workers(cfg)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=pool_size),
        options=opts,
    )
    predict_v2_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(
        BailianGrpcInferenceServicer(backend_visitor=backend_visitor), server
    )
    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    _bailian_grpc_server = server
    logging.info(
        "[BailianGrpc] Listening on 0.0.0.0:%s (predict_v2.proto, %s, max_server_workers=%s)",
        port,
        "real" if backend_visitor else "fake",
        pool_size,
    )
    return server


def start_bailian_grpc_server_in_thread(
    port: int,
    backend_visitor=None,
    bailian_grpc_config=None,
    *,
    startup_timeout_s: float = _DEFAULT_BAILIAN_GRPC_STARTUP_TIMEOUT_S,
) -> None:
    """Start Bailian gRPC in a daemon thread and block until ``server.start()`` succeeds.

    The daemon thread then runs ``wait_for_termination()`` so the process keeps serving.
    If bind/start fails or does not finish within ``startup_timeout_s``, raises the
    underlying exception or ``TimeoutError`` so callers (e.g. FastAPI startup) do not
    proceed while the port is not actually listening.
    """
    global _bailian_grpc_server, _bailian_grpc_server_thread

    started_ok = threading.Event()
    start_error: list[BaseException] = []

    def _run():
        try:
            server = start_bailian_grpc_server(
                port,
                backend_visitor=backend_visitor,
                bailian_grpc_config=bailian_grpc_config,
            )
        except BaseException as e:
            start_error.append(e)
            started_ok.set()
            return
        started_ok.set()
        if server is not None:
            server.wait_for_termination()

    _bailian_grpc_server_thread = threading.Thread(target=_run, daemon=True)
    _bailian_grpc_server_thread.start()
    if not started_ok.wait(timeout=startup_timeout_s):
        raise TimeoutError(
            f"Bailian gRPC server did not finish starting within {startup_timeout_s}s "
            f"(port {port})"
        )
    if start_error:
        raise start_error[0]


def main():
    """Standalone entry (fake mode only). Usage: python -m rtp_llm.bailian.bailian_grpc_server [--port PORT]"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Bailian gRPC server (predict_v2.proto wire, fake mode)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="gRPC port (default: 8000)"
    )
    parser.add_argument(
        "--bailian_grpc_config_json",
        type=str,
        default="",
        help="Optional JSON for BailianGrpcConfig (same shape as --bailian_grpc_config_json on main server).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    bailian_cfg = None
    if args.bailian_grpc_config_json.strip():
        from rtp_llm.ops import BailianGrpcConfig

        bailian_cfg = BailianGrpcConfig()
        bailian_cfg.from_json(args.bailian_grpc_config_json.strip())
    server = start_bailian_grpc_server(
        args.port, backend_visitor=None, bailian_grpc_config=bailian_cfg
    )
    server.wait_for_termination()


if __name__ == "__main__":
    main()
