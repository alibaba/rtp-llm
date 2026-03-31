"""
Bailian gRPC process lifecycle: bind ``grpc.Server``, start/stop, daemon thread, CLI ``main``.

Servicer implementation: ``bailian_grpc_service.BailianGrpcInferenceServicer``.
Started from FrontendApp via ``start_bailian_grpc_server_in_thread`` (daemon thread holds
``wait_for_termination``; caller blocks until ``server.start()`` succeeds).
Standalone fake: ``python -m rtp_llm.bailian.bailian_grpc_server [--port PORT]``.
"""

from __future__ import annotations

import logging
import threading
from concurrent import futures
from typing import Optional

import grpc

from rtp_llm.bailian.bailian_grpc_enqueue_loop import (
    get_bailian_grpc_enqueue_event_loop,
    set_bailian_grpc_enqueue_event_loop,
)
from rtp_llm.bailian.bailian_grpc_real_infer import _iter_enqueue_sync
from rtp_llm.bailian.bailian_grpc_service import BailianGrpcInferenceServicer
from rtp_llm.bailian.proto import predict_v2_pb2_grpc


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


_bailian_grpc_server: Optional[grpc.Server] = None
_bailian_grpc_server_thread: Optional[threading.Thread] = None
_bailian_grpc_server_lock = threading.Lock()

# Max time for grpc.Server.start() + bind before start_bailian_grpc_server_in_thread returns.
_DEFAULT_BAILIAN_GRPC_STARTUP_TIMEOUT_S = 30.0


def start_bailian_grpc_server(
    port: int,
    backend_visitor=None,
    bailian_grpc_config=None,
) -> grpc.Server:
    """Create and start Bailian gRPC server. backend_visitor=None -> fake (mock); else use enqueue."""
    global _bailian_grpc_server
    with _bailian_grpc_server_lock:
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


def stop_bailian_grpc_server(grace: Optional[float] = None) -> None:
    """Stop the global Bailian ``grpc.Server`` and clear module state (allows a later start).

    ``grace`` is passed to ``grpc.Server.stop`` (seconds to let in-flight RPCs finish); ``None``
    uses grpcio default. Safe to call multiple times or if the server was never started.

    Holds ``_bailian_grpc_server_lock`` for the full ``stop()`` so ``start_*`` cannot race on
    the same listen port during shutdown.
    """
    global _bailian_grpc_server, _bailian_grpc_server_thread
    with _bailian_grpc_server_lock:
        server = _bailian_grpc_server
        if server is None:
            return
        try:
            server.stop(grace)
            logging.info(
                "[BailianGrpc] server stopped (grace=%r); worker thread wait_for_termination returns",
                grace,
            )
        except Exception as e:
            logging.warning("[BailianGrpc] server.stop failed: %s", e, exc_info=True)
        finally:
            _bailian_grpc_server = None
            _bailian_grpc_server_thread = None


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
