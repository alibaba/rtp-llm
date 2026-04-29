"""DashScApp: standalone process hosting the DashSc gRPC server.

Parallel to ``FrontendApp`` but only serves the predict_v2 gRPC wire. Owns its own
asyncio event loop (the ``enqueue`` coroutine needs one) and its own
``BackendRPCServerVisitor`` so the two protocols can iterate independently.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import threading
import traceback
from typing import Optional

from rtp_llm.config.log_config import get_log_path
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.dash_sc.server import DashScGrpcServer
from rtp_llm.dash_sc.service import set_dash_sc_grpc_enqueue_event_loop
from rtp_llm.metrics import kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.server.backend_rpc_server_visitor import create_backend_rpc_server_visitor


class DashScApp:
    """Self-contained lifecycle for a per-rank DashSc gRPC server.

    Startup order (``start``):
      1. Build ``ModelConfig`` (no weight loading — just architecture/ports metadata).
      2. Build ``BackendRPCServerVisitor`` via the shared factory.
      3. Spin up a dedicated asyncio loop in a background thread and register it
         as the enqueue loop so servicer coroutines have somewhere to run.
      4. Call ``self._grpc_server.start_in_thread`` (blocks until bind succeeds
         or raises on bind/start error).
      5. Notify the parent via the pipe, then block the main thread waiting on
         SIGTERM/SIGINT.
    """

    def __init__(self, py_env_configs: PyEnvConfigs):
        self.py_env_configs = py_env_configs
        self.server_config = py_env_configs.server_config
        self.dash_sc_grpc_config = py_env_configs.dash_sc_grpc_config

        self._enqueue_loop: Optional[asyncio.AbstractEventLoop] = None
        self._enqueue_loop_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._grpc_server = DashScGrpcServer(dash_sc_grpc_config=self.dash_sc_grpc_config)

    def _start_enqueue_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        thread = threading.Thread(
            target=_run, name="dash_sc_enqueue_loop", daemon=True
        )
        thread.start()
        if not ready.wait(timeout=5.0):
            raise RuntimeError("dash_sc enqueue asyncio loop failed to start")
        self._enqueue_loop = loop
        self._enqueue_loop_thread = thread
        return loop

    def _stop_enqueue_loop(self) -> None:
        loop = self._enqueue_loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass
        thread = self._enqueue_loop_thread
        if thread is not None:
            thread.join(timeout=5.0)
        try:
            loop.close()
        except Exception as e:
            logging.warning("[DashScApp] close enqueue loop failed: %s", e)
        self._enqueue_loop = None
        self._enqueue_loop_thread = None

    def _install_signal_handlers(self) -> None:
        def _handler(signum, frame):
            logging.info("[DashScApp] received signal %s, shutting down", signum)
            self._shutdown_event.set()

        try:
            signal.signal(signal.SIGTERM, _handler)
            signal.signal(signal.SIGINT, _handler)
        except ValueError:
            # signal handlers can only be installed on the main thread
            logging.warning(
                "[DashScApp] signal handlers not installed (not on main thread)"
            )

    def start(self, ready_pipe_writer=None) -> None:
        try:
            model_config = ModelFactory.create_model_config(
                model_args=self.py_env_configs.model_args,
                lora_config=self.py_env_configs.lora_config,
                kv_cache_config=self.py_env_configs.kv_cache_config,
                profiling_debug_logging_config=self.py_env_configs.profiling_debug_logging_config,
                generate_env_config=self.py_env_configs.generate_env_config,
                embedding_config=self.py_env_configs.embedding_config,
                quantization_config=self.py_env_configs.quantization_config,
                render_config=self.py_env_configs.render_config,
            )

            backend_visitor = create_backend_rpc_server_visitor(
                py_env_configs=self.py_env_configs,
                model_config=model_config,
            )

            loop = self._start_enqueue_loop()
            set_dash_sc_grpc_enqueue_event_loop(loop)

            # Register py_rtp_* metrics so the access-log interceptor's kmonitor.report
            # calls find their metric objects. Idempotent — matches FrontendServer.__init__
            # so dashboards/alerts see gRPC and HTTP paths under the same metric family
            # (split via the ``protocol`` tag the interceptor injects).
            kmonitor.init()

            port = self.server_config.dash_sc_grpc_server_port
            logging.info(
                "[DashScApp] starting gRPC server rank_id=%s server_id=%s port=%s",
                self.server_config.rank_id,
                self.server_config.frontend_server_id,
                port,
            )
            self._grpc_server.start_in_thread(
                port,
                backend_visitor=backend_visitor,
                ip=self.server_config.ip,
                server_id=self.server_config.frontend_server_id,
                log_path=get_log_path(),
                backup_count=self.py_env_configs.profiling_debug_logging_config.log_file_backup_count,
                rank_id=self.server_config.rank_id,
            )
            logging.info("[DashScApp] gRPC server bound on port %s", port)
        except BaseException as e:
            error_trace = traceback.format_exc()
            logging.error(
                "[DashScApp] start failed: %s\n%s", e, error_trace
            )
            if ready_pipe_writer is not None:
                try:
                    ready_pipe_writer.send(
                        {
                            "status": "failed",
                            "message": str(e),
                            "traceback": error_trace,
                        }
                    )
                    ready_pipe_writer.close()
                except Exception as pipe_error:
                    logging.warning(
                        "[DashScApp] failed to send failure via pipe: %s",
                        pipe_error,
                    )
            self._stop_enqueue_loop()
            raise

        if ready_pipe_writer is not None:
            try:
                ready_pipe_writer.send(
                    {
                        "status": "success",
                        "message": (
                            f"DashSc gRPC server started on rank "
                            f"{self.server_config.rank_id} "
                            f"server {self.server_config.frontend_server_id}"
                        ),
                    }
                )
                ready_pipe_writer.close()
            except Exception as e:
                logging.warning(
                    "[DashScApp] failed to send success via pipe: %s", e
                )

        self._install_signal_handlers()
        logging.info("[DashScApp] entering service loop (SIGTERM/SIGINT to stop)")
        try:
            self._shutdown_event.wait()
        finally:
            self.stop()

    def stop(self) -> None:
        to = self.server_config.shutdown_timeout
        grace = None if to < 0 else float(to)
        try:
            self._grpc_server.stop(grace)
        except Exception as e:
            logging.warning("[DashScApp] grpc_server.stop failed: %s", e)
        self._stop_enqueue_loop()
