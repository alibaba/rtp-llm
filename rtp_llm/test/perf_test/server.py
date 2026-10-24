import argparse
import logging
import os
from typing import Dict, List, Optional

import requests

from rtp_llm.test.perf_test.cache_grid.runner.scheduler import configure_scheduler
from rtp_llm.test.perf_test.dataset import extract_arg
from rtp_llm.test.utils.maga_server_manager import MagaServerManager

EXISTING_SERVER_PORT_ENV = "CACHE_PERF_EXISTING_SERVER_PORT"


class EngineServer:
    """Engine server lifecycle management (start / stop / port)."""

    def __init__(self, args: argparse.Namespace, remaining_args: List[str]):
        self._args = args
        self._remaining_args = remaining_args
        self._server: Optional[MagaServerManager] = None
        self._existing_port = self._read_existing_port()

    def start(
        self,
        max_seq_len: int,
        max_concurrency: int,
        server_start_timeout: int = 1600,
        use_batch_decode_scheduler: Optional[bool] = True,
    ) -> None:
        """Assemble CLI args and start MagaServerManager."""
        if self._existing_port is not None:
            response = requests.get(
                f"http://127.0.0.1:{self._existing_port}/health", timeout=10
            )
            response.raise_for_status()
            logging.info("Using existing engine server on port %d", self._existing_port)
            return

        engine_cli = self._build_engine_cli(max_seq_len, max_concurrency)

        env: Dict[str, str] = {
            "FAKE_BALANCE_EXPERT": "1",
            "BATCH_DECODE_SCHEDULER_WARMUP_TYPE": (
                "0" if self._args.partial in (0, 1) else "1"
            ),
            "TORCH_CUDA_PROFILER_DIR": self._args.result_dir,
        }
        if use_batch_decode_scheduler is not None:
            env["USE_BATCH_DECODE_SCHEDULER"] = (
                "1" if use_batch_decode_scheduler else "0"
            )
        if getattr(self._args, "cache_profile_backend", "kineto") == "nsys":
            env["GEN_TIMELINE_SYNC"] = "0"
            engine_cli += " --gen_timeline_sync=False"

        logging.info(f"Starting server with engine CLI: {engine_cli}")
        logging.info(f"remaining_args (raw list): {self._remaining_args}")
        self._server = MagaServerManager(
            env_args=env,
            process_file_name="process.log",
            smoke_args_str=engine_cli,
        )
        try:
            if not self._server.start_server(timeout=server_start_timeout):
                self._server.print_process_log()
                raise RuntimeError(
                    "Engine server failed to start. Check process.log above for details."
                )
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.stop_server()

    def set_scheduler_mode(self, *, batch_size: int, mode: str) -> Dict[str, object]:
        """Configure BatchDecodeScheduler before issuing performance requests."""
        return configure_scheduler(
            self.port, batch_size=batch_size, mode=mode, post=requests.post, attempts=20
        )

    @property
    def port(self) -> int:
        if self._existing_port is not None:
            return self._existing_port
        assert self._server is not None, "Server not started"
        return self._server.port

    @staticmethod
    def _read_existing_port() -> Optional[int]:
        value = os.environ.get(EXISTING_SERVER_PORT_ENV)
        if value is None or not value.strip():
            return None
        try:
            port = int(value)
        except ValueError as error:
            raise ValueError(
                f"{EXISTING_SERVER_PORT_ENV} must be an integer"
            ) from error
        if not 1 <= port <= 65535:
            raise ValueError(f"{EXISTING_SERVER_PORT_ENV} must be between 1 and 65535")
        return port

    @property
    def log_file_path(self) -> Optional[str]:
        """Path to the server's process.log, or None before start."""
        return self._server.log_file_path if self._server else None

    def _build_engine_cli(self, max_seq_len: int, max_concurrency: int) -> str:
        """Assemble CLI args string for the engine server subprocess.

        remaining_args already contains all engine args (model_type, checkpoint_path,
        tp_size, etc.) that were not consumed by the perf test parser.
        We only put back the three args that the perf test consumed and may modify.
        """
        parts: List[str] = list(self._remaining_args)
        parts.extend(["--dp_size", str(self._args.dp_size)])
        parts.extend(["--max_seq_len", str(max_seq_len)])
        parts.extend(["--concurrency_limit", str(max_concurrency)])
        return " ".join(parts)

    @staticmethod
    def propagate_engine_env(remaining_args: List[str]) -> None:
        """Set engine CLI args as env vars so downstream consumers work automatically.

        create_query() and MagaServerManager.start_server() fall back to env vars
        (MODEL_TYPE, CHECKPOINT_PATH, TOKENIZER_PATH) when explicit params are not
        provided.  Setting them once here avoids threading engine-specific values
        through the perf-test layer.
        """
        for cli_key, env_key in [
            ("model_type", "MODEL_TYPE"),
            ("checkpoint_path", "CHECKPOINT_PATH"),
            ("tokenizer_path", "TOKENIZER_PATH"),
        ]:
            val = extract_arg(remaining_args, cli_key)
            if val and env_key not in os.environ:
                os.environ[env_key] = val
        if "TOKENIZER_PATH" not in os.environ and "CHECKPOINT_PATH" in os.environ:
            os.environ["TOKENIZER_PATH"] = os.environ["CHECKPOINT_PATH"]
