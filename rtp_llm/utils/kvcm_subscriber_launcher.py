from __future__ import annotations

import logging
import multiprocessing
import os
import shlex
import signal
import socket
import subprocess
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rtp_llm.config.py_config_modules import PyEnvConfigs

_CONFIG_ENV = "KVCM_SUBSCRIBER_CONFIG"
_COMMAND_ENV = "KVCM_SUBSCRIBER_COMMAND"
_ENDPOINTS_ENV = "RTP_LLM_CACHE_SUBSCRIBER_ENDPOINTS"
_HOST_IP_PORT_ENV = "KVCM_HOST_IP_PORT"
_WORLD_RANK_ENV = "KVCM_SUBSCRIBER_WORLD_RANK"
_REQUIRED_ENV = "KVCM_SUBSCRIBER_REQUIRED"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"", "0", "false", "no", "off"}

_RESTART_INTERVAL_SECONDS = 5.0
_CHILD_POLL_INTERVAL_SECONDS = 0.2
_CHILD_SHUTDOWN_TIMEOUT_SECONDS = 10.0
_RETRY_LOG_EVERY = 12


def is_kvcm_subscriber_required() -> bool:
    """Return whether Subscriber failures should fail the inference service."""

    value = os.environ.get(_REQUIRED_ENV, "").strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    logging.warning(
        "invalid %s=%r; defaulting to optional Subscriber mode",
        _REQUIRED_ENV,
        value,
    )
    return False


def _local_ip_address() -> str:
    configured = os.environ.get("HOST_IP", "").strip()
    if configured:
        return configured
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


def _subscriber_world_rank() -> int:
    value = os.environ.get(_WORLD_RANK_ENV, "0").strip()
    try:
        world_rank = int(value)
    except ValueError:
        raise ValueError(f"{_WORLD_RANK_ENV} must be an integer") from None
    if world_rank < 0:
        raise ValueError(f"{_WORLD_RANK_ENV} must be >= 0")
    return world_rank


def _model_dtype(py_env_configs: PyEnvConfigs) -> str:
    value = (py_env_configs.model_args.act_type or "").strip().lower()
    return {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }.get(value, value)


def _model_name(py_env_configs: PyEnvConfigs) -> str:
    checkpoint_path = py_env_configs.model_args.ckpt_path.rstrip("/")
    if checkpoint_path:
        return os.path.basename(checkpoint_path)
    return py_env_configs.model_args.model_type or "default"


def build_kvcm_subscriber_command(
    py_env_configs: PyEnvConfigs,
) -> tuple[str, ...] | None:
    """Build the external KVCM Subscriber command when configured.

    RTP owns only lifecycle and runtime endpoint discovery. All polling, diff,
    event conversion, and KVCM communication remain in the KVCM package.
    """

    config_path = os.environ.get(_CONFIG_ENV, "").strip()
    if not config_path:
        return None

    current_world_rank = int(py_env_configs.parallelism_config.world_rank)
    launch_world_rank = _subscriber_world_rank()
    if current_world_rank != launch_world_rank:
        logging.info(
            "skip KVCM Subscriber on world rank %s; configured launch rank is %s",
            current_world_rank,
            launch_world_rank,
        )
        return None
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"{_CONFIG_ENV} does not exist: {config_path}")

    endpoints = os.environ.get(_ENDPOINTS_ENV, "").strip()
    dp_size = int(py_env_configs.parallelism_config.dp_size)
    if not endpoints:
        if dp_size != 1:
            raise ValueError(
                f"{_ENDPOINTS_ENV} is required when DP_SIZE is greater than 1; "
                "remote DP rank addresses cannot be inferred safely by one RTP process"
            )
        endpoints = f"127.0.0.1:{py_env_configs.server_config.rpc_server_port}"
    endpoint_list = [item.strip() for item in endpoints.split(",") if item.strip()]
    if len(endpoint_list) != dp_size:
        raise ValueError(
            f"{_ENDPOINTS_ENV} must contain exactly one endpoint per DP rank; "
            f"expected {dp_size}, got {len(endpoint_list)}"
        )
    if len(set(endpoint_list)) != len(endpoint_list):
        raise ValueError(f"{_ENDPOINTS_ENV} must not contain duplicate endpoints")
    endpoints = ",".join(endpoint_list)

    host_ip_port = os.environ.get(_HOST_IP_PORT_ENV, "").strip()
    if not host_ip_port:
        host_ip = py_env_configs.server_config.ip or _local_ip_address()
        if ":" in host_ip and not host_ip.startswith("["):
            host_ip = f"[{host_ip}]"
        host_ip_port = f"{host_ip}:{py_env_configs.server_config.server_port}"

    command = shlex.split(os.environ.get(_COMMAND_ENV, "subscriber"))
    if not command:
        raise ValueError(f"{_COMMAND_ENV} must contain an executable")
    parallelism = py_env_configs.parallelism_config
    return (
        *command,
        "--config",
        config_path,
        "--engine-type",
        "rtp_llm",
        "--rtp-endpoints",
        endpoints,
        "--host-ip-port",
        host_ip_port,
        "--block-size",
        str(py_env_configs.kv_cache_config.seq_size_per_block),
        "--model-name",
        _model_name(py_env_configs),
        "--model-dtype",
        _model_dtype(py_env_configs),
        "--tensor-parallel-size",
        str(parallelism.tp_size),
        "--data-parallel-size",
        str(parallelism.dp_size),
        "--pipeline-parallel-size",
        str(parallelism.pp_size),
    )


def _signal_process_group(process, signum: int) -> None:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        return
    except OSError:
        try:
            if process.poll() is None:
                process.send_signal(signum)
        except OSError:
            pass


def _stop_subscriber_process(process) -> None:
    if process.poll() is not None:
        return
    _signal_process_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=_CHILD_SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _signal_process_group(process, signal.SIGKILL)
        process.wait()


def _log_optional_failure(
    failure_count: int,
    reason: str,
    restart_interval_s: float,
    *,
    exc_info=None,
) -> None:
    if failure_count != 1 and failure_count % _RETRY_LOG_EVERY != 0:
        return
    logging.warning(
        "optional KVCM Subscriber %s; restarting in %.1fs (failure %s)",
        reason,
        restart_interval_s,
        failure_count,
        exc_info=exc_info,
    )


def _supervise_kvcm_subscriber(
    command: tuple[str, ...],
    required: bool,
    restart_interval_s: float,
    *,
    stop_event=None,
) -> None:
    """Keep the optional sidecar alive without changing RTP's ProcessManager."""

    if stop_event is None:
        stop_event = threading.Event()

    def request_shutdown(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)

    failure_count = 0
    while not stop_event.is_set():
        process = None
        reason = ""
        exc_info = None
        try:
            process = subprocess.Popen(
                list(command),
                env=os.environ.copy(),
                start_new_session=True,
            )
            logging.info(
                "KVCM Subscriber child started with pid %s",
                process.pid,
            )
            while not stop_event.wait(_CHILD_POLL_INTERVAL_SECONDS):
                returncode = process.poll()
                if returncode is not None:
                    break

            if stop_event.is_set():
                _stop_subscriber_process(process)
                return

            returncode = process.poll()
            if returncode is None:
                continue
            if required:
                raise RuntimeError(
                    f"required KVCM Subscriber exited with code {returncode}"
                )
            reason = f"exited with code {returncode}"
        except Exception as exc:
            if process is not None:
                _stop_subscriber_process(process)
            if required:
                raise
            reason = f"failed with {exc.__class__.__name__}: {exc}"
            exc_info = (exc.__class__, exc, exc.__traceback__)

        failure_count += 1
        _log_optional_failure(
            failure_count,
            reason,
            restart_interval_s,
            exc_info=exc_info,
        )
        if stop_event.wait(restart_interval_s):
            return


def start_kvcm_subscriber(
    py_env_configs: PyEnvConfigs,
    *,
    required: bool | None = None,
) -> multiprocessing.Process | None:
    if required is None:
        required = is_kvcm_subscriber_required()

    try:
        command = build_kvcm_subscriber_command(py_env_configs)
        if command is None:
            return None

        logging.info(
            "starting external KVCM Subscriber: engine_type=rtp_llm endpoint=%s",
            command[command.index("--rtp-endpoints") + 1],
        )
        process = multiprocessing.Process(
            target=_supervise_kvcm_subscriber,
            args=(command, required, _RESTART_INTERVAL_SECONDS),
            name="kvcm_subscriber_supervisor",
            daemon=True,
        )
        process.start()
        return process
    except Exception:
        if required:
            raise
        logging.exception(
            "optional KVCM Subscriber failed to start; inference will continue "
            "without cache-state reporting"
        )
        return None
