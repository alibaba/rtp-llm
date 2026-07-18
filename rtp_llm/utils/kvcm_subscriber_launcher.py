from __future__ import annotations

import logging
import multiprocessing
import os
import shlex
import socket
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rtp_llm.config.py_config_modules import PyEnvConfigs

_CONFIG_ENV = "KVCM_SUBSCRIBER_CONFIG"
_COMMAND_ENV = "KVCM_SUBSCRIBER_COMMAND"
_ENDPOINTS_ENV = "RTP_LLM_CACHE_SUBSCRIBER_ENDPOINTS"
_HOST_IP_PORT_ENV = "KVCM_HOST_IP_PORT"
_WORLD_RANK_ENV = "KVCM_SUBSCRIBER_WORLD_RANK"


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


def _exec_kvcm_subscriber(command: tuple[str, ...]) -> None:
    os.execvpe(command[0], list(command), os.environ.copy())


def start_kvcm_subscriber(
    py_env_configs: PyEnvConfigs,
) -> multiprocessing.Process | None:
    command = build_kvcm_subscriber_command(py_env_configs)
    if command is None:
        return None

    logging.info(
        "starting external KVCM Subscriber: engine_type=rtp_llm endpoint=%s",
        command[command.index("--rtp-endpoints") + 1],
    )
    process = multiprocessing.Process(
        target=_exec_kvcm_subscriber,
        args=(command,),
        name="kvcm_subscriber",
    )
    process.start()
    return process
