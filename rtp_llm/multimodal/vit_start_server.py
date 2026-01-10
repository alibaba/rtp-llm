from typing import Optional

from setproctitle import setproctitle

from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.distribute.worker_info import g_worker_info, update_worker_info
from rtp_llm.server.vit_app import VitEndpointApp

setup_logging()


def vit_start_server(
    server_id: int,
    py_env_configs: PyEnvConfigs,
    grpc_port: int,
    http_port: Optional[int] = None,
    is_proxy_mode: bool = False,
):
    """
    Start VIT server in a separate process.
    Creates vit_process_engine inside this function for parallel creation across processes.

    Args:
        server_id: 服务器 ID
        py_env_configs: 配置对象
        grpc_port: gRPC 端口号（从外部传入）
        http_port: HTTP 端口号（从外部传入，可选）
                   如果为 None，表示工作进程模式（不启动 HTTP 服务器）
        is_proxy_mode: 是否在 proxy 模式下运行（proxy 模式下的 worker 进程不需要记录 QPS）
    """
    import logging

    from rtp_llm.model_factory import ModelFactory

    # Set server_id on the passed config
    py_env_configs.server_config.vit_server_id = server_id
    setproctitle(f"rtp_llm_vit_server_{server_id}")

    logging.info(
        f"[VIT_SERVER_{server_id}] Creating vit_process_engine... "
        f"(grpc_port={grpc_port}, http_port={http_port}, is_proxy_mode={is_proxy_mode})"
    )

    vit_process_engine = ModelFactory.create_vit_from_env(
        py_env_configs, is_proxy_mode=is_proxy_mode
    )

    if vit_process_engine is None:
        logging.error(f"[VIT_SERVER_{server_id}] Failed to create vit_process_engine")
        raise RuntimeError(
            f"Failed to create vit_process_engine for server {server_id}"
        )

    if vit_process_engine.is_embedding_task():
        logging.info(
            f"[VIT_SERVER_{server_id}] vit_process_engine is embedding task, exiting"
        )
        return

    logging.info(f"[VIT_SERVER_{server_id}] vit_process_engine created successfully")

    update_worker_info(
        py_env_configs.server_config.start_port,
        py_env_configs.server_config.worker_info_port_num,
        py_env_configs.distribute_config.remote_server_port,
    )

    app = VitEndpointApp(py_env_configs, vit_process_engine)
    app.start(
        g_worker_info,
        grpc_port=grpc_port,
        http_port=http_port,
    )
