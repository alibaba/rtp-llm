from setproctitle import setproctitle

from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.distribute.worker_info import g_worker_info, update_worker_info
from rtp_llm.server.vit_app import VitEndpointApp

setup_logging()


def vit_start_server(
    server_id: int,
    py_env_configs: PyEnvConfigs,
):
    """
    Start VIT server in a separate process.
    Creates vit_process_engine inside this function for parallel creation across processes.
    """
    import logging

    from rtp_llm.model_factory import ModelFactory

    # Set server_id on the passed config
    py_env_configs.server_config.vit_server_id = server_id
    setproctitle(f"rtp_llm_vit_server_{server_id}")

    logging.info(f"[VIT_SERVER_{server_id}] Creating vit_process_engine...")

    # Create vit_process_engine in this process for parallel creation
    vit_process_engine = ModelFactory.create_vit_from_env(py_env_configs)

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
    app.start(g_worker_info)
