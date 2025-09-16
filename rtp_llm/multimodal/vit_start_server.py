from typing import Optional

from setproctitle import setproctitle

from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.distribute.worker_info import g_worker_info, update_worker_info
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.server.vit_app import VitEndpointApp

setup_logging()


def vit_start_server(
    py_env_configs: PyEnvConfigs, vit_process_engine: Optional[MMProcessEngine]
):
    setproctitle("rtp_llm_vit_server")

    update_worker_info(
        py_env_configs.server_config.start_port,
        py_env_configs.server_config.worker_info_port_num,
        py_env_configs.distribute_config.remote_server_port,
    )

    app = VitEndpointApp(py_env_configs, vit_process_engine)
    app.start(g_worker_info)
