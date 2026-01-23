import logging

from setproctitle import setproctitle

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.model_factory import ModelFactory
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.multimodal.multimodal_mixin_factory import MultimodalMixinFactory
from rtp_llm.ops import TaskType
from rtp_llm.server.vit_app import VitEndpointApp

setup_logging()

from typing import Optional


def vit_start_server(
    server_id: int,
    py_env_configs: PyEnvConfigs,
    grpc_port: int,
    http_port: Optional[int] = None,
    is_proxy_mode: bool = False,
):
    # Set server_id on the passed config
    py_env_configs.server_config.vit_server_id = server_id
    setproctitle(f"rtp_llm_vit_server_{server_id}")

    logging.info(
        f"[VIT_SERVER_{server_id}] Creating vit_process_engine... "
        f"(grpc_port={grpc_port}, http_port={http_port}, is_proxy_mode={is_proxy_mode})"
    )

    engine_config = EngineConfig.create(py_env_configs)

    model_config = ModelFactory.create_model_config(
        model_args=py_env_configs.model_args,
        lora_config=py_env_configs.lora_config,
        kv_cache_config=engine_config.kv_cache_config,
        profiling_debug_logging_config=engine_config.profiling_debug_logging_config,
        generate_env_config=py_env_configs.generate_env_config,
        embedding_config=py_env_configs.embedding_config,
        quantization_config=py_env_configs.quantization_config,
        render_config=py_env_configs.render_config,
        eplb_config=py_env_configs.eplb_config,
        vit_config=py_env_configs.vit_config,
    )

    if (
        not model_config.mm_model_config.is_multimodal
    ) or model_config.task_type != TaskType.LANGUAGE_MODEL:
        logging.info(
            f"[VIT_SERVER_{server_id}] No multimodal model or not language model, skip start vit server"
        )
        app = VitEndpointApp(py_env_configs, None)
        app.start(grpc_port, http_port)
        return

    vit_process_engine = MultimodalMixinFactory.create_multimodal_process_engine(
        model_config=model_config,
        engine_config=engine_config,
        vit_config=py_env_configs.vit_config,
        device="cuda:0",
        server_id=server_id,
        is_proxy_mode=is_proxy_mode,
    )

    logging.info(
        f"[VIT_SERVER_{server_id}] Creating multimodal process engine finished"
    )

    app = VitEndpointApp(py_env_configs, vit_process_engine)
    app.start(
        grpc_port=grpc_port,
        http_port=http_port,
    )
