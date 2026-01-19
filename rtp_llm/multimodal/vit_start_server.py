import logging

from setproctitle import setproctitle

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.model_factory import ModelFactory
from rtp_llm.ops import TaskType
from rtp_llm.server.vit_app import VitEndpointApp

setup_logging()


def vit_start_server(py_env_configs: PyEnvConfigs, vit_server_port: int):
    setproctitle("rtp_llm_vit_server")

    from rtp_llm.multimodal.mm_process_engine import MMProcessEngine
    from rtp_llm.multimodal.multimodal_mixin_factory import MultimodalMixinFactory

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
    )

    if (
        not model_config.mm_model_config.is_multimodal
    ) or model_config.task_type != TaskType.LANGUAGE_MODEL:
        logging.info("No multimodal model or not language model, skip start vit server")
        app = VitEndpointApp(py_env_configs, None)
        app.start(vit_server_port)
        return

    model = MultimodalMixinFactory.create_multimodal_mixin(
        model_config=model_config,
        engine_config=engine_config,
        vit_config=py_env_configs.vit_config,
    )

    vit_process_engine = MMProcessEngine(
        model.mm_part,
        model.model_config,
        py_env_configs.vit_config,
        py_env_configs.profiling_debug_logging_config,
    )

    app = VitEndpointApp(py_env_configs, vit_process_engine)
    app.start(vit_server_port)
