import logging

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.multimodal.multimodal_mixin_register import (
    _multimodal_mixin_factory,
    get_multimodal_mixin_cls,
)
from rtp_llm.multimodal.multimodal_mixins import BaseMultiModalMixin
from rtp_llm.ops import TaskType


class MultimodalMixinFactory:
    @staticmethod
    def _create_multimodal_mixin(
        model_config: ModelConfig,
        engine_config: EngineConfig,
        vit_config: VitConfig,
        device: str = "cuda:0",
    ) -> BaseMultiModalMixin:
        if not model_config.mm_model_config.is_multimodal:
            logging.info("No multimodal model, skip create multimodal mixin")
            return None
        multimodal_mixin_cls = get_multimodal_mixin_cls(model_config.model_type)
        return multimodal_mixin_cls(
            model_config.compute_dtype,
            device,
            model_config.mm_related_params,
            engine_config.load_config.load_method,
            vit_config,
            model_config.ckpt_path,
        )

    @staticmethod
    def create_multimodal_process_engine(
        model_config: ModelConfig,
        engine_config: EngineConfig,
        vit_config: VitConfig,
        device: str = "cuda:0",
        server_id: int = 0,
        is_proxy_mode: bool = False,
    ) -> MMProcessEngine:
        mm_mixin = MultimodalMixinFactory._create_multimodal_mixin(
            model_config=model_config,
            engine_config=engine_config,
            vit_config=vit_config,
            device=device,
        )
        return MMProcessEngine(
            mm_mixin.mm_part,
            model_config,
            vit_config,
            engine_config.profiling_debug_logging_config,
            server_id,
            is_proxy_mode,
        )
