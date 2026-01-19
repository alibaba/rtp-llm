import logging

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.multimodal_mixin_register import (
    _multimodal_mixin_factory,
    get_multimodal_mixin_cls,
)
from rtp_llm.multimodal.multimodal_mixins import BaseMultiModalMixin
from rtp_llm.ops import TaskType


class MultimodalMixinFactory:
    @staticmethod
    def create_multimodal_mixin(
        model_config: ModelConfig, engine_config: EngineConfig, vit_config: VitConfig
    ) -> BaseMultiModalMixin:
        if not model_config.mm_model_config.is_multimodal:
            logging.info("No multimodal model, skip create multimodal mixin")
            return None
        multimodal_mixin_cls = get_multimodal_mixin_cls(model_config.model_type)
        return multimodal_mixin_cls(model_config, engine_config, vit_config)
