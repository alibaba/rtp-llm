import logging
from typing import Optional

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.multimodal.multimodal_mixin_register import (
    _multimodal_mixin_factory,
    get_multimodal_mixin_cls,
)
from rtp_llm.multimodal.multimodal_mixins import BaseMultiModalMixin
from rtp_llm.models_py.registry import is_model_registered
from rtp_llm.ops import TaskType
from rtp_llm.utils.new_loader import (
    is_new_loader_enabled,
    new_loader_unsupported_reason,
)


class MultimodalMixinFactory:
    @staticmethod
    def _create_multimodal_mixin(
        model_config: ModelConfig,
        engine_config: EngineConfig,
        vit_config: VitConfig,
        device: str = "cuda:0",
        resolved_use_new_loader: Optional[bool] = None,
    ) -> BaseMultiModalMixin:
        if not model_config.mm_model_config.is_multimodal:
            logging.info("No multimodal model, skip create multimodal mixin")
            return None
        multimodal_mixin_cls = get_multimodal_mixin_cls(model_config.model_type)
        use_new_loader = resolved_use_new_loader
        if use_new_loader is None:
            use_new_loader = is_new_loader_enabled(
                model_config,
                default_enabled=is_model_registered(model_config.model_type),
            )
            if use_new_loader and model_config.use_new_loader is None:
                unsupported_reason = new_loader_unsupported_reason(
                    model_config,
                    force_cpu_load_weights=(
                        engine_config.load_config.force_cpu_load_weights
                    ),
                    device_resource_config=engine_config.device_resource_config,
                    parallelism_config=engine_config.parallelism_config,
                )
                if unsupported_reason is not None:
                    use_new_loader = False
                    logging.warning(
                        "multimodal loader follows the language compatibility "
                        "fallback for model_type=%s (%s)",
                        model_config.model_type,
                        unsupported_reason,
                    )
        return multimodal_mixin_cls(
            model_config.compute_dtype,
            device,
            model_config.mm_related_params,
            engine_config.load_config.load_method,
            vit_config,
            model_config.ckpt_path,
            use_new_loader=use_new_loader,
        )

    @staticmethod
    def create_multimodal_process_engine(
        model_config: ModelConfig,
        engine_config: EngineConfig,
        vit_config: VitConfig,
        device: str = "cuda:0",
        server_id: int = 0,
        is_proxy_mode: bool = False,
        resolved_use_new_loader: Optional[bool] = None,
    ) -> MMProcessEngine:
        mm_mixin = MultimodalMixinFactory._create_multimodal_mixin(
            model_config=model_config,
            engine_config=engine_config,
            vit_config=vit_config,
            device=device,
            resolved_use_new_loader=resolved_use_new_loader,
        )
        return MMProcessEngine(
            mm_mixin.mm_part,
            model_config,
            vit_config,
            engine_config.profiling_debug_logging_config,
            server_id,
            is_proxy_mode,
            device=device,
        )
