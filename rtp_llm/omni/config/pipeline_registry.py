import logging
from typing import Dict, List, Optional

from rtp_llm.omni.config.stage_config import OmniPipelineConfig

logger = logging.getLogger(__name__)


class OmniPipelineRegistry:
    _registry: Dict[str, OmniPipelineConfig] = {}
    _arch_registry: Dict[str, str] = {}

    @classmethod
    def register(cls, config: OmniPipelineConfig) -> None:
        if config.model_type in cls._registry:
            raise ValueError(
                f"Pipeline already registered for model_type={config.model_type}"
            )
        cls._registry[config.model_type] = config
        cls._arch_registry[config.model_arch] = config.model_type
        logger.info(
            f"Registered omni pipeline: model_type={config.model_type}, "
            f"model_arch={config.model_arch}, stages={len(config.stages)}"
        )

    @classmethod
    def get(cls, model_type: str) -> Optional[OmniPipelineConfig]:
        return cls._registry.get(model_type)

    @classmethod
    def get_by_arch(cls, model_arch: str) -> Optional[OmniPipelineConfig]:
        model_type = cls._arch_registry.get(model_arch)
        if model_type is None:
            return None
        return cls._registry.get(model_type)

    @classmethod
    def list_all(cls) -> List[OmniPipelineConfig]:
        return list(cls._registry.values())
