import logging
from typing import Dict, Optional, Type

from rtp_llm.omni.engine.stage_processor_base import StageProcessorBase

logger = logging.getLogger(__name__)


class StageProcessorRegistry:
    _registry: Dict[str, Type[StageProcessorBase]] = {}

    @classmethod
    def register(cls, name: str, processor_cls: Type[StageProcessorBase]) -> None:
        if name in cls._registry:
            raise ValueError(f"Stage processor already registered: {name}")
        cls._registry[name] = processor_cls
        logger.info(f"Registered stage processor: {name}")

    @classmethod
    def get(cls, name: str) -> Optional[Type[StageProcessorBase]]:
        return cls._registry.get(name)

    @classmethod
    def create(cls, name: str) -> StageProcessorBase:
        processor_cls = cls._registry.get(name)
        if processor_cls is None:
            raise KeyError(f"Stage processor not found: {name}")
        return processor_cls()
