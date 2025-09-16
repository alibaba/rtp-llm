import asyncio
import logging
from typing import Optional

from typing_extensions import override

from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.ops import RtpEmbeddingOp


class EmbeddingCppEngine(BaseEngine):
    def __init__(
        self,
        model,
        engine_config: EngineConfig,
        mm_process_engine: Optional[MMProcessEngine] = None,
    ):
        super().__init__(model)
        logging.info("creating cpp embedding engine")
        self.model = model
        self.engine_config = engine_config
        self.mm_process_engine = mm_process_engine
        self.cpp_engine = RtpEmbeddingOp()

    @override
    def _stop(self) -> None:
        self.cpp_engine.stop()

    @override
    def _start(self):
        self.cpp_engine.init(
            self.model,
            self.engine_config,
            self.model.vit_config,
            self.mm_process_engine,
        )
