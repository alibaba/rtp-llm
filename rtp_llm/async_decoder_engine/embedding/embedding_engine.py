import asyncio
import logging

from typing_extensions import override

from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.ops import RtpEmbeddingOp
from rtp_llm.utils.mm_process_engine import MMProcessEngine


class EmbeddingCppEngine(BaseEngine):
    def __init__(self, model):
        super().__init__(model)
        logging.info("creating cpp embedding engine")
        self.model = model
        self.cpp_engine = RtpEmbeddingOp()

    @override
    def _stop(self) -> None:
        self.cpp_engine.stop()

    @override
    def _start(self):
        if self.model.is_multimodal():
            self.mm_engine = MMProcessEngine(self.model)
        else:
            self.mm_engine = None
        self.cpp_engine.init(self.model, self.mm_engine)
