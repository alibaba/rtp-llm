import asyncio
import logging

from typing_extensions import override

from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.async_decoder_engine.embedding.interface import EngineInputs, EngineOutputs
from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.task_type import TaskType
from rtp_llm.ops import MultimodalInputCpp, RtpEmbeddingOp
from rtp_llm.utils.mm_process_engine import MMProcessEngine


class EmbeddingCppEngine(BaseEngine):
    def __init__(self, model):
        super().__init__(model)
        logging.info("creating cpp embedding engine")
        self.model = model
        assert (
            self.model.custom_module is not None
        ), "embedding custom module should not be None"
        # self.cpp_handler = self.model.custom_module.create_cpp_handler()
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
        # self.model.custom_module.handler.init_cpp_handler()

    def decode_sync(self, inputs: EngineInputs, outputs: EngineOutputs):
        multimodal_inputs = [
            MultimodalInputCpp(i.url, i.tensor, int(i.mm_type))
            for i in inputs.multimodal_inputs
        ]
        results = self.cpp_engine.decode(
            inputs.token_ids,
            inputs.token_type_ids,
            inputs.input_lengths,
            0,
            multimodal_inputs,
        )
        outputs.outputs = results
        outputs.input_length = inputs.input_length

    @override
    async def decode(self, input: EngineInputs) -> EngineOutputs:
        output = EngineOutputs(outputs=None, input_length=0)
        await asyncio.to_thread(self.decode_sync, input, output)
        return output
