import asyncio
import logging
from typing import Dict, AsyncGenerator
from typing_extensions import override
from maga_transformer.models.base_model import BaseModel
from maga_transformer.async_decoder_engine.embedding.interface import EngineInputs, EngineOutputs
from maga_transformer.async_decoder_engine.base_engine import BaseEngine, KVCacheInfo
from maga_transformer.ops import RtpEmbeddingOp

class EmbeddingCppEngine(BaseEngine):
    def __init__(self, model: BaseModel):
        logging.info("creating cpp embedding engine")
        self.model = model
        assert self.model.custom_module is not None, "embedding custom module should not be None"
        # self.cpp_handler = self.model.custom_module.create_cpp_handler()
        self.cpp_engine = RtpEmbeddingOp()

    @override
    def stop(self) -> None:
        self.cpp_engine.stop()

    @override
    def start(self):
        self.cpp_engine.init(self.model)
        self.model.custom_module.handler.init_cpp_handler()

    def decode_sync(self, inputs: EngineInputs, outputs: EngineOutputs):
        try:
            results = self.cpp_engine.decode(inputs.token_ids, inputs.token_type_ids, inputs.input_lengths, 0)
            outputs.outputs = results
            outputs.input_length = inputs.input_length
        except Exception as e:
            raise Exception("failed to run query, error: ", e)

    @override
    async def decode(self, input: EngineInputs) -> AsyncGenerator[EngineOutputs, None]:
        output = EngineOutputs(outputs=None, input_length=0)
        await asyncio.to_thread(self.decode_sync, input, output)
        return output

    @override
    def get_kv_cache_info(self) -> KVCacheInfo:
        return KVCacheInfo(available_kv_cache=0, total_kv_cache=0)
