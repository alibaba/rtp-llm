import torch
import asyncio
import logging
from typing import List, Any, Dict
from maga_transformer.models.base_model import BaseModel
from maga_transformer.async_decoder_engine.embedding.interface import EngineInputs, EngineOutputs
from maga_transformer.ops import RtpEmbeddingOp

class EmbeddingCppEngine(object):
    def __init__(self, model: BaseModel):
        logging.info("creating cpp embedding engine")
        self.model = model
        assert self.model.custom_module is not None, "embedding custom module should not be None"
        # self.cpp_handler = self.model.custom_module.create_cpp_handler()
        self.cpp_engine = RtpEmbeddingOp()

    def start(self):
        self.cpp_engine.init(self.model.config.gpt_init_params, self.model.custom_module.handler, self.model.weight.weights, self.model.weight.global_weights)

    def decode_sync(self, inputs: EngineInputs, outputs: EngineOutputs):
        try:            
            results = self.cpp_engine.decode(inputs.token_ids, inputs.token_type_ids, inputs.input_lengths, 0)
            outputs.outputs = results
            outputs.input_length = inputs.input_length
        except Exception as e:
            raise Exception("failed to run query, error: ", e)

    async def decode(self, input: EngineInputs):
        output = EngineOutputs(outputs=None, input_length=0)
        await asyncio.to_thread(self.decode_sync, input, output)
        return output
