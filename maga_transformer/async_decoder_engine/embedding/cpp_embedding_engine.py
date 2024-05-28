import time
import torch
import asyncio
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor
from maga_transformer.models.base_model import BaseModel
from maga_transformer.metrics import kmonitor, GaugeMetrics
from maga_transformer.ops import RtpEmbeddingOp
from maga_transformer.async_decoder_engine.embedding.embedding_stream import EngineInputs, EngineOutputs

class EmbeddingCppEngine(object):
    def __init__(self, model: BaseModel):
        logging.info("creating cpp embedding engine")
        self.model = model
        assert self.model.custom_module is not None, "embedding custom module should not be None"
        # self.cpp_handler = self.model.custom_module.create_cpp_handler()
        self.cpp_engine = RtpEmbeddingOp(model.config.gpt_init_params, self.model.custom_module.handler)

    def start(self):
        self.cpp_engine.init(self.model.weight.weights, self.model.weight.global_weights)

    def decode_sync(self, inputs: EngineInputs, outputs: EngineOutputs):
        try:
            start_time = time.time()
            results = self.cpp_engine.decode(inputs.token_ids, inputs.token_type_ids, inputs.input_lengths, 0)
            end_time = time.time()
            # print("engine cost:" , end_time - start_time)        
            outputs.outputs = results
            outputs.input_length = inputs.input_length
        except Exception as e:
            raise Exception("failed to run query, error: ", e)

    async def decode(self, input: EngineInputs):
        output = EngineOutputs(outputs=[], input_length=0)
        await asyncio.to_thread(self.decode_sync, input, output)
        return output