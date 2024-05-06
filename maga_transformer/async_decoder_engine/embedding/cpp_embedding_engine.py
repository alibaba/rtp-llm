import time
import torch
import asyncio
from typing import List
from concurrent.futures import ThreadPoolExecutor
from maga_transformer.models.base_model import BaseModel
from maga_transformer.metrics import kmonitor, GaugeMetrics
from maga_transformer.async_decoder_engine.embedding.embedding_stream import EmbeddingInput, EngineInputs, EngineOutputs

class EmbeddingCppEngine(object):
    def __init__(self, model: BaseModel):
        self.model = model
        self.cpp_engine = torch.classes.FasterTransformer.RtpEmbeddingOp(model.config.gpt_init_params)
        self.executor = ThreadPoolExecutor(100)

    def start(self):
        self.cpp_engine.init(self.model.config.gpt_init_params, self.model.weight.weights, self.model.weight.global_weights)

    def decode_sync(self, inputs: EngineInputs, outputs: EngineOutputs):
        try:
            start_time = time.time()
            results = self.cpp_engine.handle(inputs.token_ids, inputs.token_type_ids, inputs.input_lengths, 0)
            end_time = time.time()
            kmonitor.report(GaugeMetrics.ASYNC_ITERATE_LANTENCY, (end_time - start_time) * 1000)
            kmonitor.report(GaugeMetrics.ASYNC_BATCH_SIZE_METRIC, len(inputs.token_ids))
            # print("engine cost:" , end_time - start_time)        
            outputs.outputs = results[:, -1].tolist()
        except Exception as e:
            raise Exception("failed to run query, error: ", e)

    async def decode(self, input: EngineInputs):
        output = EngineOutputs(outputs=[], input_length=0)
        await asyncio.to_thread(self.decode_sync, input, output)
        return output