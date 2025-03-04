# -*- coding: utf-8 -*-
# @Author: zibai.gj
# @Time  : 2024-02-26

import kserve
from kserve.model import PredictorConfig
from kserve.protocol.rest.v2_datamodels import GenerateRequest, GenerateResponse
from kserve.errors import InferenceError
from typing import Dict, Any, AsyncIterator, Optional
import argparse
import asyncio

from maga_transformer.inference import FrontendWorker


class LLMModel(kserve.Model):
    def __init__(self, predictor_config: Optional[PredictorConfig] = None):
        super().__init__('rtp-llm', predictor_config)
        self._frontend_worker = FrontendWorker()
        self.ready = True

    def preprocess(self, payload, context=None):
        raise InferenceError("USE_GENERATE_ENDPOINT_ERROR")

    async def generate(self, generate_request: GenerateRequest, headers: Dict[str, str] = None) \
            -> AsyncIterator[Any]:
        if headers.get("streaming", "false") == "true":
            return self.stream_wrap(generate_request)
        else:
            return await self.nonstream_wrap(generate_request)

    async def nonstream_wrap(self, generate_request: GenerateRequest):
        rep_gen = self._frontend_worker.inference(text=generate_request.text_input,
                                                   generate_config=generate_request.parameters)
        async for rep in rep_gen:
            pass
        return GenerateResponse(text_output=rep['response'], model_name=self.name)

    async def stream_wrap(self, generate_request: GenerateRequest):
        rep_gen = self._frontend_worker.inference(text=generate_request.text_input,
                                                   generate_config=generate_request.parameters)
        async for rep in rep_gen:
            yield rep['response']
            await asyncio.sleep(0)

    async def predict(self, input_batch, context=None):
        raise InferenceError("USE_GENERATE_ENDPOINT_ERROR")

    def postprocess(self, outputs, context=None):
        raise InferenceError("USE_GENERATE_ENDPOINT_ERROR")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
    args, _ = parser.parse_known_args()
    predictor_config = PredictorConfig(args.predictor_host, args.predictor_protocol,
                                       args.predictor_use_ssl,
                                       args.predictor_request_timeout_seconds)
    kserve.ModelServer().start(models=[LLMModel(predictor_config)])

# curl http://0.0.0.0:8080/v2/models/rtp-llm/generate_stream -d '{"text_input":"hi","parameters":{"max_new_tokens":10}}'  -H "content-type:application/json"
# curl http://0.0.0.0:8080/v2/models/rtp-llm/generate -d '{"text_input":"hi","parameters":{"max_new_tokens":10}}'  -H "content-type:application/json"
