import json
import asyncio
from typing import Any, Dict, Union, Tuple, Optional
from maga_transformer.async_decoder_engine.async_model import AsyncModel
from maga_transformer.config.exceptions import FtRuntimeException, ExceptionType
from maga_transformer.async_decoder_engine.embedding.embedding_engine import EmbeddingCppEngine
from maga_transformer.models.downstream_modules.custom_module import CustomModule
from maga_transformer.async_decoder_engine.embedding.interface import EngineOutputs
from maga_transformer.embedding.embedding_type import TYPE_STR, EmbeddingType
from maga_transformer.ops import MultimodalInputCpp

class EmbeddingEndpoint(object):
    def __init__(self, model: AsyncModel):
        assert isinstance(model.decoder_engine_, EmbeddingCppEngine)
        self.decoder_engine_: EmbeddingCppEngine = model.decoder_engine_
        assert model.model.custom_module is not None, "custom model should not be None"
        self.custom_model_: CustomModule = model.model.custom_module

    async def handle(self, request: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        if isinstance(request, str):
            request = json.loads(request)
        renderer = self.custom_model_.get_renderer(request)
        handler = self.custom_model_.get_handler()
        try:
            formated_request = renderer.render_request(request)
            batch_input = renderer.create_input(formated_request)
        except Exception as e:
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, str(e))
        try:
            batch_output = await self.decoder_engine_.decode(batch_input)
            batch_output = handler.post_process(formated_request, batch_output)
            response = await renderer.render_response(formated_request, batch_input, batch_output)
            logable_response = await renderer.render_log_response(response)
        except Exception as e:
            raise FtRuntimeException(ExceptionType.EXECUTION_EXCEPTION, str(e))
        return response, logable_response

class EmbeddingHandler(object):
    def __init__(self, request: str, custom_module: CustomModule):
        self.request = json.loads(request)
        self.custom_module = custom_module
        self.renderer = custom_module.get_renderer(self.request)
        self.handler = custom_module.get_handler()

    def create_batch_input(self):
        self.formated_request = self.renderer.render_request(self.request)
        self.batch_input = self.renderer.create_input(self.formated_request)
        return self.batch_input

    def set_embedding_type(self, type_str: EmbeddingType):
        self.request[TYPE_STR] = type_str
        self.renderer = self.custom_module.get_renderer(self.request)

    async def render_response(self, outputs):
        batch_output = EngineOutputs(outputs=outputs, input_length=self.batch_input.input_length)
        batch_output = self.handler.post_process(self.formated_request, batch_output)
        self.response = await self.renderer.render_response(self.formated_request, self.batch_input, batch_output)
        return json.dumps(self.response)

    async def render_log_response(self):
        logable_response = await self.renderer.render_log_response(self.response)
        return json.dumps(logable_response)
