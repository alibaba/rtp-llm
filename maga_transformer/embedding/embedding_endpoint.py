import json
import asyncio
from typing import Any, Dict, Union, Tuple, Optional
from maga_transformer.async_decoder_engine.async_model import AsyncModel
from maga_transformer.async_decoder_engine.embedding.embedding_engine import EmbeddingCppEngine
from maga_transformer.models.downstream_modules.custom_module import CustomModule

class EmbeddingEndpoint(object):
    def __init__(self, model: AsyncModel):
        assert isinstance(model.decoder_engine_, EmbeddingCppEngine)
        self.decoder_engine_: EmbeddingCppEngine = model.decoder_engine_
        assert model.model.custom_module is not None, "custom model should not be None"
        self.custom_model_: CustomModule = model.model.custom_module

    async def handle(self, request: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        if isinstance(request, str):
            request = json.loads(request)
        formated_request = await self.custom_model_.renderer.render_request(request)
        batch_input = await self.custom_model_.renderer.create_input(formated_request)
        batch_output = await self.decoder_engine_.decode(batch_input)
        response = await self.custom_model_.renderer.render_response(formated_request, batch_input, batch_output)
        logable_response = await self.custom_model_.renderer.render_log_response(response)
        return response, logable_response
