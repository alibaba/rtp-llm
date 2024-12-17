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
        renderer = self.custom_model_.get_renderer(request)        
        handler = self.custom_model_.get_handler()
        formated_request = renderer.render_request(request)
        batch_input = renderer.create_input(formated_request)
        batch_output = await self.decoder_engine_.decode(batch_input)
        batch_output = handler.post_process(formated_request, batch_output)
        response = await renderer.render_response(formated_request, batch_input, batch_output)        
        logable_response = await renderer.render_log_response(response)
        return response, logable_response