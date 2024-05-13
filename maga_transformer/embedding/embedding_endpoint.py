import json
import asyncio
from typing import Any, Dict
from maga_transformer.async_decoder_engine.async_model import AsyncModel
from maga_transformer.async_decoder_engine.embedding.embedding_decoder_engine import EmbeddingDecoderEngine
from maga_transformer.models.downstream_modules.custom_module import CustomModule

class EmbeddingEndpoint(object):
    def __init__(self, model: AsyncModel):
        assert isinstance(model.decoder_engine_ , EmbeddingDecoderEngine)
        self.decoder_engine_: EmbeddingDecoderEngine = model.decoder_engine_
        assert model.model.custom_module is not None, "custom model should not be None"
        self.custom_model_: CustomModule = model.model.custom_module
        assert isinstance(self.decoder_engine_ , EmbeddingDecoderEngine), f"decoder engine should be EmbeddingDecoderEngine, acutal: {type(self.decoder_engine_)}"

    async def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(request, str):
            request = json.loads(request)
        formated_request = await self.custom_model_.renderer.render_request(request)
        batch_input = await self.custom_model_.renderer.create_input(formated_request)
        batch_output = await self.decoder_engine_.decode(batch_input)
        return await self.custom_model_.renderer.render_response(formated_request, batch_input, batch_output)
