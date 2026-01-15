from typing import Any, Dict

from rtp_llm.async_decoder_engine.embedding.interface import EngineInputs, EngineOutputs
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.embedding.embedding_type import TYPE_STR, EmbeddingType
from rtp_llm.embedding.render.colbert_embedding_renderer import ColbertEmbeddingRenderer
from rtp_llm.embedding.render.custom_render import CustomRenderer
from rtp_llm.embedding.render.dense_embedding_renderer import DenseEmbeddingRenderer
from rtp_llm.embedding.render.sparse_embedding_renderer import SparseEmbeddingRenderer
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer


class BgeM3Renderer(CustomRenderer):
    def __init__(self, config: ModelConfig, tokenizer: BaseTokenizer):
        super().__init__(config, tokenizer)
        self._current_renderer = None
        self._renderer_map = {
            EmbeddingType.DENSE: DenseEmbeddingRenderer,
            EmbeddingType.SPARSE: SparseEmbeddingRenderer,
            EmbeddingType.COLBERT: ColbertEmbeddingRenderer,
        }

    def _get_renderer(self, request: Dict[str, Any]) -> CustomRenderer:
        request_type = request.get(TYPE_STR, EmbeddingType.DENSE)
        if request_type not in self._renderer_map:
            raise Exception(
                f"BgeM3Renderer get unexpected type: {request_type}, "
                f"expected one of {list(self._renderer_map.keys())}"
            )
        renderer_class = self._renderer_map[request_type]
        return renderer_class(self.config_, self.tokenizer_)

    def render_request(self, request_json: Dict[str, Any]):
        self._current_renderer = self._get_renderer(request_json)
        return self._current_renderer.render_request(request_json)

    def create_input(self, request):
        if self._current_renderer is None:
            raise Exception("render_request must be called before create_input")
        return self._current_renderer.create_input(request)

    async def render_response(
        self, request, inputs: EngineInputs, outputs: EngineOutputs
    ) -> Dict[str, Any]:
        if self._current_renderer is None:
            raise Exception("render_request must be called before render_response")
        return await self._current_renderer.render_response(request, inputs, outputs)

    async def render_log_response(self, response: Dict[str, Any]):
        if self._current_renderer is None:
            return response
        return await self._current_renderer.render_log_response(response)
