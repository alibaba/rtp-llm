import copy
from typing import Any, Dict, List, Optional, Union

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.embedding.render.embedding.api_datatype import (
    EmbeddingResponseFormat,
    EmbeddingResponseType,
    OpenAIEmbeddingRequest,
    SimilarityRequest,
)
from rtp_llm.embedding.render.embedding_renderer_base import EmbeddingRendererBase
from rtp_llm.tokenizer_factory.tokenizers import BaseTokenizer


class MiniCPMVRenderer(EmbeddingRendererBase):

    def __init__(
        self,
        config: ModelConfig,
        tokenizer: BaseTokenizer,
        vit_config: Optional[VitConfig] = None,
    ):
        super().__init__(config, tokenizer)
        self.embedding_type = EmbeddingResponseType.DENSE
        from rtp_llm.embedding.minicpmv_input_generator import MiniCPMVInputGenerator

        self.generator = MiniCPMVInputGenerator(
            config, tokenizer, vit_config=vit_config
        )

    def similar_func(
        self, left: EmbeddingResponseFormat, right: EmbeddingResponseFormat
    ) -> float:
        return float(torch.tensor(left.embedding) @ torch.tensor(right.embedding).T)

    def render_request(
        self, request_json: Dict[str, Any]
    ) -> Union[SimilarityRequest, OpenAIEmbeddingRequest]:
        if "left" in request_json:
            return SimilarityRequest(**request_json)
        else:
            return OpenAIEmbeddingRequest(**request_json)

    def embedding_func(
        self,
        request: Any,
        res: torch.Tensor,
        input_length: int,
        input_tokens: torch.Tensor,
    ) -> List[float]:
        assert isinstance(res, torch.Tensor)
        return res.tolist()

    def create_input(self, request: Union[OpenAIEmbeddingRequest, SimilarityRequest]):
        if isinstance(request, OpenAIEmbeddingRequest):
            engine_inputs = self.generator.generate(
                request.input, tokenizer_config=request.extra_configs.tokenizer_config
            )
        else:
            engine_inputs = self.generator.generate(request.left + request.right)
        return engine_inputs

    async def render_log_response(self, response: Dict[str, Any]):
        log_response = copy.copy(response)
        if "data" in log_response:
            del log_response["data"]
        return log_response
