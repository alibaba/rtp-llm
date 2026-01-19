import copy
from typing import Any, Dict, List, Union

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.embedding.render.embedding.api_datatype import (
    EmbeddingResponseFormat,
    EmbeddingResponseType,
    OpenAIEmbeddingRequest,
    SimilarityRequest,
)
from rtp_llm.embedding.render.embedding_renderer_base import EmbeddingRendererBase
from rtp_llm.tokenizer_factory.tokenizers import BaseTokenizer


class DenseEmbeddingRenderer(EmbeddingRendererBase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.embedding_type = EmbeddingResponseType.DENSE

    def render_request(
        self, request_json: Dict[str, Any]
    ) -> Union[SimilarityRequest, OpenAIEmbeddingRequest]:
        if "left" in request_json:
            return SimilarityRequest(**request_json)
        else:
            return OpenAIEmbeddingRequest(**request_json)

    def similar_func(
        self, left: EmbeddingResponseFormat, right: EmbeddingResponseFormat
    ) -> float:
        return float(torch.tensor(left.embedding) @ torch.tensor(right.embedding).T)

    def embedding_func(
        self,
        request: Any,
        res: torch.Tensor,
        input_length: int,
        input_tokens: torch.Tensor,
    ) -> List[float]:
        assert isinstance(res, torch.Tensor)
        return res.tolist()

    async def render_log_response(self, response: Dict[str, Any]):
        log_response = copy.copy(response)
        if "data" in log_response:
            del log_response["data"]
        return log_response
