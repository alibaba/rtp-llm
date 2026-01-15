from typing import Any, Dict, List, Union

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.embedding.render.embedding.api_datatype import (
    ColbertEmbeddingRequest,
    EmbeddingResponseFormat,
    EmbeddingResponseType,
    SimilarityRequest,
)
from rtp_llm.embedding.render.embedding_renderer_base import EmbeddingRendererBase
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer


class ColbertEmbeddingRenderer(EmbeddingRendererBase):
    def __init__(self, config: ModelConfig, tokenizer: BaseTokenizer):
        super().__init__(config, tokenizer)
        self.embedding_type = EmbeddingResponseType.COLBERT

    def render_request(
        self, request_json: Dict[str, Any]
    ) -> Union[SimilarityRequest, ColbertEmbeddingRequest]:
        if "left" in request_json:
            return SimilarityRequest(**request_json)
        else:
            return ColbertEmbeddingRequest(**request_json)

    def embedding_func(
        self,
        request: Any,
        res: torch.Tensor,
        input_length: int,
        input_tokens: torch.Tensor,
    ) -> List[float]:
        assert isinstance(res, torch.Tensor)
        return res[: input_length - 1].tolist()

    def similar_func(
        self, left: EmbeddingResponseFormat, right: EmbeddingResponseFormat
    ):
        left_t = torch.tensor(left.embedding)
        right_t = torch.tensor(right.embedding)
        token_scores = torch.einsum("in,jn->ij", left_t, right_t)
        scores, _ = token_scores.max(-1)
        scores = torch.sum(scores) / left_t.size(0)
        return float(scores)
