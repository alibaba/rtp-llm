from collections import defaultdict
from typing import Any, Dict, Union

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.embedding.render.embedding.api_datatype import (
    EmbeddingResponseFormat,
    EmbeddingResponseType,
    SimilarityRequest,
    SparseEmbeddingRequest,
)
from rtp_llm.embedding.render.embedding_renderer_base import EmbeddingRendererBase
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer


class SparseEmbeddingRenderer(EmbeddingRendererBase):
    def __init__(self, config: ModelConfig, tokenizer: BaseTokenizer):
        super().__init__(config, tokenizer)
        self.embedding_type = EmbeddingResponseType.SPARSE
        self.unused_tokens = set(
            [
                self.tokenizer_.cls_token_id,
                self.tokenizer_.eos_token_id,
                self.tokenizer_.pad_token_id,
                self.tokenizer_.unk_token_id,
            ]
        )

    def render_request(self, request_json: Dict[str, Any]):
        if "left" in request_json:
            return SimilarityRequest(**request_json)
        else:
            return SparseEmbeddingRequest(**request_json)

    def embedding_func(
        self,
        request: Union[SparseEmbeddingRequest, SimilarityRequest],
        res: torch.Tensor,
        input_length: int,
        input_tokens: torch.Tensor,
    ) -> Union[Dict[str, float]]:
        if len(res.shape) != 1:
            raise Exception("sparse hidden should be 1-dim")
        sparse_emb: Dict[int, float] = defaultdict(float)
        for score, id in zip(res[:input_length], input_tokens):
            score = float(score)
            id = int(id)
            if id in self.unused_tokens:
                continue
            if score > 0 and sparse_emb[id] < score:
                sparse_emb[id] = score
        if isinstance(request, SparseEmbeddingRequest) and request.return_decoded:
            return {
                self.tokenizer_.decode(key): value for key, value in sparse_emb.items()
            }
        else:
            return {str(k): v for k, v in sparse_emb.items()}

    def similar_func(
        self, left: EmbeddingResponseFormat, right: EmbeddingResponseFormat
    ) -> float:
        if not isinstance(left.embedding, dict) or not isinstance(
            right.embedding, dict
        ):
            raise Exception("sparse similaritey datatype error")
        result: float = 0
        for key in left.embedding.keys():
            if key not in right.embedding:
                continue
            result += left.embedding[key] * right.embedding[key]
        return result
