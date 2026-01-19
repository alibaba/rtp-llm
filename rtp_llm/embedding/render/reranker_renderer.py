from typing import Any, Dict, List, Tuple

import numpy as np

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.embedding.interface import EngineInputs, EngineOutputs
from rtp_llm.embedding.render.common_input_generator import CommonInputGenerator
from rtp_llm.embedding.render.custom_render import CustomRenderer
from rtp_llm.embedding.render.reranker.api_datatype import (
    RankingItem,
    VoyageRerankerRequest,
    VoyageRerankerResponse,
)
from rtp_llm.tokenizer_factory.tokenizers import BaseTokenizer


class RerankerRenderer(CustomRenderer):
    def __init__(self, config: ModelConfig, tokenizer: BaseTokenizer):
        super().__init__(config, tokenizer)
        self.generator = CommonInputGenerator(tokenizer, config)

    def render_request(self, request: Dict[str, Any]):
        return VoyageRerankerRequest(**request)

    @staticmethod
    def sigmoid(x: float):
        return float(1 / (1 + np.exp(-x)))

    def create_input(self, formated_request: VoyageRerankerRequest):
        input: List[Tuple[str, str]] = [
            (formated_request.query, doc) for doc in formated_request.documents
        ]
        return self.generator.generate(input, truncate=formated_request.truncation)

    async def render_response(
        self,
        formated_request: VoyageRerankerRequest,
        inputs: EngineInputs,
        outputs: EngineOutputs,
    ) -> Dict[str, Any]:
        if outputs.outputs is None:
            raise Exception("outputs should not be None")
        rank_items: List[RankingItem] = []
        for i in range(len(formated_request.documents)):
            rank_items.append(
                RankingItem(
                    index=i,
                    document=(
                        formated_request.documents[i]
                        if formated_request.return_documents
                        else None
                    ),
                    relevance_score=(
                        float(outputs.outputs[i])
                        if not formated_request.normalize
                        else self.sigmoid(float(outputs.outputs[i]))
                    ),
                )
            )
        if formated_request.sorted:
            rank_items.sort(key=lambda x: x.relevance_score, reverse=True)
        if formated_request.top_k is not None:
            rank_items = rank_items[: min(len(rank_items), formated_request.top_k)]
        return VoyageRerankerResponse(
            results=rank_items, total_tokens=len(inputs.token_ids)
        ).model_dump(exclude_none=True)
