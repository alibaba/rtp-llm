from typing import Any, Dict, List, Union

import torch
from pydantic import BaseModel

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.embedding.interface import EngineInputs, EngineOutputs
from rtp_llm.embedding.render.common_input_generator import CommonInputGenerator
from rtp_llm.embedding.render.custom_render import CustomRenderer
from rtp_llm.embedding.render.embedding.api_datatype import (
    EmbeddingResponseFormat,
    EmbeddingResponseType,
    OpenAIEmbeddingRequest,
    OpenAIEmbeddingResponse,
    SimilarityRequest,
    SimilarityResponse,
    Usage,
)
from rtp_llm.tokenizer_factory.tokenizers import BaseTokenizer


class EmbeddingRendererBase(CustomRenderer):
    embedding_type: EmbeddingResponseType

    def __init__(self, config: ModelConfig, tokenizer: BaseTokenizer):
        super().__init__(config, tokenizer)
        self.generator = CommonInputGenerator(tokenizer, config)

    def create_input(self, request: Union[OpenAIEmbeddingRequest, SimilarityRequest]):
        if isinstance(request, OpenAIEmbeddingRequest):
            engine_inputs = self.generator.generate(
                request.input, tokenizer_config=request.extra_configs.tokenizer_config
            )
        else:
            engine_inputs = self.generator.generate(request.left + request.right)
        return engine_inputs

    def embedding_func(
        self,
        request: BaseModel,
        res: torch.Tensor,
        input_length: int,
        input_tokens: torch.Tensor,
    ) -> Union[List[float], Dict[str, float], Dict[int, float]]:
        raise NotImplementedError

    def similar_func(
        self, left: EmbeddingResponseFormat, right: EmbeddingResponseFormat
    ) -> float:
        raise NotImplementedError

    async def _render_embedding_output(
        self, request: BaseModel, inputs: EngineInputs, outputs: EngineOutputs
    ) -> List[EmbeddingResponseFormat]:
        data: List[EmbeddingResponseFormat] = []
        bias = 0
        for i, out in enumerate(outputs.outputs):
            input_length = int(inputs.input_lengths[i])
            token_ids = inputs.token_ids[bias : bias + input_length]
            data.append(
                EmbeddingResponseFormat(
                    object=self.embedding_type,
                    embedding=self.embedding_func(
                        request, out, input_length, token_ids
                    ),
                    index=i,
                )
            )
            bias += input_length
        return data

    async def _render_similarity_output(
        self, request: SimilarityRequest, inputs: EngineInputs, outputs: EngineOutputs
    ) -> List[List[float]]:
        embedding_outputs = await self._render_embedding_output(
            request, inputs, outputs
        )
        left = embedding_outputs[: len(request.left)]
        right = embedding_outputs[len(request.left) :]
        batch_results: List[List[float]] = []
        for l_item in left:
            result: List[float] = []
            for r_item in right:
                result.append(self.similar_func(l_item, r_item))
            batch_results.append(result)
        return batch_results

    async def render_response(
        self,
        request: Union[OpenAIEmbeddingRequest, SimilarityRequest],
        inputs: EngineInputs,
        outputs: EngineOutputs,
    ) -> Dict[str, Any]:
        usage = Usage(
            prompt_tokens=outputs.input_length, total_tokens=outputs.input_length
        )
        if isinstance(request, OpenAIEmbeddingRequest):
            data = await self._render_embedding_output(request, inputs, outputs)
            return OpenAIEmbeddingResponse(data=data, usage=usage).model_dump()
        else:
            batch_results = await self._render_similarity_output(
                request, inputs, outputs
            )
            return SimilarityResponse(similarity=batch_results).model_dump()
