import json
import asyncio
from typing import AsyncGenerator, List, Set, Any, Optional, Dict, Union, Type
from maga_transformer.config.base_model_config import PyDanticModelBase
from maga_transformer.async_decoder_engine.async_model import AsyncModel
from maga_transformer.pipeline.embedding_pipeline import EmbeddingPipeline, EmbeddingResponse
from maga_transformer.embedding.api_datatype import EmbeddingResponseFormat, EmbeddingResponseType, OpenAIEmbeddingRequest, Usage, OpenAIEmbeddingResponse, SimilarityRequest, SimilarityResponse
from maga_transformer.embedding.embedding_config import EmbeddingGenerateConfig, EmbeddingType
from maga_transformer.embedding.utils import calc_similarity
from maga_transformer.utils.complete_response_async_generator import CompleteResponseAsyncGenerator
from maga_transformer.config.exceptions import FtRuntimeException, ExceptionType

class EmbeddingEndpoint(object):
    def __init__(self, model: AsyncModel):
        self.pipeline = EmbeddingPipeline(model, model.tokenizer)

    async def embedding(self, request: Union[Dict[str, Any], str, OpenAIEmbeddingRequest]) -> OpenAIEmbeddingResponse:
        request = self._request_to_base_model(request, OpenAIEmbeddingRequest)
        assert isinstance(request, OpenAIEmbeddingRequest)
        result = await self.pipeline.pipeline_async(request.input, embedding_config=request.embedding_config)
        if request.embedding_config.type == EmbeddingType.DENSE:
            res = [EmbeddingResponseFormat(embedding=output.sentence_embedding.tolist(), object=EmbeddingResponseType.DENSE, index=index) for index, output in enumerate(result.embedding_outputs)]
        elif request.embedding_config.type == EmbeddingType.SPARSE:
            res = [EmbeddingResponseFormat(embedding=output.sparse_embedding, object=EmbeddingResponseType.SPARSE, index=index) for index, output in enumerate(result.embedding_outputs)]
        elif request.embedding_config.type == EmbeddingType.COLBERT:
            res = [EmbeddingResponseFormat(embedding=output.colbert_embedding.tolist(), object=EmbeddingResponseType.COLBERT, index=index) for index, output in enumerate(result.embedding_outputs)]
        else:
            raise Exception(f"internal error, unkown emebdding type: {request.embedding_config.type}")

        return OpenAIEmbeddingResponse(data=res,
                                       model=request.model,
                                       usage=Usage(prompt_tokens=result.prompt_tokens, total_tokens=result.prompt_tokens))

    async def similarity(self, request: Union[Dict[str, Any], str, SimilarityRequest]) -> SimilarityResponse:
        request = self._request_to_base_model(request, SimilarityRequest)
        assert isinstance(request, SimilarityRequest), f"error similartity request type: {type(request)}"
        task1 = asyncio.create_task(self.embedding(OpenAIEmbeddingRequest(input=request.left,
                                                                          model=request.model,
                                                                          embedding_config=request.embedding_config)))
        task2 = asyncio.create_task(self.embedding(OpenAIEmbeddingRequest(input=request.right,
                                                                          model=request.model,
                                                                          embedding_config=request.embedding_config)))
        left_result = await task1
        right_result = await task2
        response = calc_similarity(left_result, right_result, request.embedding_config.type)
        if request.return_response:
            response.left_response = left_result
            response.right_response = right_result
        return response

    def _request_to_base_model(self, request: Any, cls: Type[Any]) -> PyDanticModelBase:
        if isinstance(request, str):
            request = json.loads(request)
        if isinstance(request, dict):
            request = cls(**request)
        if not isinstance(request, cls):
            raise FtRuntimeException(ExceptionType.ERROR_INPUT_FORMAT_ERROR, f"input_format {type(request)} is not correct, want {cls}")
        return request
