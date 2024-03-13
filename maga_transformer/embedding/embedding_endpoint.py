from threading import Lock
from typing import AsyncGenerator, List, Set, Any, Optional, Dict
from maga_transformer.async_decoder_engine.async_model import AsyncModel
from maga_transformer.pipeline.embedding_pipeline import EmbeddingPipeline, EmbeddingResponse
from maga_transformer.embedding.api_datatype import SingleEmbeddingFormat, OpenAIEmbeddingRequestFormat, Usage, OpenAIEmbeddingResponseFormat
from maga_transformer.utils.complete_response_async_generator import CompleteResponseAsyncGenerator
from maga_transformer.config.exceptions import FtRuntimeException, ExceptionType

class EmbeddingEndpoint(object):
    def __init__(self, model: AsyncModel):
        self.pipeline = EmbeddingPipeline(model, model.tokenizer)
        self.lock = Lock()

    def sentence_embedding(self, request: OpenAIEmbeddingRequestFormat) -> CompleteResponseAsyncGenerator:
        return CompleteResponseAsyncGenerator(self._sentence_embedding(request), self._collect_func)

    async def _collect_func(self, responses: AsyncGenerator[OpenAIEmbeddingResponseFormat, None]):
        res = [x async for x in responses][-1]
        return res

    async def _sentence_embedding(self, request: OpenAIEmbeddingRequestFormat) -> AsyncGenerator[OpenAIEmbeddingResponseFormat, None]:
        result = await self.pipeline.pipeline_async(request.input)
        res: List[SingleEmbeddingFormat] = []
        for index, sentence_embedding in enumerate(result.sentence_embedding):
            if sentence_embedding is None:
                raise FtRuntimeException(ExceptionType.UNKNOWN_ERROR, "sentence embedding should not be None")
            res.append(SingleEmbeddingFormat(embedding=sentence_embedding.tolist(), index=index))
        yield OpenAIEmbeddingResponseFormat(data=res,
                                            model=request.model,
                                            usage=Usage(prompt_tokens=result.prompt_tokens, total_tokens=result.prompt_tokens))