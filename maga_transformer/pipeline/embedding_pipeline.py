import torch
import asyncio
from typing import Any, Iterator, List, Optional, Union, Dict
from maga_transformer.utils.time_util import current_time_ms
from maga_transformer.metrics import kmonitor, GaugeMetrics

from maga_transformer.config.base_model_config import PyDanticModelBase
from maga_transformer.embedding.embedding_config import EmbeddingGenerateConfig
from maga_transformer.pipeline.pipeline import Pipeline
from maga_transformer.async_decoder_engine.embedding.embedding_stream import EmbeddingInput, EmbeddingOutput
from maga_transformer.config.exceptions import FtRuntimeException, ExceptionType
from maga_transformer.async_decoder_engine.embedding.embedding_decoder_engine import EmbeddingDecoderEngine

class EmbeddingResponse(PyDanticModelBase):
    embedding_outputs: List[EmbeddingOutput]
    prompt_tokens: int

class EmbeddingPipeline(Pipeline):
    @torch.inference_mode()
    async def pipeline_async( # type: ignore
        self,
        prompt: Union[List[str], str],
        images: Optional[List[str]] = None,
        embedding_config: EmbeddingGenerateConfig = EmbeddingGenerateConfig(),
        **kwargs: Any
    ) -> EmbeddingResponse:
        if isinstance(prompt, str):
            prompt = [prompt]
        begin_time = current_time_ms()
        # align images and prompts
        if images is None or len(images) == 0:
            images = []
        # do batch encode and split into embedding input per batch
        assert self.tokenizer is not None, "tokenizer should not be None"
        # truncate with tokenizer max_seq_len
        encoded = self.tokenizer(prompt, return_attention_mask=False, padding=False, return_length=True, truncation='longest_first')

        input_lengths: List[int] = encoded['length']
        token_ids: List[List[int]] = encoded['input_ids']
        token_type_ids: List[List[int]] = encoded.get("token_type_ids", [[0] * input_length for input_length in input_lengths])
        total_length = sum(input_lengths)
        # double check input length < self.model.config.max_seq_len
        for length in input_lengths:
            if length > self.model.config.max_seq_len:
                raise FtRuntimeException(ExceptionType.LONG_PROMPT_ERROR, f"one of prompt length: {length} > max_length: {self.model.config.max_seq_len}")

        kmonitor.report(GaugeMetrics.PRE_PIPELINE_RT_METRIC, current_time_ms() - begin_time)
        kmonitor.report(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC, total_length)

        inputs = [EmbeddingInput(token_ids=token_id, token_type_ids=token_type_id, input_length=input_length, embedding_config=embedding_config) \
            for token_id, token_type_id, input_length in zip(token_ids, token_type_ids, input_lengths)]
        return await self.generate_stream(inputs, total_length, **kwargs)

    async def generate_stream(self, input: List[EmbeddingInput], total_length: int, **kwargs: Any) -> EmbeddingResponse:
        assert isinstance(self.model.decoder_engine_, EmbeddingDecoderEngine)
        embedding_outputs = await self.model.decoder_engine_.decode(input)
        return EmbeddingResponse(embedding_outputs=embedding_outputs, prompt_tokens=total_length)

    def pipeline(self, prompt: Union[List[str], str], images: List[str] | None = None, **kwargs: Any) -> EmbeddingResponse:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.pipeline_async(prompt, images, **kwargs))
        return result