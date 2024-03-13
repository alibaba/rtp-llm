import torch
import asyncio
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Iterator, List, Optional, Union
from maga_transformer.utils.time_util import current_time_ms
from maga_transformer.metrics import kmonitor, GaugeMetrics

from maga_transformer.config.base_model_config import PyDanticModelBase
from maga_transformer.pipeline.pipeline import Pipeline
from maga_transformer.async_decoder_engine.embedding.embedding_stream import EmbeddingInput
from maga_transformer.async_decoder_engine.embedding.embedding_decoder_engine import EmbeddingDecoderEngine

class EmbeddingResponse(PyDanticModelBase):
    sentence_embedding: List[Optional[torch.Tensor]]
    sparse_embedding: List[Optional[torch.Tensor]]
    colbert_embedding: List[Optional[torch.Tensor]]
    prompt_tokens: int

class EmbeddingPipeline(Pipeline):
    @torch.inference_mode()
    async def pipeline_async( # type: ignore
        self,
        prompt: Union[List[str], str],
        images: Optional[List[str]] = None,
        **kwargs: Any
    ) -> EmbeddingResponse:
        if isinstance(prompt, str):
            prompt = [prompt]
        begin_time = current_time_ms()
        # align images and prompts
        if images is None or len(images) == 0:
            images = []
        generate_config_json = kwargs.pop("generate_config", {})
        generate_config = self.create_generate_config(generate_config_json, self.model.config.vocab_size, self.model.config.special_tokens, self.tokenizer, **kwargs)
        # do batch encode and split into embedding input per batch
        assert self.tokenizer is not None, "tokenizer should not be None"        
        encoded = self.tokenizer(prompt, return_attention_mask=False, padding=False, return_length=True)
        
        input_lengths: List[int] = encoded['length']
        token_ids: List[List[int]] = encoded['input_ids']
        token_type_ids: List[List[int]] = encoded.get("token_type_ids", [[0] * input_length for input_length in input_lengths])
        total_length = sum(input_lengths)
        
        kmonitor.report(GaugeMetrics.PRE_PIPELINE_RT_METRIC, current_time_ms() - begin_time)
        kmonitor.report(GaugeMetrics.INPUT_TOKEN_SIZE_METRIC, total_length)

        inputs = [EmbeddingInput(token_ids=token_id, token_type_ids=token_type_id, input_length=input_length, generate_config=generate_config) \
            for token_id, token_type_id, input_length in zip(token_ids, token_type_ids, input_lengths)]
        # check max_new_tokens > 0
        return await self.generate_stream(inputs, total_length, **kwargs)

    async def generate_stream(self, input: List[EmbeddingInput], total_length: int, **kwargs: Any) -> EmbeddingResponse:        
        assert isinstance(self.model.decoder_engine_, EmbeddingDecoderEngine)
        embedding_outputs = await self.model.decoder_engine_.decode(input)
        return EmbeddingResponse(sentence_embedding=[x.sentence_embedding for x in embedding_outputs],
                                sparse_embedding=[x.sparse_embedding for x in embedding_outputs],
                                colbert_embedding=[x.colbert_embedding for x in embedding_outputs],
                                prompt_tokens=total_length)
    
    def pipeline(self, prompt: Union[List[str], str], images: List[str] | None = None, **kwargs: Any) -> EmbeddingResponse:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.pipeline_async(prompt, images, **kwargs))
        return result

