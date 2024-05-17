import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict, Any, Union
from transformers import PreTrainedTokenizerBase

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.async_decoder_engine.embedding.embedding_stream import EngineInputs, EngineOutputs
from maga_transformer.models.downstream_modules.custom_module import CustomRenderer
from maga_transformer.models.downstream_modules.common_input_generator import CommonInputGenerator
from maga_transformer.models.downstream_modules.embedding.api_datatype import SimilarityRequest, SimilarityResponse, OpenAIEmbeddingRequest, \
    Usage, EmbeddingResponseFormat, EmbeddingResponseType, OpenAIEmbeddingResponse

def combo_to_batch(hidde_states: torch.Tensor, input_ids: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sliced_hidden_states: List[torch.Tensor] = []
    sliced_input_ids: List[torch.Tensor] = []
    hidden_bias = 0
    for input_length in input_lengths:
        sliced_hidden_states.append(hidde_states[hidden_bias: hidden_bias + input_length])
        sliced_input_ids.append(torch.IntTensor(input_ids[hidden_bias: hidden_bias + input_length]))
        hidden_bias += input_length
    batched_hidden_states = pad_sequence(sliced_hidden_states, batch_first=True)
    batched_input_ids = pad_sequence(sliced_input_ids, batch_first=True)
    # create attention mask from input_length
    max_input_length: int = max(input_lengths)
    batch_size = len(input_lengths)
    batched_attention_mask = torch.ones((batch_size, max_input_length), dtype=torch.bool, device="cuda:0")
    for b, input_length in enumerate(input_lengths):
        batched_attention_mask[b, input_length:] = 0
    return batched_input_ids, batched_hidden_states, batched_attention_mask


def combo_to_list(tensor: torch.Tensor, input_length: torch.Tensor) -> List[torch.Tensor]:
    result: List[torch.Tensor] = []
    bias = 0
    for length in input_length:
        result.append(tensor[bias: bias + length])
        bias += length
    return result


class EmbeddingRendererBase(CustomRenderer):
    embedding_type: EmbeddingResponseType
    def __init__(self, config: GptInitModelParameters, tokenizer: PreTrainedTokenizerBase):
        super().__init__(config, tokenizer)
        self.generator = CommonInputGenerator(tokenizer, config)

    async def render_request(self, request_json: Dict[str, Any]):
        if 'left' in request_json:
            return SimilarityRequest(**request_json)
        else:
            return OpenAIEmbeddingRequest(**request_json)

    async def create_input(self, request: Union[OpenAIEmbeddingRequest, SimilarityRequest]):
        if isinstance(request, OpenAIEmbeddingRequest):
            engine_inputs = await self.generator.generate(request.input)
        else:
            engine_inputs = await self.generator.generate(request.left + request.right)
        return engine_inputs

    def embedding_func(self, res: Any) -> Union[List[float], Dict[str, float]]:
        raise NotImplementedError

    def similar_func(self, left: Any, right: Any) -> float:
        raise NotImplementedError

    async def _render_embedding_output(self, request: OpenAIEmbeddingRequest, inputs: EngineInputs, outputs: EngineOutputs) -> Dict[str, Any]:
        usage = Usage(prompt_tokens=outputs.input_length, total_tokens=outputs.input_length)
        data = [
            EmbeddingResponseFormat(
                object=self.embedding_type,
                embedding=self.embedding_func(x),
                index=i)
            for i, x in enumerate(outputs.outputs)
        ]
        return OpenAIEmbeddingResponse(data=data, usage=usage).model_dump()

    async def _render_similarity_output(self, request: SimilarityRequest, inputs: EngineInputs, outputs: EngineOutputs) -> Dict[str, Any]:
        left = outputs.outputs[:len(request.left)]
        right = outputs.outputs[len(request.left): ]
        batch_results: List[List[float]] = []
        for l_item in left:
            result: List[float] = []
            for r_item in right:
                result.append(self.similar_func(l_item, r_item))                
            batch_results.append(result)
        return SimilarityResponse(similarity=batch_results).model_dump()

    async def render_response(self, request: Union[OpenAIEmbeddingRequest, SimilarityRequest], inputs: EngineInputs, outputs: EngineOutputs) -> Dict[str, Any]:
        if isinstance(request, OpenAIEmbeddingRequest):
            return await self._render_embedding_output(request, inputs, outputs)
        else:
            return await self._render_similarity_output(request, inputs, outputs)