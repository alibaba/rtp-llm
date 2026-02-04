from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from pydantic import BaseModel
from torch.nn.utils.rnn import pad_sequence

from rtp_llm.async_decoder_engine.embedding.interface import EngineInputs, EngineOutputs
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.models.downstream_modules.common_input_generator import (
    CommonInputGenerator,
)
from rtp_llm.models.downstream_modules.custom_module import CustomRenderer
from rtp_llm.models.downstream_modules.embedding.api_datatype import (
    EmbeddingResponseFormat,
    EmbeddingResponseType,
    OpenAIEmbeddingRequest,
    OpenAIEmbeddingResponse,
    SimilarityRequest,
    SimilarityResponse,
    Usage,
)


def lengths_to_slices(input_lengths: Sequence[int]) -> List[slice]:
    offset = 0
    input_slices: List[slice] = []
    for input_length in input_lengths:
        input_slices.append(slice(offset, offset + input_length))
        offset += input_length
    return input_slices


def combo_to_batch_input_ids(
    input_ids: torch.Tensor, input_slices: List[slice]
) -> torch.Tensor:
    return pad_sequence(
        [torch.IntTensor(input_ids[s]) for s in input_slices], batch_first=True
    )


def combo_to_batch_hidden_states(
    hidden_states: torch.Tensor, input_slices: List[slice]
) -> torch.Tensor:
    return pad_sequence([hidden_states[s] for s in input_slices], batch_first=True)


def combo_to_batch_moe_gating(
    moe_gating: List[Optional[torch.Tensor]], input_slices: List[slice]
) -> List[Optional[torch.Tensor]]:
    return [
        (
            pad_sequence([g[s] for s in input_slices], batch_first=True)
            if g is not None
            else None
        )
        for g in moe_gating
    ]


def generate_attention_mask(
    input_lengths: Sequence[int], device: Optional[torch.device] = None
) -> torch.Tensor:
    max_input_length: int = max(input_lengths)
    batch_size = len(input_lengths)
    batched_attention_mask = torch.ones(
        (batch_size, max_input_length), dtype=torch.bool, device=device
    )
    for b, input_length in enumerate(input_lengths):
        batched_attention_mask[b, input_length:] = 0
    return batched_attention_mask


def combo_to_batch_data(
    input_lengths: torch.Tensor, combo_data: Dict[str, Any]
) -> Dict[str, Any]:
    input_lengths_list = input_lengths.tolist()
    input_slices = lengths_to_slices(input_lengths_list)

    batch_data: Dict[str, Any] = {**combo_data}

    if "input_ids" in batch_data:
        batch_data["input_ids"] = combo_to_batch_input_ids(
            batch_data.pop("input_ids"), input_slices
        )

    if "hidden_states" in batch_data:
        batch_data["hidden_states"] = combo_to_batch_hidden_states(
            batch_data.pop("hidden_states"), input_slices
        )

    if "moe_gating" in batch_data:
        batch_data["moe_gating"] = combo_to_batch_moe_gating(
            batch_data.pop("moe_gating"), input_slices
        )

    if "attention_mask" in batch_data:
        # Get device from input_lengths or hidden_states to ensure device consistency
        device = input_lengths.device
        if "hidden_states" in batch_data:
            device = batch_data["hidden_states"].device
        batch_data["attention_mask"] = generate_attention_mask(
            input_lengths_list, device=device
        )

    return batch_data


def combo_to_batch(
    hidden_states: torch.Tensor, input_ids: torch.Tensor, input_lengths: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_lengths_list = input_lengths.tolist()
    input_slices = lengths_to_slices(input_lengths_list)
    batched_input_ids = combo_to_batch_input_ids(input_ids, input_slices)
    batched_hidden_states = combo_to_batch_hidden_states(hidden_states, input_slices)
    # Use device from hidden_states to ensure device consistency
    batched_attention_mask = generate_attention_mask(
        input_lengths_list, device=hidden_states.device
    )
    return batched_input_ids, batched_hidden_states, batched_attention_mask


def hidden_combo_to_batch(
    hidde_states: torch.Tensor, input_lengths: torch.Tensor
) -> torch.Tensor:
    sliced_hidden_states: List[torch.Tensor] = []
    hidden_bias = 0
    for input_length in input_lengths:
        sliced_hidden_states.append(
            hidde_states[hidden_bias : hidden_bias + input_length]
        )
        hidden_bias += input_length
    return pad_sequence(sliced_hidden_states, batch_first=True)


def combo_to_list(
    tensor: torch.Tensor, input_length: torch.Tensor
) -> List[torch.Tensor]:
    result: List[torch.Tensor] = []
    bias = 0
    for length in input_length:
        result.append(tensor[bias : bias + length])
        bias += length
    return result


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
