import os
import json
import torch
import copy
import numpy as np
from collections import OrderedDict
from typing import List, Dict, Any, Union

import torch.nn as nn
from transformers import PreTrainedTokenizerBase
from sentence_transformers.util import import_from_string
from sentence_transformers.models import Transformer, Normalize

from maga_transformer.utils.util import to_torch_dtype
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.async_decoder_engine.embedding.embedding_stream import EngineInputs, EngineOutputs
from maga_transformer.models.downstream_modules.custom_module import CustomModule, CustomHandler
from maga_transformer.models.downstream_modules.embedding.misc import EmbeddingRendererBase
from maga_transformer.models.downstream_modules.embedding.api_datatype import OpenAIEmbeddingRequest, \
    Usage, ALLEmbeddingResponseFormat, ALLEmbeddingResponse, EmbeddingResponseType, OpenAIEmbeddingResponse
    
class ALLEmbeddingModule(CustomModule):
    def __init__(self, config: GptInitModelParameters, tokenizer: PreTrainedTokenizerBase):
        super().__init__(config, tokenizer)
        self.renderer = ALLEmbeddingRenderer(config, tokenizer)
        self.handler = NormalHandler(config)


class ALLEmbeddingRenderer(EmbeddingRendererBase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, ** kwargs)
        self.embedding_type = EmbeddingResponseType.DENSE
    
    def embedding_func(self, x: Any) -> List[float]:
        assert isinstance(x, torch.Tensor)
        return x.tolist()

    async def _render_embedding_output(self, request: OpenAIEmbeddingRequest, inputs: EngineInputs, outputs: EngineOutputs) -> Dict[str, Any]:
        usage = Usage(prompt_tokens=outputs.input_length, total_tokens=outputs.input_length)
        data: List[ALLEmbeddingResponseFormat] = []
        bias = 0
        for i in range(len(outputs.outputs)):
            data.append(ALLEmbeddingResponseFormat(
                object=self.embedding_type,
                embedding=self.embedding_func(outputs.outputs[i]),
                token_ids=inputs.token_ids[bias: bias + inputs.input_lengths[i]].tolist(),
                index=i))
            bias += inputs.input_lengths[i]
        return ALLEmbeddingResponse(data=data, usage=usage).model_dump()

    async def render_log_response(self, response: Dict[str, Any]):
        log_response = copy.copy(response)
        if 'data' in log_response:
            del log_response['data']
        return log_response


class NormalHandler(CustomHandler):
    def __init__(self, config: GptInitModelParameters):
        super().__init__(config)
        self.is_causal = config.is_causal

    def forward(self, input_ids: torch.Tensor, hidden_states: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        input_lengths = torch.cat((torch.IntTensor([0]), input_lengths), 0)
        cum_lengths = torch.cumsum(input_lengths, dim=0)
        result = []
        for i in range(len(cum_lengths) - 1):
            tensor = hidden_states[cum_lengths[i]:cum_lengths[i+1]]
            tensor = torch.nn.functional.normalize(tensor, dim=1)
            result.append(tensor)
        return result