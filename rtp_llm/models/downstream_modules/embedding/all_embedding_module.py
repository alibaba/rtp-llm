import copy
from typing import Any, Dict, List, Union

import torch

from rtp_llm.async_decoder_engine.embedding.interface import EngineInputs, EngineOutputs
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.models.downstream_modules.custom_module import CustomHandler, CustomModule
from rtp_llm.models.downstream_modules.embedding.api_datatype import (
    AllEmbeddingRequest,
    ALLEmbeddingResponse,
    ALLEmbeddingResponseFormat,
    EmbeddingResponseType,
    OpenAIEmbeddingRequest,
    SimilarityRequest,
    Usage,
)
from rtp_llm.models.downstream_modules.embedding.misc import (
    EmbeddingRendererBase,
    hidden_combo_to_batch,
)


class ALLEmbeddingModule(CustomModule):
    def __init__(self, config: GptInitModelParameters, tokenizer: BaseTokenizer):
        super().__init__(config, tokenizer)
        self.renderer = ALLEmbeddingRenderer(config, tokenizer)
        self.handler = NormalHandler(config)


class ALLEmbeddingRenderer(EmbeddingRendererBase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.embedding_type = EmbeddingResponseType.DENSE

    def render_cpp(self, input: List[str]):
        out = self.create_input(AllEmbeddingRequest(input=input))
        return [out.token_ids, out.token_type_ids, out.input_lengths]

    def render_request(
        self, request_json: Dict[str, Any]
    ) -> Union[SimilarityRequest, OpenAIEmbeddingRequest]:
        if "left" in request_json:
            return SimilarityRequest(**request_json)
        else:
            return AllEmbeddingRequest(**request_json)

    async def render_response(
        self, request: AllEmbeddingRequest, inputs: EngineInputs, outputs: EngineOutputs
    ) -> Dict[str, Any]:
        usage = Usage(
            prompt_tokens=outputs.input_length, total_tokens=outputs.input_length
        )
        data: List[ALLEmbeddingResponseFormat] = []
        bias = 0
        if not isinstance(outputs.outputs, torch.Tensor):
            raise Exception("result should be tensor")
        for i in range(len(outputs.outputs)):
            embedding = outputs.outputs[i][: inputs.input_lengths[i]]
            token_ids = inputs.token_ids[bias : bias + inputs.input_lengths[i]].tolist()
            if self.config_.py_env_configs.embedding_config.embedding_need_norm:
                embedding = torch.nn.functional.normalize(embedding, dim=-1)
            data.append(
                ALLEmbeddingResponseFormat(
                    object=self.embedding_type,
                    embedding=embedding.tolist(),
                    token_ids=token_ids,
                    index=i,
                )
            )
            bias += inputs.input_lengths[i]
        return ALLEmbeddingResponse(data=data, usage=usage).model_dump()

    async def render_log_response(self, response: Dict[str, Any]):
        log_response = copy.copy(response)
        if "data" in log_response:
            del log_response["data"]
        return log_response


class NormalHandler(CustomHandler):
    def __init__(self, config: GptInitModelParameters):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_combo_to_batch(hidden_states, input_lengths)
