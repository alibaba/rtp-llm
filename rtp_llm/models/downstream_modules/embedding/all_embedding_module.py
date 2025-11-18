import base64
import copy
import logging
from typing import Any, Dict, List, Union

import numpy as np
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

        def numpy_to_bf16(arr: np.ndarray) -> np.ndarray:
            arr_float32 = arr.astype(np.float32)
            uint8_view = arr_float32.view(np.uint8).reshape(-1, 4)
            bf16_bytes = uint8_view[:, 2:].reshape(arr.shape + (2,))
            return np.frombuffer(bf16_bytes.tobytes(), dtype=np.uint16).reshape(
                arr.shape
            )

        for i in range(len(outputs.outputs)):
            embedding = outputs.outputs[i][:]
            token_ids = inputs.token_ids[:].tolist()
            if request.normalize:
                embedding = torch.nn.functional.normalize(embedding, dim=-1)

            if embedding is not None:
                embedding_array = embedding.numpy()
                if self.config_.data_type == "bf16":
                    embedding_array = numpy_to_bf16(embedding_array)
                embedding_base64 = base64.b64encode(embedding_array.tobytes()).decode(
                    "ascii"
                )

            data.append(
                ALLEmbeddingResponseFormat(
                    object=self.embedding_type,
                    embedding=embedding_base64,
                    token_ids=token_ids,
                    index=i,
                )
            )
            bias += inputs.input_lengths[i]
        result = ALLEmbeddingResponse(data=data, usage=usage).model_dump()
        return result

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
