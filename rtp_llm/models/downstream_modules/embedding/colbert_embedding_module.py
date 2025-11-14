import os
from typing import Any, Dict, List, Union

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.models.downstream_modules.custom_module import CustomHandler, CustomModule
from rtp_llm.models.downstream_modules.embedding.api_datatype import (
    ColbertEmbeddingRequest,
    EmbeddingResponseFormat,
    EmbeddingResponseType,
    SimilarityRequest,
)
from rtp_llm.models.downstream_modules.embedding.misc import (
    EmbeddingRendererBase,
    combo_to_batch,
)
from rtp_llm.utils.util import to_torch_dtype


class ColBertEmbeddingModule(CustomModule):
    def __init__(self, config: ModelConfig, tokenizer: BaseTokenizer):
        super().__init__(config, tokenizer)
        self.renderer = ColbertEmbeddingRenderer(config, tokenizer)
        self.handler = ColBertEmbeddingHandler(config)


class ColbertEmbeddingRenderer(EmbeddingRendererBase):
    def __init__(self, config: ModelConfig, tokenizer: BaseTokenizer):
        super().__init__(config, tokenizer)
        self.embedding_type = EmbeddingResponseType.COLBERT

    def render_request(
        self, request_json: Dict[str, Any]
    ) -> Union[SimilarityRequest, ColbertEmbeddingRequest]:
        if "left" in request_json:
            return SimilarityRequest(**request_json)
        else:
            return ColbertEmbeddingRequest(**request_json)

    def embedding_func(
        self,
        request: Any,
        res: torch.Tensor,
        input_length: int,
        input_tokens: torch.Tensor,
    ) -> List[float]:
        assert isinstance(res, torch.Tensor)
        return res[: input_length - 1].tolist()

    def similar_func(
        self, left: EmbeddingResponseFormat, right: EmbeddingResponseFormat
    ):
        left_t = torch.tensor(left.embedding)
        right_t = torch.tensor(right.embedding)
        token_scores = torch.einsum("in,jn->ij", left_t, right_t)
        scores, _ = token_scores.max(-1)
        scores = torch.sum(scores) / left_t.size(0)
        return float(scores)


class ColBertEmbeddingHandler(CustomHandler):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.colbert_linear_path_ = os.path.join(
            self.config_.ckpt_path, "colbert_linear.pt"
        )
        if not os.path.exists(self.colbert_linear_path_):
            raise Exception("failed to find colbert_linear.pt from ckpt_path")
        self.dtype_ = to_torch_dtype(self.config_.data_type)

    def init(self, tensor_map: Dict[str, torch.Tensor]) -> None:
        sparse_linear_dict = torch.load(self.colbert_linear_path_, map_location="cpu")
        self.colbert_linear = torch.nn.Linear(
            in_features=self.config_.hidden_size, out_features=self.config_.hidden_size
        )
        self.colbert_linear.load_state_dict(sparse_linear_dict)
        self.colbert_linear = self.colbert_linear.to(self.dtype_).cuda()

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        batch_input_ids, batch_hidden_states, batch_attention_mask = combo_to_batch(
            hidden_states, input_ids, input_lengths
        )
        return self.forward_internal(
            batch_input_ids, batch_hidden_states, batch_attention_mask
        )

    def forward_internal(
        self,
        batch_input_ids: torch.Tensor,
        batch_hidden_states: torch.Tensor,
        batch_attention_mask: torch.Tensor,
    ):
        colbert_vecs = self.colbert_linear(batch_hidden_states[:, 1:])
        colbert_vecs = colbert_vecs * batch_attention_mask[:, 1:][:, :, None].float()
        colbert_vecs = torch.nn.functional.normalize(colbert_vecs, dim=-1)
        return colbert_vecs
