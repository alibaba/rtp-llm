import os
import torch
import numpy as np
from numpy.typing import NDArray
from collections import defaultdict
from typing import List, Dict, Any, Union
from transformers import PreTrainedTokenizerBase

from maga_transformer.utils.util import to_torch_dtype
from maga_transformer.models.downstream_modules.custom_module import CustomModule, CustomHandler
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.models.downstream_modules.embedding.api_datatype import EmbeddingResponseFormat, EmbeddingResponseType, SparseEmbeddingRequest, SimilarityRequest
from maga_transformer.models.downstream_modules.embedding.misc import EmbeddingRendererBase, hidden_combo_to_batch

class SparseEmbeddingModule(CustomModule):
    def __init__(self, config: GptInitModelParameters, tokenizer: PreTrainedTokenizerBase):
        super().__init__(config, tokenizer)
        self.renderer = SparseEmbeddingRenderer(config, tokenizer)
        self.handler = SparseEmbeddingHandler(config)


class SparseEmbeddingRenderer(EmbeddingRendererBase):
    def __init__(self, config: GptInitModelParameters, tokenizer: PreTrainedTokenizerBase):
        super().__init__(config, tokenizer)
        self.embedding_type = EmbeddingResponseType.SPARSE
        self.unused_tokens = set([self.tokenizer_.cls_token_id,
                                  self.tokenizer_.eos_token_id,
                                  self.tokenizer_.pad_token_id,
                                  self.tokenizer_.unk_token_id])
        
    def render_request(self, request_json: Dict[str, Any]):
        if 'left' in request_json:
            return SimilarityRequest(**request_json)
        else:
            return SparseEmbeddingRequest(**request_json)
        
    def embedding_func(self, request: Union[SparseEmbeddingRequest, SimilarityRequest], res: torch.Tensor, input_length: int, input_tokens: torch.Tensor) -> Union[Dict[str, float]]:
        if len(res.shape) != 1:
            raise Exception("sparse hidden should be 1-dim")
        sparse_emb: Dict[int, float] = defaultdict(float)        
        for score, id in zip(res[:input_length], input_tokens):
            score = float(score)
            id = int(id)
            if id in self.unused_tokens:
                continue
            if score > 0 and sparse_emb[id] < score:
                sparse_emb[id] = score
        if isinstance(request, SparseEmbeddingRequest) and request.return_decoded:
            return {self.tokenizer_.decode(key): value for key,value in sparse_emb.items()}
        else:
            return {str(k): v for k, v in sparse_emb.items()}

    def similar_func(self, left: EmbeddingResponseFormat, right: EmbeddingResponseFormat) -> float:
        if not isinstance(left.embedding, dict) or not isinstance(right.embedding, dict):
            raise Exception("sparse similaritey datatype error")
        result: float = 0
        for key in left.embedding.keys():
            if key not in right.embedding:
                continue
            result += left.embedding[key] * right.embedding[key]
        return result


class SparseEmbeddingHandler(CustomHandler):
    def __init__(self, config: GptInitModelParameters):
        super().__init__(config)
        self.sparse_linear = torch.nn.Linear(in_features=self.config_.hidden_size, out_features=1)
        self.dtype_ = to_torch_dtype(self.config_.data_type)

    def tensor_info(self) -> List[str]:
        return []

    def init(self, tensor_map: Dict[str, torch.Tensor]) -> None:
        sparse_linear_path = os.path.join(self.config_.ckpt_path, 'sparse_linear.pt')
        if not os.path.exists(sparse_linear_path):
            raise Exception("sparse module should have sparse_linear.pt under ckpt_path")
        sparse_linear_dict = torch.load(sparse_linear_path, map_location='cpu')
        self.sparse_linear.load_state_dict(sparse_linear_dict)
        self.sparse_linear = self.sparse_linear.to(self.dtype_).cuda()

    def forward(self, input_ids: torch.Tensor, hidden_states: torch.Tensor, input_lengths: torch.Tensor):
        hidden_states = torch.relu(self.sparse_linear(hidden_states)).squeeze_(-1)
        return hidden_combo_to_batch(hidden_states, input_lengths)