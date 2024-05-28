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

from .misc import combo_to_list, EmbeddingRendererBase
from .api_datatype import EmbeddingResponseType

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
    def embedding_func(self, d: Any) -> List[float] | Dict[str, float]:
        assert isinstance(d, dict)
        return {self.tokenizer_.decode(idx): value for idx, value in d.items() if idx not in self.unused_tokens}

    def similar_func(self, left: Dict[int, float], right: Dict[int, float]) -> float:
        assert isinstance(left, dict) and isinstance(right, dict), "sparse similaritey datatype error"
        result: float = 0
        for key in left.keys():
            if key not in right:
                continue
            if key in self.unused_tokens:
                continue
            result += left[key] * right[key]
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
        hidden_states_list = combo_to_list(hidden_states, input_lengths)
        input_ids_list = combo_to_list(input_ids, input_lengths)
        result: List[Dict[int, float]] = []
        for hidden_states, input_ids in zip(hidden_states_list, input_ids_list):
            result.append(self._process_token_weights(hidden_states.cpu().numpy(), input_ids))
        return result

    def _process_token_weights(self, token_weights: NDArray[np.float32], input_ids: torch.Tensor) -> Dict[int, float]:
        sparse_emb: Dict[int, float] = defaultdict(float)
        for w, idx in zip(token_weights, input_ids):
            idx = int(idx)
            if w > 0 and w > sparse_emb[idx]:
                sparse_emb[idx] = w
        return sparse_emb