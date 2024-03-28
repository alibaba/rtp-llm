import os
import numpy as np
from numpy.typing import NDArray
import torch
from collections import defaultdict
from typing import List, Dict, Union, Optional
from transformers import PreTrainedTokenizerBase

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters

class SparseEmbeddingModule(object):
    def __init__(self, hidden_size: int, state_dict: Dict[str, torch.Tensor], tokenizer: PreTrainedTokenizerBase, dtype: Union[str, torch.dtype]):
        self.sparse_linear = torch.nn.Linear(in_features=hidden_size, out_features=1)
        self.sparse_linear.load_state_dict(state_dict)
        self.sparse_linear = self.sparse_linear.to(dtype).cuda()
        self.tokenizer = tokenizer

    def _process_token_weights(self, token_weights: NDArray[np.float32], input_ids: List[int]) -> Dict[str, float]:
        # conver to dict
        result: Dict[str, float] = defaultdict(float)
        unused_tokens = set([self.tokenizer.cls_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id,
                                self.tokenizer.unk_token_id])
        # token_weights = np.ceil(token_weights * 100)
        for w, idx in zip(token_weights, input_ids):
            if idx not in unused_tokens and w > 0:
                idx_str = self.tokenizer.decode(idx)
                # w = int(w)
                if w > result[idx_str]:
                    result[idx_str] = w
        return result

    def __call__(self, input_ids: torch.Tensor, hidden_states: torch.Tensor) -> List[Dict[str, float]]:
        token_weights = torch.relu(self.sparse_linear(hidden_states)).squeeze_(-1)
        all_lexical_weights = (list(map(self._process_token_weights, token_weights.cpu().numpy(), input_ids.cpu().numpy().tolist())))        
        return all_lexical_weights

def init_sparse_embedding_module(config: GptInitModelParameters, tokenizer: PreTrainedTokenizerBase, dtype: Union[str, torch.dtype]) -> Optional[SparseEmbeddingModule]:
    sparse_linear_path = os.path.join(config.ckpt_path, 'sparse_linear.pt')
    if os.path.exists(sparse_linear_path):
        sparse_linear_dict = torch.load(sparse_linear_path, map_location='cpu')
        return SparseEmbeddingModule(config.hidden_size, sparse_linear_dict, tokenizer, dtype)
    return None