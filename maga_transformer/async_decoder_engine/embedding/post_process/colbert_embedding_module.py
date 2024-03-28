import os
from numpy.typing import NDArray
import numpy as np
import torch
from typing import List, Dict, Union, Optional

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters

class ColBertEmbeddingModule(object):
    def __init__(self, hidden_size: int, state_dict: Dict[str, torch.Tensor], dtype: Union[str, torch.dtype]):
        self.colbert_linear = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.colbert_linear.load_state_dict(state_dict)
        self.colbert_linear = self.colbert_linear.to(dtype).cuda()

    def _process_colbert_vecs(self, colbert_vecs: torch.Tensor, tokens_num: int):
        # delte the vectors of padding tokens
        return colbert_vecs[:tokens_num - 1]  # we don't use the embedding of cls, so select tokens_num-1

    def __call__(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, input_length: List[int], do_normalize: bool=True) -> List[torch.Tensor]:
        colbert_vecs = self.colbert_linear(hidden_states[:, 1:])
        colbert_vecs = colbert_vecs * attention_mask[:, 1:][:, :, None].float()
        if do_normalize:
            colbert_vecs = torch.nn.functional.normalize(colbert_vecs, dim=-1)
        all_colbert_vec = (list(map(self._process_colbert_vecs, colbert_vecs.cpu(), input_length)))
        return all_colbert_vec

def init_colbert_embedding_module(config: GptInitModelParameters, dtype: Union[str, torch.dtype]) -> Optional[ColBertEmbeddingModule]:
    colbert_linear_path = os.path.join(config.ckpt_path, 'colbert_linear.pt')
    if os.path.exists(colbert_linear_path):
        sparse_linear_dict = torch.load(colbert_linear_path, map_location='cpu')
        return ColBertEmbeddingModule(config.hidden_size, sparse_linear_dict, dtype)
    return None