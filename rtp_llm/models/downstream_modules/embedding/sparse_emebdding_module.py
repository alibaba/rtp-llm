import os
from typing import Dict, List

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.embedding.render.sparse_embedding_renderer import SparseEmbeddingRenderer
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.models.downstream_modules.custom_module import CustomHandler, CustomModule
from rtp_llm.models.downstream_modules.embedding.misc import hidden_combo_to_batch
from rtp_llm.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.utils.util import to_torch_dtype


class SparseEmbeddingModule(CustomModule):
    def __init__(self, config: ModelConfig, tokenizer: BaseTokenizer):
        super().__init__(config, tokenizer)
        self.renderer = SparseEmbeddingRenderer(config, tokenizer)
        self.handler = SparseEmbeddingHandler(config)


class SparseEmbeddingHandler(CustomHandler):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.sparse_linear = torch.nn.Linear(
            in_features=self.config_.hidden_size, out_features=1
        )
        self.dtype_ = to_torch_dtype(self.config_.data_type)

    def custom_weight_info(self) -> List[CustomAtomicWeight]:
        return []

    def init(self, tensor_map: Dict[str, torch.Tensor]) -> None:
        sparse_linear_path = os.path.join(self.config_.ckpt_path, "sparse_linear.pt")
        if not os.path.exists(sparse_linear_path):
            raise Exception(
                "sparse module should have sparse_linear.pt under ckpt_path"
            )
        sparse_linear_dict = torch.load(sparse_linear_path, map_location="cpu")
        self.sparse_linear.load_state_dict(sparse_linear_dict)
        self.sparse_linear = self.sparse_linear.to(self.dtype_).cuda()

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        hidden_states = torch.relu(self.sparse_linear(hidden_states)).squeeze_(-1)
        return hidden_combo_to_batch(hidden_states, input_lengths)
