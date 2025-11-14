from typing import Dict, List

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.models.downstream_modules.custom_module import CustomHandler, CustomModule
from rtp_llm.utils.model_weight import CkptWeightInfo
from rtp_llm.utils.tensor_utils import get_first_token_from_combo_tokens
from rtp_llm.utils.util import to_torch_dtype

from .classifier import ClassifierRenderer
from .util import load_num_labels


class RobertaClassifierModule(CustomModule):
    def __init__(self, config: ModelConfig, tokenizer: BaseTokenizer):
        super().__init__(config, tokenizer)
        self.renderer = ClassifierRenderer(self.config_, self.tokenizer_)
        self.handler = RobertaClassifierHandler(self.config_)


class RobertaClassifierHandler(CustomHandler):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        num_labels = load_num_labels(self.config_.ckpt_path)
        self.out_proj = torch.nn.Linear(self.config_.hidden_size, num_labels)
        self.dense = torch.nn.Linear(self.config_.hidden_size, self.config_.hidden_size)

    def custom_weight_info(self) -> List[CustomAtomicWeight]:
        w_list = [
            "classifier.out_proj.weight",
            "classifier.out_proj.bias",
            "classifier.dense.weight",
            "classifier.dense.bias",
        ]
        weights = []
        for k in w_list:
            weights.append(
                CustomAtomicWeight(CustomAtomicWeight.prefix + k, [CkptWeightInfo(k)])
            )
        return weights

    def init(self, tensor_map: Dict[str, torch.Tensor]):
        data_type = to_torch_dtype(self.config_.data_type)
        self.out_proj.weight.data = tensor_map["classifier.out_proj.weight"]
        self.out_proj.bias.data = tensor_map["classifier.out_proj.bias"]
        self.dense.weight.data = tensor_map["classifier.dense.weight"]
        self.dense.bias.data = tensor_map["classifier.dense.bias"]
        self.out_proj = self.out_proj.to(data_type).eval().to(self.device)
        self.dense = self.dense.to(data_type).eval().to(self.device)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> List[torch.Tensor]:
        first_tokens = get_first_token_from_combo_tokens(hidden_states, input_lengths)
        return self.out_proj(torch.tanh(self.dense(first_tokens)))
