from typing import Dict, List

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.models.downstream_modules.custom_module import CustomHandler, CustomModule
from rtp_llm.utils.model_weight import CkptWeightInfo
from rtp_llm.utils.tensor_utils import first_token_indices
from rtp_llm.utils.util import to_torch_dtype

from .classifier import ClassifierRenderer
from .util import load_num_labels


class BertClassifierModule(CustomModule):

    def __init__(self, config: ModelConfig, tokenizer: BaseTokenizer):
        super().__init__(config, tokenizer)
        self.renderer = ClassifierRenderer(self.config_, self.tokenizer_)
        self.handler = BertClassifierHandler(self.config_)


class BertClassifierHandler(CustomHandler):

    pooled_row_indices = staticmethod(first_token_indices)

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        num_labels = load_num_labels(self.config_.ckpt_path)
        self.dense = torch.nn.Linear(self.config_.hidden_size, self.config_.hidden_size)
        self.activation = torch.nn.Tanh()
        self.linear = torch.nn.Linear(self.config_.hidden_size, num_labels)

    def custom_weight_info(self) -> List[CustomAtomicWeight]:
        w_list = [
            "bert.pooler.dense.bias",
            "bert.pooler.dense.weight",
            "classifier.weight",
            "classifier.bias",
        ]
        weights = []
        for k in w_list:
            weights.append(
                CustomAtomicWeight(CustomAtomicWeight.prefix + k, [CkptWeightInfo(k)])
            )
        return weights

    def pooler(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output

    def init(self, tensor_map: Dict[str, torch.Tensor]):
        data_type = to_torch_dtype(self.config_.data_type)
        self.dense.weight.data = tensor_map["bert.pooler.dense.weight"]
        self.dense.bias.data = tensor_map["bert.pooler.dense.bias"]
        # self.dense = self.dense.to(data_type).eval().cuda()
        self.dense = self.dense.to(data_type).eval().to(self.device)
        self.linear.weight.data = tensor_map["classifier.weight"]
        self.linear.bias.data = tensor_map["classifier.bias"]
        # self.linear = self.linear.to(data_type).eval().cuda()
        self.linear = self.linear.to(data_type).eval().to(self.device)
        if self.device.type == "cpu":
            self.dense = self.dense.to(torch.float32)
            self.linear = self.linear.to(torch.float32)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return self.score_pooled_rows(
            hidden_states[self.pooled_row_indices(input_lengths)]
        )

    @torch.inference_mode()
    def score_pooled_rows(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scores = self.linear(self.pooler(hidden_states))
        return scores.to(torch.float16) if self.device.type == "cpu" else scores
