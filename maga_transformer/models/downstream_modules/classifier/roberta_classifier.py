import torch
from typing import Dict, List, Any
from transformers import PreTrainedTokenizerBase

from maga_transformer.utils.util import to_torch_dtype
from maga_transformer.models.downstream_modules.custom_module import CustomModule, CustomHandler
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.utils.tensor_utils import get_first_token_from_combo_tokens

from .util import load_num_labels
from .classifier import ClassifierRenderer

class RobertaClassifierModule(CustomModule):
    def __init__(self, config: GptInitModelParameters, tokenizer: PreTrainedTokenizerBase):
        super().__init__(config, tokenizer)
        self.renderer = ClassifierRenderer(self.config_, self.tokenizer_)
        self.handler = RobertaClassifierHandler(self.config_)


class RobertaClassifierHandler(CustomHandler):
    def __init__(self, config: GptInitModelParameters):
        super().__init__(config)
        num_labels = load_num_labels(self.config_.ckpt_path)
        self.out_proj = torch.nn.Linear(self.config_.hidden_size, num_labels)
        self.dense = torch.nn.Linear(self.config_.hidden_size, self.config_.hidden_size)

    def tensor_info(self):
        return ['classifier.out_proj.weight', 'classifier.out_proj.bias', 'classifier.dense.weight', 'classifier.dense.bias']

    def init(self, tensor_map: Dict[str, torch.Tensor]):
        data_type = to_torch_dtype(self.config_.data_type)
        self.out_proj.weight.data = tensor_map['classifier.out_proj.weight']
        self.out_proj.bias.data = tensor_map['classifier.out_proj.bias']
        self.dense.weight.data = tensor_map['classifier.dense.weight']
        self.dense.bias.data = tensor_map['classifier.dense.bias']
        self.out_proj = self.out_proj.to(data_type).eval().to(self.device)
        self.dense = self.dense.to(data_type).eval().to(self.device)

    def forward(self, input_ids: torch.Tensor, hidden_states: torch.Tensor, input_lengths: torch.Tensor) -> List[torch.Tensor]:
        first_tokens = get_first_token_from_combo_tokens(hidden_states, input_lengths)
        return self.out_proj(torch.tanh(self.dense(first_tokens)))