import torch
from typing import Dict, List, Any
from transformers import PreTrainedTokenizerBase

from maga_transformer.utils.util import to_torch_dtype
from maga_transformer.models.downstream_modules.custom_module import CustomModule, CustomHandler
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.utils.tensor_utils import get_first_token_from_combo_tokens

from .util import load_num_labels
from .classifier import ClassifierRenderer


class BertClassifierModule(CustomModule):

    def __init__(self, config: GptInitModelParameters,
                 tokenizer: PreTrainedTokenizerBase):
        super().__init__(config, tokenizer)
        self.renderer = ClassifierRenderer(self.config_, self.tokenizer_)
        self.handler = BertClassifierHandler(self.config_)


class BertClassifierHandler(CustomHandler):

    def __init__(self, config: GptInitModelParameters):
        super().__init__(config)
        num_labels = load_num_labels(self.config_.ckpt_path)
        self.dense = torch.nn.Linear(self.config_.hidden_size,
                                     self.config_.hidden_size)
        self.activation = torch.nn.Tanh()
        self.linear = torch.nn.Linear(self.config_.hidden_size, num_labels)

    def tensor_info(self):
        return [
            'bert.pooler.dense.bias', 'bert.pooler.dense.weight',
            'classifier.weight', 'classifier.bias'
        ]

    def pooler(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output

    def init(self, tensor_map: Dict[str, torch.Tensor]):
        data_type = to_torch_dtype(self.config_.data_type)
        self.dense.weight.data = tensor_map['bert.pooler.dense.weight']
        self.dense.bias.data = tensor_map['bert.pooler.dense.bias']
        #self.dense = self.dense.to(data_type).eval().cuda()
        self.dense = self.dense.to(data_type).eval().to(self.device)
        self.linear.weight.data = tensor_map['classifier.weight']
        self.linear.bias.data = tensor_map['classifier.bias']
        #self.linear = self.linear.to(data_type).eval().cuda()
        self.linear = self.linear.to(data_type).eval().to(self.device)
        if self.device.type == 'cpu':
            self.dense = self.dense.to(torch.float32)
            self.linear = self.linear.to(torch.float32)

    def forward(self, input_ids: torch.Tensor, hidden_states: torch.Tensor,
                input_lengths: torch.Tensor) -> List[torch.Tensor]:
        first_tokens = get_first_token_from_combo_tokens(
            hidden_states, input_lengths)
        #first_tokens = self.pooler(first_tokens)
        #return self.linear(first_tokens)
        if self.device.type == 'cpu':
            first_tokens = self.pooler(first_tokens)
            first_tokens = self.linear(first_tokens)
            first_tokens = first_tokens.to(torch.float16)
            return first_tokens
        else:
            first_tokens = self.pooler(first_tokens)
            return self.linear(first_tokens)
