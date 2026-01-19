from typing import Optional

import torch
import torch.nn.functional as F

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.embedding.render.minicpmv_renderer import MiniCPMVRenderer
from rtp_llm.models.downstream_modules.custom_module import CustomHandler, CustomModule
from rtp_llm.tokenizer_factory.tokenizers import BaseTokenizer


class MiniCPMVHandler(CustomHandler):

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        input_lens = input_lengths.tolist()
        token_ids = 0
        reps = []
        print(f"input_lengths: {input_lengths}")
        print(f"token_ids: {token_ids}")

        for length in input_lens:
            hidden_state = hidden_states[token_ids : token_ids + length]
            attention_mask = torch.range(1, length).float().cuda()
            s = torch.sum(hidden_state * attention_mask.unsqueeze(-1), dim=0)
            d = attention_mask.sum(dim=0, keepdim=True)
            reps.append(s / d)
            token_ids += length
        reps_normalized = F.normalize(torch.stack(reps), dim=1)
        return reps_normalized


class MiniCPMVModule(CustomModule):

    def __init__(
        self,
        config: ModelConfig,
        tokenizer: BaseTokenizer,
        vit_config: Optional[VitConfig] = None,
    ):
        super().__init__(config, tokenizer)
        self.renderer = MiniCPMVRenderer(config, tokenizer, vit_config=vit_config)
        self.handler = MiniCPMVHandler(config)
