from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_loader.model_weight_info import ModelWeights

from rtp_llm.models_py.module_base import GptModelBase, set_trace_on_tty


import torch

from typing import Optional, Tuple

class GptModelExample(GptModelBase):
    def __init__(self, params: GptInitModelParameters, weight: ModelWeights) -> None:
        super().__init__(params, weight)
        print("GptModelExample initialized")

    def forward(self,
                combo_tokens: torch.Tensor,
                input_lengths: torch.Tensor,
                sequence_lengths: torch.Tensor,
                attention_mask: torch.Tensor,
                kv_cache_block_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        print("model base forward called")

        # set_trace_on_tty()

        num_tokens = combo_tokens.shape[0]
        hidden = self.params.hidden_size
        fake_hidden = torch.rand([num_tokens, hidden], dtype=torch.float32, device="cuda")

        dict_size = self.params.vocab_size
        fake_logits = torch.rand([num_tokens, dict_size], dtype=torch.float32, device="cuda")

        print("combo tokens:", combo_tokens)
        print("input lengths:", input_lengths)
        print("sequence lengths:", sequence_lengths)
        print("attention mask:", attention_mask)
        print("kv cache block id:", kv_cache_block_id)

        return (
            fake_logits,
            fake_hidden,
        )
