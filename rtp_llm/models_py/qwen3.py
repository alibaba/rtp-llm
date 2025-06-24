from typing import Optional, Tuple, List

import torch
from torch import nn

from typing_extensions import Unpack
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules import Embedding, Linear, AttentionKwargs, RMSNorm
from rtp_llm.models_py.layers.decoder_layer import Qwen3DecoderLayer
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.utils.model_weight import W

from rtp_llm.models_py.module_base import set_trace_on_tty

class Qwen3Model(nn.Module):
    def __init__(self, config: GptInitModelParameters, weights: ModelWeights):
        super().__init__()
        self.layer_num = config.layer_num
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embedding(weights.get_global_weight(W.embedding))
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, weights.weights[idx], idx) for idx in range(self.layer_num)]
        )
        self.norm = RMSNorm(weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps)
        self.lm_head = Linear(weights.get_global_weight(W.lm_head))

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[AttentionKwargs],
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers[: self.layer_num]:
            hidden_states = decoder_layer(
                hidden_states,
                **flash_attn_kwargs,
            )

        return hidden_states

__all__ = [
    "Qwen3Model",
]
