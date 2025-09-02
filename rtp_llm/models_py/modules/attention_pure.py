from typing import Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.fmha import FMHAImplBase
from rtp_llm.ops import KVCache


class CausalAttentionPure(nn.Module):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.head_num
        self.head_num = config.head_num
        self.num_key_value_groups = config.head_num // config.head_num_kv
        self.q_size = config.head_num * self.head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        attn_output = torch.empty(
            [*input_shape, 4096], device=hidden_states.device, dtype=hidden_states.dtype
        )
        attn_output = fmha_impl.forward(hidden_states, kv_cache)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return attn_output
