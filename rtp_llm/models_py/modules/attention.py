import logging
from typing import Any, Dict, Optional, Tuple, TypedDict

import torch
import torch.nn as nn
from torch import dtype as _dtype
from typing_extensions import Unpack

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.fmha import FMHAImplBase
from rtp_llm.models_py.modules.linear import Linear
from rtp_llm.models_py.modules.norm import RMSNorm
from rtp_llm.ops import KVCache
from rtp_llm.utils.model_weight import W


class CausalAttention(nn.Module):

    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.head_num
        self.head_num = config.head_num
        self.num_key_value_groups = config.head_num // config.head_num_kv
        self.q_size = config.head_num * self.head_dim

        self.qkv_proj = Linear(weights[W.attn_qkv_w], weights.get(W.attn_qkv_b, None))
        self.o_proj = Linear(weights[W.attn_o_w], weights.get(W.attn_o_b, None))

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        qkv = self.qkv_proj(hidden_states)
        attn_output = fmha_impl.forward(qkv, kv_cache)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output
