import torch
import torch.nn as nn
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.norm import RMSNorm
from torch import dtype as _dtype
from typing import Optional, Dict, TypedDict, Tuple, Any
from typing_extensions import Unpack
from rtp_llm.utils.model_weight import W
from rtp_llm.models_py.modules.linear import Linear
from rtp_llm.ops import FlashInferOp

class AttentionKwargs(TypedDict, total=False):
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    attn_params: Any

class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor], layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.head_num
        self.head_num = config.head_num
        self.num_key_value_groups = config.head_num // config.head_num_kv
        self.q_size = config.head_num * self.head_dim
        self.kv_size = config.head_num_kv * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.is_causal = True

        self.qkv_proj = Linear(weights[W.attn_qkv_w])
        self.o_proj = Linear(weights[W.attn_o_w])
        if W.q_ln_gamma in weights:
            self.q_norm = RMSNorm(weights[W.q_ln_gamma], eps=config.layernorm_eps)  # unlike olmo, only on the head dim!
            self.k_norm = RMSNorm(weights[W.k_ln_gamma], eps=config.layernorm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = None
        self.flash_infer = FlashInferOp(config)

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_by_head = q.reshape(-1, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        return q, k

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Unpack[AttentionKwargs],
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        qkv = self.qkv_proj(hidden_states)
        # TODO: need qk norm if have
        
        attn_output = torch.empty_like(hidden_states)
        # kwargs.get
        self.flash_infer.forward(qkv, attn_output, kwargs['k_cache'][self.layer_idx], kwargs['v_cache'][self.layer_idx], kwargs['attn_params'])

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output
