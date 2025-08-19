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
from rtp_llm.models_py.modules.fp8_linear import Fp8Linear
from rtp_llm.ops import PyModelInputs, PyModelOutputs, PyAttentionInputs


class CausalAttention(nn.Module):

    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor], layer_idx: int = 0
    ):
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
        
        use_fp8 = self._should_use_fp8_linear(config, weights)
        
        if use_fp8:
            self.qkv_proj = self._create_fp8_linear(weights[W.attn_qkv_w], weights[W.attn_qkv_s], weights.get(W.attn_qkv_b, None), config)
            self.o_proj = self._create_fp8_linear(weights[W.attn_o_w], weights[W.attn_o_s], weights.get(W.attn_o_b, None), config)
        else:
            self.qkv_proj = Linear(weights[W.attn_qkv_w], weights.get(W.attn_qkv_b, None))
            self.o_proj = Linear(weights[W.attn_o_w], weights.get(W.attn_o_b, None))
        if W.q_ln_gamma in weights:
            self.q_norm = RMSNorm(weights[W.q_ln_gamma], eps=config.layernorm_eps)  # unlike olmo, only on the head dim!
            self.k_norm = RMSNorm(weights[W.k_ln_gamma], eps=config.layernorm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = None
        self.flash_infer = FlashInferOp(config)

    def _should_use_fp8_linear(self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]) -> bool:

        if not hasattr(config, 'quant_config') or config.quant_config is None:
            return False
        quant_method = config.quant_config.get_method()

        fp8_methods = ['FP8', 'FP8_PER_BLOCK', 'FP8_PER_CHANNEL_COMPRESSED']
        if quant_method not in fp8_methods:
            return False
        qkv_weight = weights.get(W.attn_qkv_w)
        if qkv_weight is None:
            return False 
        
        result = qkv_weight.dtype == torch.float8_e4m3fn
               
        return qkv_weight.dtype == torch.float8_e4m3fn
    
    def _create_fp8_linear(self, weight: torch.Tensor, weight_scales: torch.Tensor, 
                          bias: Optional[torch.Tensor], config: GptInitModelParameters) -> Fp8Linear:
        return Fp8Linear(weight, weight_scales, bias, config)
    
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
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        qkv = self.qkv_proj(hidden_states)
        
        attn_output = fmha_impl.forward(qkv, kv_cache)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        output = self.o_proj(attn_output)
        return output
