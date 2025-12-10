from typing import Tuple, Union, Optional

import torch
import torch.nn.functional as F
from aiter import layernorm2d_fwd as layernorm2d_fwd
from aiter import rmsnorm2d_fwd as rms_norm
from torch import nn

from rtp_llm.models_py.modules.base.common.norm import (
    BaseAddBiasResLayerNorm,
    BaseLayerNorm,
    BaseNorm,
    BaseQKNorm
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


class LayerNorm(BaseLayerNorm):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, beta, eps)

    def forward(self, hidden_states: torch.Tensor):
        output = torch.empty_like(hidden_states)
        rtp_llm_ops.layernorm(
            output, hidden_states, self.weight.data, self.beta, self.variance_epsilon, 0
        )
        return output


class RMSNorm(BaseNorm):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, eps)

    def forward(self, hidden_states: torch.Tensor, output: Optional[torch.Tensor] = None) -> torch.Tensor:
        return rms_norm(hidden_states, self.weight.data, self.variance_epsilon)


class AddBiasResLayerNorm(BaseAddBiasResLayerNorm):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, beta, eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        bias: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if hidden_states.shape[0] > 32 and hidden_states.shape[1] <= 768:
            return layernorm2d_fwd(
                hidden_states,
                self.weight.data,
                bias,
                self.variance_epsilon,
                x_bias=None,
            )
        else:
            rtp_llm_ops.fused_add_layernorm(
                hidden_states,
                residual,
                bias,
                self.weight.data,
                self.beta,
                self.variance_epsilon,
                0,
            )
            return hidden_states


class AddBiasResLayerNormTorch(BaseAddBiasResLayerNorm):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, beta, eps)

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, bias: torch.Tensor
    ):
        hidden_states = hidden_states + bias + residual
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(dim=-1, keepdim=True)
        squared_sum = (hidden_states**2).mean(dim=-1, keepdim=True)

        x_normalized = (hidden_states - mean) / torch.sqrt(
            (squared_sum - (mean**2)) + self.variance_epsilon
        )
        return (self.weight * x_normalized + self.beta).to(input_dtype)


class QKRMSNorm(BaseQKNorm):
    def __init__(
        self,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        head_num: int,
        kv_head_num: int,
        size_per_head: int,
        eps: float = 1e-6,
    ):
        super().__init__(q_weight, k_weight, head_num, kv_head_num, size_per_head, eps)
        self.q_norm = RMSNorm(q_weight, eps)
        self.k_norm = RMSNorm(k_weight, eps)

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_by_head = q.reshape(-1, self.size_per_head)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.reshape(-1, self.size_per_head)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        return q, k

    def forward(self, hidden_states: torch.Tensor):
        q, k, v = hidden_states.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q, k)
        output = torch.cat([q, k, v], dim=-1)
        return output


class FusedQKRMSNorm(BaseQKNorm):
    def __init__(
        self,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        head_num: int,
        kv_head_num: int,
        size_per_head: int,
        eps: float = 1e-6,
    ):
        super().__init__(q_weight, k_weight, head_num, kv_head_num, size_per_head, eps)

    def forward(self, hidden_states: torch.Tensor):
        m, n = hidden_states.shape
        rtp_llm_ops.fused_qk_rmsnorm(
            hidden_states,
            self.q_weight,
            self.k_weight,
            self.variance_epsilon,
            self.head_num,
            self.kv_head_num,
            m,
            n,
            self.size_per_head,
        )
        return hidden_states
