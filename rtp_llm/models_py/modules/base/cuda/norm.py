from typing import Optional, Tuple

import flashinfer
import torch
from torch import nn

from rtp_llm.models_py.modules.base.common.norm import (
    BaseAddBiasResLayerNorm,
    BaseNorm,
    BaseAddNorm,
    BaseQKNorm
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


class RMSNorm(BaseNorm):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6, enable_pdl: bool = False):
        super().__init__(weight, eps)
        self.enable_pdl = enable_pdl

    def forward(self, hidden_states: torch.Tensor, output: Optional[torch.Tensor] = None) -> torch.Tensor:
        return flashinfer.norm.rmsnorm(hidden_states, self.weight.data, eps=self.variance_epsilon,
                                       out=output, enable_pdl=self.enable_pdl)


class FusedAddRMSNorm(BaseAddNorm):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6, enable_pdl: bool = False):
        super().__init__(weight, eps)
        self.enable_pdl = enable_pdl

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        flashinfer.fused_add_rmsnorm(hidden_states, residual, self.weight.data,
                                     eps=self.variance_epsilon, enable_pdl=self.enable_pdl)
        return hidden_states


class QKRMSNorm(BaseQKNorm):
    def __init__(
        self,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        head_num: int,
        kv_head_num: int,
        size_per_head: int,
        eps: float = 1e-6,
        enable_pdl: bool = False
    ):
        super().__init__(q_weight, k_weight, head_num, kv_head_num, size_per_head, eps)
        self.enable_pdl = enable_pdl

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        m, n = hidden_states.shape
        qkv = hidden_states.reshape(m, (self.head_num + self.kv_head_num * 2), self.size_per_head)
        q = qkv[:, :self.head_num, :]
        k = qkv[:, self.head_num:self.head_num + self.kv_head_num, :]
        flashinfer.norm.rmsnorm(q, self.q_weight, eps=self.variance_epsilon, out=q, enable_pdl=self.enable_pdl)
        flashinfer.norm.rmsnorm(k, self.k_weight, eps=self.variance_epsilon, out=k, enable_pdl=self.enable_pdl)
        return qkv.reshape(m, n)


class FusedQKRMSNorm(BaseQKNorm):
    def __init__(
        self,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        head_num: int,
        kv_head_num: int,
        size_per_head: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__(q_weight, k_weight, head_num, kv_head_num, size_per_head, eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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


class AddBiasResLayerNorm(BaseAddBiasResLayerNorm):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, beta, eps)

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, bias: torch.Tensor
    ):
        rtp_llm_ops.fused_add_layernorm(
            hidden_states,
            residual,
            bias,
            self.weight.data,
            self.beta,
            self.variance_epsilon,
        )
        return hidden_states
