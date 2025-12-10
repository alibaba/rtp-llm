from typing import Optional, Tuple

import torch
from torch import nn

from rtp_llm.ops.compute_ops import rtp_llm_ops


class BaseNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, output: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError()


class BaseAddNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class BaseQKNorm(nn.Module):
    def __init__(self, q_weight: torch.Tensor, k_weight: torch.Tensor, head_num: int, kv_head_num: int, size_per_head: int, eps: float = 1e-6):
        super().__init__()
        self.q_weight = q_weight
        self.k_weight = k_weight
        self.head_num = head_num
        self.kv_head_num = kv_head_num
        self.size_per_head = size_per_head
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class BaseAddBiasResLayerNorm(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.beta = beta
        self.variance_epsilon = eps

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()


class BaseLayerNorm(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.beta = beta
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


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


class RMSNormTorch(BaseNorm):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, eps)

    def forward(self, hidden_states: torch.Tensor, output: Optional[torch.Tensor] = None) -> torch.Tensor:
        if output is None:
            output = torch.empty_like(hidden_states)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        output = self.weight * hidden_states.to(input_dtype)
        return output


class AddRMSNormTorch(BaseAddNorm):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, eps)
        self.rmsnorm_torch = RMSNormTorch(weight, eps)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + residual
        return self.rmsnorm_torch(hidden_states)


class QKRMSNormTorch(BaseQKNorm):
    def __init__(self, q_weight: torch.Tensor, k_weight: torch.Tensor, head_num: int, kv_head_num: int, size_per_head: int, eps: float = 1e-6):
        super().__init__(q_weight, k_weight, head_num, kv_head_num, size_per_head, eps)
        self.q_rmsnorm_torch = RMSNormTorch(q_weight, eps)
        self.k_rmsnorm_torch = RMSNormTorch(k_weight, eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        m, n = hidden_states.shape
        qkv = hidden_states.reshape(m, (self.head_num + self.kv_head_num * 2), self.size_per_head)
        q = qkv[:, :self.head_num, :]
        k = qkv[:, self.head_num:self.head_num + self.kv_head_num, :]
        self.q_rmsnorm_torch(q, q)
        self.k_rmsnorm_torch(k, k)
        return qkv.reshape(m, n)


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


class LayerNormTorch(BaseLayerNorm):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, beta, eps)

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(dim=-1, keepdim=True)
        squared_sum = (hidden_states**2).mean(dim=-1, keepdim=True)

        x_normalized = (hidden_states - mean) / torch.sqrt(
            (squared_sum - (mean**2)) + self.variance_epsilon
        )
        return (self.weight * x_normalized + self.beta).to(input_dtype)


class LayerNorm(BaseLayerNorm):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, beta, eps)

    def forward(self, hidden_states: torch.Tensor):
        output = torch.empty_like(hidden_states)
        rtp_llm_ops.layernorm(
            output,
            hidden_states,
            self.weight.data,
            self.beta,
            self.variance_epsilon,
        )
        return output
