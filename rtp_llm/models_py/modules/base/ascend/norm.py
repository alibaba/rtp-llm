import torch

from rtp_llm.models_py.modules.base.common.norm import (
    BaseAddBiasResLayerNorm,
    BaseLayerNorm,
    BaseNorm,
    BaseResNorm,
    RMSNormTorch,
    RMSResNormTorch,
)


class RMSNorm(BaseNorm):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, eps)
        self._impl = RMSNormTorch(weight, eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._impl(hidden_states)


class RMSResNorm(BaseResNorm):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, eps)
        self._impl = RMSResNormTorch(weight, eps)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        return self._impl(hidden_states, residual)


class LayerNorm(BaseLayerNorm):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, beta, eps)

    def forward(self, hidden_states: torch.Tensor):
        return torch.nn.functional.layer_norm(
            hidden_states,
            [hidden_states.shape[-1]],
            self.weight.float(),
            self.beta.float(),
            self.variance_epsilon,
        ).to(hidden_states.dtype)


class AddBiasResLayerNorm(BaseAddBiasResLayerNorm):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, beta, eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        bias: torch.Tensor,
    ):
        hidden_states = hidden_states + bias + residual
        return torch.nn.functional.layer_norm(
            hidden_states,
            [hidden_states.shape[-1]],
            self.weight.float(),
            self.beta.float(),
            self.variance_epsilon,
        ).to(hidden_states.dtype)


class QKRMSNorm(torch.nn.Module):
    def __init__(
        self,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        head_num: int,
        kv_head_num: int,
        size_per_head: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.q_norm = RMSNorm(q_weight, eps)
        self.k_norm = RMSNorm(k_weight, eps)
        self.head_num = head_num
        self.kv_head_num = kv_head_num
        self.size_per_head = size_per_head
        self.q_size = self.head_num * self.size_per_head
        self.kv_size = self.kv_head_num * self.size_per_head

    def forward(self, hidden_states):
        q, k, v = hidden_states.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.reshape(-1, self.size_per_head)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.reshape(-1, self.size_per_head)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        return torch.cat([q, k, v], dim=-1)


class FusedQKRMSNorm(QKRMSNorm):
    pass
