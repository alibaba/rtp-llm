from typing import Tuple, Union

import torch
import torch.nn.functional as F
from aiter import layernorm2d_fwd as layernorm2d_fwd
from aiter import rms_norm
from aiter import rmsnorm2d_fwd_with_add as fused_add_rmsnorm
from aiter import (
    rmsnorm2d_fwd_with_add_dynamicquant as fused_add_rmsnorm_quant,
)
from aiter import (
    rmsnorm2d_fwd_with_dynamicquant as fused_rmsnorm_quant,
)
from torch import nn

from rtp_llm.models_py.modules.base.common.norm import (
    BaseAddBiasResLayerNorm,
    BaseLayerNorm,
    BaseNorm,
    BaseResNorm,
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

    def forward(self, hidden_states: torch.Tensor):
        return rms_norm(hidden_states, self.weight.data, self.variance_epsilon)


class RMSResNorm(BaseResNorm):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, eps)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        output = torch.empty_like(hidden_states)
        residual_out = torch.empty_like(hidden_states)
        fused_add_rmsnorm(
            output,
            hidden_states,
            residual,
            residual_out,
            self.weight.data,
            self.variance_epsilon,
            0, #use_model_sensitive_rmsnorm
        )
        # NOTE: copy_ may introduce extra overhead.
        residual.copy_(residual_out)
        return output


class AddBiasResLayerNorm(BaseAddBiasResLayerNorm):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, beta, eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        bias: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if bias.numel() == 0:
            bias = torch.zeros(
                hidden_states.shape[-1],
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        if hidden_states.shape[0] > 32 and hidden_states.shape[1] <= 768:
            hidden_states = hidden_states + residual
            x_bias = bias if bias.numel() > 0 else None
            return layernorm2d_fwd(
                hidden_states,
                self.weight.data,
                self.beta,
                self.variance_epsilon,
                x_bias=x_bias,
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


class QKRMSNorm(nn.Module):
    def __init__(
        self,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        head_num: int,
        kv_head_num: int,
        size_per_head: float = 128,
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
        self.variance_epsilon = eps

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

    def forward(self, hidden_states):
        q, k, v = hidden_states.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q, k)
        output = torch.cat([q, k, v], dim=-1)
        return output


class FusedQKRMSNorm(nn.Module):
    def __init__(
        self,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        head_num: int,
        kv_head_num: int,
        size_per_head: float = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.q_weight = q_weight
        self.k_weight = k_weight
        self.eps = eps
        self.head_num = head_num
        self.kv_head_num = kv_head_num
        self.size_per_head = size_per_head
        self.q_size = self.head_num * self.size_per_head
        self.kv_size = self.kv_head_num * self.size_per_head

    def forward(self, hidden_states):
        m, n = hidden_states.shape
        rtp_llm_ops.fused_qk_rmsnorm(
            hidden_states,
            self.q_weight,
            self.k_weight,
            self.eps,
            self.head_num,
            self.kv_head_num,
            m,
            n,
            self.size_per_head,
        )
        return hidden_states


class RMSNormFusedQuant(nn.Module):
    """ROCm-only RMSNorm + per-token FP8 quant 2-in-1 fused module."""

    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        m, k = hidden_states.shape
        out_fp8 = torch.empty(
            (m, k), dtype=torch.float8_e4m3fnuz, device=hidden_states.device
        )
        out_scale = torch.empty(
            (m, 1), dtype=torch.float32, device=hidden_states.device
        )
        fused_rmsnorm_quant(
            out_fp8,
            hidden_states,
            out_scale,
            self.weight.data,
            self.variance_epsilon,
            0,      # use_model_sensitive_rmsnorm
            0,      # group_size (0 = per-token)
            False,  # shuffle_scale
        )
        return out_fp8, out_scale


class RMSResNormFusedQuant(nn.Module):
    """ROCm-only residual_add + RMSNorm + per-token FP8 quant 3-in-1 fused module."""

    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual):
        m, k = hidden_states.shape
        out_fp8 = torch.empty(
            (m, k), dtype=torch.float8_e4m3fnuz, device=hidden_states.device
        )
        out_scale = torch.empty(
            (m, 1), dtype=torch.float32, device=hidden_states.device
        )
        residual_out = torch.empty_like(residual)
        fused_add_rmsnorm_quant(
            out_fp8,
            hidden_states,
            residual,
            residual_out,
            out_scale,
            self.weight.data,
            self.variance_epsilon,
            0,      # use_model_sensitive_rmsnorm
            0,      # group_size
            False,  # shuffle_scale
        )
        return out_fp8, out_scale, residual_out

