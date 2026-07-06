"""XPU-specific normalization implementations.

Uses vllm-xpu-kernels ops when available, PyTorch fallbacks otherwise.
"""
from typing import Optional, Tuple

import torch
from torch import nn

from rtp_llm.models_py.modules.base.common.norm import (
    BaseAddBiasResLayerNorm,
    BaseNorm,
    BaseResNorm,
)

try:
    from rtp_llm.models_py.modules.base.xpu.vllm_xpu_ops import is_available as _vllm_available
except ImportError:
    _vllm_available = lambda: False


def _can_use_vllm(tensor: torch.Tensor) -> bool:
    """Check if vllm-xpu-kernels ops can run on this tensor's device."""
    return _vllm_available() and tensor.is_xpu


class RMSNorm(BaseNorm):
    """XPU RMSNorm using vllm-xpu-kernels."""

    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, eps)

    def forward(
        self, hidden_states: torch.Tensor, output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if _can_use_vllm(hidden_states):
            if output is None:
                output = torch.empty_like(hidden_states)
            torch.ops._C.rms_norm(output, hidden_states, self.weight.data, self.variance_epsilon)
            return output
        # PyTorch fallback
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        result = (self.weight * hidden_states).to(input_dtype)
        if output is not None:
            output.copy_(result)
            return output
        return result


class RMSResNorm(BaseResNorm):
    """XPU fused add + RMSNorm.

    Semantics (matching vllm fused_add_rms_norm):
    - residual is updated IN-PLACE to: residual + hidden_states
    - Returns RMSNorm(new_residual)
    """

    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, eps)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        if _can_use_vllm(hidden_states):
            torch.ops._C.fused_add_rms_norm(hidden_states, residual, self.weight.data, self.variance_epsilon)
            return hidden_states, residual
        # PyTorch fallback — must update both residual and hidden_states in-place
        # to match vllm fused_add_rms_norm semantics:
        #   residual <- residual + hidden_states
        #   hidden_states <- RMSNorm(new_residual)
        residual.add_(hidden_states)
        input_dtype = residual.dtype
        r_float = residual.to(torch.float32)
        variance = r_float.pow(2).mean(-1, keepdim=True)
        normed = r_float * torch.rsqrt(variance + self.variance_epsilon)
        result = (self.weight * normed).to(input_dtype)
        hidden_states.copy_(result)
        return hidden_states, residual


class QKRMSNorm(nn.Module):
    """XPU QK-RMSNorm using composition of RMSNorm."""

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
        self.size_per_head = int(size_per_head)
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
        if _can_use_vllm(hidden_states):
            # Normalize Q and K, write back to hidden_states
            q_slice = hidden_states[..., :self.q_size]
            q_flat = q_slice.reshape(-1, self.size_per_head)
            q_out = torch.empty_like(q_flat)
            torch.ops._C.rms_norm(q_out, q_flat, self.q_norm.weight.data, self.variance_epsilon)
            q_slice.copy_(q_out.view_as(q_slice))

            ks = self.q_size
            k_slice = hidden_states[..., ks:ks + self.kv_size]
            k_flat = k_slice.reshape(-1, self.size_per_head)
            k_out = torch.empty_like(k_flat)
            torch.ops._C.rms_norm(k_out, k_flat, self.k_norm.weight.data, self.variance_epsilon)
            k_slice.copy_(k_out.view_as(k_slice))

            return hidden_states
        
        q_slice, k_slice, _ = hidden_states.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q_slice, k_slice)
        q_slice.copy_(q)
        k_slice.copy_(k)
        return hidden_states

# FusedQKRMSNorm - same as QKRMSNorm for XPU (no special fused kernel)
FusedQKRMSNorm = QKRMSNorm


class AddBiasResLayerNorm(BaseAddBiasResLayerNorm):
    """XPU AddBiasResLayerNorm with empty-bias guard.

    When bias.numel()==0 (e.g. models without attention output bias), skip the
    bias addition to avoid a shape-mismatch crash on XPU.
    """

    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, beta, eps)

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, bias: torch.Tensor
    ):
        if bias.numel() == 0:
            hidden_states = hidden_states + residual
        else:
            hidden_states = hidden_states + bias + residual
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(dim=-1, keepdim=True)
        centered = hidden_states - mean
        variance = centered.pow(2).mean(dim=-1, keepdim=True)
        x_normalized = centered / torch.sqrt(
            variance + self.variance_epsilon
        )
        return (self.weight * x_normalized + self.beta).to(input_dtype)
