import functools
import math
import os
from typing import Optional, Tuple

import flashinfer
import torch
from torch import nn

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.modules.base.common.norm import (
    BaseAddBiasResLayerNorm,
    BaseNorm,
    BaseResNorm,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


@functools.lru_cache(maxsize=None)
def _qk_vec8_device_supported(device_index: int) -> bool:
    # Tensor metadata may change between calls; only immutable device capability
    # is cached. Normal warmup resolves this before CUDA Graph capture.
    return torch.cuda.get_device_capability(device_index) == (12, 0)


def _qk_packed_bf16_tensor_contract(input, q_weight, k_weight, eps) -> bool:
    if (
        not input.is_cuda
        or input.dtype != torch.bfloat16
        or input.dim() != 2
        or input.shape[1] != 4096
        or not 1 <= input.shape[0] <= 2048
        or input.stride() != (4096, 1)
        or not input.is_contiguous()
        or input.data_ptr() % 16
        or not isinstance(eps, (int, float))
    ):
        return False
    try:
        if not math.isfinite(eps) or eps <= 0:
            return False
    except OverflowError:
        return False
    begin = input.data_ptr()
    end = begin + input.numel() * input.element_size()
    for weight in (q_weight, k_weight):
        if (
            weight.device != input.device
            or weight.dtype != torch.bfloat16
            or weight.shape != (128,)
            or weight.stride() != (1,)
            or not weight.is_contiguous()
            or weight.data_ptr() % 16
            or (weight.data_ptr() < end and begin < weight.data_ptr() + 256)
        ):
            return False
    return True


def _qk_vec8_tensor_contract(input, q_weight, k_weight, eps) -> bool:
    return _qk_packed_bf16_tensor_contract(
        input, q_weight, k_weight, eps
    ) and _qk_vec8_device_supported(input.device.index)


class RMSNorm(BaseNorm):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, eps)

    def forward(
        self, hidden_states: torch.Tensor, output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        stream_id = torch.cuda.current_stream().cuda_stream
        if output is None:
            output = torch.empty_like(hidden_states)
        rtp_llm_ops.rmsnorm(
            output, hidden_states, self.weight.data, self.variance_epsilon, stream_id
        )
        return output


class RMSResNorm(BaseResNorm):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, eps)

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        stream_id = torch.cuda.current_stream().cuda_stream
        rtp_llm_ops.fused_add_rmsnorm(
            hidden_states, residual, self.weight.data, self.variance_epsilon, stream_id
        )
        return hidden_states, residual


class QKRMSNorm(nn.Module):
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
        size_per_head: int = 128,
        eps: float = 1e-6,
        enable_pdl: bool = False,
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
        self.enable_pdl = enable_pdl
        # Resolve the explicit A/B switch before warmup / graph capture. PPU uses
        # the existing portable fused kernel; CUDA/ROCm keep their current route.
        self._use_ppu_fused_qk = (
            os.environ.get("RTP_LLM_PPU_FUSED_QK_RMSNORM", "0") == "1"
            and get_device_type() == DeviceType.Ppu
        )

    def forward(self, hidden_states: torch.Tensor):
        if (
            self._use_ppu_fused_qk
            and self.enable_pdl is False
            and self.head_num == 16
            and self.kv_head_num == 8
            and self.size_per_head == 128
            and hasattr(rtp_llm_ops, "fused_qk_rmsnorm")
            and _qk_packed_bf16_tensor_contract(
                hidden_states, self.q_weight, self.k_weight, self.eps
            )
        ):
            rtp_llm_ops.fused_qk_rmsnorm(
                hidden_states,
                self.q_weight,
                self.k_weight,
                self.eps,
                self.head_num,
                self.kv_head_num,
                hidden_states.shape[0],
                hidden_states.shape[1],
                self.size_per_head,
            )
            return hidden_states

        if (
            self.enable_pdl is False
            and self.head_num == 16
            and self.kv_head_num == 8
            and self.size_per_head == 128
            and getattr(flashinfer.norm, "_USE_CUDA_NORM", False) is True
            and hasattr(rtp_llm_ops, "fused_qk_rmsnorm_cuda_vec8")
            and _qk_vec8_tensor_contract(
                hidden_states, self.q_weight, self.k_weight, self.eps
            )
        ):
            rtp_llm_ops.fused_qk_rmsnorm_cuda_vec8(
                hidden_states, self.q_weight, self.k_weight, self.eps
            )
            return hidden_states

        assert hidden_states.dim() == 2
        m, n = hidden_states.shape
        qkv = hidden_states.reshape(
            m, (self.head_num + self.kv_head_num * 2), self.size_per_head
        )
        q = qkv[:, : self.head_num, :]
        k = qkv[:, self.head_num : self.head_num + self.kv_head_num, :]
        flashinfer.norm.rmsnorm(
            q, self.q_weight, eps=self.eps, out=q, enable_pdl=self.enable_pdl
        )
        flashinfer.norm.rmsnorm(
            k, self.k_weight, eps=self.eps, out=k, enable_pdl=self.enable_pdl
        )
        return qkv.reshape(m, n)


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
