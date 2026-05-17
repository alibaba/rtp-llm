from typing import Optional, Tuple

import flashinfer
import torch
from torch import nn

from rtp_llm.models_py.modules.base.common.norm import (
    BaseAddBiasResLayerNorm,
    BaseNorm,
    BaseResNorm,
)
from rtp_llm.models_py.modules.base.cuda.stream import current_cuda_stream_id
from rtp_llm.ops.compute_ops import rtp_llm_ops

_rmsnorm_custom_op = None


def _is_fake_or_meta_tensor(x: torch.Tensor) -> bool:
    if x.is_meta:
        return True
    try:
        from torch._subclasses.fake_tensor import FakeTensor

        return isinstance(x, FakeTensor)
    except Exception:
        return False


def _rmsnorm_eager(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    output = torch.empty_like(hidden_states)
    rtp_llm_ops.rmsnorm(output, hidden_states, weight, eps, current_cuda_stream_id())
    return output


try:
    _rmsnorm_custom_op = torch.library.custom_op(
        "rtp_llm_dsv4::rmsnorm",
        _rmsnorm_eager,
        mutates_args=(),
    )

    @_rmsnorm_custom_op.register_fake
    def _rmsnorm_fake(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.empty_like(hidden_states)

except Exception:
    _rmsnorm_custom_op = None


def rmsnorm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compile-friendly wrapper around the RTP-LLM RMSNorm pybind op.

    The eager path still launches the existing CUDA kernel.  During
    torch.compile/FakeTensor tracing the pybind op cannot be executed because
    it reads tensor data pointers, so return the correctly shaped tensor and
    keep this function as a stable FX call target for GraphFX fusion passes.
    """
    if output is None and _rmsnorm_custom_op is not None:
        return _rmsnorm_custom_op(hidden_states, weight, float(eps))
    if output is None:
        output = torch.empty_like(hidden_states)
    if _is_fake_or_meta_tensor(hidden_states):
        return output
    stream_id = current_cuda_stream_id()
    rtp_llm_ops.rmsnorm(output, hidden_states, weight, eps, stream_id)
    return output


class RMSNorm(BaseNorm):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, eps)

    def forward(
        self, hidden_states: torch.Tensor, output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return rmsnorm(
            hidden_states,
            self.weight.data,
            self.variance_epsilon,
            output=output,
        )


class RMSResNorm(BaseResNorm):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, eps)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor):
        stream_id = current_cuda_stream_id()
        rtp_llm_ops.fused_add_rmsnorm(
            hidden_states, residual, self.weight.data, self.variance_epsilon, stream_id
        )
        return hidden_states


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

    def forward(self, hidden_states: torch.Tensor):
        assert hidden_states.dim() == 2
        m, n = hidden_states.shape
        qkv = hidden_states.reshape(m, (self.head_num + self.kv_head_num * 2), self.size_per_head)
        q = qkv[:, :self.head_num, :]
        k = qkv[:, self.head_num:self.head_num + self.kv_head_num, :]
        flashinfer.norm.rmsnorm(q, self.q_weight, eps=self.eps, out=q, enable_pdl=self.enable_pdl)
        flashinfer.norm.rmsnorm(k, self.k_weight, eps=self.eps, out=k, enable_pdl=self.enable_pdl)
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
