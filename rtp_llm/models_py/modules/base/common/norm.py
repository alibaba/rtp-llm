from typing import Optional, Tuple

import torch
from torch import nn

from rtp_llm.ops.compute_ops import rtp_llm_ops


class BaseNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class BaseResNorm(nn.Module):
    """Fused residual-add + RMSNorm base class.

    .. note:: Breaking change — ``forward`` now returns ``(normed, residual)``
        rather than a single ``Tensor``. Callers must unpack both values; the
        returned ``residual`` is the post-add residual to feed into the next
        layer. The prior single-Tensor return would silently bind the tuple
        to one variable on older callsites, so any un-migrated consumer will
        hit an obvious shape/type mismatch at use site.
    """

    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RMSResNormTorch(BaseResNorm):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__(weight, eps)
        self.rmsnorm_torch = RMSNormTorch(weight, eps)

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states + residual
        normed = self.rmsnorm_torch(residual)
        return normed, residual


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


class LayerwiseQKRMSNorm(nn.Module):
    """Per-layer (a.k.a. per-token) QK RMSNorm.

    Used by MiniMax-M2 (`qk_norm_type="per_layer"`): the RMSNorm is computed
    over the FULL Q (head_num*head_dim) and K (kv_head_num*head_dim) dims
    BEFORE per-head reshape. This is mathematically distinct from the
    per-head QKRMSNorm where variance is taken per-head over head_dim.
    Pipeline:
        sumsq_qk (Triton)  →  all_reduce([m, 2] fp32, TP group)  →  apply_qk (Triton)
    The all_reduce step is skipped for ``tp_size == 1``.

    Inputs:
        q_weight: full gamma for Q, shape [head_num_total*head_dim].
        k_weight: full gamma for K, shape [kv_head_num_total*head_dim].
        head_num / kv_head_num: LOCAL counts on this TP rank.
        tp_size / tp_rank: attention TP info. With tp_size>1 the variance
            must be reduced across the TP group so each rank sees the
            global mean(x^2) over the full H*D / Hkv*D dimension.
    """

    def __init__(
        self,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        head_num: int,
        kv_head_num: int,
        size_per_head: int = 128,
        eps: float = 1e-6,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        super().__init__()
        self.head_num = head_num
        self.kv_head_num = kv_head_num
        self.size_per_head = size_per_head
        self.eps = eps
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.q_size = head_num * size_per_head
        self.kv_size = kv_head_num * size_per_head
        self.q_total_elts = int(q_weight.numel())
        self.kv_total_elts = int(k_weight.numel())
        if tp_size > 1:
            expected_q = self.q_size * tp_size
            expected_kv = self.kv_size * tp_size
            assert (
                self.q_total_elts == expected_q
            ), f"q_weight numel {self.q_total_elts} != expected {expected_q}"
            assert self.kv_total_elts == expected_kv, (
                f"k_weight numel {self.kv_total_elts} != expected {expected_kv}. "
                f"LayerwiseQKRMSNorm does not support duplicate KV yet "
                f"(kv_head_num must be divisible by tp_size)."
            )
            self.q_weight = q_weight[
                tp_rank * self.q_size : (tp_rank + 1) * self.q_size
            ].contiguous()
            self.k_weight = k_weight[
                tp_rank * self.kv_size : (tp_rank + 1) * self.kv_size
            ].contiguous()
        else:
            assert (
                self.q_total_elts == self.q_size
            ), f"q_weight numel {self.q_total_elts} != {self.q_size}"
            assert (
                self.kv_total_elts == self.kv_size
            ), f"k_weight numel {self.kv_total_elts} != {self.kv_size}"
            self.q_weight = q_weight
            self.k_weight = k_weight

    @torch.no_grad()
    def forward(self, qk: torch.Tensor) -> torch.Tensor:
        from rtp_llm.models_py.triton_kernels.common.layerwise_qk_norm import (
            rmsnorm_qk_apply,
            rmsnorm_qk_sumsq,
        )

        sumsq = rmsnorm_qk_sumsq(qk, self.q_size, self.kv_size)
        if self.tp_size > 1:
            from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

            sumsq = all_reduce(sumsq.to(torch.float64), group=Group.TP).to(
                torch.float32
            )
        return rmsnorm_qk_apply(
            qk,
            self.q_weight,
            self.k_weight,
            sumsq,
            self.q_total_elts,
            self.kv_total_elts,
            self.eps,
        )
