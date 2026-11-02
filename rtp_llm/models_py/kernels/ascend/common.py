"""Ascend implementations of common Qwen3.5 model operators.

These operators deliberately use PyTorch operations supported by torch_npu.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the Gated Delta Net decay and input gates.

    Ascend does not provide the Triton runtime used by the original operator,
    so its equivalent is expressed with regular PyTorch operations.
    """

    g = -torch.exp(A_log.float()) * F.softplus(
        a.float() + dt_bias.float(), beta=beta, threshold=threshold
    )
    beta_output = torch.sigmoid(b.float()).to(b.dtype)
    return g.unsqueeze(0), beta_output.unsqueeze(0)


class RmsNormGated(torch.nn.Module):
    """Grouped RMSNorm followed by a SiLU or sigmoid gate."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        group_size: Optional[int] = None,
        eps: float = 1e-6,
        activation: str = "silu",
    ):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.eps = eps
        self.activation = activation
        self.group_size = weight.shape[-1] if group_size is None else group_size

        if bias is not None:
            assert bias.shape[-1] == weight.shape[-1], (
                "Bias dimension must be equal to weight dimension, "
                "weight_shape: {}, bias_shape: {}"
            ).format(weight.shape, bias.shape)
        assert weight.shape[-1] % self.group_size == 0, (
            "Weight dimension must be divisible by group size, "
            "weight_shape: {}, group_size: {}"
        ).format(weight.shape, self.group_size)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.weight.shape[-1], (
            "Input dimension must be equal to weight dimension, "
            "input_shape: {}, weight_shape: {}"
        ).format(x.shape, self.weight.shape)

        assert gate.shape == x.shape, "Gate shape must be equal to input shape"
        input_dtype = x.dtype
        x_float = x.float()
        num_groups = x.shape[-1] // self.group_size
        grouped_shape = x.shape[:-1] + (num_groups, self.group_size)
        grouped_x = x_float.reshape(grouped_shape)
        variance = grouped_x.pow(2).mean(dim=-1, keepdim=True)
        normalized = grouped_x * torch.rsqrt(variance + self.eps)
        output = normalized.reshape_as(x_float) * self.weight.float()
        if self.bias is not None:
            output = output + self.bias.float()

        gate_float = gate.float()
        if self.activation == "sigmoid":
            output = output * torch.sigmoid(gate_float)
        else:
            output = output * F.silu(gate_float)
        return output.to(input_dtype)
