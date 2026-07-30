"""Pure-Torch primitives used only by the Kimi K3 reference model."""

from __future__ import annotations

import torch
from torch import nn


class KimiRMSNorm(nn.Module):
    """RMSNorm with fp32 variance accumulation, matching K3."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"RMSNorm expected last dim {self.hidden_size}, got {x.shape[-1]}"
            )
        x_float = x.float()
        normalized = x_float * torch.rsqrt(
            x_float.square().mean(dim=-1, keepdim=True) + self.eps
        )
        # The checkpoint implementation rounds the normalized activation back
        # to BF16 before applying the BF16 affine weight.  This rounding point
        # is observable at routed-expert boundaries and must remain exact.
        return self.weight * normalized.to(dtype=x.dtype)


class SituAndMul(nn.Module):
    """K3 SiTU gated activation.

    ``x`` contains concatenated gate and up projections.  Computation is kept
    in fp32 to match the checkpoint implementation and cast back at the end.
    """

    def __init__(self, beta: float = 1.0, linear_beta: float | None = None):
        super().__init__()
        if beta <= 0:
            raise ValueError(f"SiTU beta must be positive, got {beta}")
        if linear_beta is not None and linear_beta <= 0:
            raise ValueError(
                f"SiTU linear_beta must be positive when set, got {linear_beta}"
            )
        self.beta = float(beta)
        self.linear_beta = None if linear_beta is None else float(linear_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError(
                f"SiTU expects an even gate/up dimension, got {x.shape[-1]}"
            )
        gate, up = x.float().chunk(2, dim=-1)
        activated_gate = (
            self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        )
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (activated_gate * up).to(dtype=x.dtype)


class KimiRMSGated(nn.Module):
    """Per-head RMSNorm followed by K3's sigmoid output gate."""

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.head_dim = int(head_dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if x.shape != gate.shape or x.shape[-1] != self.head_dim:
            raise ValueError(
                "KimiRMSGated requires matching [...,head_dim] inputs, got "
                f"x={tuple(x.shape)} gate={tuple(gate.shape)}"
            )
        input_dtype = x.dtype
        x_float = x.float()
        normalized = x_float * torch.rsqrt(
            x_float.square().mean(dim=-1, keepdim=True) + self.eps
        )
        output = normalized * self.weight.float() * torch.sigmoid(gate.float())
        return output.to(dtype=input_dtype)
