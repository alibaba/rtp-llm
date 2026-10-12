"""Common activation functions that are architecture-independent base operations."""

from abc import ABC, abstractmethod

import torch
from torch.nn import functional as F


def gelu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
    """GeGLU with GELU rounded to the input dtype before multiplication."""
    if (
        gate_up.is_cuda
        and torch.version.hip is None
        and gate_up.dtype == torch.bfloat16
        and gate_up.dim() == 2
        and gate_up.shape[1] % 2 == 0
        and gate_up.numel() > 0
        and gate_up.stride(1) == 1
        and gate_up.stride(0) > 0
    ):
        from rtp_llm.models_py.triton_kernels.common.activation import gelu_tanh_and_mul

        output = gate_up.new_empty((gate_up.shape[0], gate_up.shape[1] // 2))
        return gelu_tanh_and_mul(output, gate_up)
    gate, up = gate_up.chunk(2, dim=-1)
    return F.gelu(gate, approximate="tanh") * up


class GeluAndMul(torch.nn.Module):
    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        return gelu_and_mul(gate_up)


class SiluAndMulBase(torch.nn.Module):
    """Base class for silu_and_mul operation."""

    @abstractmethod
    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        """
        Perform SiLU activation and element-wise multiplication.

        Args:
            output: Output tensor to write result to
            gate_up: Input tensor with concatenated gate and up projections
        """
        pass
