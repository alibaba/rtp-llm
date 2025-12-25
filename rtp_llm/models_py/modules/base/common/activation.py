"""Common activation functions that are architecture-independent base operations."""

from abc import ABC, abstractmethod

import torch


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
