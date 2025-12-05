"""Common activation functions that are architecture-independent base operations."""

from abc import ABC, abstractmethod

import torch


class SiluAndMulBase(ABC):
    """Base class for silu_and_mul operation."""

    @abstractmethod
    def silu_and_mul(self, output: torch.Tensor, gate_up: torch.Tensor) -> None:
        """
        Perform SiLU activation and element-wise multiplication.

        Args:
            output: Output tensor to write result to
            gate_up: Input tensor with concatenated gate and up projections
        """
        pass
