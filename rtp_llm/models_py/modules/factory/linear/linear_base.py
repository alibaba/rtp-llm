"""Linear strategy base class

Defines the unified interface for all Linear strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn


class LinearBase(nn.Module, ABC):
    """Linear strategy base class

    Each strategy is both a strategy checker and a Linear implementation.
    It inherits from nn.Module and implements forward() directly.
    """

    @classmethod
    @abstractmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        """Determine whether this strategy can handle the given configuration

        Args:
            quant_config: Quantization configuration (required)
            weight: Weight tensor
            weight_scales: Weight scales tensor (None for non-FP8)
            weight_scale_2: Second weight scale tensor (for NVFP4, can be None)
            input_scale: Input scale tensor (for NVFP4, can be None)

        Returns:
            Whether this configuration can be handled
        """
        pass

    @abstractmethod
    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_config: object = None,
        weight_scale_2: Optional[torch.Tensor] = None,
    ):
        """Initialize the Linear module with weights

        Args:
            weight: Weight tensor
            weight_scales: Weight scales tensor
            input_scales: Input scales tensor
            bias: Bias tensor
            quant_config: Quantization configuration (required)ers
            weight_scale_2: Second weight scale tensor (for FP4, can be None)
        """
        super().__init__()

    def maybe_cache_quant_scale(self, max_len: int) -> None:
        """For quantized linear gemm input (fp8, fp4, etc),
        further quant scale calculation is not needed and can be constructed by simply filling ones.
        This method is used to cache the quant scale with given max length.

        Args:
            max_len: max input length to cache.
        """
        pass

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            input: Input tensor

        Returns:
            Output tensor
        """
        pass

    def __repr__(self) -> str:
        """Return string representation of the strategy"""
        return f"{self.__class__.__name__}"
