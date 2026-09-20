"""Linear strategy base class

Defines the unified interface for all Linear strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.ops import HWKernelConfig


class LinearBase(nn.Module, ABC):
    """Linear strategy base class

    Each strategy is both a strategy checker and a Linear implementation.
    It inherits from nn.Module and implements forward() directly.
    """

    supports_deferred_bias = False
    supports_fused_bias_gelu_quant = False
    supports_prequantized_activation = False

    @classmethod
    @abstractmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional["HWKernelConfig"] = None,
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
            quant_config: Quantization configuration (required)
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

    def forward_with_bias_gelu(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass followed by GELU.

        Backends with a fused GEMM+bias+GELU epilogue can override this method.
        The default implementation preserves existing device behavior.
        """
        return F.gelu(self.forward(input))

    def forward_without_bias(self, input: torch.Tensor) -> torch.Tensor:
        """Forward while deferring bias to a following fused epilogue."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support deferred bias"
        )

    def forward_with_bias_gelu_quantized(
        self, input: torch.Tensor
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Return fused GELU output in backend-native quantized form when supported."""
        return None

    def forward_quantized(
        self,
        input: torch.Tensor,
        input_scales: torch.Tensor,
        apply_bias: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} does not accept pre-quantized activations"
        )

    def forward_quantized_with_bias_gelu_quantized(
        self, input: torch.Tensor, input_scales: torch.Tensor
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        return None

    def __repr__(self) -> str:
        """Return string representation of the strategy"""
        return f"{self.__class__.__name__}"
