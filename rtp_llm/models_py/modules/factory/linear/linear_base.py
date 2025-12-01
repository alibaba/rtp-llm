"""Linear strategy base class

Defines the unified interface for all Linear strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters


class LinearBase(nn.Module, ABC):
    """Linear strategy base class

    Each strategy is both a strategy checker and a Linear implementation.
    It inherits from nn.Module and implements forward() directly.
    """

    @classmethod
    @abstractmethod
    def can_handle(
        cls,
        config: Optional[GptInitModelParameters],
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
    ) -> bool:
        """Determine whether this strategy can handle the given configuration

        Args:
            config: Model initialization parameters (can be None for non-FP8)
            weight: Weight tensor
            weight_scales: Weight scales tensor (None for non-FP8)

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
        config: Optional[GptInitModelParameters] = None,
    ):
        """Initialize the Linear module with weights

        Args:
            weight: Weight tensor
            weight_scales: Weight scales tensor
            input_scales: Input scales tensor
            bias: Bias tensor
            config: Model initialization parameters
        """
        super().__init__()

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
