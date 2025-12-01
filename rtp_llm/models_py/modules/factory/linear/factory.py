"""Linear factory class

Uses strategy pattern to create appropriate Linear instances based on configuration.
"""

import logging
from typing import Dict, List, Optional

import torch
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters

from .linear_base import LinearBase

logger = logging.getLogger(__name__)


class LinearFactory:
    """Linear factory class

    Responsible for creating appropriate Linear instances based on configuration.
    Uses a list of registered strategies to find the right implementation.
    """

    _strategies: List[type] = []

    @classmethod
    def register(cls, strategy_class: type) -> None:
        """Register a strategy class

        Args:
            strategy_class: Strategy class (not instance) to register
        """
        cls._strategies.append(strategy_class)
        logger.debug(f"Registered Linear strategy: {strategy_class.__name__}")

    @classmethod
    def clear(cls) -> None:
        """Clear all registered strategies (mainly for testing)"""
        cls._strategies.clear()

    @classmethod
    def create_linear_from_weights(
        cls,
        weights: Dict[str, torch.Tensor],
        weight_key: str,
        scale_key: Optional[str] = None,
        bias_key: Optional[str] = None,
        config: Optional[GptInitModelParameters] = None,
    ) -> nn.Module:
        """Create Linear layer from weight dictionary

        Args:
            weights: Weight dictionary
            weight_key: Key for weight tensor
            scale_key: Key for scale tensor (optional)
            bias_key: Key for bias tensor (optional)
            config: Model initialization parameters (optional)

        Returns:
            Linear module instance

        Raises:
            ValueError: If no suitable strategy is found or multiple strategies match
        """
        weight = weights[weight_key]
        weight_scales = weights.get(scale_key) if scale_key else None
        bias = weights.get(bias_key)

        # Find all candidate strategies that can handle this configuration
        candidates = [
            strategy_class
            for strategy_class in cls._strategies
            if strategy_class.can_handle(config, weight, weight_scales)
        ]

        if not candidates:
            raise ValueError(
                f"No suitable Linear strategy found for weight_key={weight_key}, "
                f"weight.dtype={weight.dtype}, "
                f"has_scales={weight_scales is not None}, "
                f"config={config}"
            )

        # Check uniqueness - should only have one matching strategy
        if len(candidates) > 1:
            strategy_names = [cls.__name__ for cls in candidates]
            raise ValueError(
                f"Multiple Linear strategies found for weight_key={weight_key}: {strategy_names}. "
                f"Each configuration should have exactly one matching strategy."
            )

        # Use the single matching strategy
        selected_class = candidates[0]
        logger.debug(f"Selected Linear strategy: {selected_class.__name__}")

        # Get input_scales if available
        input_scales = None

        # Create instance directly with all parameters
        instance = selected_class(
            weight=weight,
            weight_scales=weight_scales,
            input_scales=input_scales,
            bias=bias,
            config=config,
        )

        return instance
