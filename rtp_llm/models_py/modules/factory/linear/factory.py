"""Linear factory class

Uses strategy pattern to create appropriate Linear instances based on configuration.
"""

import logging
from typing import Dict, List, Optional, Type

import torch
from torch import nn

from rtp_llm.ops import HWKernelConfig

from .linear_base import LinearBase

try:
    # Fix nvidia-cutlass-dsl cutlass module path on sm100 or upper device
    import os
    import sys

    import nvidia_cutlass_dsl

    pkg_dir = nvidia_cutlass_dsl.__path__[0]
    python_packages_dir = os.path.join(pkg_dir, "python_packages")

    if os.path.isdir(python_packages_dir) and python_packages_dir not in sys.path:
        sys.path.insert(0, python_packages_dir)
        print(f"[sitecustomize] Added to sys.path: {python_packages_dir}")
except ImportError:
    pass  # nvidia-cutlass-dsl not installed

logger = logging.getLogger(__name__)


class LinearFactory:
    """Linear factory class

    Responsible for creating appropriate Linear instances based on configuration.
    Uses a list of registered strategies to find the right implementation.
    """

    _strategies: List[Type[LinearBase]] = []

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
        quant_config: object = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        weight_scale_2_key: Optional[str] = None,
        input_scale_key: Optional[str] = None,
    ) -> LinearBase:
        """Create Linear layer from weight dictionary

        Args:
            weights: Weight dictionary
            weight_key: Key for weight tensor
            scale_key: Key for scale tensor (optional)
            bias_key: Key for bias tensor (optional)
            quant_config: Quantization configuration (required)
            weight_scale_2_key: Key for second weight scale tensor (for FP4, optional)
            input_scale_key: Key for input scale tensor (for FP4, optional)

        Returns:
            Linear module instance

        Raises:
            ValueError: If no suitable strategy is found or multiple strategies match
        """
        weight = weights[weight_key]
        weight_scales = weights.get(scale_key) if scale_key else None
        bias = weights.get(bias_key) if bias_key else None
        weight_scale_2 = (
            weights.get(weight_scale_2_key, None) if weight_scale_2_key else None
        )
        input_scale = weights.get(input_scale_key, None) if input_scale_key else None

        return cls.create_linear(
            weight,
            bias,
            weight_scales,
            quant_config,
            hw_kernel_config,
            weight_scale_2,
            input_scale,
        )

    @classmethod
    def create_linear(
        cls,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        weight_scales: Optional[torch.Tensor],
        quant_config: object,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ):
        candidates = [
            strategy_class
            for strategy_class in cls._strategies
            if strategy_class.can_handle(
                quant_config,
                weight,
                weight_scales,
                hw_kernel_config,
                weight_scale_2,
                input_scale,
            )
        ]

        if not candidates:
            raise ValueError(
                f"No suitable Linear strategy found for:"
                f"weight.dtype={weight.dtype}, "
                f"has_scales={weight_scales is not None}, "
                f"quant_config={quant_config}"
            )

        # Check uniqueness - should only have one matching strategy
        if len(candidates) > 1:
            strategy_names = [cls.__name__ for cls in candidates]
            raise ValueError(
                f"Multiple Linear strategies found: {strategy_names}. "
                f"Each configuration should have exactly one matching strategy."
            )

        # Use the single matching strategy
        selected_class = candidates[0]
        logger.debug(f"Selected Linear strategy: {selected_class.__name__}")

        # Get input_scales if available

        # Create instance directly with all parameters
        instance = selected_class(
            weight=weight,
            weight_scales=weight_scales,
            input_scales=input_scale,
            bias=bias,
            quant_config=quant_config,
            weight_scale_2=weight_scale_2,
        )

        return instance

    @staticmethod
    def should_use_fp8_linear(
        quant_config: object,
        weights: Dict[str, torch.Tensor],
        weight_key: str,
    ) -> bool:
        """Check if FP8 linear layer should be used."""
        if quant_config is None:
            return False
        # Check quantization method if available
        quant_method = quant_config.get_method()
        fp8_methods = [
            "FP8",
            "FP8_PER_BLOCK",
            "FP8_PER_CHANNEL_COMPRESSED",
            "FP8_PER_CHANNEL_QUARK",
            "FP8_PER_TENSOR_COMPRESSED",
            "FP8_PER_BLOCK_QUARK",
        ]
        if quant_method not in fp8_methods:
            return False

        # Check if weight is FP8 format
        weight = weights.get(weight_key)
        if weight is None:
            return False

        return (
            weight.dtype == torch.float8_e4m3fn or weight.dtype == torch.float8_e4m3fnuz
        )

    @staticmethod
    def should_use_fp4_linear(
        quant_config: object,
        weights: Dict[str, torch.Tensor],
        weight_key: str,
    ) -> bool:
        """Check if NVFP4 linear layer should be used."""
        if quant_config is None:
            return False
        # Check quantization method if available
        quant_method = quant_config.get_method()
        fp4_methods = [
            "FP4",
            "modelopt_fp4",
        ]
        if quant_method not in fp4_methods:
            return False

        # Check if weight is FP8 format
        weight = weights.get(weight_key)
        if weight is None:
            return False

        return weight.dtype == torch.uint8
