"""
Factory class for creating Linear layers with FP8 or regular precision.
"""

from typing import Dict, Optional

import torch
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules import Linear, utils

if utils.is_hip():
    from rtp_llm.models_py.modules.rocm.fp8_linear import (
        Fp8DeepGEMMLinear,
        Fp8PTPCLinear,
    )

    FP8_LINEAR_AVAILABLE = True
    FP8_PTPC_AVAILABLE = True


elif utils.is_cuda():
    from rtp_llm.models_py.modules.fp8_linear import Fp8PerTensorLinear

    try:
        from rtp_llm.models_py.modules.fp8_linear import Fp8DeepGEMMLinear

        FP8_LINEAR_AVAILABLE = True
    except ImportError:
        Fp8DeepGEMMLinear = None
        FP8_LINEAR_AVAILABLE = False
else:
    Fp8DeepGEMMLinear = None
    FP8_LINEAR_AVAILABLE = False
    FP8_PTPC_AVAILABLE = False


class LinearFactory:
    """Factory for creating Linear layers with automatic FP8/regular precision selection."""

    @staticmethod
    def should_use_fp8_linear(
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        weight_key: str,
    ) -> bool:
        """Check if FP8 linear layer should be used."""
        if not hasattr(config, "quant_config") or config.quant_config is None:
            return False

        # Check quantization method if available
        if hasattr(config.quant_config, "get_method"):
            quant_method = config.quant_config.get_method()
            fp8_methods = [
                "FP8",
                "FP8_PER_BLOCK",
                "FP8_PER_CHANNEL_COMPRESSED",
                "FP8_PER_TENSOR_COMPRESSED",
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
    def create_linear(
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        config: Optional[GptInitModelParameters] = None,
        force_fp8: bool = False,
    ) -> nn.Module:
        """Create Linear layer (FP8 or regular)."""
        if force_fp8 or (
            weight_scales is not None
            and (
                weight.dtype == torch.float8_e4m3fn
                or weight.dtype == torch.float8_e4m3fnuz
            )
        ):
            if weight_scales is None:
                raise ValueError("FP8 linear layer requires weight_scales")
            if config is None:
                raise ValueError("FP8 linear layer requires config")
            else:
                quant_config = config.quant_config
                if quant_config.get_method() == "FP8_PER_BLOCK":
                    if has_deep_gemm():
                        return Fp8PerBlockLinear(weight, weight_scales, bias)
                    else:
                        raise RuntimeError(
                            "No available fp8 gemm backend for Fp8PerBlockLinear"
                        )
                elif quant_config.get_method() in [
                    "FP8_PER_TENSOR_COMPRESSED",
                    "FP8_DYNAMIC_PER_TENSOR",
                ]:
                    return Fp8PerTensorLinear(
                        weight, weight_scales, input_scales, bias, config.quant_config
                    )
                else:
                    if not FP8_LINEAR_AVAILABLE:
                        raise RuntimeError(
                            "FP8 DeepGEMMLinear layer requested but not available"
                        )

                    if (
                        quant_config
                        and quant_config.get_method() == "FP8_PER_CHANNEL_COMPRESSED"
                    ):
                        return Fp8PTPCLinear(weight, weight_scales, bias, config)
                    else:
                        return Fp8DeepGEMMLinear(weight, weight_scales, bias, config)
        else:
            return Linear(weight, bias)

    @staticmethod
    def create_linear_from_weights(
        weights: Dict[str, torch.Tensor],
        weight_key: str,
        scale_key: Optional[str] = None,
        bias_key: Optional[str] = None,
        config: Optional[GptInitModelParameters] = None,
    ) -> nn.Module:
        """Create Linear layer from weight dictionary."""
        weight = weights[weight_key]
        weight_scales = weights.get(scale_key) if scale_key else None
        bias = weights.get(bias_key)

        # Auto-detect FP8 usage
        use_fp8 = LinearFactory.should_use_fp8_linear(config, weights, weight_key)

        return LinearFactory.create_linear(
            weight=weight,
            weight_scales=weight_scales,
            bias=bias,
            config=config,
            force_fp8=use_fp8,
        )

    @staticmethod
    def create_merged_linear(
        weights: Dict[str, torch.Tensor],
        weight_keys: list,
        scale_keys: Optional[list] = None,
        bias_keys: Optional[list] = None,
        config: Optional[GptInitModelParameters] = None,
        dim: int = -1,
    ) -> nn.Module:
        """Create merged Linear layer (e.g., gate_up_proj)."""
        # Check FP8 usage based on first weight
        use_fp8 = LinearFactory.should_use_fp8_linear(config, weights, weight_keys[0])

        # Merge weights
        weight_tensors = [weights[key] for key in weight_keys]
        merged_weight = torch.cat(weight_tensors, dim=dim)

        # Merge scales if needed
        merged_scales = None
        if use_fp8 and scale_keys:
            scale_tensors = [weights[key] for key in scale_keys]
            merged_scales = torch.cat(scale_tensors, dim=dim)

        # Merge bias if exists
        merged_bias = None
        if bias_keys:
            bias_tensors = []
            for key in bias_keys:
                bias = weights.get(key)
                if bias is not None:
                    bias_tensors.append(bias)

            if bias_tensors:
                merged_bias = torch.cat(bias_tensors, dim=dim)

        return LinearFactory.create_linear(
            weight=merged_weight,
            weight_scales=merged_scales,
            bias=merged_bias,
            config=config,
            force_fp8=use_fp8,
        )
