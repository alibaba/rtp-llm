from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import torch

if TYPE_CHECKING:
    from rtp_llm.config.quant_config import QuantizationConfig

# Type alias for quantization dtype
QuantDtype = Union[None, torch.dtype, str]


@dataclass
class FusedMoEQuantConfig:
    # The post quantization activation type.
    quant_dtype: QuantDtype = None
    per_act_token_quant: bool = False
    per_out_ch_quant: bool = False
    block_shape: Optional[list[int]] = None

    def __post_init__(self):
        assert (
            not self.per_act_token_quant or self.block_shape is None
        ), "illegal quantization"

    def __str__(self) -> str:
        if not self.is_quantized:
            return "FusedMoEQuantConfig(no_quant)"

        # Format quant_dtype
        if isinstance(self.quant_dtype, torch.dtype):
            dtype_str = str(self.quant_dtype).replace("torch.", "")
        else:
            dtype_str = str(self.quant_dtype)

        # Determine quantization type
        quant_type_parts = [dtype_str]
        if self.per_act_token_quant:
            quant_type_parts.append("per_token")
        elif self.is_block_quantized:
            quant_type_parts.append(f"per_block{self.block_shape}")
        else:
            quant_type_parts.append("per_tensor")

        if self.per_out_ch_quant:
            quant_type_parts.append("per_out_ch")

        return f"FusedMoEQuantConfig({', '.join(quant_type_parts)})"

    @property
    def is_quantized(self) -> bool:
        return self.quant_dtype is not None

    @property
    def is_per_act_token(self) -> bool:
        return self.per_act_token_quant

    @property
    def is_block_quantized(self) -> bool:
        return self.block_shape is not None

    @property
    def is_per_tensor(self) -> bool:
        return not self.per_act_token_quant and self.block_shape is None

    def scale_shape(
        self,
        max_tokens: int,
        hidden_dim: int,
    ) -> Optional[tuple[int, int]]:
        if self.is_quantized:
            if self.is_block_quantized:
                assert self.block_shape is not None
                _, block_k = self.block_shape
                k_tiles = (hidden_dim + block_k - 1) // block_k
                return (max_tokens, k_tiles)
            elif self.is_per_act_token:
                return (max_tokens, 1)
            else:
                return (1, 1)
        else:
            return None

    def batched_scale_shape(
        self,
        num_experts: int,
        max_tokens: int,
        hidden_dim: int,
    ) -> Optional[tuple[int, int, int]]:
        if self.is_quantized:
            scale_shape = self.scale_shape(max_tokens, hidden_dim)
            assert scale_shape is not None
            return (num_experts, *scale_shape)
        else:
            return None

    @classmethod
    def from_quantization_config(
        cls, quant_config: Optional["QuantizationConfig"]
    ) -> "FusedMoEQuantConfig":
        """Create FusedMoEQuantConfig from QuantizationConfig.

        Args:
            quant_config: QuantizationConfig instance or None

        Returns:
            FusedMoEQuantConfig instance
        """
        if quant_config is None:
            return cls(quant_dtype=None)

        quant_method = quant_config.get_method()

        # Handle FP8_PER_BLOCK quantization
        if quant_method == "FP8_PER_BLOCK":
            block_size = quant_config.group_size()
            return cls(
                quant_dtype=torch.float8_e4m3fn,
                per_act_token_quant=False,
                per_out_ch_quant=False,
                block_shape=[block_size, block_size],
            )

        # Handle FP8_DYNAMIC_PER_TENSOR and FP8_PER_TENSOR quantization
        if quant_method in ["FP8_DYNAMIC_PER_TENSOR", "FP8_PER_TENSOR"]:
            return cls(
                quant_dtype=torch.float8_e4m3fn,
                per_act_token_quant=True,
                per_out_ch_quant=False,
                block_shape=None,
            )

        # For other quantization methods, return no quantization
        # (or extend this method to support more types as needed)
        return cls(quant_dtype=None)
