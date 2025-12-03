from dataclasses import dataclass
from typing import Optional, Union

import torch

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
