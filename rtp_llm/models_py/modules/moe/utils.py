from dataclasses import dataclass
from math import prod
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


def resize_cache(x: torch.Tensor, v: tuple[int, ...]) -> torch.Tensor:
    """
    Shrink the given tensor and apply the given view to it.  This is
    used to resize the intermediate fused_moe caches.
    """
    assert (
        prod(v) <= x.numel()
    ), f"{v} ({prod(v)}) <= {x.shape} ({x.numel()})"  # CUDAGRAPH unfriendly?
    return x.flatten()[: prod(v)].view(*v)


def normalize_scales_shape(scales: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if scales is not None:
        if scales.numel() == 1:
            scales = scales.view(1, 1)
        else:
            scales = scales.view(-1, scales.size(-1))
    return scales


def moe_kernel_quantize_input(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    quant_dtype: Union[None, torch.dtype, str],
    per_act_token_quant: bool,
    block_shape: Optional[list[int]] = None,
    is_fp4_scale_swizzled: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if quant_dtype == torch.float8_e4m3fn:
        raise NotImplementedError("float8 not supported yet")
    elif quant_dtype == torch.int8:
        raise NotImplementedError("int8 not supported yet")
    elif quant_dtype == torch.uint8:  # nvfp4
        raise NotImplementedError("nvfp4 not supported yet")
    elif quant_dtype == "mxfp4":
        raise NotImplementedError("mxfp4 not supported yet")
    else:
        return A, A_scale
