from typing import Optional, Union

import torch


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
