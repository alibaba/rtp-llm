from dataclasses import dataclass

import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops


@dataclass
class QuantizeFp8Tensor:
    quantized_value: torch.Tensor
    quantized_scale: torch.Tensor


def atex_minmax_pertensor_quant_fp8(x: torch.Tensor) -> QuantizeFp8Tensor:
    # kernel dispatch
    if x.dtype == torch.float16:
        o1, o2 = rtp_llm_ops.atex_minmax_pertensor_quant_fp16_fp8(x)
        return QuantizeFp8Tensor(quantized_value=o1, quantized_scale=o2)

    elif x.dtype == torch.bfloat16:
        o1, o2 = rtp_llm_ops.atex_minmax_pertensor_quant_bf16_fp8(x)
        return QuantizeFp8Tensor(quantized_value=o1, quantized_scale=o2)

    else:
        raise TypeError(f"Unsupported dtype {x.dtype}")
