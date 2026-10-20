"""SM120 prefill up projection with the original BF16 gate/product boundaries."""

import torch
import triton

from .gated_residual_medium_up_mix_triton import _hc_medium_up_mix


def try_fp8_prefill_up_mix(x, weight, scale, normed, branches):
    if not (
        x.is_cuda
        and torch.version.hip is None
        and x.ndim >= 2
        and x.shape[-1] == 320
        and (4093 <= x.numel() // 320 <= 4096 or x.numel() // 320 == 12288)
        and branches == 4
        and x.dtype == torch.bfloat16
        and weight.shape == (10240, 320)
        and weight.dtype == torch.float8_e4m3fn
        and scale.dtype == torch.float32
        and scale.numel() == 1
        and normed.shape == (*x.shape[:-1], 10240)
        and normed.dtype == torch.bfloat16
        and x.device == weight.device == scale.device == normed.device
        and all(
            t.is_contiguous() and not t.requires_grad
            for t in (x, weight, scale, normed)
        )
        and torch.cuda.get_device_capability(x.device) == (12, 0)
    ):
        return None
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import scaled_fp8_per_tensor_quant

    rows = x.reshape(-1, 320)
    padding = (-rows.shape[0]) % 16
    padded = (
        torch.empty(
            (rows.shape[0] + padding, 320), device=x.device, dtype=torch.float8_e4m3fn
        )
        if padding
        else None
    )
    quantized, input_scale = scaled_fp8_per_tensor_quant(rows, output=padded)
    output = x.new_empty((rows.shape[0], 2560))
    # The shared MMA keeps K=320, the two FP32 scales, BF16 logits, sigmoid,
    # products and sequential branch sum. Only the measured prefill tile differs.
    _hc_medium_up_mix[(triton.cdiv(rows.shape[0], 32), 40)](
        quantized,
        weight,
        input_scale,
        scale,
        normed.reshape(-1, 10240),
        output,
        M=rows.shape[0],
        BM=32,
        BN=64,
        BK=64,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
    )
    return output.reshape(*x.shape[:-1], 2560)
