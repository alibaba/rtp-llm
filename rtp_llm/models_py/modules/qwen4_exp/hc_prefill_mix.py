"""Qwen4 Prefill HC up projection and BF16 branch consumer.

The FP8 producer uses the existing per-tensor quantizer and MMA, retaining
the materialized BF16 logits, sigmoid and product boundaries of the ordinary
HC implementation.
"""

import torch
import triton


def supports_hc_prefill(normed, weight, scale):
    return (
        normed.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(normed.device) == (12, 0)
        and normed.ndim == 2
        and normed.shape[1] == 10240
        and 8192 <= normed.shape[0] <= 16384
        and weight.shape == (10240, 320)
        and normed.dtype == torch.bfloat16
        and weight.dtype == torch.float8_e4m3fn
        and weight.device == normed.device
        and all(t.is_contiguous() and not t.requires_grad for t in (weight, normed))
        and isinstance(scale, torch.Tensor)
        and scale.device == normed.device
        and scale.dtype == torch.float32
        and scale.numel() == 1
    )


def supports_hc_up_mix(mix, weight, normed, scale):
    return (
        supports_hc_prefill(normed, weight, scale)
        and mix.ndim == 2
        and mix.shape == (normed.shape[0], 320)
        and mix.dtype == torch.bfloat16
        and mix.device == normed.device
        and mix.is_contiguous()
        and not mix.requires_grad
    )


def hc_up_mix(mix, weight, normed, scale=None):
    if not supports_hc_up_mix(mix, weight, normed, scale):
        raise ValueError("unsupported Qwen4 Prefill HC up/mix geometry")
    rows = mix.shape[0]
    output = mix.new_empty((rows, 2560))
    grid = (triton.cdiv(rows, 32), 40)
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import scaled_fp8_per_tensor_quant
    from rtp_llm.models_py.triton_kernels.gated_residual.medium_up_mix import (
        _hc_medium_up_mix,
    )

    padding = (-rows) % 16
    padded = (
        torch.empty((rows + padding, 320), device=mix.device, dtype=weight.dtype)
        if padding
        else None
    )
    quantized, input_scale = scaled_fp8_per_tensor_quant(mix, output=padded)
    _hc_medium_up_mix[grid](
        quantized,
        weight,
        input_scale,
        scale,
        normed,
        output,
        M=rows,
        BM=32,
        BN=64,
        BK=64,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
    )
    return output
