"""Host weight quantization contract for force_cpu_load_weights.

This is weight preparation only, never an FP8 inference backend. Preserve the
main loader's float32 division and scale floor rather than substitute the CUDA
activation quantizer (reciprocal multiplication and a different scale floor).
"""

import torch


@torch.no_grad()
def quantize_weight_per_tensor_cpu(weight: torch.Tensor):
    if weight.device.type != "cpu":
        raise ValueError("CPU FP8 weight preparation requires host storage")
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("CPU FP8 weight preparation requires unquantized floats")
    if weight.numel() == 0:
        raise ValueError("CPU FP8 weight preparation requires a nonempty tensor")
    finfo = torch.finfo(torch.float8_e4m3fn)
    maximum = weight.abs().max().to(torch.float32)
    scale = maximum / finfo.max
    floor = torch.tensor(1.0, dtype=torch.float32) / (
        torch.tensor(512.0, dtype=torch.float32) * finfo.max
    )
    scale = torch.maximum(floor, scale)
    quantized = (
        (weight.float() / scale).clamp(finfo.min, finfo.max).to(torch.float8_e4m3fn)
    )
    return quantized.contiguous(), scale.to(torch.float32)
