"""Weight encoding for ROCm online per-channel FP8 linear layers."""

import torch


def encode_rocm_ptpc_weight(weight: torch.Tensor):
    """Return canonical E4M3FN row weights and FP32 scales, prior to device conversion."""
    if weight.ndim != 2 or weight.dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ):
        raise ValueError("expected a 2D floating point weight")
    rows, width = weight.shape
    if not rows or not width:
        raise ValueError("expected a non-empty per-channel weight")
    quantized = torch.empty_like(weight, dtype=torch.float8_e4m3fn,
                                 memory_format=torch.contiguous_format)
    scales = torch.empty((rows, 1), dtype=torch.float32, device=weight.device)
    # Reproduce checkpoint loading, not the activation quantizer. In particular,
    # the 448 division rounds in the source dtype before the scale becomes FP32.
    # Chunk the FP32 quotient so wide models do not need a second weight-sized buffer.
    chunk = max(1, (1 << 20) // width)
    for offset in range(0, rows, chunk):
        values = weight[offset:offset + chunk]
        scale = (values.abs().amax(dim=1, keepdim=True).clamp_min(1e-4) / 448.0).float()
        quantized[offset:offset + chunk].copy_((values / scale).to(torch.float8_e4m3fn))
        scales[offset:offset + chunk].copy_(scale)
    return quantized, scales
