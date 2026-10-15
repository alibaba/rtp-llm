"""BF16x2 INT8 quantization and dequantization with per-group BF16 scales (SM90+).

Inputs must be finite; decode scales must be nonnegative and finite. Output
storage must not overlap inputs. Extreme magnitudes may overflow on decode or
reduction. These value/aliasing constraints are caller preconditions.
"""

import torch


def support(x: torch.Tensor, group_size: int = 16) -> bool:
    """Check BF16 x[M,K] layout, group size and NVIDIA CUDA capability without scanning values."""
    return (
        torch.version.hip is None
        and group_size in (16, 32, 64, 128)
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.is_contiguous()
        and x.shape[1] % group_size == 0
        and x.data_ptr() % 16 == 0
        and torch.cuda.get_device_capability(x.device)[0] >= 9
    )


def quantize(x, q, scales, *, group_size=16):
    """Write INT8 q[M,K] and BF16 scales[M,K/G] from BF16 x[M,K].

    G is group_size, one of 16, 32, 64 or 128, and must divide K.
    Scale and reciprocal round to BF16; codes round to nearest-even and clamp
    to [-127,127]. All tensors are contiguous and allocated by the caller.
    """
    if not support(x, group_size):
        raise ValueError("Unsupported BF16 input or group_size")
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    rtp_llm_ops.fast_bf16_int8_quantize(x, q, scales, group_size)


def dequantize(q, scales, out, *, group_size=16):
    """Decode INT8 q[M,K] with BF16 scales[M,K/G] into BF16 out[M,K], G=group_size."""
    dequantize_reduce(q.unsqueeze(0), scales.unsqueeze(0), out, group_size=group_size)


def dequantize_reduce(q, scales, out, *, group_size=16):
    """Decode q[R,M,K] and scales[R,M,K/G], then sum into BF16 out[M,K].

    G is group_size and R is the number of sources, reduced in index order.
    Each product and each ordered addition rounds to BF16. R can be any
    positive count; sources must be contiguous individually, with optional
    padding between sources (INT8 source stride must be 16-byte aligned).
    Uses one kernel and no intermediate tensors; launches on the current stream.
    """
    if not support(out, group_size):
        raise ValueError("Unsupported BF16 output or group_size")
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    rtp_llm_ops.fast_bf16_int8_dequantize_reduce(q, scales, out, group_size)
