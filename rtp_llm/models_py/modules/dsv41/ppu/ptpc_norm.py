"""PTPC RMSNorm fusion with the canonical ATen reduction kept intact.

The following FP8 quantizer makes a one-ULP BF16 norm change observable.
Reuse the existing exact square/epilogue provider rather than changing the
non-power-of-two reduction tree or reciprocal-square-root implementation.
"""

import importlib.util

import torch

from rtp_llm.models_py.modules.dsv41.ppu.ptpc_quant import ptpc_fusion_enabled


PTPC_NORM_WIDTHS = (512, 1280, 5120)


def can_fuse_ptpc_rms_norm(hidden, gamma):
    """Metadata-only admission for the component-qualified inference shapes."""
    return (
        hidden.is_cuda
        and ptpc_fusion_enabled()
        and hidden.dtype == torch.bfloat16
        and hidden.ndim in (2, 3)
        and hidden.is_contiguous()
        and hidden.shape[-1] in PTPC_NORM_WIDTHS
        and gamma.dtype == torch.bfloat16
        and gamma.ndim == 1
        and gamma.numel() == hidden.shape[-1]
        and gamma.is_contiguous()
        and gamma.device == hidden.device
        and not torch.is_grad_enabled()
        and importlib.util.find_spec("triton") is not None
    )


def ptpc_rms_norm(hidden, gamma, eps):
    """Return canonical RMSNorm with the original FP32 operation boundaries.

On supported inference tensors the provider combines widening with square,
retains ATen mean/add/rsqrt, and combines the two separately rounded FP32
multiplications with the final BF16 store. Other inputs use the original
arithmetic, including autograd and strided/odd-width component inputs.
"""
    if can_fuse_ptpc_rms_norm(hidden, gamma):
        if hidden.numel() == 0:
            return torch.empty_like(hidden)
        from rtp_llm.models_py.modules.dsv41._v41_fused_qkv import (
            verify_rmsnorm_bf16,
        )

        return verify_rmsnorm_bf16(hidden, gamma, eps)
    values = hidden.float()
    inverse = torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    return (values * inverse * gamma.float()).to(hidden.dtype)
