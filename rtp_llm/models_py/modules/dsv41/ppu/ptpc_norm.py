"""PTPC RMSNorm fusion preserving the canonical ATen reduction tree.

The following FP8 quantizer makes a one-ULP BF16 norm change observable.
Qualified Torch shapes use the tree-preserving kernel. Other shapes retain
the existing exact square/ATen reduction/epilogue provider.
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

    Supported Torch2.11 inference shapes use one kernel with the same
    lane-local accumulation, shuffle tree and separate FP32 rounding. The
    original exact provider and portable arithmetic cover other metadata.
    """
    if can_fuse_ptpc_rms_norm(hidden, gamma):
        if hidden.numel() == 0:
            return torch.empty_like(hidden)
        from rtp_llm.models_py.modules.dsv41.ppu.ptpc_norm_quant import (
            ptpc_rms_norm_candidate,
        )

        normalized = ptpc_rms_norm_candidate(hidden, gamma, eps)
        if normalized is not None:
            return normalized
        from rtp_llm.models_py.modules.dsv41._v41_fused_qkv import (
            verify_rmsnorm_bf16,
        )

        return verify_rmsnorm_bf16(hidden, gamma, eps)
    values = hidden.float()
    inverse = torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    return (values * inverse * gamma.float()).to(hidden.dtype)
