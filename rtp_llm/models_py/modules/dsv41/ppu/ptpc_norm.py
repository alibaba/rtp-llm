"""Compatibility ABI for PTPC callers; ordinary norm uses shared dispatch.

The explicit norm/quant producer remains in ptpc_norm_quant.py. Ordinary
calls no longer select a standalone ATen-tree kernel or a Torch epilogue.
"""

import importlib.util

import torch
from rtp_llm.models_py.modules.dsv41.ppu.ptpc_quant import ptpc_fusion_enabled
from rtp_llm.models_py.modules.dsv41.ppu.rmsnorm import rms_norm

PTPC_NORM_WIDTHS = (512, 1280, 5120)


def can_fuse_ptpc_rms_norm(hidden, gamma, *, allow_strided=False):
    """Metadata-only admission for the component-qualified inference shapes."""
    return (
        hidden.is_cuda
        and ptpc_fusion_enabled()
        and hidden.dtype == torch.bfloat16
        and hidden.ndim in (2, 3)
        and (not allow_strided or (not hidden.is_conj() and not hidden.is_neg()))
        and (
            hidden.is_contiguous()
            or (
                allow_strided
                and hidden.ndim == 2
                and hidden.shape[-1] in (512, 1280)
                and hidden.stride(1) == 1
                and hidden.stride(0) >= hidden.shape[-1]
                and not hidden.is_conj()
                and not hidden.is_neg()
            )
        )
        and hidden.shape[-1] in PTPC_NORM_WIDTHS
        and gamma.dtype == torch.bfloat16
        and gamma.ndim == 1
        and gamma.numel() == hidden.shape[-1]
        and gamma.is_contiguous()
        and gamma.device == hidden.device
        and not torch.is_grad_enabled()
        and importlib.util.find_spec("triton") is not None
    )


def ptpc_rms_norm(hidden, gamma, eps, *, allow_strided=False):
    """Keep the keyword ABI; layout selects strided norm or FlashInfer."""
    return rms_norm(hidden, gamma, eps)
