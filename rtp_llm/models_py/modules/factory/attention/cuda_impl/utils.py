"""Shared utilities for CUDA attention implementations."""

import functools
from typing import Optional

import torch

# Constants
DEFAULT_TRT_WORKSPACE_SIZE_MB = (
    512  # Memory workspace size in MB, todo(Yingyi): read from config
)

DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_MB = 512

DEFAULT_XQA_WORKSPACE_SIZE_MB = 248

# Global workspace buffers shared across all implementations
_g_trt_workspace_buffer: Optional[torch.Tensor] = None
_g_py_flashinfer_workspace_buffer: Optional[torch.Tensor] = None
_g_xqa_workspace_buffer: Optional[torch.Tensor] = None


@functools.cache
def is_sm_100() -> bool:
    """Check if current GPU is SM 10.0 (Blackwell architecture)."""
    return torch.cuda.get_device_capability()[0] in [10]


def get_trt_workspace_buffer(device: str = "cuda:0") -> torch.Tensor:
    """Get or create the global TRT workspace buffer.

    This workspace buffer is shared across all TRT attention implementations
    to avoid allocating 512MB per instance.

    Args:
        device: CUDA device to allocate buffer on (default: "cuda:0")

    Returns:
        Workspace buffer tensor of size DEFAULT_TRT_WORKSPACE_SIZE_MB
    """
    global _g_trt_workspace_buffer
    if _g_trt_workspace_buffer is None:
        _g_trt_workspace_buffer = torch.zeros(
            DEFAULT_TRT_WORKSPACE_SIZE_MB * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )
    return _g_trt_workspace_buffer


def get_py_flashinfer_workspace_buffer(device: str = "cuda:0") -> torch.Tensor:
    """Get or create the global PyFlashInfer workspace buffer.

    This workspace buffer is shared across all PyFlashInfer attention implementations
    to avoid allocating 512MB per instance.

    Args:
        device: CUDA device to allocate buffer on (default: "cuda:0")

    Returns:
        Workspace buffer tensor of size DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_MB
    """
    global _g_py_flashinfer_workspace_buffer
    if _g_py_flashinfer_workspace_buffer is None:
        _g_py_flashinfer_workspace_buffer = torch.zeros(
            DEFAULT_PY_FLASHINFER_WORKSPACE_SIZE_MB * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )
    return _g_py_flashinfer_workspace_buffer


def get_xqa_workspace_buffer(device: str = "cuda:0") -> torch.Tensor:
    """Get or create the global XQA workspace buffer.

    This workspace buffer is shared across all XQA attention implementations
    to avoid allocating 248MB per instance.

    Args:
        device: CUDA device to allocate buffer on (default: "cuda:0")

    Returns:
        Workspace buffer tensor of size DEFAULT_XQA_WORKSPACE_SIZE_MB
    """
    global _g_xqa_workspace_buffer
    if _g_xqa_workspace_buffer is None:
        _g_xqa_workspace_buffer = torch.zeros(
            DEFAULT_XQA_WORKSPACE_SIZE_MB * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )
    return _g_xqa_workspace_buffer
