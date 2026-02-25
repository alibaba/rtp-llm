"""Shared utilities for CUDA attention implementations."""

import functools
from typing import Optional

import torch

# Constants
DEFAULT_WORKSPACE_SIZE_MB = (
    512  # Memory workspace size in MB, todo(Yingyi): read from config
)

# Global workspace buffers shared across all implementations
_g_workspace_buffer: Optional[torch.Tensor] = None


@functools.cache
def is_sm_100() -> bool:
    """Check if current GPU is SM 10.0 (Blackwell architecture)."""
    return torch.cuda.get_device_capability()[0] in [10]


def get_workspace_buffer(device: str = "cuda") -> torch.Tensor:
    """Get or create the global workspace buffer.

    This workspace buffer is shared across all attention implementations
    to avoid allocating 512MB per instance.

    Args:
        device: CUDA device to allocate buffer on (default: "cuda:0")

    Returns:
        Workspace buffer tensor of size DEFAULT_WORKSPACE_SIZE_MB
    """
    global _g_workspace_buffer
    if _g_workspace_buffer is None:
        _g_workspace_buffer = torch.zeros(
            DEFAULT_WORKSPACE_SIZE_MB * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )
    return _g_workspace_buffer
