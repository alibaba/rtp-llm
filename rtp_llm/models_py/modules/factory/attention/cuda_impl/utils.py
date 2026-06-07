"""Shared utilities for CUDA attention implementations."""

import functools

import torch


@functools.cache
def is_sm_100() -> bool:
    """Check if current GPU is SM 10.0 (Blackwell architecture)."""
    return torch.cuda.get_device_capability()[0] in [10]


def force_py_flashinfer() -> bool:
    """Force the pure-Python FlashInfer attention impls even on Blackwell.

    On B300/SM103 the C++ FMHA ops (TRT / C++ FlashInfer prefill) are either
    "unsupported architecture" or segfault during prepare(); the pure-Python
    flashinfer library path works for dense GQA. Enabled by LOAD_PYTHON_MODEL.
    """
    import os as _os
    return _os.environ.get("LOAD_PYTHON_MODEL", "").strip().lower() in ("1", "true", "yes")
