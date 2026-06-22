"""Shared utilities for CUDA attention implementations."""

from rtp_llm.models_py.utils.arch import is_blackwell, is_sm10x, is_sm12x

__all__ = ["is_blackwell", "is_sm10x", "is_sm12x"]
