"""Compatibility imports for the packed projection now shared with ROCm."""

from rtp_llm.models_py.quant_methods.groupwise_triton import _project, project

__all__ = ["project"]
