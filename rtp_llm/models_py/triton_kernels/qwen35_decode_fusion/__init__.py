"""Qwen3.5 decode short-kernel fusions (everything except the MoE router).

Import kernels from their modules. Package import is intentionally lightweight.

Enable every decode fusion with ``RTP_QWEN35_DECODE_FUSION=1``.
"""

from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.env import (
    DECODE_FUSION_ENV,
    is_decode_fusion_enabled,
)

__all__ = ["DECODE_FUSION_ENV", "is_decode_fusion_enabled"]

