"""Production support, state ABI, and diagnostics for Kimi K3.

Pure-Torch correctness models live in :mod:`rtp_llm.models_py.modules.kimi_k3.reference`.
"""

from rtp_llm.models_py.modules.kimi_k3.kda_state import (
    KDAExecutionMode,
    KimiKDAState,
)
from rtp_llm.models_py.modules.kimi_k3.mxfp4 import (
    MXFP4_GROUP_SIZE,
    dequantize_mxfp4,
)

__all__ = [
    "KDAExecutionMode",
    "KimiKDAState",
    "MXFP4_GROUP_SIZE",
    "dequantize_mxfp4",
]
