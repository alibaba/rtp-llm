"""Kimi K3 modeling components.

``model_desc.kimi_k3`` owns the RTP model and decoder composition. KDA, MLA
and K3-specific mathematical primitives live in this package. Dense MLP and
sequence-parallel execution reuse framework modules under ``modules.hybrid``
and ``models_py.distributed``; KDA uses ``LinearCacheConverter`` directly.
"""

from rtp_llm.models_py.modules.kimi_k3.mxfp4 import (
    MXFP4_GROUP_SIZE,
    dequantize_mxfp4,
)


def __getattr__(name: str):
    if name in ("KDAExecutionMode", "KimiKDAState"):
        from rtp_llm.models_py.modules.kimi_k3.kda.state import (
            KDAExecutionMode,
            KimiKDAState,
        )

        return {
            "KDAExecutionMode": KDAExecutionMode,
            "KimiKDAState": KimiKDAState,
        }[name]
    raise AttributeError(name)

__all__ = [
    "KDAExecutionMode",
    "KimiKDAState",
    "MXFP4_GROUP_SIZE",
    "dequantize_mxfp4",
]
