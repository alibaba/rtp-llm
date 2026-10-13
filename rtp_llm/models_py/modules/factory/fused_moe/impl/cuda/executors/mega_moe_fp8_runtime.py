"""FusedMoe-compatible wrapper for GLM-5 FP8xFP8 MegaMoE."""

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8 import (
    MegaMoeFp8Backend,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_runtime import (
    MegaMoeWrapper,
)


class MegaMoeFp8Wrapper(MegaMoeWrapper):
    def _get_mega_moe_cls(self):
        return MegaMoeFp8Backend
