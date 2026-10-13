"""FP4 routed and FP8 shared experts using the unified MegaMoE kernel."""

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_runtime import (
    SharedExpertMegaMoeWrapper,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_shared import (
    MegaMoeSharedBackend,
)


class MegaMoeSEWrapper(SharedExpertMegaMoeWrapper):
    require_full_ffn_weights = True

    def _get_mega_moe_cls(self):
        return MegaMoeSharedBackend


__all__ = ["MegaMoeSEWrapper"]
