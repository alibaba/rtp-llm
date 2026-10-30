"""GenericMoeLayer adapter for routed-only GLM-5 GC MoE."""

from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_wrapper import MegaMoeWrapper

from .gc_moe import GLM5GCMoE


class GCMoeWrapper(MegaMoeWrapper):
    def _get_mega_moe_cls(self):
        return GLM5GCMoE
