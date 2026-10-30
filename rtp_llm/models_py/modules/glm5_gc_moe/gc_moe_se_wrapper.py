"""GenericMoeLayer adapter for GLM-5 GC MoE fused FP8 shared experts."""

from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_se_wrapper import MegaMoeSEWrapper

from .gc_moe_se import GLM5GCMoESE


class GCMoeSEWrapper(MegaMoeSEWrapper):
    def _get_mega_moe_cls(self):
        return GLM5GCMoESE
