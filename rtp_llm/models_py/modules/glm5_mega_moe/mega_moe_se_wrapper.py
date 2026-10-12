"""FP4 routed and FP8 shared experts using the unified MegaMoE kernel."""

from .mega_moe_se import GLM5MegaMoESE
from .mega_moe_wrapper import SharedExpertMegaMoeWrapper


class MegaMoeSEWrapper(SharedExpertMegaMoeWrapper):
    require_full_ffn_weights = True

    def _get_mega_moe_cls(self):
        return GLM5MegaMoESE


__all__ = ["MegaMoeSEWrapper"]
