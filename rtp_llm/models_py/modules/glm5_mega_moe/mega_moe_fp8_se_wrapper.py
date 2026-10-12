"""FP8 routed and shared experts using the common wrapper lifecycle."""

from .mega_moe_fp8_se import GLM5MegaMoEFP8SE
from .mega_moe_wrapper import SharedExpertMegaMoeWrapper


class MegaMoeFp8SEWrapper(SharedExpertMegaMoeWrapper):
    def _get_mega_moe_cls(self):
        return GLM5MegaMoEFP8SE
