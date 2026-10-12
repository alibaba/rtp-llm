"""FusedMoe-compatible wrapper for GLM-5 FP8xFP8 MegaMoE."""

from .mega_moe_fp8 import GLM5MegaMoEFP8
from .mega_moe_wrapper import MegaMoeWrapper


class MegaMoeFp8Wrapper(MegaMoeWrapper):
    def _get_mega_moe_cls(self):
        return GLM5MegaMoEFP8
