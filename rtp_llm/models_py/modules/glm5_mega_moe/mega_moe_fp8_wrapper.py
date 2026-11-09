"""FusedMoe-compatible wrapper for GLM-5 FP8xFP8 MegaMoE."""

import torch

from .mega_moe_fp8 import GLM5MegaMoEFP8
from .mega_moe_wrapper import MegaMoeWrapper


class MegaMoeFp8Wrapper(MegaMoeWrapper):
    """Route GLM-5 MoE through DeepGEMM ``fp8_fp8_mega_moe``."""

    def _get_mega_moe_cls(self):
        return GLM5MegaMoEFP8

    def clone_for_cuda_graph(self) -> "MegaMoeFp8Wrapper":
        clone = object.__new__(type(self))
        torch.nn.Module.__init__(clone)
        clone.mega_moe = self.mega_moe.clone_for_cuda_graph()
        clone.expert_num = self.expert_num
        return clone
