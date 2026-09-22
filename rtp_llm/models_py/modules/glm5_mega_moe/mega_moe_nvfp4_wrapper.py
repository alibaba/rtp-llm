"""FusedMoe-compatible wrapper for NVFP4xNVFP4 MegaMoE."""

import torch

from .mega_moe_nvfp4 import GLM5MegaMoENVFP4
from .mega_moe_wrapper import MegaMoeWrapper


class MegaMoeNvfp4Wrapper(MegaMoeWrapper):
    """Route experts through DeepGEMM ``nvfp4_nvfp4_mega_moe``."""

    def _get_mega_moe_cls(self):
        return GLM5MegaMoENVFP4

    def clone_for_cuda_graph(self) -> "MegaMoeNvfp4Wrapper":
        clone = object.__new__(type(self))
        torch.nn.Module.__init__(clone)
        clone.mega_moe = self.mega_moe.clone_for_cuda_graph()
        clone.expert_num = self.expert_num
        return clone
