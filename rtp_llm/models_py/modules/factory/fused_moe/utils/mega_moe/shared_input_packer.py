"""Add shared-L1 scale staging to the common routed input packer."""

from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.backend_input_packer import (
    TorchMegaMoeInputPacker,
    get_mega_moe_input_packer,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_fp8_scale import (
    stage_shared_fp8_input_scales,
)


class MegaMoeSeInputPacker:
    def __init__(self, routed_packer):
        self.routed_packer = routed_packer
        self.name = routed_packer.name

    def pack(self, x, weights, indices, buf, tokens: int, block_m: int) -> None:
        self.routed_packer.pack(x, weights, indices, buf, tokens)
        stage_shared_fp8_input_scales(buf.x_sf, buf.shared_l1_acts_sf, tokens, block_m)


class TorchMegaMoeSeInputPacker(MegaMoeSeInputPacker):
    def __init__(self):
        super().__init__(TorchMegaMoeInputPacker())


def get_mega_moe_se_input_packer():
    return MegaMoeSeInputPacker(get_mega_moe_input_packer())
