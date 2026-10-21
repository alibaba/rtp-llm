"""Prepared MegaMoE launch policy over the common input packers."""

from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.input_packer import (
    FusedMegaMoEInputPacker,
    TorchMegaMoEInputPacker,
)


class TorchMegaMoeInputPacker(TorchMegaMoEInputPacker):
    def __init__(self):
        super().__init__(sanitize_non_finite=False)


def get_mega_moe_input_packer():
    """Keep prepared recipes on their fixed, non-sanitizing launch policy."""
    from rtp_llm.models_py.triton_kernels.moe.mega_moe_backend_input_pack import (
        fused_pack_mega_moe_inputs,
    )

    return FusedMegaMoEInputPacker(pack_inputs=fused_pack_mega_moe_inputs)
