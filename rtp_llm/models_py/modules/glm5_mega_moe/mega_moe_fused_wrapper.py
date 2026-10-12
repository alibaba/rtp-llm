"""FP4 routed and FP8 shared experts using the dedicated fused kernel."""

from .mega_moe_fused import GLM5MegaMoEFused
from .mega_moe_wrapper import SharedExpertMegaMoeWrapper


class MegaMoeFusedWrapper(SharedExpertMegaMoeWrapper):
    # Backend initialization still inspects the generic shared-weight surface.
    consume_shared_weights = False
    validate_shared_dtype = True

    def _get_mega_moe_cls(self):
        return GLM5MegaMoEFused

    def _mega_forward(self):
        return self.mega_moe.forward_with_shared_expert
