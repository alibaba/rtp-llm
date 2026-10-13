"""FP4 routed and FP8 shared experts using the dedicated fused kernel."""

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fused import (
    MegaMoeFusedBackend,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_runtime import (
    SharedExpertMegaMoeWrapper,
)


class MegaMoeFusedWrapper(SharedExpertMegaMoeWrapper):
    # Backend initialization still inspects the generic shared-weight surface.
    consume_shared_weights = False
    validate_shared_dtype = True

    def _get_mega_moe_cls(self):
        return MegaMoeFusedBackend

    def _mega_forward(self):
        return self.mega_moe.forward_with_shared_expert
