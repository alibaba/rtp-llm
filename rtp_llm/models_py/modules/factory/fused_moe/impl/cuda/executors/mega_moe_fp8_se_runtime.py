"""FP8 routed and shared experts using the common wrapper lifecycle."""

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8_se import (
    MegaMoeFp8SharedBackend,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_runtime import (
    SharedExpertMegaMoeWrapper,
)


class MegaMoeFp8SEWrapper(SharedExpertMegaMoeWrapper):
    def _get_mega_moe_cls(self):
        return MegaMoeFp8SharedBackend
