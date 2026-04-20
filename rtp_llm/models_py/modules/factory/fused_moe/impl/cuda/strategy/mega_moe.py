"""MegaMoe strategy: fp8_fp4_mega_moe fused EP+GEMM+SwiGLU kernel.

Activated when USE_MEGA_MOE=1.  Requires SM100 (GB200/Blackwell) and
moe_intermediate_size % 512 == 0.

Priority = MEGA_MOE_router(6)*10 + MEGA_MOE_executor(9) = 69,
which beats every other strategy so it wins when conditions are met.
"""

from typing import Any

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy


class MegaMoeStrategy(MoeStrategy):
    """Strategy that uses the DeepGEMM fp8_fp4_mega_moe fused kernel."""

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_executor import (
            MegaMoeExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.mega_moe_passthrough_router import (
            MegaMoePassthroughRouter,
        )

        return StrategyAttributes(
            router_class=MegaMoePassthroughRouter,
            executor_class=MegaMoeExecutor,
            quant_config=FusedMoEQuantConfig(),
        )
