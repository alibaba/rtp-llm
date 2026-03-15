"""ROCm BF16 no quantization strategies"""

from typing import Any

from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy


class RocmBf16PureTPStrategy(MoeStrategy):
    """ROCm BF16 (no quantization) pure TP strategy"""

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.rocm_moe import (
            RocmExpertsBf16,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.pure_tp_router import (
            PureTpRouterNoQuant,
        )

        quant_config = FusedMoEQuantConfig(quant_dtype=None)
        return StrategyAttributes(
            router_class=PureTpRouterNoQuant,
            executor_class=RocmExpertsBf16,
            quant_config=quant_config,
        )
