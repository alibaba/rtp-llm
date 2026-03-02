"""DCU BF16 no-quantization strategy."""

from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy


class DcuBf16PureTPStrategy(MoeStrategy):
    """DCU BF16 pure-TP strategy backed by vllm-dcu fused_experts."""

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.dcu.executors.dcu_moe import (
            DcuExpertsBf16,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.dcu.routers.pure_tp_router import (
            PureTpRouterNoQuant,
        )

        quant_config = FusedMoEQuantConfig(quant_dtype=None)
        return StrategyAttributes(
            router_class=PureTpRouterNoQuant,
            executor_class=DcuExpertsBf16,
            quant_config=quant_config,
        )
