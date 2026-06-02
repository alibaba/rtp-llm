"""Rocm FP8 PerChannel quantization strategies"""

from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm._utils import (
    get_rocm_fp8_dtype,
)


class RocmFp8PerChannelPureTPStrategy(MoeStrategy):
    """Rocm FP8 PerChannel(PTPC) pure TP strategy"""

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.rocm_moe import (
            RocmExpertsFp8PerChannel,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.pure_tp_router import (
            PureTpRouterFusedQuant,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=get_rocm_fp8_dtype(),
            per_act_token_quant=True,
            per_out_ch_quant=True,
        )
        return StrategyAttributes(
            router_class=PureTpRouterFusedQuant,
            executor_class=RocmExpertsFp8PerChannel,
            quant_config=quant_config,
        )
