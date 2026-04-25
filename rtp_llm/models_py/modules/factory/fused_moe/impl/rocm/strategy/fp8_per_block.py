"""Rocm FP8 PerBlock quantization strategies"""

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


class RocmFp8PerBlockPureTPStrategy(MoeStrategy):
    """Rocm FP8 PerBlock pure TP strategy"""

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.rocm_moe import (
            RocmExpertsFp8PerBlock,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.pure_tp_router import (
            PureTpRouterFp8PerBlockPassthrough,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=get_rocm_fp8_dtype(),
            per_act_token_quant=False,
            per_out_ch_quant=False,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=PureTpRouterFp8PerBlockPassthrough,
            executor_class=RocmExpertsFp8PerBlock,
            quant_config=quant_config,
        )
