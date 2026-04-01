"""Rocm FP4 PerGroup quantization strategies"""

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy


class RocmFp4PerGroupPureTPStrategy(MoeStrategy):
    """Rocm FP4 PerGroup pure TP strategy"""

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.rocm_moe import (
            RocmExpertsFp4PerGroup,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.pure_tp_router import (
            PureTpRouterFp4PerGroupPassthrough,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float4_e2m1fn_x2,
            per_act_token_quant=False,
            per_out_ch_quant=False,
            block_shape=None,
        )
        return StrategyAttributes(
            router_class=PureTpRouterFp4PerGroupPassthrough,
            executor_class=RocmExpertsFp4PerGroup,
            quant_config=quant_config,
        )