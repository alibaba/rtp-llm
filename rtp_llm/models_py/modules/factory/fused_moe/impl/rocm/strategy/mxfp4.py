"""ROCm MXFP4 quantization strategies."""

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy


class RocmMXFp4PureTPStrategy(MoeStrategy):
    """ROCm MXFP4 pure TP strategy."""

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.rocm_moe import (
            RocmExpertsMXFp4,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.pure_tp_router import (
            PureTpRouterMXFp4Passthrough,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float4_e2m1fn_x2,
            per_act_token_quant=False,
            per_out_ch_quant=False,
            block_shape=None,
        )
        return StrategyAttributes(
            router_class=PureTpRouterMXFp4Passthrough,
            executor_class=RocmExpertsMXFp4,
            quant_config=quant_config,
        )
