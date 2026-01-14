"""CUDA W4A8 INT4 PerChannel quantization strategies"""

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy


class CudaW4a8Int4PerChannelEpLowLatencyStrategy(MoeStrategy):
    """CUDA W4A8 INT4 PerChannel EP low latency strategy"""

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_w4a8_moe import (
            CutlassBatchedExpertsW4a8Int4,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouter,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=True,
        )
        return StrategyAttributes(
            router_class=DeepEpLowLatencyRouter,
            executor_class=CutlassBatchedExpertsW4a8Int4,
            quant_config=quant_config,
        )


class CudaW4a8Int4PerChannelEpNormalStrategy(MoeStrategy):
    """CUDA W4A8 INT4 PerChannel EP normal mode strategy"""

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_w4a8_moe import (
            CutlassExpertsW4a8Int4,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouterW4a8Int4PerChannel,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=True,
        )
        return StrategyAttributes(
            router_class=DeepepNormalRouterW4a8Int4PerChannel,
            executor_class=CutlassExpertsW4a8Int4,
            quant_config=quant_config,
        )


class CudaW4a8Int4PerChannelNoDPStrategy(MoeStrategy):
    """CUDA W4A8 INT4 PerChannel single GPU strategy"""

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_w4a8_moe import (
            CutlassExpertsW4a8Int4,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterW4a8Int4PerChannel,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=True,
        )
        return StrategyAttributes(
            router_class=PureTpRouterW4a8Int4PerChannel,
            executor_class=CutlassExpertsW4a8Int4,
            quant_config=quant_config,
        )
