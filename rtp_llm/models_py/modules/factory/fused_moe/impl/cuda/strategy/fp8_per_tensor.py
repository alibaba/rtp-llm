"""CUDA FP8 PerTensor quantization strategies"""

from typing import Any

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy


class CudaFp8PerTensorEpLowLatencyStrategy(MoeStrategy):
    """CUDA FP8 PerTensor EP low latency strategy"""

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_moe import (
            CutlassBatchedExpertsFp8,
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
            executor_class=CutlassBatchedExpertsFp8,
            quant_config=quant_config,
        )


class CudaFp8PerTensorEpNormalStrategy(MoeStrategy):
    """CUDA FP8 PerTensor EP normal mode strategy"""

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_moe import (
            CutlassExpertsFp8,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouterFp8PerTensor,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=True,
        )
        return StrategyAttributes(
            router_class=DeepepNormalRouterFp8PerTensor,
            executor_class=CutlassExpertsFp8,
            quant_config=quant_config,
        )


class CudaFp8PerTensorNoDPStrategy(MoeStrategy):
    """CUDA FP8 PerTensor single GPU strategy"""

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_moe import (
            CutlassExpertsFp8,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterFp8PerTensor,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=True,
        )
        return StrategyAttributes(
            router_class=PureTpRouterFp8PerTensor,
            executor_class=CutlassExpertsFp8,
            quant_config=quant_config,
        )
