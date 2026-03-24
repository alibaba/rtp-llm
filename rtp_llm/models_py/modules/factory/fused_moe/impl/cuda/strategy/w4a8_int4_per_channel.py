"""CUDA W4A8 INT4 PerChannel quantization strategies"""

from typing import Any

import torch

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
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)


class CudaW4a8Int4PerChannelNoDPStrategy(MoeStrategy):
    """CUDA W4A8 INT4 PerChannel single GPU strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "W4A8_INT4_PER_CHANNEL")
        checker.check(config.moe_strategy == "w4a8_int4_per_channel_no_dp" or config.moe_strategy == "auto")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_w4a8_moe import (
            CutlassExpertsW4a8Int4PerChannel,
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
            executor_class=CutlassExpertsW4a8Int4PerChannel,
            quant_config=quant_config,
        )


class CudaW4a8Int4PerChannelEpLowLatencyStrategy(MoeStrategy):
    """CUDA W4A8 INT4 PerChannel EP low latency strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "W4A8_INT4_PER_CHANNEL")
        checker.check(config.moe_strategy == "w4a8_int4_per_channel_ep_low_latency" or config.moe_strategy == "auto")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_w4a8_moe import (
            CutlassBatchedExpertsW4a8Int4PerChannel,
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
            executor_class=CutlassBatchedExpertsW4a8Int4PerChannel,
            quant_config=quant_config,
        )


class CudaW4a8Int4PerChannelEpNormalStrategy(MoeStrategy):
    """CUDA W4A8 INT4 PerChannel EP normal mode strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "W4A8_INT4_PER_CHANNEL")
        checker.check(config.moe_strategy == "w4a8_int4_per_channel_ep_normal" or config.moe_strategy == "auto")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_w4a8_moe import (
            CutlassExpertsW4a8Int4PerChannel,
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
            executor_class=CutlassExpertsW4a8Int4PerChannel,
            quant_config=quant_config,
        )
