"""PPU fused-MoE strategies for compressed W8A8 INT8 checkpoints."""

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


class PpuW8A8Int8DpNormalDeepGemmStrategy(MoeStrategy):
    """INT8 DeepEP Normal communication with W8A8 INT8 expert GEMMs."""

    QUANT_METHOD = "W8A8_INT8_PER_CHANNEL_COMPRESSED"
    STRATEGY = "w8a8_int8_dp_normal_deepgemm"

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(config.moe_strategy in ("auto", cls.STRATEGY))
        checker.check(resolver.get_quant_method(config) == cls.QUANT_METHOD)
        # DeepEP strategies require expert-parallel dispatch; exclude pure-TP mode.
        checker.check(not resolver.use_all_gather(config))

    def get_attributes(self) -> StrategyAttributes:
        from ..executors.deepgemm_hybrid_executor import DeepGemmInt8HybridExecutor
        from ..routers.deepep_normal_router import PpuDeepepNormalRouterW8A8Int8

        return StrategyAttributes(
            router_class=PpuDeepepNormalRouterW8A8Int8,
            executor_class=DeepGemmInt8HybridExecutor,
            quant_config=FusedMoEQuantConfig(
                quant_dtype=torch.int8,
                per_act_token_quant=True,
                per_out_ch_quant=True,
            ),
        )


class PpuW8A8Int8EpLowLatencyDeepGemmStrategy(MoeStrategy):
    """INT8 DeepEP low-latency communication and masked expert GEMMs."""

    QUANT_METHOD = "W8A8_INT8_PER_CHANNEL_COMPRESSED"
    STRATEGY = "w8a8_int8_ep_low_latency_deepgemm"

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(config.moe_strategy in ("auto", cls.STRATEGY))
        checker.check(resolver.get_quant_method(config) == cls.QUANT_METHOD)
        # DeepEP strategies require expert-parallel dispatch; exclude pure-TP mode.
        checker.check(not resolver.use_all_gather(config))

    def get_attributes(self) -> StrategyAttributes:
        from ..executors.deepgemm_masked_executor import DeepGemmInt8MaskedExecutor
        from ..routers.deepep_low_latency_router import (
            PpuDeepEpLowLatencyRouterW8A8Int8,
        )

        return StrategyAttributes(
            router_class=PpuDeepEpLowLatencyRouterW8A8Int8,
            executor_class=DeepGemmInt8MaskedExecutor,
            quant_config=FusedMoEQuantConfig(
                quant_dtype=torch.int8,
                per_act_token_quant=True,
                per_out_ch_quant=True,
            ),
        )
