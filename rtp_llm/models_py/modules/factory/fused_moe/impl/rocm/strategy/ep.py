"""ROCm Expert Parallelism strategies"""

from typing import Any, Dict

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


class RocmEpNormalStrategy(MoeStrategy):
    """ROCm EP normal mode strategy"""

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.deepep_normal_fused_moe_executor import (
            FusedMoeExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.deepep_normal_router import (
            DeepepNormalRouter,
        )

        quant_config = FusedMoEQuantConfig(quant_dtype=None)
        return StrategyAttributes(
            router_class=DeepepNormalRouter,
            executor_class=FusedMoeExecutor,
            quant_config=quant_config,
        )


class RocmEpLowLatencyStrategy(MoeStrategy):
    """ROCm EP low latency strategy (not supported)"""

    def create_router(self, config: MoEConfigAdapter) -> Any:
        raise ValueError("deepep_low_latency for rocm moe is not yet supported")

    def create_executor(
        self, config: MoEConfigAdapter, weights: Dict[str, torch.Tensor]
    ) -> Any:
        raise ValueError("deepep_low_latency for rocm moe is not yet supported")

    def get_attributes(self) -> StrategyAttributes:
        # Not actually used, but needed for interface completeness
        # Don't set router_class and executor_class since this strategy is not supported
        # This will raise an error if called, which is expected since the strategy is not supported
        return StrategyAttributes(
            router_class=None,
            executor_class=None,
            quant_config=FusedMoEQuantConfig(quant_dtype=None),
        )
