"""ROCm Expert Parallelism strategies"""

from typing import Any, Dict

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory.fused_moe.strategies.base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.strategies.priority_attributes import (
    StrategyAttributes,
)


class RocmEpNormalStrategy(MoeStrategy):
    """ROCm EP normal mode strategy"""

    def create_router(self, config: GptInitModelParameters) -> Any:
        from rtp_llm.models_py.modules.rocm.moe.routers.deepep_normal_router import (
            DeepepNormalRouter,
        )

        return DeepepNormalRouter(config)

    def create_executor(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ) -> Any:
        from rtp_llm.models_py.modules.rocm.moe.executors.deepep_normal_fused_moe_executor import (
            FusedMoeExecutor,
        )

        return FusedMoeExecutor(config, weights)

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.rocm.moe.executors.deepep_normal_fused_moe_executor import (
            FusedMoeExecutor,
        )
        from rtp_llm.models_py.modules.rocm.moe.routers.deepep_normal_router import (
            DeepepNormalRouter,
        )

        return StrategyAttributes(
            router_class=DeepepNormalRouter,
            executor_class=FusedMoeExecutor,
        )


class RocmEpLowLatencyStrategy(MoeStrategy):
    """ROCm EP low latency strategy (not supported)"""

    def create_router(self, config: GptInitModelParameters) -> Any:
        raise ValueError("deepep_low_latency for rocm moe is not yet supported")

    def create_executor(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ) -> Any:
        raise ValueError("deepep_low_latency for rocm moe is not yet supported")

    def get_attributes(self) -> StrategyAttributes:
        # Not actually used, but needed for interface completeness
        # Don't set router_class and executor_class since this strategy is not supported
        # This will raise an error if called, which is expected since the strategy is not supported
        return StrategyAttributes(
            router_class=None,
            executor_class=None,
        )
