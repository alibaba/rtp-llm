"""Ascend BF16 fallback MoE strategy using naive PyTorch implementation.

This is a pure PyTorch fallback for Ascend (no Triton dependency).
For production use, replace with a properly optimized executor.
"""

from typing import Any, Dict

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy


class AscendBf16FallbackStrategy(MoeStrategy):
    """Ascend BF16 fallback MoE strategy using pure PyTorch."""

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.common.executor.batched_triton_executor import (
            BatchedTritonExperts,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.common.router.batched_data_router import (
            BatchedDataRouter,
        )

        quant_config = FusedMoEQuantConfig(quant_dtype=None)
        return StrategyAttributes(
            router_class=BatchedDataRouter,
            executor_class=BatchedTritonExperts,
            quant_config=quant_config,
        )

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(not resolver.has_quantization(config))
