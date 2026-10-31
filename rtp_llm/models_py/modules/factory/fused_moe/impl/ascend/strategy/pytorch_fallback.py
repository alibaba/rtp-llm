"""Ascend BF16 fallback MoE strategy.

Candidate for non-quantized MoE configs that AscendCannStrategy rejects
(e.g. fp16 act_type or non-auto MOE_STRATEGY): routes through the shared
BatchedDataRouter + BatchedTritonExperts execution chain (triton-ascend).
Not numerically validated on NPU yet; when triton is unimportable the
strategy skips itself and selection fails with an explicit error.
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
    """Fallback candidate for non-quantized configs the CANN strategy rejects."""

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
