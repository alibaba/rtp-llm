"""ATREX deterministic MoE strategy for ROCm"""

import os
from typing import Any

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


class AtrexMoeStrategy(MoeStrategy):
    """ATREX deterministic MoE strategy.

    Uses ATREX Triton Gluon kernels with split_reduce mode
    for deterministic MoE computation. Activated by setting
    environment variable USE_ATREX_MOE=1.
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(
            os.environ.get("USE_ATREX_MOE", "0") == "1",
        )

    def can_handle(self, config: MoEConfigAdapter) -> bool:
        if os.environ.get("USE_ATREX_MOE", "0") != "1":
            return False
        return True

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.atrex_fused_moe_executor import (
            AtrexFusedMoeExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.atrex_passthrough_router import (
            AtrexPassthroughRouter,
        )

        quant_config = FusedMoEQuantConfig(quant_dtype=None)
        return StrategyAttributes(
            router_class=AtrexPassthroughRouter,
            executor_class=AtrexFusedMoeExecutor,
            quant_config=quant_config,
        )
