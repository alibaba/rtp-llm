"""Torch distributed-based EP strategy (no DeepEP dependency)."""

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


class TorchDistEpNormalStrategy(MoeStrategy):
    """ROCm EP normal mode using torch.distributed all_to_all (no DeepEP)."""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Only applicable when DeepEP is NOT available."""
        try:
            import deep_ep  # noqa: F401

            # DeepEP available, prefer DeepepNormalRouter
            checker.check(False)
        except ImportError:
            pass

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.deepep_normal_fused_moe_executor import (
            FusedMoeExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.torch_dist_ep_router import (
            TorchDistEpRouter,
        )

        quant_config = FusedMoEQuantConfig(quant_dtype=None)
        return StrategyAttributes(
            router_class=TorchDistEpRouter,
            executor_class=FusedMoeExecutor,
            quant_config=quant_config,
        )
