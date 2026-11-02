"""Strategy registration for the first CANN MoE implementation."""

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
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)


class AscendCannStrategy(MoeStrategy):
    """BF16 eager MoE for single-card and pure-TP Ascend deployments."""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(resolver.is_bf16(config))
        checker.check(not resolver.has_quantization(config))
        checker.check(resolver.is_pure_tp_mode(config))
        # NOTE: upstream gated MoE out of cuda-graph deployments; the CANN ops
        # used here (npu_moe_gating_top_k / npu grouped GEMM) are aclnn calls
        # that are capturable by aclgraph. Validation pending for graph mode.
        checker.check(not config.moe_config.fake_balance_expert)
        checker.check(config.moe_strategy == "auto")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.ascend.executors import (
            NpuFusedExpertsExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.ascend.routers import (
            NpuPureTpRouter,
        )

        return StrategyAttributes(
            router_class=NpuPureTpRouter,
            executor_class=NpuFusedExpertsExecutor,
            quant_config=FusedMoEQuantConfig(quant_dtype=None),
        )
