"""CUDA MXFP8 (1x32) MoE strategy (pure-TP, no DP)."""

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


class CudaMxfp8NoDPStrategy(MoeStrategy):
    """MXFP8 1x32 MoE via grouped fp8_fp4 contiguous GEMM (pure-TP / no DP)."""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(resolver.get_quant_method(config) == "MXFP8")
        checker.check(
            config.moe_strategy in ("mxfp8", "mxfp8_no_dp", "auto")
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mxfp8_contiguous_executor import (
            Mxfp8ContiguousExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterMxfp8,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[1, 32],
        )
        return StrategyAttributes(
            router_class=PureTpRouterMxfp8,
            executor_class=Mxfp8ContiguousExecutor,
            quant_config=quant_config,
        )
