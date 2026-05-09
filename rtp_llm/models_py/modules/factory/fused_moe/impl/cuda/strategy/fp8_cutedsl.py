"""CUDA MXFP8 (FlashInfer CuteDSL) MoE strategies.

These strategies wire the MXFP8 cutedsl executor to the existing DeepEP
routers. They opt in via ``--moe_strategy fp8_cutedsl_*``; the activation type
stays BF16 because online quantization is performed inside the executor.
"""

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


def _fp8_cutedsl_quant_config() -> FusedMoEQuantConfig:
    """Quant-config used by the cutedsl FP8 path.

    quant_dtype=None signals upstream routers to skip dispatch-time quantization
    (the executor quantizes per-expert with MXFP8). block_shape=[32, 32]
    advertises the MXFP8 group_size so anything that inspects it can act
    accordingly.
    """
    return FusedMoEQuantConfig(quant_dtype=None)


class CudaFp8CutedslEpLowLatencyStrategy(MoeStrategy):
    """MXFP8 cutedsl strategy paired with DeepEP low-latency router."""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(config.moe_strategy == "fp8_cutedsl_ep_low_latency")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutedsl_fp8_executor import (
            CutedslFp8Executor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouter,
        )

        return StrategyAttributes(
            router_class=DeepEpLowLatencyRouter,
            executor_class=CutedslFp8Executor,
            quant_config=_fp8_cutedsl_quant_config(),
        )


class CudaFp8CutedslEpNormalStrategy(MoeStrategy):
    """MXFP8 cutedsl strategy paired with DeepEP normal router (no quant)."""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(config.moe_strategy == "fp8_cutedsl_ep_normal")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutedsl_fp8_executor import (
            CutedslFp8Executor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouterFp8Cutedsl,
        )

        return StrategyAttributes(
            router_class=DeepepNormalRouterFp8Cutedsl,
            executor_class=CutedslFp8Executor,
            quant_config=_fp8_cutedsl_quant_config(),
        )
