"""PPU fused-MoE strategies for FP8 per-block [128, 128] checkpoints."""

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


class PpuFp8PerBlockDpNormalDeepGemmStrategy(MoeStrategy):
    """FP8 DeepEP Normal communication with FP8 expert GEMMs."""

    QUANT_METHOD = "FP8_PER_BLOCK"
    STRATEGY = "fp8_per_block_dp_normal_deepgemm"
    strategy_name = STRATEGY

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(config.moe_strategy in ("auto", cls.STRATEGY))
        checker.check(resolver.get_quant_method(config) == cls.QUANT_METHOD)
        # DeepEP strategies require expert-parallel dispatch; exclude pure-TP mode.
        checker.check(not resolver.use_all_gather(config))

    def get_attributes(self) -> StrategyAttributes:
        from ..executors.deepgemm_hybrid_executor import DeepGemmFp8HybridExecutor
        from ..routers.deepep_normal_router import PpuDeepepNormalRouterFp8PerBlock

        return StrategyAttributes(
            router_class=PpuDeepepNormalRouterFp8PerBlock,
            executor_class=DeepGemmFp8HybridExecutor,
            quant_config=FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn,
                block_shape=[128, 128],
            ),
        )


class PpuFp8PerBlockEpLowLatencyDeepGemmStrategy(MoeStrategy):
    """FP8 DeepEP low-latency communication with FP8 masked expert GEMMs."""

    QUANT_METHOD = "FP8_PER_BLOCK"
    STRATEGY = "fp8_per_block_ep_low_latency_deepgemm"
    strategy_name = STRATEGY

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(config.moe_strategy in ("auto", cls.STRATEGY))
        checker.check(resolver.get_quant_method(config) == cls.QUANT_METHOD)
        # DeepEP strategies require expert-parallel dispatch; exclude pure-TP mode.
        checker.check(not resolver.use_all_gather(config))

    def get_attributes(self) -> StrategyAttributes:
        from ..executors.deepgemm_masked_executor import DeepGemmFp8MaskedExecutor
        from ..routers.deepep_low_latency_router import (
            PpuDeepEpLowLatencyRouterFp8PerBlock,
        )

        return StrategyAttributes(
            router_class=PpuDeepEpLowLatencyRouterFp8PerBlock,
            executor_class=DeepGemmFp8MaskedExecutor,
            quant_config=FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn,
                block_shape=[128, 128],
            ),
        )


class PpuFp8PerBlockPureTpStrategy(MoeStrategy):
    """FP8 per-block expert GEMMs on the pure-TP / single-GPU path.

    Used when DeepEP is unavailable (e.g. no EP ranks or the deep_ep
    extension cannot load); all experts live on this rank and the router
    all-reduces over the TP group.
    """

    QUANT_METHOD = "FP8_PER_BLOCK"
    STRATEGY = "fp8_per_block_pure_tp"
    strategy_name = STRATEGY

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.kernels.ppu.deepgemm_wrapper import (
            has_deep_gemm_fp8_grouped,
        )

        resolver = MoeConfigResolver()
        checker.check(config.moe_strategy in ("auto", cls.STRATEGY))
        checker.check(resolver.get_quant_method(config) == cls.QUANT_METHOD)
        checker.check(resolver.use_all_gather(config))
        checker.check(resolver.is_single_gpu(config) or resolver.is_tp_equal_ep(config))
        checker.check(has_deep_gemm_fp8_grouped())

    @property
    def priority(self) -> int:
        # CudaFp8PerBlockNoDPStrategy matches the same topology under
        # ``auto`` but its CUDA executor cannot run on PPU; win the tie.
        return super().priority + 1

    def get_attributes(self) -> StrategyAttributes:
        from ..executors.deepgemm_hybrid_executor import DeepGemmFp8HybridExecutor
        from ..routers.pure_tp_router import PpuPureTpRouterFp8PerBlock

        return StrategyAttributes(
            router_class=PpuPureTpRouterFp8PerBlock,
            executor_class=DeepGemmFp8HybridExecutor,
            quant_config=FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn,
                block_shape=[128, 128],
            ),
        )
