"""CUDA FP8 PerBlock quantization strategies"""

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


class CudaFp8PerBlockNoDPStrategy(MoeStrategy):
    """CUDA FP8 PerBlock single GPU strategy"""

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_continous_executor import (
            DeepGemmContinousExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterFp8PerBlock,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=PureTpRouterFp8PerBlock,
            executor_class=DeepGemmContinousExecutor,
            quant_config=quant_config,
        )


class CudaFp8PerBlockEpLowLatencyStrategy(MoeStrategy):
    """CUDA FP8 PerBlock EP low latency strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor import (
            DeepGemmMaskedExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouter,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=DeepEpLowLatencyRouter,
            executor_class=DeepGemmMaskedExecutor,
            quant_config=quant_config,
        )


class CudaFp8PerBlockEpNormalStrategy(MoeStrategy):
    """CUDA FP8 PerBlock EP normal mode strategy"""

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_continous_executor import (
            DeepGemmContinousExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouterFp8PerBlock,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=DeepepNormalRouterFp8PerBlock,
            executor_class=DeepGemmContinousExecutor,
            quant_config=quant_config,
        )


class CudaFp8PerBlockPureTpMaskedStrategy(MoeStrategy):
    """CUDA FP8 PerBlock Pure TP with contiguous-to-masked conversion.

    This strategy combines PureTpRouter (simple TP parallelism with contiguous output)
    with DeepGemmContinuousToMaskedExecutor (GPU-based layout conversion + masked GEMM).

    Benefits:
    - Simplicity of Pure TP communication (no DeepEP complexity)
    - Performance of DeepGemmMasked computation
    - End-to-end GPU execution without CPU-GPU synchronization
    - Suitable for single GPU or TP=EP scenarios
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_contiguous_to_masked_executor import (
            DeepGemmContinuousToMaskedExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterFp8PerBlock,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=PureTpRouterFp8PerBlock,
            executor_class=DeepGemmContinuousToMaskedExecutor,
            quant_config=quant_config,
        )
