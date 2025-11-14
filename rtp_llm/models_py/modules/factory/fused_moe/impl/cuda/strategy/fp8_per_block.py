"""CUDA FP8 PerBlock quantization strategies"""

from typing import Any, Dict

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import MoEConfigAdapter
from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy
from rtp_llm.config.model_config import ModelConfig


class CudaFp8PerBlockNoDPStrategy(MoeStrategy):
    """CUDA FP8 PerBlock single GPU strategy"""

    def create_router(self, config: MoEConfigAdapter) -> Any:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepgeemm_coutinous_router import (
            PureTpRouter,
        )

        return PureTpRouter(config)

    def create_executor(
        self, config: MoEConfigAdapter, weights: Dict[str, torch.Tensor]
    ) -> Any:
        # maybe use DeepGemmMaskedExecutor with reorder for small token size
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepep_normal_executor import (
            DeepGemmContinousExecutor,
        )

        return DeepGemmContinousExecutor(config, weights)

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepep_normal_executor import (
            DeepGemmContinousExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepgeemm_coutinous_router import (
            PureTpRouter,
        )

        return StrategyAttributes(
            router_class=PureTpRouter,
            executor_class=DeepGemmContinousExecutor,
        )


class CudaFp8PerBlockEpLowLatencyStrategy(MoeStrategy):
    """CUDA FP8 PerBlock EP low latency strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")


    def create_router(self, config: MoEConfigAdapter) -> Any:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouter,
        )

        return DeepEpLowLatencyRouter(
            config,
            use_fp8_dispatch=True,
            zero_copy=False,
            async_finish=False,
            return_recv_hook=False,
        )

    def create_executor(
        self, config: MoEConfigAdapter, weights: Dict[str, torch.Tensor]
    ) -> Any:
        from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
            FusedMoEQuantConfig,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor import (
            DeepGemmMaskedExecutor,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn, block_shape=[128, 128]
        )

        return DeepGemmMaskedExecutor(
            config,
            weights,
            quant_config=quant_config,
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor import (
            DeepGemmMaskedExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouter,
        )

        return StrategyAttributes(
            router_class=DeepEpLowLatencyRouter,
            executor_class=DeepGemmMaskedExecutor,
        )


class CudaFp8PerBlockEpNormalStrategy(MoeStrategy):
    """CUDA FP8 PerBlock EP normal mode strategy"""

    def create_router(self, config: MoEConfigAdapter) -> Any:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouter,
        )

        return DeepepNormalRouter(config)

    def create_executor(
        self, config: MoEConfigAdapter, weights: Dict[str, torch.Tensor]
    ) -> Any:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepep_normal_executor import (
            DeepGemmContinousExecutor,
        )

        return DeepGemmContinousExecutor(config, weights)

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepep_normal_executor import (
            DeepGemmContinousExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouter,
        )

        return StrategyAttributes(
            router_class=DeepepNormalRouter,
            executor_class=DeepGemmContinousExecutor,
        )
