"""CUDA FP8 PerBlock quantization strategies"""

from typing import Any, Dict

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory.fused_moe.strategies.base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.strategies.priority_attributes import (
    StrategyAttributes,
)


class CudaFp8PerBlockNoDPStrategy(MoeStrategy):
    """CUDA FP8 PerBlock single GPU strategy"""

    def create_router(self, config: GptInitModelParameters) -> Any:
        from rtp_llm.models_py.modules.cuda.moe.routers.deepgeemm_coutinous_router import (
            PureTpRouter,
        )

        return PureTpRouter(config)

    def create_executor(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ) -> Any:
        # maybe use DeepGemmMaskedExecutor with reorder for small token size
        from rtp_llm.models_py.modules.cuda.moe.executors.deepep_normal_executor import (
            DeepGemmContinousExecutor,
        )

        return DeepGemmContinousExecutor(config, weights)

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.cuda.moe.executors.deepep_normal_executor import (
            DeepGemmContinousExecutor,
        )
        from rtp_llm.models_py.modules.cuda.moe.routers.deepgeemm_coutinous_router import (
            PureTpRouter,
        )

        return StrategyAttributes(
            router_class=PureTpRouter,
            executor_class=DeepGemmContinousExecutor,
        )


class CudaFp8PerBlockEpLowLatencyStrategy(MoeStrategy):
    """CUDA FP8 PerBlock EP low latency strategy"""

    def create_router(self, config: GptInitModelParameters) -> Any:
        from rtp_llm.models_py.modules.cuda.moe.routers.deepep_low_latency_router import (
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
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ) -> Any:
        from rtp_llm.models_py.modules.cuda.moe.executors.deepgemm_masked_executor import (
            DeepGemmMaskedExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.quant_config import (
            FusedMoEQuantConfig,
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
        from rtp_llm.models_py.modules.cuda.moe.executors.deepgemm_masked_executor import (
            DeepGemmMaskedExecutor,
        )
        from rtp_llm.models_py.modules.cuda.moe.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouter,
        )

        return StrategyAttributes(
            router_class=DeepEpLowLatencyRouter,
            executor_class=DeepGemmMaskedExecutor,
        )


class CudaFp8PerBlockEpNormalStrategy(MoeStrategy):
    """CUDA FP8 PerBlock EP normal mode strategy"""

    def create_router(self, config: GptInitModelParameters) -> Any:
        from rtp_llm.models_py.modules.cuda.moe.routers.deepep_normal_router import (
            DeepepNormalRouter,
        )

        return DeepepNormalRouter(config)

    def create_executor(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ) -> Any:
        from rtp_llm.models_py.modules.cuda.moe.executors.deepep_normal_executor import (
            DeepGemmContinousExecutor,
        )

        return DeepGemmContinousExecutor(config, weights)

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.cuda.moe.executors.deepep_normal_executor import (
            DeepGemmContinousExecutor,
        )
        from rtp_llm.models_py.modules.cuda.moe.routers.deepep_normal_router import (
            DeepepNormalRouter,
        )

        return StrategyAttributes(
            router_class=DeepepNormalRouter,
            executor_class=DeepGemmContinousExecutor,
        )
