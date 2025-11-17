"""CUDA strategies without quantization"""

from typing import Any, Dict

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory.fused_moe.strategies.base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.strategies.priority_attributes import (
    StrategyAttributes,
)


class CudaNoQuantEpLowLatencyStrategy(MoeStrategy):
    """CUDA EP low latency mode without quantization strategy"""

    def create_router(self, config: GptInitModelParameters) -> Any:
        from rtp_llm.models_py.modules.cuda.moe.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouter,
        )

        return DeepEpLowLatencyRouter(
            config,
            use_fp8_dispatch=False,
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

        quant_config = FusedMoEQuantConfig(quant_dtype=None)

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
