"""CUDA strategies without quantization"""

from typing import Any, Dict

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy


class CudaNoQuantEpLowLatencyStrategy(MoeStrategy):
    """CUDA EP low latency mode without quantization strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method is None)

    def create_router(self, config: MoEConfigAdapter) -> Any:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
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
        self, config: MoEConfigAdapter, weights: Dict[str, torch.Tensor]
    ) -> Any:
        from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
            FusedMoEQuantConfig,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor import (
            DeepGemmMaskedExecutor,
        )

        quant_config = FusedMoEQuantConfig(quant_dtype=None)

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


class CudaNoQuantCppStrategy(MoeStrategy):
    """CUDA CPP mode without quantization strategy"""

    def create_router(self, config: MoEConfigAdapter) -> Any:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepgeemm_coutinous_router import (
            PureTpRouter,
        )

        return PureTpRouter(
            config,
            use_fp8=False,
            need_recompute_topk_ids=False,
        )

    def create_executor(
        self, config: MoEConfigAdapter, weights: Dict[str, torch.Tensor]
    ) -> Any:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.f16_cpp_executor import (
            CppMoeExecutor,
        )

        return CppMoeExecutor(
            config,
            weights,
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.f16_cpp_executor import (
            CppMoeExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepgeemm_coutinous_router import (
            PureTpRouter,
        )

        return StrategyAttributes(
            router_class=PureTpRouter,
            executor_class=CppMoeExecutor,
        )
