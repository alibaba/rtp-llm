"""CUDA FP4 Per-Group quantization strategies"""

from typing import Any, Dict

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import MoEConfigAdapter
from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy
from rtp_llm.config.model_config import ModelConfig


class CudaFp4NoDPStrategy(MoeStrategy):
    """CUDA FP4 PerGroup single GPU strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "modelopt_fp4")
        checker.check(not config.moe_config.use_deepep_moe)

    def create_router(self, config: MoEConfigAdapter) -> Any:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepgeemm_coutinous_router import (
            PureTpRouter,
        )

        return PureTpRouter(config, False)

    def create_executor(
        self, config: MoEConfigAdapter, weights: Dict[str, torch.Tensor]
    ) -> Any:
        # maybe use DeepGemmMaskedExecutor with reorder for small token size
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.trtllm_fp4_executor import (
            TrtllmFp4Executor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
            FusedMoEQuantConfig,
        )
        
        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.uint8, block_shape=[16, 16]
        )
        return TrtllmFp4Executor(config, weights, quant_config)

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.trtllm_fp4_executor import (
            TrtllmFp4Executor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepgeemm_coutinous_router import (
            PureTpRouter,
        )

        return StrategyAttributes(
            router_class=PureTpRouter,
            executor_class=TrtllmFp4Executor,
        )


class CudaFp4EpLowLatencyStrategy(MoeStrategy):
    """CUDA FP4 PerGroup EP low latency strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "modelopt_fp4")


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
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutedsl_fp4_executor import (
            CutedslFp4Executor,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.uint8, block_shape=[16, 16]
        )
        return CutedslFp4Executor(
            config,
            weights,
            quant_config=quant_config,
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutedsl_fp4_executor import (
            CutedslFp4Executor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouter,
        )

        return StrategyAttributes(
            router_class=DeepEpLowLatencyRouter,
            executor_class=CutedslFp4Executor,
        )


class CudaFp4EpNormalStrategy(MoeStrategy):
    """CUDA FP4 PerGroup EP normal mode strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "modelopt_fp4")
        checker.check(config.moe_config.use_deepep_moe)
        checker.check(not config.moe_config.use_deepep_low_latency)

    
    def create_router(self, config: MoEConfigAdapter) -> Any:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouter,
        )

        return DeepepNormalRouter(config, use_fp8=False)

    def create_executor(
        self, config: MoEConfigAdapter, weights: Dict[str, torch.Tensor]
    ) -> Any:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.trtllm_fp4_executor import (
            TrtllmFp4Executor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
            FusedMoEQuantConfig,
        )
        
        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.uint8, block_shape=[16, 16]
        )
        return TrtllmFp4Executor(config, weights, quant_config)

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.trtllm_fp4_executor import (
            TrtllmFp4Executor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouter,
        )

        return StrategyAttributes(
            router_class=DeepepNormalRouter,
            executor_class=TrtllmFp4Executor,
        )
