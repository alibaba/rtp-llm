"""CUDA strategies without quantization"""

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


class CudaNoQuantEpLowLatencyStrategy(MoeStrategy):
    """CUDA EP low latency mode without quantization strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method is None)
        checker.check(
            config.moe_strategy == "no_auant_ep_low_latency"
            or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor import (
            DeepGemmMaskedExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouter,
        )

        quant_config = FusedMoEQuantConfig(quant_dtype=None)
        return StrategyAttributes(
            router_class=DeepEpLowLatencyRouter,
            executor_class=DeepGemmMaskedExecutor,
            quant_config=quant_config,
        )


class CudaNoQuantCppStrategy(MoeStrategy):
    """CUDA CPP mode without quantization strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(
            config.moe_strategy == "no_auant_cpp" or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_executor import (
            TritonFusedMoeExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterNoQuant,
        )

        quant_config = FusedMoEQuantConfig(quant_dtype=None)
        return StrategyAttributes(
            router_class=PureTpRouterNoQuant,
            executor_class=TritonFusedMoeExecutor,
            quant_config=quant_config,
        )


class CudaNoQuantDpNormalStrategy(MoeStrategy):
    """CUDA CPP mode without quantization strategy and dp normal mode"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(
            config.moe_strategy == "no_auant_dp_normal" or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_executor import (
            TritonFusedMoeExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouterNoQuant,
        )

        quant_config = FusedMoEQuantConfig(quant_dtype=None)
        return StrategyAttributes(
            router_class=DeepepNormalRouterNoQuant,
            executor_class=TritonFusedMoeExecutor,
            quant_config=quant_config,
        )


class CudaNoQuantDpNormalDeepGemmStrategy(MoeStrategy):
    """CUDA DeepEP Normal mode without quantization, using deepgemm bf16 grouped GEMM.

    Instead of fused_moe_kernel, this strategy uses the hybrid bf16 deepgemm path
    (masked for decode, contiguous for prefill):
    ep_scatter(_v2)_bf16 → deepgemm bf16 grouped GEMM → ep_gather
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import has_deep_gemm
        from rtp_llm.models_py.utils.arch import get_sm

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method is None)
        # Opt-in only: must be explicitly requested via --moe_strategy. Not part of
        # "auto" selection because the bf16 deepgemm path is not yet benchmarked on
        # CUDA — keeping it out of auto avoids changing the default CUDA MoE path.
        checker.check(config.moe_strategy == "no_quant_dp_normal_deepgemm")
        checker.check(has_deep_gemm())
        checker.check(get_sm()[0] >= 9)
        # executor dispatches masked/contiguous at runtime — incompatible with CUDA Graph replay
        checker.check(not config.enable_cuda_graph)

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_bf16_hybrid_executor import (
            DeepGemmBf16HybridExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouterNoQuant,
        )

        quant_config = FusedMoEQuantConfig(quant_dtype=None)
        return StrategyAttributes(
            router_class=DeepepNormalRouterNoQuant,
            executor_class=DeepGemmBf16HybridExecutor,
            quant_config=quant_config,
        )
