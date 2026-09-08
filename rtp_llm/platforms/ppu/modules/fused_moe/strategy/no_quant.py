"""PPU fused-MoE strategies without quantization."""

from typing import Any

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


class PpuNoQuantDpNormalDeepGemmStrategy(MoeStrategy):
    """PPU BF16 DeepEP Normal communication with DeepGEMM expert compute."""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(config.moe_strategy == "no_quant_dp_normal_deepgemm")
        if config.moe_strategy != "no_quant_dp_normal_deepgemm":
            return

        from rtp_llm.platforms.ppu.kernels.int8.deepgemm_wrapper import (
            has_deep_gemm_bf16_grouped,
        )

        resolver = MoeConfigResolver()
        checker.check(resolver.get_quant_method(config) is None)
        checker.check(has_deep_gemm_bf16_grouped())
        checker.check(not config.enable_cuda_graph)
        # DeepEP Normal needs expert-parallel dispatch; exclude pure-TP mode.
        checker.check(not resolver.use_all_gather(config))

    def get_attributes(self) -> StrategyAttributes:
        from ..executors.deepgemm_hybrid_executor import DeepGemmBf16HybridExecutor
        from ..routers.deepep_normal_router import PpuDeepepNormalRouterNoQuant

        return StrategyAttributes(
            router_class=PpuDeepepNormalRouterNoQuant,
            executor_class=DeepGemmBf16HybridExecutor,
            quant_config=FusedMoEQuantConfig(quant_dtype=None),
        )


class PpuNoQuantEpLowLatencyDeepGemmStrategy(MoeStrategy):
    """PPU BF16 DeepEP low-latency communication with masked DeepGEMM."""

    STRATEGY = "no_quant_ep_low_latency_deepgemm"

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(config.moe_strategy in ("auto", cls.STRATEGY))
        checker.check(resolver.get_quant_method(config) is None)
        checker.check(not resolver.use_all_gather(config))

    @property
    def priority(self) -> int:
        # PPU registers after the open-source CUDA strategies. BF16 low-latency
        # uses the same router/executor type priority as CudaNoQuantEpLowLatencyStrategy,
        # so bump it by one to make the first-class PPU implementation win under
        # moe_strategy=auto without relying on registration order.
        return super().priority + 1

    def get_attributes(self) -> StrategyAttributes:
        from ..executors.deepgemm_masked_executor import DeepGemmBf16MaskedExecutor
        from ..routers.deepep_low_latency_router import PpuDeepEpLowLatencyRouterNoQuant

        return StrategyAttributes(
            router_class=PpuDeepEpLowLatencyRouterNoQuant,
            executor_class=DeepGemmBf16MaskedExecutor,
            quant_config=FusedMoEQuantConfig(quant_dtype=None),
        )
