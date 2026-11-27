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

    strategy_name = "no_auant_ep_low_latency"

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


class CudaBf16MegaMoeStrategy(MoeStrategy):
    """BF16 MegaMoE: one fused kernel owns dispatch, both GEMMs and combine.

    EP > 1 only, and chosen over the Triton/CPP paths when available because it
    is both batch invariant as shipped and 1.9-7.7x faster per MoE layer than
    the TP/EP grouped-GEMM path it replaces.
    """

    strategy_name = "bf16_mega_moe"

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
            _mega_moe_available,
        )

        resolver = MoeConfigResolver()
        checker.check(resolver.get_quant_method(config) is None)
        # Explicit only. Admitting it under "auto" would silently move every
        # unquantized EP model off its current MoE path, which is a much wider
        # change than this strategy has been validated for; engine_config
        # selects it by name for the configurations that have been.
        checker.check(config.moe_strategy == cls.strategy_name)
        checker.check(not config.has_redundant_experts)
        checker.check(config.ep_size > 1)
        checker.check(config.world_size == config.ep_size)
        checker.check(_mega_moe_available())

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.bf16_mega_moe import (
            Bf16MegaMoeExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.bf16_mega_moe_router import (
            Bf16MegaMoeRouter,
        )

        return StrategyAttributes(
            router_class=Bf16MegaMoeRouter,
            executor_class=Bf16MegaMoeExecutor,
            quant_config=FusedMoEQuantConfig(quant_dtype=None),
        )


class CudaNoQuantCppStrategy(MoeStrategy):
    """CUDA CPP mode without quantization strategy"""

    strategy_name = "no_quant_cpp"

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method is None)
        checker.check(
            config.moe_strategy == "no_quant_cpp" or config.moe_strategy == "auto"
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


class CudaNoQuantPureDPStrategy(MoeStrategy):
    """BF16 batch-invariant MoE over the pure DP router."""

    strategy_name = "no_quant_pure_dp"

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(config.batch_invariant)
        checker.check(config.moe_strategy == cls.strategy_name)

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_executor import (
            TritonFusedMoeExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_dp_router import (
            PureDpRouterNoQuant,
        )

        return StrategyAttributes(
            router_class=PureDpRouterNoQuant,
            executor_class=TritonFusedMoeExecutor,
            quant_config=FusedMoEQuantConfig(quant_dtype=None),
        )


class CudaNoQuantDpNormalStrategy(MoeStrategy):
    """CUDA CPP mode without quantization strategy and dp normal mode"""

    strategy_name = "no_auant_dp_normal"

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method is None)
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
