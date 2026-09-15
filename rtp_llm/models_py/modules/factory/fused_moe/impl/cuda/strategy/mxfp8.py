"""CUDA MXFP8 (1x32) MoE strategy (pure-TP, no DP)."""

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


class CudaMxfp8NoDPStrategy(MoeStrategy):
    """MXFP8 1x32 MoE via grouped fp8_fp4 contiguous GEMM (pure-TP / no DP)."""


    strategy_name = "mxfp8_no_dp"
    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(resolver.get_quant_method(config) == "MXFP8")
        checker.check(config.moe_strategy in ("mxfp8", "mxfp8_no_dp", "auto"))
        # When DeepEP is requested, defer to CudaMxfp8EpNormalStrategy. Needed
        # because PURE_TP has a higher router priority than DEEPEP_NORMAL, so
        # without this gate the pure-TP path would always win the tie-break.
        checker.check(not config.moe_config.use_deepep_moe)

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mxfp8_contiguous_executor import (
            Mxfp8ContiguousExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterMxfp8,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[1, 32],
        )
        return StrategyAttributes(
            router_class=PureTpRouterMxfp8,
            executor_class=Mxfp8ContiguousExecutor,
            quant_config=quant_config,
        )


class CudaMxfp8EpNormalStrategy(MoeStrategy):
    """MXFP8 1x32 MoE via DeepEP normal dispatch/combine (EP, no all_reduce).

    Enabled with ``use_deepep_moe=True`` (and ``use_all_gather=False``): the
    full (TP-replicated) hidden states are TP-sliced, DeepEP all-to-all
    dispatched to expert-owning ranks, run through the same MXFP8 grouped-GEMM
    executor on each rank's local experts, then combined + TP all_gathered
    back. Replaces the pure-TP path's full-hidden TP all_reduce, which is the
    long-context MoE communication bottleneck for MiniMax-M3."""


    strategy_name = "mxfp8_ep_normal"
    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(resolver.get_quant_method(config) == "MXFP8")
        checker.check(config.moe_strategy in ("mxfp8", "mxfp8_ep_normal", "auto"))
        checker.check(config.moe_config.use_deepep_moe)
        checker.check(not resolver.use_all_gather(config))

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mxfp8_deepep_executor import (
            Mxfp8DeepepExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouterMxfp8,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[1, 32],
        )
        return StrategyAttributes(
            router_class=DeepepNormalRouterMxfp8,
            executor_class=Mxfp8DeepepExecutor,
            quant_config=quant_config,
        )


class CudaMxfp8EpLowLatencyStrategy(MoeStrategy):
    """MXFP8 1x32 MoE via DeepEP low-latency dispatch/combine."""


    strategy_name = "mxfp8_ep_low_latency"
    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(resolver.get_quant_method(config) == "MXFP8")
        checker.check(config.moe_strategy in ("mxfp8", "mxfp8_ep_low_latency", "auto"))
        checker.check(config.moe_config.use_deepep_moe)
        checker.check(config.moe_config.use_deepep_low_latency)
        checker.check(not resolver.use_all_gather(config))

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mxfp8_deepep_executor import (
            Mxfp8LowLatencyExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouter,
        )

        # Dispatch BF16 activations. MXFP8 grouped GEMM quantizes activation
        # internally with recipe=(1, 32), matching the weight scale layout.
        quant_config = FusedMoEQuantConfig()
        return StrategyAttributes(
            router_class=DeepEpLowLatencyRouter,
            executor_class=Mxfp8LowLatencyExecutor,
            quant_config=quant_config,
        )


class _CudaMegaMoeWrapperStrategy(MoeStrategy):
    """Base for the GLM5 mega-kernel strategies.

    These answer only an exact ``MOE_STRATEGY`` request, never ``auto``. Before
    the factory owned their selection they were reachable exclusively through an
    explicit strategy name, and letting them win ``auto`` would silently move
    every eligible EP deployment onto a fused kernel.
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(config.moe_strategy == cls.strategy_name)

    @classmethod
    def _executor_class(cls) -> Any:
        raise NotImplementedError

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.fp8_fp8_router import (
            MegaMoeFp8Router,
        )

        return StrategyAttributes(
            router_class=MegaMoeFp8Router,
            executor_class=self._executor_class(),
            quant_config=FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn,
                block_shape=[1, 32],
            ),
        )


class CudaMegaMoeFp8Strategy(_CudaMegaMoeWrapperStrategy):
    """MXFP8 routed experts via DeepGEMM ``fp8_fp8_mega_moe`` (EP)."""

    strategy_name = "mega_moe_fp8"

    @classmethod
    def _executor_class(cls) -> Any:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_mxfp8 import (
            MegaMoeFp8Executor,
        )

        return MegaMoeFp8Executor


class CudaMegaMoeFp8SEStrategy(_CudaMegaMoeWrapperStrategy):
    """MXFP8 routed experts plus in-kernel FP8 shared expert (EP)."""

    strategy_name = "mega_moe_fp8_se"

    @classmethod
    def _executor_class(cls) -> Any:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_mxfp8 import (
            MegaMoeFp8SEExecutor,
        )

        return MegaMoeFp8SEExecutor


class CudaMegaMoeFusedStrategy(_CudaMegaMoeWrapperStrategy):
    """FP4 routed experts plus in-kernel FP4 shared expert (EP)."""

    strategy_name = "mega_moe_fused"

    @classmethod
    def _executor_class(cls) -> Any:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_mxfp8 import (
            MegaMoeFusedExecutor,
        )

        return MegaMoeFusedExecutor
