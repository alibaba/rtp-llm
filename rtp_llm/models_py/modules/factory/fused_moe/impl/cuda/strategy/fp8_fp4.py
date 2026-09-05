"""CUDA strategies for FP8-activation/FP4-weight MoE backends."""

from typing import Any, Type

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


class _CudaFp8Fp4Strategy(MoeStrategy):
    supported_moe_quant_method = "FP8_FP4"
    strategy_name: str
    requires_ep: bool
    requires_shared: bool = False

    @classmethod
    def get_executor_class(cls) -> Type:
        raise NotImplementedError

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(config.moe_strategy in ("auto", cls.strategy_name))
        checker.check(getattr(config, "moe_quant_method", None) == "FP8_FP4")
        checker.check((config.ep_size > 1) == cls.requires_ep)
        if cls.requires_shared:
            checker.check(config.n_shared_experts > 0)
            checker.check(not config.has_shared_expert_gate)

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.fp8_fp4_router import (
            Fp8Fp4Router,
            MegaMoeRouter,
        )

        router_class = MegaMoeRouter if self.requires_ep else Fp8Fp4Router
        return StrategyAttributes(
            router_class=router_class,
            executor_class=self.get_executor_class(),
            quant_config=FusedMoEQuantConfig(
                quant_dtype="fp8_fp4",
                block_shape=[128, 32],
            ),
        )


class CudaMegaMoeSEStrategy(_CudaFp8Fp4Strategy):
    strategy_name = "mega_moe_se"
    requires_ep = True
    requires_shared = True

    @classmethod
    def get_executor_class(cls) -> Type:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
            MegaMoeSEExecutor,
        )

        return MegaMoeSEExecutor


class CudaMegaMoeStrategy(_CudaFp8Fp4Strategy):
    strategy_name = "mega_moe"
    requires_ep = True

    @classmethod
    def get_executor_class(cls) -> Type:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
            MegaMoeExecutor,
        )

        return MegaMoeExecutor


class CudaGroupedFp4Strategy(_CudaFp8Fp4Strategy):
    strategy_name = "grouped_fp4"
    requires_ep = False

    @classmethod
    def get_executor_class(cls) -> Type:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.grouped_fp4 import (
            GroupedFp4Executor,
        )

        return GroupedFp4Executor


class CudaLocalLoopStrategy(_CudaFp8Fp4Strategy):
    strategy_name = "local_loop"
    requires_ep = False

    @classmethod
    def get_executor_class(cls) -> Type:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.local_loop import (
            LocalLoopExecutor,
        )

        return LocalLoopExecutor
