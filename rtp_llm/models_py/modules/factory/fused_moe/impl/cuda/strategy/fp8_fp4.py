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
        checker.check(getattr(config, "mega_moe_policy", None) is None)
        checker.check(config.moe_strategy in ("auto", cls.strategy_name))
        checker.check(config.moe_quant_method == "FP8_FP4")
        checker.check(not config.has_redundant_experts)
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


class CudaMegaMoeStrategy(_CudaFp8Fp4Strategy):
    """One routed executor for native FP4 and prepared GLM5 checkpoints."""

    supported_moe_quant_method = (None, "FP8_FP4", "FP8_PER_BLOCK", "modelopt_fp4")
    strategy_name = "mega_moe"
    requires_ep = True
    requires_policy = False

    @classmethod
    def check_conditions(cls, checker, config):
        policy = getattr(config, "mega_moe_policy", None)
        if policy is None:
            if not checker.check(not cls.requires_policy):
                return
            super().check_conditions(checker, config)
            return
        checker.check(config.moe_strategy == cls.strategy_name)
        checker.check(not config.has_redundant_experts)
        checker.check(config.ep_size >= 1)
        if cls.requires_shared:
            checker.check(config.n_shared_experts > 0)
            checker.check(not config.has_shared_expert_gate)
            checker.check(config.parallelism_config.get_ffn_tp_size() == 1)

    @classmethod
    def get_executor_class(cls) -> Type:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
            MegaMoeExecutor,
        )

        return MegaMoeExecutor


class CudaMegaMoeSEStrategy(CudaMegaMoeStrategy):
    strategy_name = "mega_moe_se"
    requires_shared = True

    @classmethod
    def check_conditions(cls, checker, config):
        super().check_conditions(checker, config)
        if config.moe_strategy == "auto":
            from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
                _mega_moe_se_available,
            )

            checker.check(_mega_moe_se_available())

    @classmethod
    def get_executor_class(cls) -> Type:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
            MegaMoeSEExecutor,
            _mega_moe_se_available,
        )

        if not _mega_moe_se_available(allow_single_rank=True):
            # Preserve main's fallback when DeepGEMM has no shared_recipe API:
            # the model layer evaluates the shared expert independently.
            return CudaMegaMoeStrategy.get_executor_class()
        return MegaMoeSEExecutor


class CudaMegaMoeFp8Strategy(CudaMegaMoeStrategy):
    strategy_name = "mega_moe_fp8"
    requires_policy = True


class CudaMegaMoeFp8SEStrategy(CudaMegaMoeFp8Strategy):
    strategy_name = "mega_moe_fp8_se"
    requires_shared = True

    @classmethod
    def get_executor_class(cls) -> Type:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
            MegaMoeSEExecutor,
        )

        return MegaMoeSEExecutor


class CudaMegaMoeFusedStrategy(CudaMegaMoeStrategy):
    strategy_name = "mega_moe_fused"
    requires_policy = True
    requires_shared = True

    @classmethod
    def get_executor_class(cls) -> Type:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
            MegaMoeFusedBackend,
        )

        return MegaMoeFusedBackend


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
