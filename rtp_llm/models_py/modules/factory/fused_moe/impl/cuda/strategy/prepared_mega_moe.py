"""Prepared MegaMoE recipes using the common fused-executor routing protocol."""

from rtp_llm.config.mega_moe import FUSED_SHARED_STRATEGIES
from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.fp8_fp4_router import (
    Fp8Fp4Router,
)


class PreparedMegaMoeRouter(Fp8Fp4Router):
    @classmethod
    def check_conditions(cls, checker, config):
        checker.check(getattr(config, "mega_moe_policy", None) is not None)


class CudaPreparedMegaMoeStrategy(MoeStrategy):
    supported_moe_quant_method = (None, "FP8_FP4", "FP8_PER_BLOCK", "modelopt_fp4")

    def __init__(self, strategy_name):
        self.strategy_name = strategy_name

    def check_conditions(self, checker, config):
        checker.check(config.moe_strategy == self.strategy_name)
        checker.check(getattr(config, "mega_moe_policy", None) is not None)
        if self.strategy_name in FUSED_SHARED_STRATEGIES:
            checker.check(config.n_shared_experts > 0)
            checker.check(not config.has_shared_expert_gate)
            checker.check(config.parallelism_config.get_ffn_tp_size() == 1)

    def get_attributes(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.prepared_mega_moe import (
            PreparedMegaMoeExecutor,
        )

        return StrategyAttributes(
            router_class=PreparedMegaMoeRouter,
            executor_class=PreparedMegaMoeExecutor,
            quant_config=FusedMoEQuantConfig(),
        )
