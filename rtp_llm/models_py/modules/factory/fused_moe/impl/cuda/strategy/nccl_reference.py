"""Never selected by auto: portable bring-up path using Torch NCCL BF16."""
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import StrategyAttributes
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import FusedMoEQuantConfig


class CudaBf16NcclReferenceStrategy(MoeStrategy):
    strategy_name = "nccl_bf16_reference"

    @classmethod
    def check_conditions(cls, checker, config):
        checker.check(config.moe_strategy == cls.strategy_name)

    def get_attributes(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_dp_bf16_tp_router import PureDpRouterBf16Tp
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_executor import TritonFusedMoeExecutor
        return StrategyAttributes(router_class=PureDpRouterBf16Tp, executor_class=TritonFusedMoeExecutor, quant_config=FusedMoEQuantConfig(quant_dtype=None))
