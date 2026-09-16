"""Explicit opt-in strategy for BF16 activations and FP8 block-quantized weights."""

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8 import (
    MegaMoeFp8Executor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.fp8_fp4_router import (
    Fp8Fp4Router,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)


class MegaMoeFp8Router(Fp8Fp4Router):
    """Reuse the fused executor's passthrough routing contract."""

    @classmethod
    def check_conditions(cls, checker, config):
        checker.check(MoeConfigResolver().get_quant_method(config) == "FP8_PER_BLOCK")


class CudaMegaMoeFp8Strategy(MoeStrategy):
    strategy_name = "mega_moe_fp8"
    supported_moe_quant_method = "FP8_PER_BLOCK"

    @classmethod
    def check_conditions(cls, checker, config):
        checker.check(config.moe_strategy == cls.strategy_name)
        checker.check(MoeConfigResolver().get_quant_method(config) == "FP8_PER_BLOCK")

    def get_attributes(self):
        return StrategyAttributes(
            router_class=MegaMoeFp8Router,
            executor_class=MegaMoeFp8Executor,
            quant_config=FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn, block_shape=[128, 128]
            ),
        )
