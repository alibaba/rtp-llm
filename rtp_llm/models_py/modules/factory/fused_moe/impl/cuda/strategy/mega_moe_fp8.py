"""Explicit opt-in strategies for BF16 activations and FP8 block-quantized weights."""

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
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
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8_se import (
    MegaMoeFp8SEExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.fp8_fp4_router import (
    Fp8Fp4Router,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)


class MegaMoeFp8Router(Fp8Fp4Router):
    """Pass routing tensors, or raw gate scores, to the fused FP8 executor."""

    @classmethod
    def check_conditions(cls, checker, config):
        checker.check(MoeConfigResolver().get_quant_method(config) == "FP8_PER_BLOCK")
        checker.check(config.ep_size > 1)

    @property
    def supports_gate_pack(self) -> bool:
        return True

    def prepare_gate_pack(self, a1, gate_payload):
        return ExpertForwardPayload(
            expert_x=a1,
            expert_x_origin_dtype=a1.dtype,
            gate_payload=gate_payload,
        )


class _CudaMegaMoeFp8Strategy(MoeStrategy):
    supported_moe_quant_method = "FP8_PER_BLOCK"
    requires_shared = False

    @classmethod
    def get_executor_class(cls):
        raise NotImplementedError

    @classmethod
    def check_conditions(cls, checker, config):
        checker.check(config.moe_strategy == cls.strategy_name)
        checker.check(MoeConfigResolver().get_quant_method(config) == "FP8_PER_BLOCK")
        if cls.requires_shared:
            checker.check(getattr(config, "n_shared_experts", 0) == 1)
            checker.check(bool(getattr(config, "has_shared_expert_gate", False)))

    def get_attributes(self):
        return StrategyAttributes(
            router_class=MegaMoeFp8Router,
            executor_class=self.get_executor_class(),
            quant_config=FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn, block_shape=[128, 128]
            ),
        )


class CudaMegaMoeFp8Strategy(_CudaMegaMoeFp8Strategy):
    strategy_name = "mega_moe_fp8"

    @classmethod
    def get_executor_class(cls):
        return MegaMoeFp8Executor


class CudaMegaMoeFp8SEStrategy(_CudaMegaMoeFp8Strategy):
    strategy_name = "mega_moe_fp8_se"
    requires_shared = True

    @classmethod
    def get_executor_class(cls):
        return MegaMoeFp8SEExecutor
