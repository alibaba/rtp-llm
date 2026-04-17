"""Torch distributed-based EP strategy for FP4 quantized weights."""

import os
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


class TorchDistEpFp4Strategy(MoeStrategy):
    """ROCm EP mode with FP4 per-group quantized weights.

    Combines TorchDistEpRouter (all_to_all dispatch/combine) with
    RocmExpertsFp4PerGroup executor (aiter fused_moe kernel).
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()

        # Must have USE_TORCH_DIST_EP=1
        checker.check(os.environ.get("USE_TORCH_DIST_EP", "0") == "1")

        # Must be EP enabled
        checker.check(resolver.is_ep_enabled(config))

        # Must be FP4 quantization
        quant_method = resolver.get_quant_method(config)
        checker.check(
            quant_method
            in (
                "FP4_PER_GROUP",
                "FP4_PER_GROUP_QUARK",
                "modelopt_fp4",
            )
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.rocm_moe import (
            RocmExpertsFp4PerGroup,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.torch_dist_ep_router import (
            TorchDistEpRouter,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float4_e2m1fn_x2,
            per_act_token_quant=False,
            per_out_ch_quant=False,
            block_shape=None,
        )
        return StrategyAttributes(
            router_class=TorchDistEpRouter,
            executor_class=RocmExpertsFp4PerGroup,
            quant_config=quant_config,
        )
