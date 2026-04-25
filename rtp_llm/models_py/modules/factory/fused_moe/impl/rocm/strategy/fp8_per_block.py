"""Rocm FP8 PerBlock quantization strategies"""

from typing import Any, Dict

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    FusedMoeDataRouter,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy


class RocmFp8PerBlockPureTPStrategy(MoeStrategy):
    """Rocm FP8 PerBlock pure TP strategy"""

    def _get_block_shape(self, config: MoEConfigAdapter) -> list[int]:
        model_quant_config = config.model_config.quant_config
        if model_quant_config is not None and hasattr(model_quant_config, "group_size"):
            gs = model_quant_config.group_size()
            return [gs, gs]
        return [128, 128]

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.rocm_moe import (
            RocmExpertsFp8PerBlock,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.pure_tp_router import (
            PureTpRouterFp8PerBlockPassthrough,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fnuz,
            per_act_token_quant=False,
            per_out_ch_quant=False,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=PureTpRouterFp8PerBlockPassthrough,
            executor_class=RocmExpertsFp8PerBlock,
            quant_config=quant_config,
        )
