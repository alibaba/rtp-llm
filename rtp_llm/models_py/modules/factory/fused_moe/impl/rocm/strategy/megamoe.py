"""ROCm FlyDSL 2-stage fused MegaMoE strategy."""

import logging
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

logger = logging.getLogger(__name__)


class RocmMegaMoeStrategy(MoeStrategy):
    """ROCm FlyDSL 2-stage fused MegaMoE (gfx942 plain FP8).

    Gated on the USE_MEGAMOE env toggle (surfaced as config.use_megamoe) plus
    EP being enabled and FlyDSL being importable. When active it has the
    highest priority (router MEGAMOE=8, executor MEGAMOE_FUSED=9).
    """

    def can_handle(self, config: MoEConfigAdapter) -> bool:
        self._config = config
        return super().can_handle(config)

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        checker.check(bool(getattr(config, "use_megamoe", False)))

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm._utils import (
            get_rocm_fp8_dtype,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.megamoe_executor import (
            MegaMoeFusedExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.megamoe_router import (
            MegaMoePassthroughRouter,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=get_rocm_fp8_dtype(),
            per_act_token_quant=True,
            per_out_ch_quant=False,
            block_shape=None,
        )

        return StrategyAttributes(
            router_class=MegaMoePassthroughRouter,
            executor_class=MegaMoeFusedExecutor,
            quant_config=quant_config,
        )
