"""MoRI EP router + FP4 per-group executor strategy (ROCm).

Added 2026-04-21 by a debugging session to enable launching rtp-llm with
`QUANTIZATION=FP4_PER_GROUP_QUARK` together with `USE_MORI_EP=1`.

Background
----------
Before this file existed, no registered strategy covered the combination
(MoRI intranode EP router, FP4-per-group executor):

- ``RocmEpNormalStrategy`` always pairs its routers (DeepEP / MoRI) with
  ``RocmExpertsBf16``, whose ``check_conditions`` requires
  ``quant_method is None``, so it rejects every FP4 config.
- ``RocmFp4PerGroupPureTPStrategy`` is pure-TP and uses
  ``PureTpRouterBase`` which requires ``use_all_gather=True``; the FP4 launch
  script sets ``USE_ALL_GATHER=0`` because we do want EP-style dispatch.
- ``TorchDistEpFp4Strategy`` only applies when ``USE_TORCH_DIST_EP=1``, which
  is NOT the case for MoRI launches.

Result: ``StrategyRegistry.get_strategy`` raised ``ValueError: No suitable
MOE strategy found for configuration`` on every rank and the server died
during startup.

Fix
---
This strategy pairs :class:`MoriEpIntranodeRouter` with
:class:`RocmExpertsFp4PerGroup`. The MoRI router leaves the activation
in bf16 (its ``prepare`` does not quantize — it only performs
dispatch/combine), exactly like ``TorchDistEpRouter`` in the torch-dist
FP4 strategy, so the FP4 executor can quantize activations internally in
the same way.
"""

import logging
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

logger = logging.getLogger(__name__)


class MoriEpFp4Strategy(MoeStrategy):
    """ROCm EP mode combining MoRI intranode router with FP4-per-group executor.

    Only applicable when ``USE_MORI_EP=1`` is set in the environment, EP is
    enabled (``ep_size > 1``) and ``quant_method`` is one of the FP4 per-group
    variants (``FP4_PER_GROUP``, ``FP4_PER_GROUP_QUARK`` or ``modelopt_fp4``).
    The actual ``can_handle`` also asks the router / executor classes to
    validate their own conditions (``MoriEpIntranodeRouter`` needs mori to be
    available; ``RocmExpertsFp4PerGroup`` double-checks the quant method).
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()

        use_mori_ep = os.environ.get("USE_MORI_EP", "0").lower() in ("1", "true", "on")
        checker.check(use_mori_ep)

        checker.check(resolver.is_ep_enabled(config))

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
        from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.mori_ep_intranode_router import (
            MoriEpIntranodeRouter,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float4_e2m1fn_x2,
            per_act_token_quant=False,
            per_out_ch_quant=False,
            block_shape=None,
        )
        return StrategyAttributes(
            router_class=MoriEpIntranodeRouter,
            executor_class=RocmExpertsFp4PerGroup,
            quant_config=quant_config,
        )
