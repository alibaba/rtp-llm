import logging
from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.distributed.megamoe_wrapper import MegaMoeWrapper
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoeDataRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import RouterType


class MegaMoePassthroughRouter(FusedMoeDataRouter):
    """Passthrough router for the FlyDSL 2-stage fused MegaMoE.

    The FlyDSL FusedMoEZeroCopyFp8 executor performs the *entire* MoE pipeline
    internally (dispatch + GEMM1 fused, then GEMM2 + combine fused). The
    rtp-llm prepare/execute/finalize boundary therefore cannot map onto the
    FlyDSL fusion boundary. This router hands the raw (undispatched) tokens and
    the *global* routing tables straight to the executor, and finalize simply
    returns the already-combined output.
    """

    @classmethod
    def router_type(cls) -> RouterType:
        return RouterType.MEGAMOE

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(bool(getattr(config, "use_megamoe", False)))
        checker.check(resolver.is_ep_enabled(config))
        checker.check(MegaMoeWrapper.supported())

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config)
        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        logging.debug(
            "[MegaMoePassthroughRouter] prepare tokens=%d ep_rank=%d",
            a1.shape[0],
            self.ep_rank,
        )
        # Pass raw tokens + global routing straight through; the FlyDSL executor
        # owns dispatch/GEMM/combine end-to-end.
        return ExpertForwardPayload(
            expert_x=a1,
            expert_x_scale=None,
            expert_x_origin_dtype=a1.dtype,
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
            expert_ids_are_local=False,
        )

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        logging.debug("[MegaMoePassthroughRouter] finalize ep_rank=%d", self.ep_rank)
        out = payload.fused_expert_output
        if (
            extra_finalize_args is not None
            and "original_num_tokens" in extra_finalize_args
        ):
            original_num_tokens = extra_finalize_args["original_num_tokens"]
            if out.shape[0] > original_num_tokens:
                out = out[:original_num_tokens]
        return out
