"""Ascend NPU W8A8_MXFP8 MoE strategy (W8A8_MXFP8: Weight & Activation FP8 Quantization).

Flat router (NpuPureTpRouter semantics) + CANN-native executor: the router
passes the un-permuted hidden states and top-k ids/weights straight through
and the executor performs the fused dispatch
(``npu_moe_init_routing_v2``) / combine (``npu_moe_token_unpermute``), so
no torch bucketing / boolean-mask compaction / index_add runs on the hot
path.  The activation MX quantization happens inside the executor, so
``a1_scale/a2_scale`` stay None through ``prepare()``.

The existing ``AscendBf16FallbackStrategy`` requires "no quantization", so
without this strategy a quantized config would find no candidate and
``StrategyRegistry.get_strategy`` would raise.
"""

from typing import Any, Optional

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoeDataRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import RouterType


class AscendW8A8MXFP8MoeRouter(FusedMoeDataRouter):
    """Flat router for the MXFP8 CANN executor (single GPU / tp==ep).

    ``prepare`` only validates and forwards the flat ``[tokens, hidden]``
    activations with the top-k ids/weights; the fused dispatch/combine and
    the router-weight application happen inside the executor
    (init_routing_v2 / token_unpermute), so ``finalize`` reduces to the TP
    collective.
    """

    @classmethod
    def router_type(cls) -> RouterType:
        return RouterType.PURE_TP

    def __init__(self, config: Any, quant_config: FusedMoEQuantConfig) -> None:
        super().__init__(config, quant_config)
        self.tp_size = config.tp_size

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        if a1_scale is not None or a2_scale is not None:
            raise NotImplementedError(
                "Ascend W8A8_MXFP8 MoE does not take activation scales: "
                "the MX quantization happens inside the executor"
            )
        if a1.dim() != 2 or not a1.is_contiguous():
            raise ValueError(
                "Ascend W8A8_MXFP8 MoE input must be a contiguous "
                "[tokens, hidden] tensor"
            )
        if topk_weights.shape != topk_ids.shape:
            raise ValueError("topk_weights and topk_ids must have identical shapes")
        if topk_ids.size(0) != a1.size(0):
            raise ValueError("top-k token dimension does not match hidden states")
        if topk_ids.dtype != torch.int32:
            raise TypeError(
                f"Ascend W8A8_MXFP8 MoE requires int32 topk_ids, got {topk_ids.dtype}"
            )
        if topk_weights.dtype != torch.float32:
            raise TypeError(
                "Ascend W8A8_MXFP8 MoE requires float32 topk_weights, "
                f"got {topk_weights.dtype}"
            )

        return ExpertForwardPayload(
            expert_x=a1,
            expert_x_origin_dtype=a1.dtype,
            expert_x_scale=None,
            expert_tokens_meta=ExpertTokensMetadata(),
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
        )

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "Ascend W8A8_MXFP8 MoE applies router weights in token unpermute"
            )
        output = payload.fused_expert_output
        if self.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        # Unlike the pure-TP bf16 router, quantization IS expected here
        # (ModelSlim / W8A8_MXFP8); activation quantization happens inside the
        # executor so prepare()'s a1_scale/a2_scale=None assertion still holds.
        checker.check(
            resolver.get_quant_method(config) in ("ASCEND_W8A8_MXFP8",)
        )
        checker.check(resolver.is_single_gpu(config) or resolver.is_tp_equal_ep(config))


class AscendW8A8MXFP8MoeStrategy(MoeStrategy):
    """Ascend W8A8_MXFP8 MoE strategy."""

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.ascend.executors.w8a8_mxfp8_executor import (
            AscendW8A8MXFP8Executor,
        )

        return StrategyAttributes(
            router_class=AscendW8A8MXFP8MoeRouter,
            executor_class=AscendW8A8MXFP8Executor,
            quant_config=FusedMoEQuantConfig(quant_dtype=None),
        )

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(resolver.is_bf16(config))
        # Static (ModelSlim) path only in this stage; the dynamic load-quant
        # path (FP8_PER_BLOCK) is enabled by a follow-up TODO.
        checker.check(resolver.get_quant_method(config) in ("ASCEND_W8A8_MXFP8",))
