from dataclasses import dataclass
from typing import Any, Optional

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FinalizeArgs,
    FusedMoeDataRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import RouterType


@dataclass(frozen=True, slots=True)
class _BatchedRoutingPlan:
    packed_rows: torch.Tensor
    routed: torch.Tensor


class BatchedDataRouter(FusedMoeDataRouter):
    """Router for the batched expert-output layout.

    Keeps its own TP all-reduce in ``finalize``: its batched combine contract is
    not the pure-TP partial-output contract that GenericMoeLayer's unified
    shared-expert reduction needs, so it does not advertise deferral.

    The expert block is ``local_experts x num_tokens x hidden``.
    """

    @classmethod
    def router_type(cls):
        return RouterType.BATCHED_DATA

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(not resolver.has_quantization(config))
        checker.check(resolver.is_tp_equal_ep(config))

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config)

        self.ep_rank = config.ep_rank
        self.num_local_experts = config.expert_num // config.ep_size

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        assert (
            a1.dim() == 2 and topk_weights.dim() == 2 and topk_ids.dim() == 2
        ), "a1, topk_weights, and topk_ids must be rank-2 tensors"
        assert a1.size(0) == topk_ids.size(
            0
        ), "a1 and topk_ids must have the same token count"
        assert (
            topk_weights.shape == topk_ids.shape
        ), "topk_weights and topk_ids must have the same shape"
        assert (
            a1_scale is None and a2_scale is None
        ), "BatchedDataRouter does not support quantized MoE"

        num_tokens = a1.size(0)
        num_experts = self.num_local_experts
        token_ids = torch.arange(num_tokens, device=a1.device, dtype=torch.int32)

        # Shapes below depend only on num_tokens, keeping the plan CUDA-Graph
        # capturable: no boolean indexing, no device-to-host expert counts.
        slots = topk_ids.to(dtype=torch.int64, copy=True)
        slots.sub_(num_experts * self.ep_rank)
        routed = (slots >= 0) & (slots < num_experts)
        # Column num_experts is scratch: non-local ids cannot hit a real column.
        slots.masked_fill_(~routed, num_experts)
        match = torch.zeros(
            (num_tokens, num_experts + 1), dtype=torch.bool, device=a1.device
        ).scatter_(1, slots, routed)
        # Unrouted slots point at placeholder row 0 and are masked in finalize.
        seen = match.cumsum(0, dtype=torch.int32)
        packed_rows = torch.where(
            routed, slots * num_tokens + torch.gather(seen, 1, slots) - 1, 0
        )

        # Initialize padding with valid token ids, then overwrite the packed
        # live rows. The extra row absorbs every non-local slot.
        token_indices = token_ids.expand(num_experts + 1, -1).clone()
        positions = torch.where(routed, packed_rows, num_experts * num_tokens)
        token_indices.view(-1).scatter_(
            0,
            positions.reshape(-1),
            token_ids.view(-1, 1).expand_as(positions).reshape(-1),
        )
        expert_num_tokens = match[:, :num_experts].sum(0, dtype=torch.int32)
        return ExpertForwardPayload(
            expert_x=a1[token_indices[:num_experts]],
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=expert_num_tokens
            ),
            router_context=_BatchedRoutingPlan(packed_rows, routed),
        )

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[FinalizeArgs],
    ) -> torch.Tensor:
        plan = payload.router_context
        if not isinstance(plan, _BatchedRoutingPlan):
            raise TypeError("BatchedDataRouter requires its prepared routing context")
        packed_rows, routed = plan.packed_rows, plan.routed

        expert_output = payload.fused_expert_output
        num_tokens, top_k = packed_rows.size()
        expected_shape = (self.num_local_experts, num_tokens)
        assert expert_output.shape[:2] == expected_shape, (
            f"Expected expert block {expected_shape}, "
            f"got {tuple(expert_output.shape[:2])}"
        )

        hidden_dim = expert_output.size(-1)
        rows = (
            expert_output.reshape(-1, hidden_dim)
            .index_select(0, packed_rows.reshape(-1))
            .view(num_tokens, top_k, hidden_dim)
        )
        # Mask before any arithmetic: unrouted slots gathered the placeholder
        # row 0, which the executor may never have written.
        rows.masked_fill_(~routed.unsqueeze(-1), 0)
        if not apply_router_weight_on_input:
            rows.mul_(topk_weights.to(rows.dtype).unsqueeze(-1))
        # Reducing over top-k, not experts, fixes the float accumulation order.
        output = rows.sum(1)

        if self.tp_size > 1:
            output = all_reduce(output, Group.TP)
        return output
