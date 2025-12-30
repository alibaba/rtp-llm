# Note: A simple non-batched FusedMoeDataRouter implementation temporary, perhaps delete this file in the future.

from typing import Any, Optional

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoeDataRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import RouterType


def normalize_scales_shape(scales: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if scales is not None:
        if scales.numel() == 1:
            scales = scales.view(1, 1)
        else:
            scales = scales.view(-1, scales.size(-1))
    return scales


class TopKWeightAndReduceNaiveBatched(object):
    def __init__(self, rank: int):
        self.rank = rank

    def apply(
        self,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        assert fused_expert_output.ndim == 3
        num_tokens = topk_ids.size(0)
        num_local_experts = fused_expert_output.size(0)
        K = fused_expert_output.size(-1)

        output = torch.zeros(
            (num_tokens, K),
            device=fused_expert_output.device,
            dtype=fused_expert_output.dtype,
        )

        assert output.size() == (
            num_tokens,
            K,
        ), f"Expected output size {(num_tokens, K)}, but got {output.size()}"

        first_expert = num_local_experts * self.rank
        last_expert = first_expert + num_local_experts

        for expert_id in range(first_expert, last_expert):
            matching_tokens = topk_ids == expert_id
            topks = torch.any(matching_tokens, dim=1).flatten()
            rows = torch.count_nonzero(matching_tokens)
            rhs = fused_expert_output[expert_id - first_expert, :rows, :]
            if not apply_router_weight_on_input:
                rhs.mul_(topk_weights[matching_tokens].view(rhs.size(0), 1))
            output[topks] = output[topks] + rhs

        return output


class BatchedDataRouter(FusedMoeDataRouter):
    @classmethod
    def router_type(cls):
        return RouterType.BATCHED_DATA

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if BatchedDataRouter can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(not resolver.has_quantization(config))

        checker.check(resolver.is_single_gpu(config) or resolver.is_tp_equal_ep(config))

    def __init__(
        self, max_num_tokens: int, num_local_experts: int, ep_rank: int, tp_size: int
    ):
        super().__init__()
        self.max_num_tokens = max_num_tokens
        self.num_local_experts = num_local_experts // tp_size
        self.ep_rank = ep_rank
        self.tp_size = tp_size

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
    ) -> ExpertForwardPayload:
        assert a1.dim() == 2
        assert topk_ids.dim() == 2
        assert a1.size(0) == topk_ids.size(0)
        assert a1_scale is None and a2_scale is None, "not support quanted moe"

        _, hidden_dim = a1.size()

        tokens_per_expert = torch.zeros(
            self.num_local_experts, dtype=torch.int, device=a1.device
        )

        if quant_config.quant_dtype is None:
            b_type = a1.dtype
        else:
            b_type = quant_config.quant_dtype

        assert isinstance(b_type, torch.dtype)

        b_a1 = torch.zeros(
            (self.num_local_experts, self.max_num_tokens, hidden_dim),
            dtype=b_type,
            device=a1.device,
        )

        first_expert = self.num_local_experts * self.ep_rank
        last_expert = first_expert + self.num_local_experts

        for expert_id in range(first_expert, last_expert):
            topks = torch.any(topk_ids == expert_id, dim=1).flatten()
            rows = torch.count_nonzero(topks.flatten())
            if rows == 0:
                continue
            idx = expert_id - first_expert
            tokens_per_expert[idx] = rows
            rhs = a1[topks]
            b_a1[idx, :rows, :] = rhs

        return ExpertForwardPayload(
            expert_x=b_a1,
            expert_x_scale=None,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=tokens_per_expert,
                expert_num_tokens_cpu=None,
            ),
        )

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        weight_and_reduce_impl = TopKWeightAndReduceNaiveBatched(self.ep_rank)
        output = weight_and_reduce_impl.apply(
            fused_expert_output=payload.fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
        if self.tp_size > 1:
            output = all_reduce(output, Group.TP)
        return output
