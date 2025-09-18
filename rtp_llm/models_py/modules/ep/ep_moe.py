"""
EP MoE implementation with clean router+executor pattern.
This module provides a standardized interface similar to FusedMoE.
"""

from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.moe.fused_moe import (
    ExpertForwardPayload,
    FusedMoeDataRouter,
    FusedMoeExpertExecutor,
    TopKWeightAndReduce,
)


class EPDataRouter(FusedMoeDataRouter):
    """EP implementation of data router for expert selection and data dispatch."""

    def __init__(self, config: GptInitModelParameters):
        self.config = config
        self.top_k = config.moe_k

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        quant_config,
    ) -> ExpertForwardPayload:
        """Prepare data for expert computation."""
        return ExpertForwardPayload(
            expert_x=a1,
            expert_x_scale=a1_scale,
            expert_tokens_meta=None,
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
        )

    def finalize(
        self,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: TopKWeightAndReduce,
        extra_finalize_args: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        """Finalize expert outputs and apply routing weights."""
        return fused_expert_output


class EPExpertExecutor(FusedMoeExpertExecutor):
    """EP implementation of expert executor."""

    def __init__(self, config: GptInitModelParameters, weights, layer_idx: int):
        super().__init__(None)
        self.config = config
        # Use the proven legacy implementation for computation
        from rtp_llm.models_py.modules.ep.layers import LegacyEPMoE

        self.legacy_ep_moe = LegacyEPMoE(config, weights, layer_idx)

    @property
    def local_num_experts(self) -> int:
        """Number of experts in this executor."""
        return self.legacy_ep_moe.num_experts

    def finalize_weight_and_reduce_impl(self) -> TopKWeightAndReduce:
        """Get the weight and reduce implementation."""
        from rtp_llm.models_py.modules.moe.topk_weight_and_reduce import (
            TopKWeightAndReduceNaiveBatched,
        )

        return TopKWeightAndReduceNaiveBatched(rank=self.config.tp_rank)

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        global_num_experts: int,
        expert_map,
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Dict[str, Any],
    ) -> torch.Tensor:
        """Execute expert computation using legacy implementation."""
        hidden_states = payload.expert_x
        router_logits = extra_expert_args["router_logits"]
        return self.legacy_ep_moe(hidden_states, router_logits)


class EPMoE(nn.Module):
    """
    Expert Parallel MoE with router+executor interface.
    This provides a clean interface that delegates to the proven legacy implementation.
    """

    def __init__(self, router: EPDataRouter, executor: EPExpertExecutor):
        super().__init__()
        self.router = router
        self.executor = executor

    def forward(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass using router+executor pattern."""
        from rtp_llm.models_py.modules.ep.topk import select_experts

        # Router: expert selection
        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=self.router.top_k,
            use_grouped_topk=False,
            renormalize=True,
        )

        # Router: prepare payload
        payload = self.router.prepare(
            hidden_states,
            None,
            None,
            topk_weights,
            topk_ids,
            self.executor.local_num_experts,
            None,
        )

        # Executor: run computation
        output = self.executor.execute(
            payload,
            "silu",
            self.executor.local_num_experts,
            None,
            None,
            False,
            {"router_logits": router_logits},
        )

        # Router: finalize (currently passthrough)
        weight_and_reduce_impl = self.executor.finalize_weight_and_reduce_impl()
        return self.router.finalize(
            output, topk_weights, topk_ids, False, weight_and_reduce_impl, None
        )


def create_ep_moe_instance(
    config: GptInitModelParameters, weights, layer_idx: int
) -> EPMoE:
    """
    Factory function for creating EP MoE instances.

    Args:
        config: Model configuration parameters
        weights: Model weights
        layer_idx: Layer index

    Returns:
        EPMoE instance with router and executor
    """
    router = EPDataRouter(config)
    executor = EPExpertExecutor(config, weights, layer_idx)
    return EPMoE(router=router, executor=executor)
