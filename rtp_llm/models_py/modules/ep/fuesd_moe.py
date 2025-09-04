from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, final

import torch


# from rtp_llm.models_py.modules.moe.utils import FusedMoEQuantConfig
# TODO(baowending.bwd): remove it later
class FusedMoEQuantConfig:
    pass


@dataclass
class ExpertTokensMetadata:
    """
    Metadata regarding expert-token routing.
    """

    expert_num_tokens: torch.Tensor
    expert_num_tokens_cpu: Optional[torch.Tensor]


@dataclass
class ExpertForwardPayload:
    """
    Represents the data payload dispatched to experts for computation.
    """

    expert_x: torch.Tensor
    expert_x_scale: Optional[torch.Tensor] = None
    expert_tokens_meta: Optional[ExpertTokensMetadata] = None
    expert_topk_ids: Optional[torch.Tensor] = None
    expert_topk_weights: Optional[torch.Tensor] = None
    extra_finalize_args: Optional[Dict[str, Any]] = None


class TopKWeightAndReduce(ABC):
    @abstractmethod
    def apply(
        self,
        output: Optional[torch.Tensor],
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        raise NotImplementedError


class FusedMoeDataRouter(ABC):
    @abstractmethod
    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        quant_config: FusedMoEQuantConfig,
    ) -> ExpertForwardPayload:
        raise NotImplementedError

    @abstractmethod
    def finalize(
        self,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: TopKWeightAndReduce,
        extra_finalize_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        raise NotImplementedError


class FusedMoeExpertExecutor(ABC):
    def __init__(
        self,
        quant_config: Optional[FusedMoEQuantConfig],
    ):
        if quant_config is not None:
            self.quant_config = quant_config
        else:
            self.quant_config = FusedMoEQuantConfig()

    @property
    def local_num_experts(self) -> int:
        raise NotImplementedError

    def finalize_weight_and_reduce_impl(self) -> TopKWeightAndReduce:
        raise NotImplementedError

    @abstractmethod
    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        raise NotImplementedError


@final
class FusedMoe(torch.nn.Module):
    def __init__(
        self,
        router: FusedMoeDataRouter,
        fused_experts: FusedMoeExpertExecutor,
    ):
        super().__init__()
        self.router = router
        self.fused_experts = fused_experts

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        inplace: bool = False,
        activation: str = "silu",
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        extra_expert_args: Optional[Dict[str, Any]] = None,
        extra_prepare_args: Optional[Dict[str, Any]] = None,
        extra_finalize_args: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        a1 = hidden_states
        output = torch.zeros_like(a1)
        local_num_experts = self.fused_experts.local_num_experts
        if global_num_experts == -1:
            global_num_experts = local_num_experts
        payload = self.router.prepare(
            a1,
            a1_scale,
            a2_scale,
            topk_ids,
            global_num_experts,
            self.fused_experts.quant_config,
        )
        if payload.expert_topk_ids is None:
            payload.expert_topk_ids = topk_ids
        if payload.expert_topk_weights is None:
            payload.expert_topk_weights = topk_weights
        fused_out = None
        if payload.expert_x.numel() == 0:
            # This happens when none of the tokens from the all2all reach this
            # EP rank. Also, note that this is only relevant for CUDAGraph
            # incompatible all2all kernels like the DeepEP high-throughput
            # kernels. CUDAGraph compatible all2all kernels like the pplx
            # kernels and the DeepEP low-latency kernels are always batched
            # and can never run into the tensor.numel() == 0 case.
            fused_out = torch.empty_like(payload.expert_x).to(dtype=a1.dtype)
        else:
            fused_out = self.fused_experts.execute(
                payload,
                activation=activation,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
                a2_scale=a2_scale,
                apply_router_weight_on_input=apply_router_weight_on_input,
                extra_expert_args=extra_expert_args,
            )
        output = self.router.finalize(
            output,
            fused_out,
            payload.expert_topk_weights,
            payload.expert_topk_ids,
            apply_router_weight_on_input,
            self.fused_experts.finalize_weight_and_reduce_impl(),
            extra_finalize_args,
        )
        return output
