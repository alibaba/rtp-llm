from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, final

import torch

from rtp_llm.models_py.modules.moe.utils import FusedMoEQuantConfig


@dataclass
class ExpertTokensMetadata:
    """
    Metadata regarding expert-token routing.
    """

    expert_num_tokens: Optional[torch.Tensor]
    expert_num_tokens_cpu: Optional[Union[List[int], torch.Tensor]]


@dataclass
class ExpertForwardPayload:
    """
    Represents the data payload dispatched to experts for computation.
    """

    expert_x: torch.Tensor
    expert_x_origin_dtype: Optional[torch.dtype] = None
    expert_x_scale: Optional[torch.Tensor] = None
    expert_tokens_meta: Optional[ExpertTokensMetadata] = None
    expert_topk_ids: Optional[torch.Tensor] = None
    expert_topk_weights: Optional[torch.Tensor] = None


class TopKWeightAndReduce(ABC):
    @abstractmethod
    def apply(
        self,
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
        weight_and_reduce_impl: Union[TopKWeightAndReduce, None],
        extra_finalize_args: Optional[Dict[str, Any]],
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

    def finalize_weight_and_reduce_impl(self) -> Union[TopKWeightAndReduce, None]:
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
        expert_num: int,
    ):
        super().__init__()
        self.router = router
        self.fused_experts = fused_experts
        self.expert_num = expert_num

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        inplace: bool = False,
        activation: str = "silu",
        expert_map: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        extra_expert_args: Optional[Dict[str, Any]] = None,
        extra_finalize_args: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        a1 = hidden_states

        payload = self.router.prepare(
            a1,
            a1_scale,
            a2_scale,
            topk_weights,
            topk_ids,
            self.expert_num,
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
                global_num_experts=self.expert_num,
                expert_map=expert_map,
                a2_scale=a2_scale,
                apply_router_weight_on_input=apply_router_weight_on_input,
                extra_expert_args=extra_expert_args,
            )

        # pass a1.shape to finalize for shape check
        if extra_finalize_args is None:
            extra_finalize_args = {"a1_shape": a1.shape}
        else:
            extra_finalize_args.update({"a1_shape": a1.shape})

        output = self.router.finalize(
            fused_out,
            payload.expert_topk_weights,
            payload.expert_topk_ids,
            apply_router_weight_on_input,
            (
                None
                if self.fused_experts.finalize_weight_and_reduce_impl is None
                else self.fused_experts.finalize_weight_and_reduce_impl()
            ),
            extra_finalize_args,
        )

        assert output.shape == hidden_states.shape

        return output
