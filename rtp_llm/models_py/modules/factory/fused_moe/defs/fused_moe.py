from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, final

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import (
    ExecutorType,
    RouterType,
)


@dataclass
class ExpertTokensMetadata:
    """
    Metadata regarding expert-token routing.
    """

    expected_m: Optional[int] = None
    expert_num_tokens: Optional[torch.Tensor] = None
    expert_num_tokens_cpu: Optional[Union[List[int], torch.Tensor]] = None


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


@dataclass
class CombineForwardPayload:
    """
    Represents the data payload for combining the expert outputs.
    """

    fused_expert_output: torch.Tensor


class FusedMoeDataRouter(ABC):
    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        """Initialize FusedMoeDataRouter with standard parameters.

        Args:
            config: MOE configuration adapter
            quant_config: Quantization configuration
        """
        self.config = config
        self.quant_config = quant_config

    @classmethod
    def router_type(cls) -> RouterType:
        raise NotImplementedError

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if this router can handle the given configuration.

        Subclasses should override this method to check router-specific conditions.

        Args:
            checker: ConditionChecker instance from MoeStrategy
            config: Model initialization parameters
        """
        raise NotImplementedError

    @abstractmethod
    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        raise NotImplementedError

    @abstractmethod
    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        raise NotImplementedError


class FusedMoeExpertExecutor(ABC):
    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        """Initialize FusedMoeExpertExecutor with standard parameters.

        Args:
            config: MOE configuration adapter
            quant_config: Quantization configuration
            weights: Model weights dictionary
        """
        self.config = config
        self.quant_config = quant_config
        self.weights = weights

    @classmethod
    def executor_type(cls) -> ExecutorType:
        raise NotImplementedError

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if this executor can handle the given configuration.

        Subclasses should override this method to check executor-specific conditions.

        Args:
            checker: ConditionChecker instance from MoeStrategy
            config: Model initialization parameters
        """
        pass

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int64

    @abstractmethod
    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
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

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return self.fused_experts.topk_ids_dtype

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

        expert_payload = self.router.prepare(
            a1,
            a1_scale,
            a2_scale,
            topk_weights,
            topk_ids,
        )

        if expert_payload.expert_topk_ids is None:
            expert_payload.expert_topk_ids = topk_ids
        if expert_payload.expert_topk_weights is None:
            expert_payload.expert_topk_weights = topk_weights

        if expert_payload.expert_x.numel() == 0:
            # This happens when none of the tokens from the all2all reach this
            # EP rank. Also, note that this is only relevant for CUDAGraph
            # incompatible all2all kernels like the DeepEP high-throughput
            # kernels. CUDAGraph compatible all2all kernels like the pplx
            # kernels and the DeepEP low-latency kernels are always batched
            # and can never run into the tensor.numel() == 0 case.
            combine_payload = CombineForwardPayload(
                fused_expert_output=torch.empty_like(
                    expert_payload.expert_x, dtype=a1.dtype
                )
            )
        else:
            combine_payload = self.fused_experts.execute(
                expert_payload,
                activation=activation,
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

        extra_finalize_args.update({"original_num_tokens": hidden_states.size(0)})

        output = self.router.finalize(
            combine_payload,
            expert_payload.expert_topk_weights,
            expert_payload.expert_topk_ids,
            apply_router_weight_on_input,
            extra_finalize_args,
        )

        assert (
            output.shape == hidden_states.shape
        ), f"output batch size mismatch: expected {hidden_states.shape}, got {output.shape}"

        return output
