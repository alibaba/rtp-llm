from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import prod
from typing import Any, Dict, Optional, final

import torch
from libth_transformer import rtp_llm_ops  # type: ignore

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules import FusedMoEQuantConfig, resize_cache


def _moe_problem_size(
    a1: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    """
    Extract the MoE problem size from the given tensor arguments:
    - a: The hidden states, input to the MoE layer.
    - w1: The first set of expert weights.
    - w2: The second set of expert weights.
    - topk_ids: The topk ids.

    Note: extracting the problem shape from the weight and activation tensors is
    not obvious.  It needs to be done this way specifically due to subtle issues
    with particular kernels, e.g. the int4 kernels divide the trailing dimension
    by two, so it's not "correct" to extract N or K from the trailing dimension
    of w1 or w2.  Similarly, some kernels transpose the weights, so this needs
    to be kept in mind.
    """
    assert w1.dim() == 3 and w2.dim() == 3
    E, N, _ = w1.size()
    K = w2.size(1)

    if a1.dim() == 2:
        # Make sure we are using the correct a1 (pre-permute).
        assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
        M = a1.size(0)
    else:
        assert a1.dim() == 3
        assert a1.size(0) == E, f"{a1.size(0)} == {E}"
        M = a1.size(1)  # type: ignore This is max_num_tokens

    assert topk_ids.dim() == 2
    topk = topk_ids.size(1)

    return E, M, N, K, topk


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
        topk_ids: torch.Tensor,
        num_experts: int,
        quant_config: FusedMoEQuantConfig,
    ) -> ExpertForwardPayload:
        raise NotImplementedError

    @abstractmethod
    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: TopKWeightAndReduce,
        extra_finalize_args: Optional[dict[str, Any]],
    ) -> None:
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

    @abstractmethod
    def workspace_shapes(
        self,
        a: torch.Tensor,
        aq: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: Optional[ExpertTokensMetadata],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype]:
        """
        Compute the shapes for the temporary and final outputs of the two gemms
        and activation in the fused expert function.  Since the gemms are
        independent, the workspace for the first gemm can be shared with the
        workspace for the last gemm.

        Returns a tuple of:
        - workspace13 shape tuple: must be large enough to hold the
          result of either expert gemm.
        - workspace2 shape tuple: must be large enough to hold the
          result of the activation function.
        - output shape tuple: must be exact size of the final gemm output.
        - Workspace type: The dtype to use for the workspace tensors.
        - Note: in order for activation chunking to work, the first dimension
          of each tuple must be the number of tokens.
        """
        raise NotImplementedError

    def activation(
        self, activation: str, output: torch.Tensor, input: torch.Tensor
    ) -> None:
        if activation == "SiGLU":
            stream_id = torch.cuda.current_stream().cuda_stream
            rtp_llm_ops.silu_and_mul(output, input, stream_id)
        else:
            raise ValueError(f"Unsupported activation type: {activation}")

    def finalize_weight_and_reduce_impl(self) -> TopKWeightAndReduce:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        output: torch.Tensor,
        payload: ExpertForwardPayload,
        w1: torch.Tensor,
        w2: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ):
        raise NotImplementedError


@final
class FusedMoe(torch.nn.Module):
    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        router: FusedMoeDataRouter,
        fused_experts: FusedMoeExpertExecutor,
    ):
        super().__init__()
        self.router = router
        self.fused_experts = fused_experts

    def _do_fused_experts(
        self,
        fused_out: Optional[torch.Tensor],
        a1: torch.Tensor,
        payload: ExpertForwardPayload,
        w1: torch.Tensor,
        w2: torch.Tensor,
        activation: str,
        global_num_experts: int,
        local_num_experts: int,
        expert_map: Optional[torch.Tensor],
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        assert (
            payload.expert_topk_ids is not None
        ), "expert_topk_ids must be provided in the payload"
        _, M, N, K, top_k = _moe_problem_size(
            payload.expert_x, w1, w2, payload.expert_topk_ids
        )

        (workspace13_shape, workspace2_shape, fused_out_shape, workspace_dtype) = (
            self.fused_experts.workspace_shapes(
                a1,
                payload.expert_x,
                M,
                N,
                K,
                top_k,
                global_num_experts,
                local_num_experts,
                payload.expert_tokens_meta,
            )
        )

        workspace13 = torch.empty(
            prod(workspace13_shape), device=a1.device, dtype=workspace_dtype
        )
        workspace2 = torch.empty(
            prod(workspace2_shape), device=a1.device, dtype=workspace_dtype
        )

        assert (
            fused_out is None or fused_out.shape == fused_out_shape
        ), f"fused_out {fused_out.shape} but expected {fused_out_shape}"

        if fused_out is None:
            # reuse workspace13 for the output
            fused_out = resize_cache(workspace13, fused_out_shape)

        self.fused_experts.apply(
            fused_out,
            payload,
            w1,
            w2,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=w1_zp,
            w2_zp=w2_zp,
            a2_scale=a2_scale,
            workspace13=workspace13,
            workspace2=workspace2,
            apply_router_weight_on_input=apply_router_weight_on_input,
            extra_expert_args=extra_expert_args,
        )

        return fused_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        inplace: bool = False,
        activation: str = "silu",
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        w1_zp: Optional[torch.Tensor] = None,
        w2_zp: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        extra_expert_args: Optional[Dict[str, Any]] = None,
        extra_prepare_args: Optional[Dict[str, Any]] = None,
        extra_finalize_args: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        a1 = hidden_states
        output = torch.zeros_like(a1)

        local_num_experts = w1.size(0)
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
            fused_out = self._do_fused_experts(
                fused_out,
                a1=a1,
                payload=payload,
                w1=w1,
                w2=w2,
                activation=activation,
                global_num_experts=global_num_experts,
                local_num_experts=local_num_experts,
                expert_map=expert_map,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_zp=w1_zp,
                w2_zp=w2_zp,
                a2_scale=a2_scale,
                apply_router_weight_on_input=apply_router_weight_on_input,
                extra_expert_args=extra_expert_args,
            )

        self.router.finalize(
            output,
            fused_out,
            payload.expert_topk_weights,
            payload.expert_topk_ids,
            apply_router_weight_on_input,
            self.fused_experts.finalize_weight_and_reduce_impl(),
            extra_finalize_args,
        )

        return output
