from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
    final,
)

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

SKIP_TP_ALLREDUCE_ARG: Final[Literal["skip_tp_allreduce"]] = "skip_tp_allreduce"
ROW_SCATTER_TARGET_ARG: Final[Literal["row_scatter_target"]] = "row_scatter_target"


class FinalizeArgs(TypedDict, total=False):
    """Private, optional arguments passed from ``FusedMoe`` to routers."""

    a1_shape: torch.Size
    original_num_tokens: int
    skip_tp_allreduce: bool
    row_scatter_target: torch.Tensor


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
    expert_ids_are_local: bool = False
    router_context: object | None = None


@dataclass
class CombineForwardPayload:
    """
    Represents the data payload for combining the expert outputs.
    """

    fused_expert_output: torch.Tensor
    router_context: object | None = None


def should_skip_tp_allreduce(
    extra_finalize_args: Optional[FinalizeArgs],
) -> bool:
    """Return whether a pure-TP router should leave reduction to its caller.

    GenericMoeLayer uses this internal finalize argument when it combines the
    routed and shared-expert partial outputs before issuing one TP all-reduce.
    Keeping the flag in ``extra_finalize_args`` avoids changing the finalize
    interface for routers that do not use TP all-reduce.
    """

    return bool(
        extra_finalize_args is not None
        and extra_finalize_args.get(SKIP_TP_ALLREDUCE_ARG, False)
    )


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
        # Keep the legacy field for router-local logic. Callers that need the
        # size of the Group.TP collective must use tp_collective_size below.
        self.tp_size = config.tp_size

    @property
    def tp_collective_size(self) -> int:
        """Return the TP size used by this router's Group.TP collective."""

        return self.config.tp_size

    @classmethod
    def router_type(cls) -> RouterType:
        raise NotImplementedError

    @property
    def supports_skip_tp_allreduce(self) -> bool:
        """Whether ``finalize`` consumes ``skip_tp_allreduce``.

        A router must only override this capability when its finalize path
        delegates to the shared skip decision (or an equivalent implementation).
        """
        return False

    @property
    def supports_row_scatter_finalize(self) -> bool:
        """Whether ``finalize`` consumes ``row_scatter_target``.

        Only routers whose combine output is already fully reduced per token,
        and which therefore reassemble the TP token slices with an all_gather,
        can trade that all_gather for a scatter-add into the caller's buffer.
        """
        return False

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
        extra_finalize_args: Optional[FinalizeArgs],
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
        extra_finalize_args: Optional[FinalizeArgs] = None,
        skip_tp_allreduce: bool = False,
        row_scatter_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if skip_tp_allreduce and not self.router.supports_skip_tp_allreduce:
            raise ValueError(
                "skip_tp_allreduce is only supported by routers that "
                "advertise supports_skip_tp_allreduce"
            )

        if (
            row_scatter_target is not None
            and not self.router.supports_row_scatter_finalize
        ):
            raise ValueError(
                "row_scatter_target is only supported by routers that "
                "advertise supports_row_scatter_finalize"
            )

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
        combine_payload.router_context = expert_payload.router_context

        # Finalize arguments are a private per-call protocol. Copy caller
        # input before adding derived values so a reusable dict cannot retain
        # state from a previous forward.
        finalize_args: FinalizeArgs = {
            **(extra_finalize_args or {}),
            "a1_shape": a1.shape,
        }

        # Pure-TP routers normally reduce their routed output in finalize().
        # GenericMoeLayer can set this flag to combine routed and shared-expert
        # partial outputs first, reducing the number of small TP collectives.
        finalize_args.update(
            {
                "original_num_tokens": hidden_states.size(0),
                SKIP_TP_ALLREDUCE_ARG: skip_tp_allreduce,
            }
        )
        if row_scatter_target is not None:
            finalize_args[ROW_SCATTER_TARGET_ARG] = row_scatter_target

        output = self.router.finalize(
            combine_payload,
            expert_payload.expert_topk_weights,
            expert_payload.expert_topk_ids,
            apply_router_weight_on_input,
            finalize_args,
        )

        assert (
            output.shape == hidden_states.shape
        ), f"output batch size mismatch: expected {hidden_states.shape}, got {output.shape}"

        return output
