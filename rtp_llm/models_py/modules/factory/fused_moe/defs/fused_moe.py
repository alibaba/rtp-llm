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


class FinalizeArgs(TypedDict, total=False):
    """Private, optional arguments passed from ``FusedMoe`` to routers."""

    a1_shape: torch.Size
    original_num_tokens: int
    skip_tp_allreduce: bool


@dataclass(frozen=True)
class ExpertGatePayload:
    """Raw gate output accepted by route-and-pack capable backends.

    The payload is model-agnostic: a hash-routed gate supplies ``input_ids``
    and ``tid2eid``; a score-routed gate supplies ``bias``.  Backends that do
    not advertise ``supports_gate_pack`` continue to consume materialized
    top-k weights and ids through the ordinary ``prepare`` path.
    """

    scores: torch.Tensor
    topk: int
    score_func: str
    route_scale: float
    norm_eps: float = 1.0e-12
    bias: Optional[torch.Tensor] = None
    input_ids: Optional[torch.Tensor] = None
    tid2eid: Optional[torch.Tensor] = None


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
    gate_payload: Optional[ExpertGatePayload] = None


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
    def supports_gate_pack(self) -> bool:
        """Whether this router can pass raw gate output to its executor."""

        return False

    def prepare_gate_pack(
        self,
        a1: torch.Tensor,
        gate_payload: ExpertGatePayload,
    ) -> ExpertForwardPayload:
        raise NotImplementedError(
            f"{type(self).__name__} does not support fused gate packing"
        )

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
    includes_shared_expert = False
    execute_empty_inputs = False

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

    @property
    def supports_gate_pack(self) -> bool:
        """Whether ``execute`` accepts an ``ExpertGatePayload``."""

        return False

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
        strategy_name: str = "",
    ):
        super().__init__()
        self.router = router
        self.fused_experts = fused_experts
        self.expert_num = expert_num
        self.strategy_name = strategy_name

    @property
    def includes_shared_expert(self) -> bool:
        return bool(self.fused_experts.includes_shared_expert)

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return self.fused_experts.topk_ids_dtype

    @property
    def supports_gate_pack(self) -> bool:
        return bool(
            self.router.supports_gate_pack and self.fused_experts.supports_gate_pack
        )

    def forward_gate_pack(
        self,
        hidden_states: torch.Tensor,
        gate_payload: ExpertGatePayload,
        activation: str = "silu",
        expert_map: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        extra_expert_args: Optional[Dict[str, Any]] = None,
        extra_finalize_args: Optional[FinalizeArgs] = None,
        skip_tp_allreduce: bool = False,
    ) -> torch.Tensor:
        if not self.supports_gate_pack:
            raise RuntimeError(
                f"strategy {self.strategy_name!r} does not support fused gate packing"
            )
        expert_payload = self.router.prepare_gate_pack(hidden_states, gate_payload)
        return self._execute_payload(
            hidden_states,
            expert_payload,
            activation=activation,
            expert_map=expert_map,
            a2_scale=a2_scale,
            apply_router_weight_on_input=apply_router_weight_on_input,
            extra_expert_args=extra_expert_args,
            extra_finalize_args=extra_finalize_args,
            skip_tp_allreduce=skip_tp_allreduce,
        )

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

        return self._execute_payload(
            hidden_states,
            expert_payload,
            activation=activation,
            expert_map=expert_map,
            a2_scale=a2_scale,
            apply_router_weight_on_input=apply_router_weight_on_input,
            extra_expert_args=extra_expert_args,
            extra_finalize_args=extra_finalize_args,
            skip_tp_allreduce=skip_tp_allreduce,
        )

    def _execute_payload(
        self,
        hidden_states: torch.Tensor,
        expert_payload: ExpertForwardPayload,
        *,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[Dict[str, Any]],
        extra_finalize_args: Optional[FinalizeArgs],
        skip_tp_allreduce: bool,
    ) -> torch.Tensor:
        if skip_tp_allreduce and not self.router.supports_skip_tp_allreduce:
            raise ValueError(
                "skip_tp_allreduce is only supported by routers that "
                "advertise supports_skip_tp_allreduce"
            )

        if (
            expert_payload.expert_x.numel() == 0
            and self.fused_experts.execute_empty_inputs is not True
        ):
            # This happens when none of the tokens from the all2all reach this
            # EP rank. Also, note that this is only relevant for CUDAGraph
            # incompatible all2all kernels like the DeepEP high-throughput
            # kernels. CUDAGraph compatible all2all kernels like the pplx
            # kernels and the DeepEP low-latency kernels are always batched
            # and can never run into the tensor.numel() == 0 case.
            combine_payload = CombineForwardPayload(
                fused_expert_output=torch.empty_like(
                    expert_payload.expert_x, dtype=hidden_states.dtype
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

        if (
            expert_payload.expert_topk_weights is None
            or expert_payload.expert_topk_ids is None
        ):
            raise RuntimeError("router/executor did not provide top-k weights and ids")

        # Finalize arguments are a private per-call protocol. Copy caller
        # input before adding derived values so a reusable dict cannot retain
        # state from a previous forward.
        finalize_args: FinalizeArgs = {
            **(extra_finalize_args or {}),
            "a1_shape": hidden_states.shape,
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
