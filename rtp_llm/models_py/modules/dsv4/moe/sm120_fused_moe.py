"""DSV4 SM120 local expert implementation on the generic ``FusedMoe`` API.

The Pro 5000 / SM120 software stack does not provide the same monolithic
communication-and-compute MoE kernel used on supported DeepEP platforms.
Keep those concerns composable instead:

* the common Factory PureCP router owns CP all-gather/reduce-scatter;
* this module presents the local SM120 expert partition as a generic
  :class:`FusedMoeExpertExecutor`;
* the executor reuses DSV4's FlashInfer GroupedFP4 implementation (or the
  correctness local-loop fallback) without exposing it to the communication
  layer.

The resulting ``FusedMoe`` computes a *partial* routed-expert output for one
expert partition.  A CP reduce-scatter or the legacy EP all-reduce combines
those partials across ranks.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional, Type

import torch

from rtp_llm.models_py.modules.factory import FusedMoeFactory
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FinalizeArgs,
    FusedMoe,
    FusedMoeDataRouter,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import (
    ExecutorType,
    RouterType,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_cp_router import (
    PureCpRouterNoQuant,
)

from .strategies.base import MoeCfg, RoutedExpertsStrategy


class Sm120LocalFusedMoeRouter(FusedMoeDataRouter):
    """Pass-through router for one already-routed SM120 compute chunk.

    Cross-rank communication deliberately lives outside this router.  It
    adapts one decode/DP chunk to the generic ``FusedMoe`` payload contract and
    leaves the partial output untouched for the outer all-reduce.
    """

    def __init__(
        self,
        config: Any,
        quant_config: Optional[FusedMoEQuantConfig] = None,
    ) -> None:
        super().__init__(config, quant_config or FusedMoEQuantConfig())

    @classmethod
    def router_type(cls) -> RouterType:
        return RouterType.PURE_TP

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        # DSV4 constructs this adapter through the Factory strategy override;
        # it is intentionally not registered as a process-wide strategy.
        return None

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        if a2_scale is not None:
            raise ValueError("DSV4 SM120 FusedMoe does not consume a2_scale")
        return ExpertForwardPayload(
            expert_x=a1,
            expert_x_origin_dtype=a1.dtype,
            expert_x_scale=a1_scale,
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
            expert_ids_are_local=False,
        )

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[FinalizeArgs],
    ) -> torch.Tensor:
        return payload.fused_expert_output


class Sm120GroupedFp4FusedMoeExecutor(FusedMoeExpertExecutor):
    """Local DSV4 expert partition exposed as a FusedMoe executor.

    ``local_strategy`` is normally ``GroupedFP4Strategy``.  It owns only the
    rank-local expert weights, so global top-k ids are masked and remapped to
    its local id space before execution.  Keeping that transformation here
    gives every caller (CP, all-reduce fallback, or experimental all-to-all)
    the same global-id input contract.
    """

    def __init__(
        self,
        config: Any,
        quant_config: FusedMoEQuantConfig,
        weights: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        super().__init__(config, quant_config, weights or {})
        self.cfg: MoeCfg = config.dsv4_cfg
        self.local_strategy: RoutedExpertsStrategy = config.local_strategy
        self.uses_grouped_fp4 = bool(config.uses_grouped_fp4)

    @classmethod
    def executor_type(cls) -> ExecutorType:
        # This is a FusedMoe-compatible executor composed from the available
        # SM120 grouped kernels, not the SM100 DeepGEMM executor.
        return ExecutorType.FUSED_MOE

    @classmethod
    def check_conditions(cls, checker: Any, config: Any) -> None:
        # Internal DSV4 executor; selection is handled by the DSV4 SM120
        # strategy while construction follows the Factory executor contract.
        return None

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        if activation not in ("silu", "swiglu"):
            raise ValueError(f"unsupported DSV4 SM120 activation: {activation}")
        if expert_map is not None:
            raise ValueError("DSV4 SM120 FusedMoe does not consume expert_map")
        if a2_scale is not None:
            raise ValueError("DSV4 SM120 FusedMoe does not consume a2_scale")
        if apply_router_weight_on_input:
            raise ValueError(
                "DSV4 SM120 applies router weights during expert-output gather"
            )
        if payload.expert_topk_ids is None or payload.expert_topk_weights is None:
            raise ValueError("SM120 FusedMoe requires top-k ids and weights")

        x = payload.expert_x
        weights = payload.expert_topk_weights
        indices = payload.expert_topk_ids.to(torch.int64)
        cfg = self.cfg

        if self.uses_grouped_fp4:
            local_ids = indices - cfg.local_expert_start
            valid = (local_ids >= 0) & (local_ids < cfg.n_local_experts)
            local_weights = weights * valid.to(weights.dtype)
            local_ids = local_ids.clamp(0, cfg.n_local_experts - 1)

            # Experimental MXFP8 all-to-all can pass a pre-quantized input
            # and its linear scale.  Normal CP/all-reduce calls enter through
            # ``forward`` and let GroupedFP4 quantize BF16 locally.
            if payload.expert_x_scale is None:
                output = self.local_strategy(x, local_weights, local_ids)
            else:
                output = self.local_strategy.forward_sm120_eager(
                    x,
                    local_weights,
                    local_ids,
                    input_scale=payload.expert_x_scale,
                )
        else:
            output = self.local_strategy.forward_local_range(
                x,
                weights,
                indices,
                local_start=cfg.local_expert_start,
                local_end=cfg.local_expert_end,
            )

        return CombineForwardPayload(fused_expert_output=output.float())


@dataclass(frozen=True)
class Sm120FactoryConfig:
    """Small adapter exposing DSV4's config through the Factory contract."""

    dsv4_cfg: MoeCfg
    local_strategy: RoutedExpertsStrategy
    uses_grouped_fp4: bool
    expert_num: int
    tp_size: int
    ep_size: int
    ep_rank: int
    dp_size: int
    parallelism_config: Any

    @classmethod
    def from_cfg(
        cls,
        cfg: MoeCfg,
        local_strategy: RoutedExpertsStrategy,
        uses_grouped_fp4: bool,
    ) -> "Sm120FactoryConfig":
        # The common PureCP router reads physical TP from this field and uses
        # Group.TP for collectives.  Decode never instantiates that router.
        parallelism = SimpleNamespace(tp_size=max(int(cfg.moe_tp_size), 1))
        return cls(
            dsv4_cfg=cfg,
            local_strategy=local_strategy,
            uses_grouped_fp4=bool(uses_grouped_fp4),
            expert_num=cfg.n_routed_experts,
            tp_size=1 if cfg.cp_enabled else max(int(cfg.moe_tp_size), 1),
            ep_size=cfg.ep_size,
            ep_rank=cfg.ep_rank,
            dp_size=1,
            parallelism_config=parallelism,
        )


class Sm120FactoryStrategy(MoeStrategy):
    """Factory strategy adapter for DSV4's SM120 executor.

    DSV4 supplies a model-specific config/weight adapter, while router and
    executor lifecycle remains the common FusedMoeFactory contract.
    """

    def __init__(self, router_cls: Type[FusedMoeDataRouter]) -> None:
        self.router_cls = router_cls
        self.quant_config = FusedMoEQuantConfig()

    def get_attributes(self) -> StrategyAttributes:
        return StrategyAttributes(
            router_class=self.router_cls,
            executor_class=Sm120GroupedFp4FusedMoeExecutor,
            quant_config=self.quant_config,
        )

    def create_router(self, config: Any) -> FusedMoeDataRouter:
        return self.router_cls(config, self.quant_config)

    def create_executor(
        self, config: Any, weights: dict[str, torch.Tensor]
    ) -> FusedMoeExpertExecutor:
        return Sm120GroupedFp4FusedMoeExecutor(config, self.quant_config, weights)


def build_sm120_fused_moe(
    cfg: MoeCfg,
    local_strategy: RoutedExpertsStrategy,
    *,
    uses_grouped_fp4: bool,
    router_cls: Type[FusedMoeDataRouter] = Sm120LocalFusedMoeRouter,
) -> FusedMoe:
    """Build an SM120 FusedMoe through the common Factory contract.

    Decode passes the local pass-through router; prefill CP passes the common
    ``PureCpRouterNoQuant``.  The expert executor and Factory lifecycle are
    shared by both modes.
    """

    factory_config = Sm120FactoryConfig.from_cfg(cfg, local_strategy, uses_grouped_fp4)
    return FusedMoeFactory().create_fused_moe(
        factory_config,
        {},
        strategy_override=Sm120FactoryStrategy(router_cls),
    )


__all__ = [
    "Sm120GroupedFp4FusedMoeExecutor",
    "Sm120LocalFusedMoeRouter",
    "Sm120FactoryConfig",
    "Sm120FactoryStrategy",
    "build_sm120_fused_moe",
]
