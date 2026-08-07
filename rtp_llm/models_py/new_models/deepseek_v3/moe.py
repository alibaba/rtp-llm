"""
MoE layers for DeepSeek V3.2, new-loader style.

Key design:
  - BaseMoEExperts owns the routed experts and their EP/TP loading contract.
  - DeepSeekV32MoEBlock wraps gate + SelectTopk(/GroupTopK) + experts + shared_expert.
    Mirrors GenericMoeLayer.forward but with new-loader submodules.
  - Shared expert reuses the common TP/FP8-safe DeepSeek MLP.
"""

import logging
import math
from enum import Enum, IntEnum
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules import FakeBalanceExpert, GroupTopK, SelectTopk
from rtp_llm.models_py.quant_methods.base import QuantizationConfig

from .mlp import DeepSeekV32MLP

logger = logging.getLogger(__name__)


class ScoringFunc(IntEnum):
    SOFTMAX = 0
    SIGMOID = 1


class TopKMethod(str, Enum):
    GREEDY = "greedy"
    GROUP_LIMITED_GREEDY = "group_limited_greedy"
    NOAUX_TC = "noaux_tc"


def normalize_topk_method(topk_method: str | TopKMethod) -> TopKMethod:
    if isinstance(topk_method, TopKMethod):
        return topk_method
    if not isinstance(topk_method, str):
        raise TypeError(f"topk_method must be a string, got {topk_method!r}")
    normalized = "greedy" if topk_method == "gready" else topk_method
    try:
        return TopKMethod(normalized)
    except ValueError as exc:
        raise ValueError(f"unsupported DeepSeek topk_method={topk_method!r}") from exc


def _mask_scores_by_group(
    scores_for_choice: torch.Tensor,
    group_scores: torch.Tensor,
    n_group: int,
    topk_group: int,
) -> torch.Tensor:
    """Mask experts outside the selected routing groups."""
    group_size = scores_for_choice.shape[-1] // n_group
    selected_groups = torch.topk(
        group_scores,
        k=topk_group,
        dim=-1,
        sorted=False,
    ).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups, True)
    grouped_scores = scores_for_choice.view(-1, n_group, group_size)
    return grouped_scores.masked_fill(
        ~group_mask.unsqueeze(-1),
        float("-inf"),
    ).view_as(scores_for_choice)


def _validate_routing_args(
    *,
    num_experts: int,
    top_k: int,
    n_group: int,
    topk_group: int,
    grouped: bool,
) -> None:
    if not 0 < top_k <= num_experts:
        raise ValueError(f"top_k={top_k} must be in [1, {num_experts}]")
    if not grouped:
        return
    if n_group <= 0 or num_experts % n_group:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by n_group={n_group}"
        )
    if not 0 < topk_group <= n_group:
        raise ValueError(f"topk_group={topk_group} must be in [1, n_group={n_group}]")
    selected_capacity = topk_group * (num_experts // n_group)
    if not 0 < top_k <= selected_capacity:
        raise ValueError(
            f"top_k={top_k} exceeds selected-group capacity {selected_capacity}"
        )


def _select_deepseek_topk(
    router_logits_fp32: torch.Tensor,
    *,
    top_k: int,
    scoring_func: int | ScoringFunc,
    n_group: int,
    topk_group: int,
    group_limited: bool,
    renormalize: bool,
    routed_scaling_factor: float,
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference-correct DeepSeek-V2 routing for the non-noaux path.

    DeepSeek-V2 normalizes top-k weights *or* applies routed scaling. Public
    V2 configurations do not combine non-unit scaling with normalization, but
    keeping the reference branch explicit avoids silently adopting V3 noaux
    semantics for a future V2 variant.
    """
    if router_logits_fp32.dim() != 2:
        raise ValueError(
            "router logits must have shape [tokens, experts], got "
            f"{tuple(router_logits_fp32.shape)}"
        )
    num_experts = router_logits_fp32.shape[-1]
    if validate:
        _validate_routing_args(
            num_experts=num_experts,
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            grouped=group_limited,
        )
    scoring_func = ScoringFunc(scoring_func)
    if scoring_func == ScoringFunc.SOFTMAX:
        scores = torch.softmax(router_logits_fp32, dim=-1)
    elif scoring_func == ScoringFunc.SIGMOID:
        scores = torch.sigmoid(router_logits_fp32)
    else:
        raise ValueError(f"unsupported DeepSeek scoring_func={scoring_func!r}")

    candidate_scores = scores
    if group_limited:
        group_size = num_experts // n_group
        group_scores = scores.view(-1, n_group, group_size).amax(dim=-1)
        candidate_scores = _mask_scores_by_group(
            scores,
            group_scores,
            n_group,
            topk_group,
        )

    topk_weights, topk_ids = torch.topk(
        candidate_scores,
        k=top_k,
        dim=-1,
        sorted=False,
    )
    if renormalize and top_k > 1:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(
            1e-20
        )
    else:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids


def _select_deepseek_noaux_topk(
    router_logits_fp32: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    top_k: int,
    n_group: int,
    topk_group: int,
    renormalize: bool,
    routed_scaling_factor: float,
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference DeepSeek-V3 noaux_tc routing used when GroupTopK is unavailable."""
    if router_logits_fp32.dim() != 2:
        raise ValueError(
            "router logits must have shape [tokens, experts], got "
            f"{tuple(router_logits_fp32.shape)}"
        )
    num_experts = router_logits_fp32.shape[-1]
    if validate and correction_bias.shape != (num_experts,):
        raise ValueError(
            "correction_bias must have shape [experts], got "
            f"{tuple(correction_bias.shape)} for {num_experts} experts"
        )
    if validate:
        _validate_routing_args(
            num_experts=num_experts,
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            grouped=True,
        )
    group_size = num_experts // n_group

    scores = torch.sigmoid(router_logits_fp32)
    scores_for_choice = scores + correction_bias
    grouped_scores = scores_for_choice.view(-1, n_group, group_size)
    group_scores = torch.topk(
        grouped_scores,
        k=min(2, group_size),
        dim=-1,
        sorted=False,
    ).values.sum(dim=-1)
    candidate_scores = _mask_scores_by_group(
        scores_for_choice,
        group_scores,
        n_group,
        topk_group,
    )
    topk_ids = torch.topk(
        candidate_scores,
        k=top_k,
        dim=-1,
        sorted=False,
    ).indices
    topk_weights = scores.gather(1, topk_ids)
    if renormalize and top_k > 1:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(
            1e-20
        )
    return topk_weights * routed_scaling_factor, topk_ids


class DeepSeekV32MoeGate(RtpModule):
    """Router gate that owns both `weight` and `e_score_correction_bias`.

    Matches HF ckpt keys:
      model.layers.{i}.mlp.gate.weight                  -> weight
      model.layers.{i}.mlp.gate.e_score_correction_bias -> e_score_correction_bias

    Not TP-sharded (num_experts is small relative to hidden dim).
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        has_correction_bias: bool,
        params_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        if has_correction_bias:
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(num_experts, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.register_parameter("e_score_correction_bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep the gate GEMM in the model dtype to match GenericMoeLayer and
        # avoid materializing a full FP32 gate-weight copy on every layer and
        # forward. Routing itself consumes FP32 logits below.
        return F.linear(x, self.weight, None)


class DeepSeekV32MoEBlock(RtpModule):
    """Full MoE block: gate + router + routed experts + shared expert.

    Mirrors GenericMoeLayer.forward: routing → FusedMoe → (optional) shared expert.
    """

    _reference_router_warning_emitted = False

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        top_k: int,
        layer_idx: int,
        tp_size: int,
        tp_rank: int,
        ep_size: int,
        ep_rank: int,
        model_config: Any,
        parallelism_config: Any,
        moe_config: Any,
        quant_config: Optional[QuantizationConfig],
        params_dtype: torch.dtype,
        # Config fields for GroupTopK (DeepSeek V3 style)
        has_shared_expert: bool = True,
        shared_expert_intermediate_size: int = 0,
        scoring_func: int | ScoringFunc = ScoringFunc.SIGMOID,
        routed_scaling_factor: float = 1.0,
        n_group: int = 1,
        topk_group: int = 1,
        topk_method: str | TopKMethod = TopKMethod.GREEDY,
        has_moe_norm: bool = False,
        correction_bias: bool = False,
        prefix: str = "mlp",
    ):
        super().__init__()
        if moe_config is None:
            fake_balance_expert = False
        else:
            fake_balance_expert = moe_config.fake_balance_expert
        if not isinstance(fake_balance_expert, bool):
            raise TypeError("moe_config.fake_balance_expert must be a bool")
        try:
            scoring_func = ScoringFunc(scoring_func)
        except ValueError as exc:
            raise ValueError(
                f"unsupported DeepSeek scoring_func={scoring_func}"
            ) from exc
        topk_method = normalize_topk_method(topk_method)
        if correction_bias != (topk_method == TopKMethod.NOAUX_TC):
            raise ValueError(
                "correction_bias must be enabled exactly for noaux_tc routing"
            )
        if correction_bias and scoring_func != ScoringFunc.SIGMOID:
            raise ValueError("noaux_tc routing requires sigmoid scoring")
        grouped = topk_method in {
            TopKMethod.GROUP_LIMITED_GREEDY,
            TopKMethod.NOAUX_TC,
        }
        _validate_routing_args(
            num_experts=num_experts,
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            grouped=grouped,
        )
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.has_moe_norm = has_moe_norm
        self.correction_bias = correction_bias
        self.group_limited = topk_method == TopKMethod.GROUP_LIMITED_GREEDY

        routing_config = {
            "hidden_size": (model_config.hidden_size, hidden_size),
            "expert_num": (model_config.expert_num, num_experts),
            "moe_k": (model_config.moe_k, top_k),
            "has_moe_norm": (
                model_config.has_moe_norm,
                has_moe_norm,
            ),
            "moe_n_group": (model_config.moe_n_group, n_group),
            "moe_topk_group": (
                model_config.moe_topk_group,
                topk_group,
            ),
            "scoring_func": (
                ScoringFunc(model_config.scoring_func),
                scoring_func,
            ),
            "routed_scaling_factor": (
                float(model_config.routed_scaling_factor),
                float(routed_scaling_factor),
            ),
        }
        mismatches = [
            f"{name}=ModelConfig({actual!r})/constructor({expected!r})"
            for name, (actual, expected) in routing_config.items()
            if not (
                math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
                if name == "routed_scaling_factor"
                else actual == expected
            )
        ]
        if mismatches:
            raise ValueError(
                f"DeepSeek routing config mismatch at layer {layer_idx}: "
                + ", ".join(mismatches)
            )
        device_type = get_device_type()
        fast_select_topk_candidate = (
            device_type == DeviceType.Cuda
            and not correction_bias
            and scoring_func == ScoringFunc.SOFTMAX
            and not self.group_limited
            and routed_scaling_factor == 1.0
            and not (top_k == 1 and has_moe_norm)
        )
        # SelectTopk and FusedMoe both consume ModelConfig. The equality check
        # above keeps their sizing/normalization contract identical to the
        # canonical config.json-derived constructor arguments.
        self._use_fast_select_topk = fast_select_topk_candidate

        # Router gate: hidden → num_experts (not TP-sharded, small).
        # Custom wrapper owns `weight` AND `e_score_correction_bias` so the
        # HF key model.layers.{i}.mlp.gate.e_score_correction_bias loads
        # cleanly via streaming dispatch.
        self.gate = DeepSeekV32MoeGate(
            hidden_size=hidden_size,
            num_experts=num_experts,
            has_correction_bias=correction_bias,
            params_dtype=params_dtype,
        )
        self.select_topk = (
            SelectTopk(config=model_config) if self._use_fast_select_topk else None
        )
        # GroupTopK is a CUDA-only fused op. ROCm and CPU use the exact
        # reference algorithm below instead of failing during construction.
        self._use_fast_group_topk = (
            correction_bias
            and device_type == DeviceType.Cuda
            and n_group <= 32
            and top_k <= 32
            and not (top_k == 1 and has_moe_norm)
        )
        self.group_topk = GroupTopK() if self._use_fast_group_topk else None
        if (
            not self._use_fast_select_topk
            and not self._use_fast_group_topk
            and not type(self)._reference_router_warning_emitted
        ):
            logger.warning(
                "DeepSeek layer %d uses reference PyTorch MoE routing "
                "(device=%s method=%s scoring_func=%d groups=%d/%d scaling=%s "
                "renormalize=%s)",
                layer_idx,
                device_type,
                topk_method,
                scoring_func,
                topk_group,
                n_group,
                routed_scaling_factor,
                has_moe_norm,
            )
            type(self)._reference_router_warning_emitted = True
        if fake_balance_expert:
            if parallelism_config is None:
                raise ValueError(
                    "fake_balance_expert requires a complete parallelism_config"
                )
            if device_type != DeviceType.Cuda:
                raise RuntimeError("fake_balance_expert is supported only on CUDA")
            self.fake_balance_expert = FakeBalanceExpert(
                expert_num=num_experts,
                moe_k=top_k,
                dp_rank=parallelism_config.dp_rank,
                dp_size=parallelism_config.dp_size,
                ep_size=ep_size,
            )
        else:
            self.fake_balance_expert = None

        if has_shared_expert != (shared_expert_intermediate_size > 0):
            raise ValueError(
                "has_shared_expert must match whether "
                "shared_expert_intermediate_size is positive"
            )

        # Routed experts
        self.experts = BaseMoEExperts(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            tp_size=tp_size,
            tp_rank=tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            params_dtype=params_dtype,
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            quant_config=quant_config,
            layer_idx=layer_idx,
            prefix=f"{prefix}.experts",
        )

        # Shared expert. HF key is `mlp.shared_experts.{gate,up,down}_proj`
        # (PLURAL). DeepSeek-V3.2 has no shared-expert gating — the routed
        # MoE output is summed directly with shared_experts(hidden_states).
        if has_shared_expert:
            self.shared_experts = DeepSeekV32MLP(
                hidden_size=hidden_size,
                intermediate_size=shared_expert_intermediate_size,
                tp_size=tp_size,
                tp_rank=tp_rank,
                quant_config=quant_config,
                params_dtype=params_dtype,
                reduce_output=True,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() != 2:
            raise ValueError(
                "DeepSeek MoE expects a two-dimensional token matrix, got "
                f"{tuple(hidden_states.shape)}"
            )
        router_logits = self.gate(hidden_states)
        router_logits_fp32 = router_logits.float()
        topk_ids_dtype = self.experts.topk_ids_dtype

        if self.correction_bias:
            if self._use_fast_group_topk:
                if not router_logits_fp32.is_cuda:
                    raise RuntimeError(
                        "DeepSeek CUDA grouped top-k router received a CPU tensor"
                    )
                topk_weights = torch.empty(
                    (hidden_states.shape[0], self.top_k),
                    dtype=torch.float32,
                    device=hidden_states.device,
                )
                topk_ids = torch.empty(
                    (hidden_states.shape[0], self.top_k),
                    dtype=topk_ids_dtype,
                    device=hidden_states.device,
                )
                self.group_topk(
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    scores=router_logits_fp32,
                    correction_bias=self.gate.e_score_correction_bias,
                    n_group=self.n_group,
                    topk_group=self.topk_group,
                    topk=self.top_k,
                    renormalize=self.has_moe_norm,
                    routed_scaling_factor=self.routed_scaling_factor,
                )
            else:
                selected_weights, selected_ids = _select_deepseek_noaux_topk(
                    router_logits_fp32,
                    self.gate.e_score_correction_bias,
                    top_k=self.top_k,
                    n_group=self.n_group,
                    topk_group=self.topk_group,
                    renormalize=self.has_moe_norm,
                    routed_scaling_factor=self.routed_scaling_factor,
                    validate=False,
                )
                topk_weights = selected_weights
                topk_ids = selected_ids.to(dtype=topk_ids_dtype)
        else:
            if self._use_fast_select_topk:
                if not router_logits_fp32.is_cuda:
                    raise RuntimeError(
                        "DeepSeek CUDA top-k router received a CPU tensor"
                    )
                topk_weights = torch.empty(
                    (hidden_states.shape[0], self.top_k),
                    dtype=torch.float32,
                    device=hidden_states.device,
                )
                topk_ids = torch.empty(
                    (hidden_states.shape[0], self.top_k),
                    dtype=topk_ids_dtype,
                    device=hidden_states.device,
                )
                self.select_topk(router_logits_fp32, topk_ids, topk_weights)
            else:
                selected_weights, selected_ids = _select_deepseek_topk(
                    router_logits_fp32,
                    top_k=self.top_k,
                    scoring_func=self.scoring_func,
                    n_group=self.n_group,
                    topk_group=self.topk_group,
                    group_limited=self.group_limited,
                    renormalize=self.has_moe_norm,
                    routed_scaling_factor=self.routed_scaling_factor,
                    validate=False,
                )
                topk_weights = selected_weights
                topk_ids = selected_ids.to(dtype=topk_ids_dtype)

        if self.fake_balance_expert is not None:
            self.fake_balance_expert(topk_ids, topk_weights)

        experts_output = self.experts(hidden_states, topk_weights, topk_ids)

        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
            experts_output = experts_output + shared_output

        return experts_output
