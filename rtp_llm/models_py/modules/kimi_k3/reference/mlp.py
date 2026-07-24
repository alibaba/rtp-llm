"""SiTU dense and latent-MoE correctness modules for Kimi K3."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.modules.kimi_k3.reference.common import (
    KimiRMSNorm,
    SituAndMul,
)


class KimiMLPReference(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        situ_beta: float = 4.0,
        situ_linear_beta: float | None = 25.0,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SituAndMul(situ_beta, situ_linear_beta)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = torch.cat(
            (self.gate_proj(hidden_states), self.up_proj(hidden_states)), dim=-1
        )
        return self.down_proj(self.act_fn(gate_up))


class KimiBlockSparseMLPReference(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        situ_beta: float = 4.0,
        situ_linear_beta: float | None = 25.0,
    ):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = SituAndMul(situ_beta, situ_linear_beta)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w2(
            self.act_fn(
                torch.cat((self.w1(hidden_states), self.w3(hidden_states)), dim=-1)
            )
        )


class KimiMoEGateReference(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        *,
        routed_scaling_factor: float = 1.0,
        renormalize: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
    ):
        super().__init__()
        if top_k <= 0 or top_k > num_experts:
            raise ValueError(f"invalid top_k={top_k} for {num_experts} experts")
        if num_experts % num_expert_group != 0:
            raise ValueError("num_experts must be divisible by num_expert_group")
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.renormalize = bool(renormalize)
        self.num_expert_group = int(num_expert_group)
        self.topk_group = int(topk_group)
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(num_experts))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        scores = F.linear(flat.float(), self.weight.float()).sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias.float()
        if self.num_expert_group > 1 and self.num_expert_group > self.topk_group:
            grouped = scores_for_choice.reshape(
                flat.shape[0], self.num_expert_group, -1
            )
            group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
            selected_groups = group_scores.topk(
                self.topk_group, dim=-1, sorted=False
            ).indices
            group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
            group_mask.scatter_(1, selected_groups, True)
            expert_mask = group_mask.unsqueeze(-1).expand_as(grouped).reshape_as(
                scores_for_choice
            )
            scores_for_choice = scores_for_choice.masked_fill(
                ~expert_mask, float("-inf")
            )
        topk_ids = scores_for_choice.topk(
            self.top_k, dim=-1, sorted=False
        ).indices
        topk_weight = scores.gather(1, topk_ids)
        if self.top_k > 1 and self.renormalize:
            topk_weight = topk_weight / (
                topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            )
        return topk_ids, topk_weight * self.routed_scaling_factor


class KimiSparseMoeBlockReference(nn.Module):
    """K3 routed latent experts plus a full-hidden shared expert."""

    def __init__(
        self,
        hidden_size: int,
        routed_hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        top_k: int,
        num_shared_experts: int,
        *,
        rms_norm_eps: float = 1e-6,
        latent_moe_use_norm: bool = True,
        routed_scaling_factor: float = 1.0,
        renormalize: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
        situ_beta: float = 4.0,
        situ_linear_beta: float | None = 25.0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.routed_hidden_size = int(routed_hidden_size)
        self.latent_moe_use_norm = bool(latent_moe_use_norm)
        self.gate = KimiMoEGateReference(
            hidden_size,
            num_experts,
            top_k,
            routed_scaling_factor=routed_scaling_factor,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
        )
        self.experts = nn.ModuleList(
            [
                KimiBlockSparseMLPReference(
                    routed_hidden_size,
                    moe_intermediate_size,
                    situ_beta=situ_beta,
                    situ_linear_beta=situ_linear_beta,
                )
                for _ in range(num_experts)
            ]
        )
        self.routed_expert_down_proj = nn.Linear(
            hidden_size, routed_hidden_size, bias=False
        )
        self.routed_expert_up_proj = nn.Linear(
            routed_hidden_size, hidden_size, bias=False
        )
        if self.latent_moe_use_norm:
            self.routed_expert_norm = KimiRMSNorm(
                routed_hidden_size, eps=rms_norm_eps
            )
        self.shared_experts = KimiMLPReference(
            hidden_size,
            moe_intermediate_size * num_shared_experts,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        topk_ids, topk_weight = self.gate(hidden_states)
        flat_identity = hidden_states.reshape(-1, self.hidden_size)
        routed_input = self.routed_expert_down_proj(flat_identity)
        routed_output = routed_input.new_zeros(routed_input.shape)

        for expert_idx, expert in enumerate(self.experts):
            matches = (topk_ids == expert_idx).nonzero(as_tuple=False)
            if matches.numel() == 0:
                continue
            token_indices = matches[:, 0]
            slot_indices = matches[:, 1]
            expert_output = expert(routed_input[token_indices])
            weighted = expert_output * topk_weight[
                token_indices, slot_indices
            ].to(dtype=expert_output.dtype).unsqueeze(-1)
            routed_output.index_add_(0, token_indices, weighted)

        if self.latent_moe_use_norm:
            routed_output = self.routed_expert_norm(routed_output)
        routed_output = self.routed_expert_up_proj(routed_output).reshape_as(
            hidden_states
        )
        return routed_output + self.shared_experts(hidden_states)
