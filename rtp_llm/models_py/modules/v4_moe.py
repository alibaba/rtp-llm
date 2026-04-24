"""DeepSeek-V4 MoE building blocks.

Three V4-specific pieces (rest is shared with V3):

1. **`V4Gate`**: scoring with `sqrt(softplus(x))` (the new `scoring_func == "sqrtsoftplus"`),
   plus the `noaux_tc` topk path with no `n_group / topk_group` constraint
   (V4 dropped node-grouped routing).

2. **`V4HashGate`**: deterministic per-token-id expert lookup for the first
   `num_hash_layers=3` MoE layers. The lookup table `tid2eid: [vocab_size, n_activated]`
   is loaded from the checkpoint (`layers.{i}.ffn.gate.tid2eid`).

3. **`V4Expert`**: SwiGLU with optional clamp (linear ∈ [-L, L], gate ≤ L) where
   L = `swiglu_limit = 10.0` for routed experts; shared experts use no clamp.

Reference: `inference/model.py:Gate`, `Expert`, `MoE` from the official V4 release.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def sqrt_softplus(x: torch.Tensor) -> torch.Tensor:
    """V4 routing score: `sqrt(softplus(x))`. Always non-negative.

    Equivalent to `softplus(x).sqrt()` but uses fp32 internally for stability,
    matching the official reference's `scores.float() → softplus → sqrt`.
    """
    return F.softplus(x.float()).sqrt()


def v4_score(scores: torch.Tensor, score_func: int) -> torch.Tensor:
    """Apply the V4 scoring function. ScoreFunc enum from cpp/config/ModelConfig.h:
    0 = softmax, 1 = sigmoid, 2 = sqrt_softplus.
    """
    if score_func == 0:
        return scores.softmax(dim=-1)
    elif score_func == 1:
        return scores.sigmoid()
    elif score_func == 2:
        return sqrt_softplus(scores)
    else:
        raise ValueError(f"unknown score_func {score_func}")


class V4Gate(nn.Module):
    """Score-based MoE gate (V4 noaux_tc topk over sqrt(softplus) scores).

    Differences vs DeepSeek-V3 noaux_tc:
    - Score function is sqrt(softplus) (V4) instead of sigmoid (V3).
    - No `n_group / topk_group` node-grouped constraint — pure top-k over all
      `n_routed_experts` (=256 for V4-Flash, 384 for V4-Pro).
    - `e_score_correction_bias` (the `.bias` parameter) shifts scores for *expert
      selection* (topk argmax) but does NOT affect the routing weights returned
      to combine expert outputs — same trick as V3.
    - Weights are renormalized to sum to 1 across the chosen experts before
      multiplying by `route_scale`.
    """

    def __init__(
        self,
        dim: int,
        n_routed_experts: int,
        n_activated_experts: int,
        score_func: int,
        route_scale: float,
    ):
        super().__init__()
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.score_func = score_func
        self.route_scale = route_scale
        # Compute in fp32 — store gate weight as fp32 to match official.
        self.weight = nn.Parameter(
            torch.empty(n_routed_experts, dim, dtype=torch.float32)
        )
        self.bias = nn.Parameter(torch.zeros(n_routed_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [N, dim] (already flattened over batch+seq)
        returns: (weights [N, k], indices [N, k])
        """
        scores = F.linear(x.float(), self.weight)  # [N, n_routed]
        scores = v4_score(scores, self.score_func)
        original_scores = scores
        scores = scores + self.bias  # bias only affects topk argmax
        indices = scores.topk(self.n_activated_experts, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != 0:
            # softmax already sums to 1 over all experts; for sigmoid/sqrt_softplus
            # we must renormalize over the chosen top-k.
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.route_scale
        return weights, indices


class V4HashGate(nn.Module):
    """Hash-routing gate for the first `num_hash_layers` MoE layers.

    Expert assignment is fixed at training time via `tid2eid: [vocab_size,
    n_activated]`. At inference, indices come from `tid2eid[input_ids]` — no
    score computation, no topk.

    Routing weights still come from `sqrt_softplus(linear(x, W))` so that the
    expert *combination* remains data-dependent; only which experts are
    queried is tied to the token id.
    """

    def __init__(
        self,
        dim: int,
        n_routed_experts: int,
        n_activated_experts: int,
        vocab_size: int,
        score_func: int,
        route_scale: float,
    ):
        super().__init__()
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.score_func = score_func
        self.route_scale = route_scale
        self.weight = nn.Parameter(
            torch.empty(n_routed_experts, dim, dtype=torch.float32)
        )
        # tid2eid is loaded from checkpoint as int32; per token id, the n_activated
        # expert ids it always routes to.
        self.tid2eid = nn.Parameter(
            torch.empty(vocab_size, n_activated_experts, dtype=torch.int32),
            requires_grad=False,
        )

    def forward(
        self, x: torch.Tensor, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [N, dim]; input_ids: [N] (flattened)"""
        scores = F.linear(x.float(), self.weight)
        scores = v4_score(scores, self.score_func)
        indices = self.tid2eid[input_ids].long()  # [N, k]
        weights = scores.gather(1, indices)
        if self.score_func != 0:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.route_scale
        return weights, indices


class V4Expert(nn.Module):
    """SwiGLU FFN with optional clamp (V4 routed expert).

    `up = clamp(W_3 x, -L, L)` and `gate = clamp(W_1 x, max=L)` then `silu(gate) * up`.
    Compute in fp32 for stability (matches official reference).

    L (`swiglu_limit`) = 10.0 for routed experts in V4-Flash. The shared expert
    uses no clamp (`swiglu_limit = 0`).
    """

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        swiglu_limit: float = 0.0,
    ):
        super().__init__()
        self.swiglu_limit = swiglu_limit
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(
        self, x: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class V4MoE(nn.Module):
    """V4 layer-level MoE wrapper.

    Holds gate (hash or score), routed experts, and one shared expert. This is
    the **non-fused** PyTorch reference path — production should call into
    `models_py/modules/factory/fused_moe/` for DeepGEMM MegaMoE etc.

    Per-layer dispatch:
    - `is_hash_layer = layer_id < model_config.moe_hash_routing_layers` (e.g. first 3 layers)
    - `is_hash_layer == True`  → `V4HashGate` (needs input_ids)
    - `is_hash_layer == False` → `V4Gate`     (pure score-based topk)

    Sharded across TP ranks: each rank holds `n_routed_experts // world_size`
    experts; routing indices that fall outside the local range are skipped.
    """

    def __init__(
        self,
        layer_id: int,
        dim: int,
        moe_inter_dim: int,
        n_routed_experts: int,
        n_activated_experts: int,
        n_shared_experts: int,
        score_func: int,
        route_scale: float,
        swiglu_limit: float,
        moe_hash_routing_layers: int,
        vocab_size: int,
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = dim
        assert n_routed_experts % world_size == 0, (
            f"n_routed_experts={n_routed_experts} must be divisible by world_size={world_size}"
        )
        self.n_routed_experts = n_routed_experts
        self.n_local_experts = n_routed_experts // world_size
        self.n_activated_experts = n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

        is_hash = layer_id < moe_hash_routing_layers
        if is_hash:
            self.gate = V4HashGate(
                dim,
                n_routed_experts,
                n_activated_experts,
                vocab_size,
                score_func,
                route_scale,
            )
        else:
            self.gate = V4Gate(
                dim,
                n_routed_experts,
                n_activated_experts,
                score_func,
                route_scale,
            )

        # Local routed experts; non-local slots are None to save memory.
        self.experts = nn.ModuleList(
            [
                V4Expert(dim, moe_inter_dim, swiglu_limit=swiglu_limit)
                if self.experts_start_idx <= i < self.experts_end_idx
                else None
                for i in range(n_routed_experts)
            ]
        )
        assert n_shared_experts == 1, "V4 has exactly 1 shared expert"
        # Shared expert: no swiglu clamp.
        self.shared_experts = V4Expert(dim, moe_inter_dim, swiglu_limit=0.0)

    def forward(
        self, x: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """x: [b, s, dim]; input_ids: [b, s] for hash gate"""
        shape = x.shape
        x_flat = x.view(-1, self.dim)
        ids_flat = input_ids.flatten()
        if isinstance(self.gate, V4HashGate):
            weights, indices = self.gate(x_flat, ids_flat)
        else:
            weights, indices = self.gate(x_flat)

        y = torch.zeros_like(x_flat, dtype=torch.float32)
        # Group tokens by their assigned local experts.
        counts = torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts
        ).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x_flat[idx], weights[idx, top, None])
        # NOTE: production path needs `dist.all_reduce(y)` here when world_size > 1;
        # the caller is responsible (this module stays comm-agnostic for testability).
        y = y + self.shared_experts(x_flat).float()
        return y.type_as(x).view(shape)
