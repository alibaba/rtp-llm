"""Configuration shared by the explicit PPU routed implementations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PpuMoeConfig:
    """Immutable dimensions and ownership for PPU routed experts.

    Frozen because strategies cache stuff keyed off it; mutating after
    construction would silently invalidate those caches.
    """

    layer_id: int
    dim: int
    moe_inter_dim: int
    n_routed_experts: int
    n_activated_experts: int  # topk
    swiglu_limit: float
    ep_size: int
    ep_rank: int
    n_local_experts: int
    local_expert_start: int
    local_expert_end: int
    max_tokens_per_rank: int
    tp_size: int = 1
