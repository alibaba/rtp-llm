"""Operand configuration for the Hopper grouped-FP8 kernels."""

from dataclasses import dataclass

import torch.nn as nn


@dataclass(frozen=True)
class MoeCfg:
    """Per-layer MoE configuration shared across all strategies.

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
    is_decode_role: bool = False
    enable_cuda_graph: bool = False


class RoutedExpertsStrategy(nn.Module):
    """Grouped compute module driven by the common fused-MoE executor."""

    def __init__(self, cfg: MoeCfg):
        super().__init__()
        self.cfg = cfg
