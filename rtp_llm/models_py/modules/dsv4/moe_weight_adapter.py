"""Adapt DeepSeek-V4 checkpoint weights to the generic fused-MoE contract."""

from typing import Dict

import torch

from rtp_llm.utils.model_weight import W


def adapt_dsv4_moe_weights(
    weights: Dict[str, torch.Tensor],
    moe_inter_dim: int,
    n_shared_experts: int,
) -> Dict[str, torch.Tensor]:
    """Move model-specific weight keys to their canonical MoE equivalents."""

    if W.moe_w1 in weights:
        return weights

    routed_w1 = weights.pop(W.v4_routed_w1_w)
    routed_w3 = weights.pop(W.v4_routed_w3_w)
    routed_s1 = weights.pop(W.v4_routed_w1_s)
    routed_s3 = weights.pop(W.v4_routed_w3_s)
    expert_count, _, packed_hidden = routed_w1.shape
    combined_shape = (expert_count, 2 * moe_inter_dim, packed_hidden)
    combined_w13 = torch.empty(
        combined_shape,
        dtype=routed_w1.dtype,
        device=routed_w1.device,
    )
    combined_s13 = torch.empty(
        (expert_count, 2 * moe_inter_dim, routed_s1.size(-1)),
        dtype=routed_s1.dtype,
        device=routed_s1.device,
    )
    combined_w13[:, :moe_inter_dim].copy_(routed_w1)
    combined_w13[:, moe_inter_dim:].copy_(routed_w3)
    combined_s13[:, :moe_inter_dim].copy_(routed_s1)
    combined_s13[:, moe_inter_dim:].copy_(routed_s3)
    weights[W.moe_w1] = combined_w13
    weights[W.moe_s1] = combined_s13
    weights[W.moe_w2] = weights.pop(W.v4_routed_w2_w)
    weights[W.moe_s2] = weights.pop(W.v4_routed_w2_s)

    weights[W.moe_gate] = weights.pop(W.v4_router_w)
    if W.v4_router_bias in weights:
        weights[W.moe_gate_bias] = weights.pop(W.v4_router_bias)
    if W.v4_router_tid2eid in weights:
        weights[W.moe_gate_tid2eid] = weights.pop(W.v4_router_tid2eid)

    if n_shared_experts > 0:
        weights[W.ffn_w13] = weights.pop(W.v4_shared_w13_w)
        weights[W.ffn_s13] = weights.pop(W.v4_shared_w13_s)
        weights[W.ffn_w2] = weights.pop(W.v4_shared_w2_w)
        weights[W.ffn_s2] = weights.pop(W.v4_shared_w2_s)
    return weights
