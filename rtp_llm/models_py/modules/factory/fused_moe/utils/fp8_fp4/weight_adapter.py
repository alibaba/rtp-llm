"""Pack split gate/up weights into the canonical fused-MoE contract."""

from typing import Dict, Mapping

import torch

from rtp_llm.utils.model_weight import W


def adapt_split_moe_weights(
    weights: Dict[str, torch.Tensor],
    moe_inter_dim: int,
    n_shared_experts: int,
    weight_names: Mapping[str, str],
) -> Dict[str, torch.Tensor]:
    """Consume caller-named tensors and pack routed W1 in gate/up order.

    The caller supplies checkpoint key names; shared W13 is already packed.
    Copying the halves also supports scale dtypes without ``torch.cat``.
    """

    if W.moe_w1 in weights:
        return weights

    routed_w1 = weights.pop(weight_names["routed_gate"])
    routed_w3 = weights.pop(weight_names["routed_up"])
    routed_s1 = weights.pop(weight_names["routed_gate_scale"])
    routed_s3 = weights.pop(weight_names["routed_up_scale"])
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
    weights[W.moe_w2] = weights.pop(weight_names["routed_down"])
    weights[W.moe_s2] = weights.pop(weight_names["routed_down_scale"])

    weights[W.moe_gate] = weights.pop(weight_names["router"])
    if weight_names.get("router_bias") in weights:
        weights[W.moe_gate_bias] = weights.pop(weight_names["router_bias"])
    if weight_names.get("router_tid2eid") in weights:
        weights[W.moe_gate_tid2eid] = weights.pop(weight_names["router_tid2eid"])

    if n_shared_experts > 0:
        weights[W.ffn_w13] = weights.pop(weight_names["shared_gate_up"])
        weights[W.ffn_s13] = weights.pop(weight_names["shared_gate_up_scale"])
        weights[W.ffn_w2] = weights.pop(weight_names["shared_down"])
        weights[W.ffn_s2] = weights.pop(weight_names["shared_down_scale"])
    return weights
