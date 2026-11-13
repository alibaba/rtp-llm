"""Pack split gate/up weights into the canonical fused-MoE contract."""

from typing import Dict, Mapping

import torch

from rtp_llm.resident_weights import should_consume_loaded
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

    # Packing consumes the loaded routed tensors, and the fused executor then consumes the
    # packed ones. On a resident restart the executor hands back its own final weights, so
    # neither step may run and none of these names is in the dict any more.
    routed = [
        weight_names[k]
        for k in (
            "routed_gate",
            "routed_up",
            "routed_gate_scale",
            "routed_up_scale",
            "routed_down",
            "routed_down_scale",
        )
    ]
    if should_consume_loaded(weights, routed):
        _pack_routed(weights, moe_inter_dim, weight_names)

    weights[W.moe_gate] = weights.pop(weight_names["router"])
    if weight_names.get("router_bias") in weights:
        weights[W.moe_gate_bias] = weights.pop(weight_names["router_bias"])
    if weight_names.get("router_tid2eid") in weights:
        weights[W.moe_gate_tid2eid] = weights.pop(weight_names["router_tid2eid"])
    if weight_names.get("router_bias_vl") in weights:
        weights[W.moe_gate_bias_vl] = weights.pop(weight_names["router_bias_vl"])

    if n_shared_experts > 0:
        # Absent on a resident restart when a fused executor consumed them through
        # resident_init_tensors().
        for src, dst in (
            ("shared_gate_up", W.ffn_w13),
            ("shared_gate_up_scale", W.ffn_s13),
            ("shared_down", W.ffn_w2),
            ("shared_down_scale", W.ffn_s2),
        ):
            if weight_names[src] in weights:
                weights[dst] = weights.pop(weight_names[src])
    return weights


def _pack_routed(
    weights: Dict[str, torch.Tensor],
    moe_inter_dim: int,
    weight_names: Mapping[str, str],
) -> None:
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
