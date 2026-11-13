"""Shared wo_a entry; numerical providers and BF16 policies stay separate."""

import torch


def project_wo_a(output, projection, *, bf16_projection, prefill=False):
    """Dispatch by the resident projection protocol without converting weights.

    PTPC keeps its existing quantizer/provider and Prefill-only direct opt-in.
    Other projections use the caller's BF16 policy, including its admission,
    layout, fallback and telemetry. Provider errors propagate without retry.
    """
    if hasattr(projection, "forward_grouped"):
        if prefill:
            return projection.forward_grouped(output, prefill=True)
        return projection.forward_grouped(output)
    return bf16_projection(output, projection.weight)


def batched_wo_a(output, weight, groups, rank):
    """Write token-major wo_b inputs with one BF16 BMM and view transposes.

    Prefill and Graph callers validate their own geometry, device and layout
    before entering this operation. No packing copy or arithmetic changes.
    """
    projected = output.new_empty((output.shape[0], groups, rank))
    grouped_weight = weight.view(groups, rank, output.shape[-1])
    torch.bmm(
        output.transpose(0, 1),
        grouped_weight.transpose(1, 2),
        out=projected.transpose(0, 1),
    )
    return projected
