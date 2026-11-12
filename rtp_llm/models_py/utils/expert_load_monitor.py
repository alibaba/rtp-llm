"""Expert token counts accumulated into the current step's output."""

from __future__ import annotations

from typing import Optional

import torch


def record_expert_load(
    topk_ids: torch.Tensor,
    expert_token_counts: torch.Tensor,
) -> None:
    """Add destination-visible assignments, including padding, to step counts."""
    from rtp_llm.ops import compute_ops

    compute_ops.record_expert_load(topk_ids, expert_token_counts)


class ExpertLoadContext:
    """A layer's step count tensor, indexed by local expert ID."""

    def __init__(self, expert_token_counts: torch.Tensor):
        self.expert_token_counts = expert_token_counts

    def record_ids(self, ids: torch.Tensor) -> None:
        record_expert_load(ids, self.expert_token_counts)


def expert_load_for_layer(
    expert_token_counts: Optional[torch.Tensor], layer_id: int
) -> Optional[ExpertLoadContext]:
    if expert_token_counts is None:
        return None
    return ExpertLoadContext(expert_token_counts[layer_id])


def collect_eplb_layers(model: torch.nn.Module):
    """Discover routed layers through their shared MoE interface."""
    layers = []
    for module in model.modules():
        if module is model:
            continue
        get_layer = getattr(module, "get_eplb_layer", None)
        if callable(get_layer):
            layer = get_layer()
            if layer is None:
                raise ValueError("EPLB routing maps are missing for a routed layer")
            layers.append(layer)
    if not layers:
        raise NotImplementedError(f"{type(model).__name__} does not expose EPLB layers")
    if len({layer.layer_id for layer in layers}) != len(layers):
        raise ValueError("EPLB requires unique routed layer IDs")
    return sorted(layers, key=lambda layer: layer.layer_id)


def bind_model_expert_load(model: torch.nn.Module, counts: torch.Tensor) -> None:
    """Bind persistent count views before capture without model forward hooks."""
    if counts.ndim != 2 or counts.dtype != torch.int32 or not counts.is_contiguous():
        raise ValueError(
            "Expert counts must be a contiguous int32 [layer, expert] tensor"
        )
    modules = [
        module
        for module in model.modules()
        if module is not model and callable(getattr(module, "bind_expert_load", None))
    ]
    for module in modules:
        if not 0 <= module.expert_load_layer_id < counts.shape[0]:
            raise ValueError("Expert count buffer does not cover all routed layers")
    with torch.inference_mode(False):
        for module in modules:
            module.bind_expert_load(
                ExpertLoadContext(counts[module.expert_load_layer_id])
            )
