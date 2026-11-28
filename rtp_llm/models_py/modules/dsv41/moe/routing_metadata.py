"""Token-only router inputs shared by the layers of one model forward."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class GateRoutingMetadata:
    input_ids: torch.Tensor
    image_mask: torch.Tensor
    image_mask_column: torch.Tensor
    safe_hash_input_ids: Optional[torch.Tensor]
    _slices: Dict[Tuple[int, int], "GateRoutingMetadata"] = field(
        default_factory=dict, repr=False
    )

    def slice(self, start: int, end: int) -> "GateRoutingMetadata":
        """Reuse MoE chunk views across layers, never across model forwards."""
        key = (start, end)
        chunk = self._slices.get(key)
        if chunk is None:
            chunk = GateRoutingMetadata(
                self.input_ids[start:end],
                self.image_mask[start:end],
                self.image_mask_column[start:end],
                (
                    self.safe_hash_input_ids[start:end]
                    if self.safe_hash_input_ids is not None
                    else None
                ),
            )
            self._slices[key] = chunk
        return chunk


def prepare_gate_routing_metadata(
    input_ids: torch.Tensor,
    *,
    vocab_size: int,
    has_hash_routing: bool,
) -> GateRoutingMetadata:
    # Preserve RTP's visual sentinel contract (all ids >= vocab_size), including
    # the rank-local CP padding/token order. No host read or dynamic indexing.
    flat_ids = input_ids if input_ids.ndim == 1 else input_ids.reshape(-1)
    image_mask = flat_ids >= vocab_size
    return GateRoutingMetadata(
        flat_ids,
        image_mask,
        image_mask.unsqueeze(-1),
        torch.where(image_mask, 0, flat_ids) if has_hash_routing else None,
    )


def prepare_model_gate_routing_metadata(
    model: Any, input_ids: torch.Tensor
) -> Optional[GateRoutingMetadata]:
    # Text-only forwards already bypass these operations via fused gate/pack.
    # A fresh local is created during every eager forward / graph capture. CUDA
    # graph replay then reruns the comparison on the current input-id buffer.
    if not getattr(model, "_has_visual_tokens", False):
        return None
    return prepare_gate_routing_metadata(
        input_ids,
        vocab_size=int(model.args.vocab_size),
        has_hash_routing=int(model.args.n_hash_layers) > 0,
    )
