from __future__ import annotations

from typing import Dict, List, Optional

import torch


def resolve_aux_hidden_states_layers(
    layer_ids: Optional[torch.Tensor],
    num_layers: int,
) -> List[int]:
    if layer_ids is None or layer_ids.numel() == 0:
        if num_layers < 4:
            raise ValueError(
                "Default aux hidden states layers require at least 4 layers, "
                f"got {num_layers}"
            )
        layers = [1, num_layers // 2 - 1, num_layers - 4]
    else:
        layers = [int(x) for x in layer_ids.detach().cpu().tolist()]

    if not layers:
        raise ValueError("aux_hidden_states_layers must not be empty")

    seen = set()
    for idx, layer_id in enumerate(layers):
        if layer_id < 0 or layer_id >= num_layers:
            raise ValueError(
                f"aux_hidden_states_layers[{idx}]={layer_id} out of range "
                f"[0, {num_layers})"
            )
        if layer_id in seen:
            raise ValueError(
                f"aux_hidden_states_layers contains duplicate layer id {layer_id}"
            )
        seen.add(layer_id)
    return layers


class AuxHiddenStatesCapture:
    def __init__(self, layers: List[int], template: torch.Tensor):
        self.layers = layers
        self.layer_offsets: Dict[int, int] = {
            layer_id: idx for idx, layer_id in enumerate(layers)
        }
        self.flat_dim = int(template.size(-2) * template.size(-1))
        self.token_count = int(template.numel() // max(self.flat_dim, 1))
        self.tensor = torch.empty(
            (self.token_count, len(layers) * self.flat_dim),
            dtype=template.dtype,
            device=template.device,
        )

    def maybe_capture(self, layer_id: int, hidden: torch.Tensor) -> None:
        offset = self.layer_offsets.get(layer_id)
        if offset is None:
            return
        flat = hidden.reshape(self.token_count, self.flat_dim)
        start = offset * self.flat_dim
        self.tensor[:, start : start + self.flat_dim].copy_(flat)


def make_aux_hidden_states_layers_tensor(
    layers: List[int],
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(layers, dtype=torch.int32, device=device)
