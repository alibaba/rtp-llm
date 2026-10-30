"""MiniMax-M3.1 OAI gated activation reference."""

import torch


def swiglu_oai(
    gate_up: torch.Tensor, alpha: float, limit: float, gate_first: bool = True
) -> torch.Tensor:
    if gate_up.shape[-1] % 2:
        raise ValueError("MiniMax-M3.1 gate/up dimension must be even")
    first, second = gate_up.chunk(2, dim=-1)
    gate, up = (first, second) if gate_first else (second, first)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(gate * alpha) * (up + 1)
