"""Shared expert gate/up layout normalization."""

import torch


def split_moe_w13_gate_up(
    w13: torch.Tensor,
    inter_dim: int,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return gate/up views from an explicitly declared W.moe_w1 layout."""

    if w13.size(-2) != 2 * inter_dim:
        raise ValueError(f"W.moe_w1 has {w13.size(-2)} rows, expected {2 * inter_dim}")
    first = w13[..., :inter_dim, :]
    second = w13[..., inter_dim:, :]
    if layout == "gate_up":
        return first, second
    if layout == "up_gate":
        return second, first
    raise ValueError(f"moe_w1_layout must be 'gate_up' or 'up_gate', got {layout!r}")


def normalize_moe_w13_gate_up(
    w13: torch.Tensor,
    s13: torch.Tensor,
    inter_dim: int,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return weight/scale tensors in the gate|up order required by DeepGEMM."""

    if layout == "gate_up":
        # Validate both tensors even when no reorder is needed.
        split_moe_w13_gate_up(w13, inter_dim, layout)
        split_moe_w13_gate_up(s13, inter_dim, layout)
        return w13, s13
    gate_w, up_w = split_moe_w13_gate_up(w13, inter_dim, layout)
    gate_s, up_s = split_moe_w13_gate_up(s13, inter_dim, layout)
    return (
        torch.cat((gate_w, up_w), dim=-2),
        # UE8M0 scales are stored as uint8 or float8_e8m0fnu. Reorder their
        # bytes because torch.cat does not support the latter CUDA dtype.
        torch.cat((gate_s.view(torch.uint8), up_s.view(torch.uint8)), dim=-2).view(
            s13.dtype
        ),
    )
