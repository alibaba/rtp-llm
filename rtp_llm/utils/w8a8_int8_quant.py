"""CPU W8A8 INT8 checkpoint helpers for Qwen MoE and attention weights.

The recipe stores each selected floating-point weight as symmetric INT8 with
one FP32 scale per output channel.  For a dense ``[N, K]`` tensor, rows are
the output channels.  For Qwen's stacked routed-expert ``[E, N, K]`` tensors,
every ``[N, K]`` expert slice follows the same rule.  Qwen3.5 checkpoint
conversion uses the complete suffix recipe below.  The tensor layout is also
applicable to compatible Qwen3.6 checkpoints, but Qwen3.6 conversion has not
been validated by this module alone.

This module intentionally depends only on PyTorch.  It is shared by offline
checkpoint conversion and CPU-side loading paths, so it must not import
platform-specific code, internal-source modules, configuration, or ops.
"""

from __future__ import annotations

import torch

WEIGHT_SUFFIX = ".weight"
SCALE_SUFFIX = ".weight_scale"
EPSILON = 1.0e-10

# Fused routed-expert tensors deliberately omit ``.weight``.  Keep their
# checkpoint key intact and append only ``.weight_scale`` for their scale.
STACKED_EXPERT_SUFFIXES = (
    ".mlp.experts.gate_up_proj",
    ".mlp.experts.down_proj",
)
DENSE_WEIGHT_SUFFIXES = (
    ".mlp.shared_expert.gate_proj.weight",
    ".mlp.shared_expert.up_proj.weight",
    ".mlp.shared_expert.down_proj.weight",
    ".linear_attn.in_proj_qkv.weight",
    ".linear_attn.in_proj_z.weight",
    ".linear_attn.out_proj.weight",
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
)


def quantize_weight_per_output_channel(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return INT8 weights and FP32 ``[..., 1]`` output-channel scales.

    The final dimension is K; each preceding index selects an independent
    output channel.  This is the exact CPU recipe used by the Qwen3.5 offline
    converter: ``absmax.clamp_min(1e-10) / 127``, then round and clamp to the
    signed INT8 range.  Non-finite validation belongs to the caller so this
    primitive preserves the converter's established API behavior.
    """

    if weight.ndim < 2 or not weight.is_floating_point():
        raise ValueError(
            f"W8A8 requires a floating tensor with ndim >= 2, got "
            f"dtype={weight.dtype}, shape={tuple(weight.shape)}"
        )
    rows = weight.reshape(-1, weight.shape[-1]).float()
    scales = rows.abs().amax(dim=1, keepdim=True).clamp_min(EPSILON).div_(127.0)
    quantized = torch.round(rows / scales).clamp_(-128, 127).to(torch.int8)
    return quantized.reshape(weight.shape), scales.reshape(*weight.shape[:-1], 1)


def is_recipe_weight(name: str) -> bool:
    """Return whether ``name`` is an explicitly approved W8A8 recipe key."""

    if name.endswith(SCALE_SUFFIX):
        return False
    return name.endswith(STACKED_EXPERT_SUFFIXES + DENSE_WEIGHT_SUFFIXES)


def scale_name_for(weight_name: str) -> str:
    """Return the compressed-tensors FP32 scale key for a recipe weight."""

    if not is_recipe_weight(weight_name):
        raise ValueError(f"not a recipe weight: {weight_name}")
    base = (
        weight_name[: -len(WEIGHT_SUFFIX)]
        if weight_name.endswith(WEIGHT_SUFFIX)
        else weight_name
    )
    return base + SCALE_SUFFIX


__all__ = [
    "DENSE_WEIGHT_SUFFIXES",
    "EPSILON",
    "SCALE_SUFFIX",
    "STACKED_EXPERT_SUFFIXES",
    "WEIGHT_SUFFIX",
    "is_recipe_weight",
    "quantize_weight_per_output_channel",
    "scale_name_for",
]
