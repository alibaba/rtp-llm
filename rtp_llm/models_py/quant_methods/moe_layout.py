"""Runtime MoE layout preparation owned by the module's backend.

This consumes module projection semantics, not checkpoint or legacy W.* names.
The AITER kernels are shared with OldLoader; only ownership and dispatch move.
"""

from typing import Literal

import torch

from rtp_llm.device.architecture import is_gfx950

_FLOAT8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
)


def _rocm_moe_layout(
    tensor: torch.Tensor, *, projection: str, is_scale: bool, gate_first: bool = False
) -> torch.Tensor:
    from aiter.ops.shuffle import shuffle_weight

    original_ndim = tensor.dim()
    result = tensor.unsqueeze(-1) if original_ndim == 2 else tensor
    if projection == "gate_up" and not gate_first:
        # Module checkpoint storage is [up, gate]; the ROCm executor consumes
        # [gate, up]. Byte concatenation preserves FP8 payloads on ROCm builds
        # where torch.cat does not implement the float8 dtype.
        half = result.shape[1] // 2
        pieces = (result[:, half:, :], result[:, :half, :])
        if result.dtype in _FLOAT8_DTYPES:
            result = (
                torch.cat([part.view(torch.uint8) for part in pieces], dim=1)
                .contiguous()
                .view(result.dtype)
            )
        else:
            result = torch.cat(pieces, dim=1).contiguous()
    if not is_scale:
        return shuffle_weight(result, (16, 16))
    if result.dtype == torch.uint8:
        if not is_gfx950(device=tensor.device):
            raise RuntimeError("Quark MXFP4 MoE scale shuffle requires gfx950 (MI355).")
        from aiter.utility.fp4_utils import e8m0_shuffle

        if result.dim() == 3:
            experts, rows, _ = result.shape
            result = e8m0_shuffle(result.contiguous().view(experts * rows, -1)).view(
                experts, rows, -1
            )
        else:
            result = e8m0_shuffle(result)
        if original_ndim == 2:
            result = result.squeeze(-1)
    return result


def prepare_moe_weight_layout(
    tensor: torch.Tensor,
    *,
    projection: Literal["gate_up", "down"],
    is_scale: bool = False,
    gate_first: bool = False,
) -> torch.Tensor:
    """Convert one complete module-owned projection to its runtime layout.

    Call once after quantization and EP/TP placement, before publishing weights
    to an executor. Reloaded EPLB modules use the same conversion. CPU and CUDA
    preserve their layout; ROCm uses the existing AITER preparation kernels.
    """
    if projection not in ("gate_up", "down"):
        raise ValueError(f"Unknown MoE projection {projection!r}")
    if not isinstance(is_scale, bool):
        raise TypeError("is_scale must be a bool")
    # Per-tensor scales have no fused gate/up rows and must remain unchanged.
    if is_scale and tensor.dim() <= 1:
        return tensor
    if tensor.device.type != "cuda" or torch.version.hip is None:
        return tensor
    return _rocm_moe_layout(
        tensor, projection=projection, is_scale=is_scale, gate_first=gate_first
    )


def undo_rocm_moe_matrix_tiles(tensor: torch.Tensor) -> torch.Tensor:
    """Inverse of the old writer's AITER shuffle_weight(layout=(16,16))."""
    experts, n, k = tensor.shape
    if n % 16 or k % 32:
        raise ValueError("Invalid AITER shuffled expert extents")
    lanes = 16 // tensor.element_size()
    # FP8 indexing/permutation is byte-safe: this only moves whole elements.
    return (
        tensor.reshape(experts, n // 16, k // 32, 32 // lanes, 16, lanes)
        .permute(0, 1, 4, 2, 3, 5)
        .contiguous()
        .reshape_as(tensor)
    )
