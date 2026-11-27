"""W8A8_MXFP8 (E8M0 scale) layout helpers for Ascend NPU.

MUST NOT import torch_npu at top level (imported from shared modules that
load on every platform); get_e8m0_dtype resolves torch_npu lazily at call time.
Reference: vllm-ascend w8a8_mxfp8.py process_weights_after_loading:
  2D dense: [N, kp]    -> [kp_pad // 2, N, 2]
  3D MoE:   [E, N, kp] -> [E, kp_pad // 2, N, 2]
"""

import torch
import torch.nn.functional as F

__all__ = ["MXFP8_GROUP_SIZE", "get_e8m0_dtype", "swizzle_scale_to_npu_layout"]

# MXFP8 block size: one E8M0 scale per 32 elements along K.
MXFP8_GROUP_SIZE = 32


def get_e8m0_dtype():
    """Logical dtype of E8M0 scales (prefer torch_npu, fall back to torch)."""
    try:
        import torch_npu  # lazy: keep this module importable on non-NPU hosts

        dtype = getattr(torch_npu, "float8_e8m0fnu", None)
        if dtype is not None:
            return dtype
    except ImportError:
        pass
    return getattr(torch, "float8_e8m0fnu", None)


def swizzle_scale_to_npu_layout(scale: torch.Tensor) -> torch.Tensor:
    """Swizzle E8M0 scales into the pair-split layout required by npu_quant_matmul.

    Must run AFTER the TP split (pair-split spans the whole K-group dim, an
    earlier swizzle would be broken by the split).

    Args:
        scale: uint8 E8M0 scales, 2D ``[N, kp]`` or 3D MoE ``[E, N, kp]``,
            where ``kp = K // 32``.

    Returns:
        2D ``[kp_pad // 2, N, 2]`` or 3D ``[E, kp_pad // 2, N, 2]``; odd ``kp``
        is zero-padded to even (the pad column belongs to no weight group).
        Layout permutation only; E8M0 values unchanged.
    """
    if scale.dim() == 2:
        n, kp = scale.shape
        if kp % 2 != 0:
            scale = F.pad(scale, (0, 1))
            kp += 1
        # [N, kp] -> [N, kp//2, 2] -> [kp//2, N, 2]
        return scale.reshape(n, kp // 2, 2).transpose(0, 1).contiguous()

    if scale.dim() == 3:
        e, n, kp = scale.shape
        if kp % 2 != 0:
            scale = F.pad(scale, (0, 1))
            kp += 1
        # [E, N, kp] -> [E, N, kp//2, 2] -> [E, kp//2, N, 2]
        return scale.reshape(e, n, kp // 2, 2).transpose(1, 2).contiguous()

    raise ValueError(
        f"swizzle_scale_to_npu_layout expects 2D/3D scale, got shape {tuple(scale.shape)}"
    )
