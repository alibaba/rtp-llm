"""FP8 layout preparation; no re-quantization of the loaded expert weights."""

import torch

from rtp_llm.models_py.modules.factory.fused_moe.utils.weight_layout import (
    normalize_moe_w13_gate_up,
)


def expand_fp8_scale(scale, nrows, kcols):
    """Express 128-value UE8M0 groups as identical 32-value subgroups."""
    from deep_gemm.utils.math import pack_ue8m0_to_int, unpack_ue8m0_from_int

    if scale.dtype != torch.int32:
        raise ValueError(
            "MegaMoE FP8 expects packed UE8M0 scales from the FP8 weight loader"
        )
    values = unpack_ue8m0_from_int(scale.contiguous())
    if values.shape[-2] == (nrows + 127) // 128:
        values = values.repeat_interleave(128, dim=-2)[..., :nrows, :]
    if values.shape[-1] == (kcols + 127) // 128:
        values = values.repeat_interleave(4, dim=-1)
    if values.shape[-2:] != (nrows, kcols // 32):
        raise ValueError("FP8 scale geometry does not match the expert weight")
    return pack_ue8m0_to_int(values.contiguous())


def _reuse_weight_storage(source, packed):
    weight, scale = packed
    if (
        source.dtype != weight.dtype
        or source.device != weight.device
        or source.numel() != weight.numel()
    ):
        raise ValueError(
            "Packed expert weight must preserve dtype, device and element count"
        )
    extent = 1 + sum((n - 1) * s for n, s in zip(weight.shape, weight.stride()))
    available = (
        source.untyped_storage().nbytes() // source.element_size()
        - source.storage_offset()
    )
    if (
        any(s < 0 for s in weight.stride())
        or extent != weight.numel()
        or extent > available
    ):
        raise ValueError(
            "Packed expert weight requires a dense layout fitting the original storage"
        )
    destination = source.as_strided(weight.shape, weight.stride())
    if destination.data_ptr() != weight.data_ptr():
        destination.copy_(weight)
    elif destination.stride() != weight.stride():
        raise ValueError("Aliased packed weight has an incompatible layout")
    return destination, scale.transpose(-1, -2).contiguous().transpose(-1, -2)


@torch.no_grad()
def prepare_mega_moe_fp8_weights(w13, s13, w2, s2, inter_dim, layout):
    """Pack once at executor construction; the selected backend owns the storage.

    Changing the backend requires reloading the model. Scale expansion duplicates
    existing values; the kernel's internal SwiGLU quantization is a separate step.
    """
    from deep_gemm import mega_fp8

    if w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        raise ValueError("MegaMoE FP8 requires E4M3 expert weights")
    if (
        w13.ndim != 3
        or w2.shape != (w13.shape[0], w13.shape[2], inter_dim)
        or w13.shape[1] != 2 * inter_dim
    ):
        raise ValueError("MegaMoE expert weight geometry is inconsistent")
    s13 = expand_fp8_scale(s13, w13.shape[1], w13.shape[2])
    s2 = expand_fp8_scale(s2, w2.shape[1], w2.shape[2])
    gate_up, s13 = normalize_moe_w13_gate_up(w13, s13, inter_dim, layout)
    l1, l2 = mega_fp8.transform_weights_for_mega_moe_fp8((gate_up, s13), (w2, s2))
    return _reuse_weight_storage(w13, l1), _reuse_weight_storage(w2, l2)
