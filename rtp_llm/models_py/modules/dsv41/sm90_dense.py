"""MXFP8 scale utilities and legacy class name for native DeepGEMM."""

import torch


def expand_block32_scales(scales: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Expand checkpoint blocks to per-row g32 without converting E8M0 values."""
    if scales.dtype != torch.float8_e8m0fnu:
        raise TypeError("V4.1 MXFP8 weight scales must be UE8M0")
    expected = ((rows + 31) // 32, (cols + 31) // 32)
    if tuple(scales.shape) != expected:
        raise ValueError(f"Expected block32 scales {expected}, got {scales.shape}")
    # Float8 index/repeat kernels are not implemented by all supported PyTorch
    # versions. Moving the bytes also preserves every checkpoint exponent.
    return (
        scales.view(torch.uint8)
        .repeat_interleave(32, dim=0)[:rows]
        .contiguous()
        .view(torch.float8_e8m0fnu)
    )


def unpack_ue8m0_input_scales(scales: torch.Tensor) -> torch.Tensor:
    """Losslessly decode RTP packed scales to the SM90 FP32-scale ABI.

    The C++ launcher accepts packed int32 only for MXMMA, not Hopper WGMMA.
    Casting the UE8M0 bytes also preserves byte 0 (2**-127) and byte 255 (NaN),
    unlike an unchecked exponent-bit shift. Packed tail groups cover padded K.
    """
    if scales.dtype != torch.int32 or scales.ndim != 2:
        raise ValueError("Expected a 2-D packed int32 UE8M0 scale tensor")
    packed = scales.contiguous()
    # PyTorch ignores singleton strides when deciding contiguity. A TMA
    # [M,1] scale view can therefore retain stride (1,align(M,4)), whereas
    # dtype reinterpretation requires its last stride to be exactly one.
    # The storage is contiguous here; canonicalizing its metadata is safe and
    # avoids another copy, including the [1,1] and empty-row cases.
    packed = packed.as_strided(packed.shape, (packed.shape[1], 1))
    return packed.view(torch.float8_e8m0fnu).float()


# Preserve callers importing the historical adapter name without another backend.
from .sm90_deepgemm_dense import Sm90DeepGEMMMXFP8Linear

Sm90MXFP8Linear = Sm90DeepGEMMMXFP8Linear
