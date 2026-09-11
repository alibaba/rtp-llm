"""SGLang masked SwiGLU/MXFP4 with the PPU DeepGEMM scale contract."""

import math

import torch
from rtp_llm.platforms.ppu.kernels.cuda.sglang_jit import load_sglang_kernel


def is_supported(input, masked_m):
    return (
        input.is_cuda
        and torch.cuda.get_device_name(input.device) == "ZW-M890P"
        and input.dtype == torch.bfloat16
        and input.ndim == 3
        and input.shape[-1] > 0
        and input.shape[-1] % 512 == 0
        and input.is_contiguous()
        and masked_m.device == input.device
        and masked_m.dtype == torch.int32
        and masked_m.shape == (input.shape[0],)
        and masked_m.is_contiguous()
    )


def silu_mul_masked_mxfp4(input, masked_m, swiglu_limit=None, expected_m=None):
    """Quantize valid expert rows; leave padding undefined and counts unchanged.

    ``masked_m`` must contain counts in [0, slot_capacity], as produced by
    DeepEP. ``expected_m`` is only a launch hint, never a capacity limit.
    Return packed [E,M,H/2] and logical mn-major uint16 [E,M,H/64] scales.
    """
    if not is_supported(input, masked_m):
        raise ValueError(
            "Masked MXFP4 SwiGLU requires M890P contiguous BF16 [E,M,2H], "
            "H aligned to 256, and contiguous device int32 [E] counts"
        )
    if swiglu_limit is not None and (
        not math.isfinite(swiglu_limit) or swiglu_limit <= 0
    ):
        raise ValueError("swiglu_limit must be positive and finite, or None")
    e, m, two_h = input.shape
    h = two_h // 2
    quant = torch.empty((e, m, h // 2), dtype=torch.uint8, device=input.device)
    physical_scale = torch.empty(
        (e, h // 64, m), dtype=torch.uint16, device=input.device
    )
    if e and m:
        hint = m if expected_m is None else expected_m
        if not isinstance(hint, int) or hint <= 0:
            raise ValueError("expected_m must be a positive host integer")
        apply_clamp = "true" if swiglu_limit is not None else "false"
        module = load_sglang_kernel(
            f"silu_mul_masked_mxfp4_{apply_clamp}",
            "csrc/elementwise/silu_and_mul_masked_post_quant_mxfp4.cuh",
            f"SiluMulMxfp4EP<256,{apply_clamp}>::run",
            torch.cuda.get_device_capability(input.device),
            ("-use_fast_math",),
        )
        module.forward(
            input, quant, physical_scale, masked_m, swiglu_limit or 0.0, hint
        )
    return quant, physical_scale.transpose(-1, -2)
