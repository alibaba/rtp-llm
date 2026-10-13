"""gpt-oss MXFP4 -> BF16 dequantization for GPUs without native FP4 GEMM.

The sm103 (B300) lane consumes the packed MXFP4 experts directly through
DeepGEMM. sm90 (H20) and sm12x (RTX Pro 5000) have no MXFP4 compute path,
so their bring-up dequantizes the packed nibbles + UE8M0 group-32 scales to
BF16 once at layer construction and runs the generic Triton fused-MoE
executor. The decode matches the byte-level oracle verified in
rtp_llm/models/test/gpt_oss_math_test.py.
"""

import os

import torch

# E2M1 magnitudes by 3-bit code; sign bit is the top nibble bit.
_E2M1_MAGS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def dequant_mxfp4_stacked(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize stacked expert MXFP4 tensors to BF16 on their device.

    Args:
        packed: ``[..., K/2]`` uint8 nibble-packed E2M1 weights (the loader's
            ``deinterleave_stack_moe_w1``/``flatten_stack_moe`` output).
        scales: ``[..., K/32]`` uint8 UE8M0 group scales for the same layout.

    Returns:
        ``[..., K]`` BF16 tensor on ``packed.device``.
    """
    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError(
            f"mxfp4 dequant expects uint8 tensors, got {packed.dtype}/{scales.dtype}"
        )
    k2 = packed.shape[-1]
    if k2 * 2 != scales.shape[-1] * 32:
        raise ValueError(
            f"packed K/2={k2} and scales K/32={scales.shape[-1]} disagree"
        )
    device = packed.device
    mags = _E2M1_MAGS.to(device=device, dtype=torch.float32)
    low = (packed & 0xF).long()
    high = (packed >> 4).long()
    lo = mags[low & 7] * torch.where((low & 8) > 0, -1.0, 1.0)
    hi = mags[high & 7] * torch.where((high & 8) > 0, -1.0, 1.0)
    out = torch.stack([lo, hi], dim=-1).flatten(-2)
    k = out.shape[-1]
    groups = k // 32
    scale_f = torch.pow(2.0, scales.to(torch.float32) - 127.0)
    out = out.reshape(*out.shape[:-1], groups, 32) * scale_f.unsqueeze(-1)
    return out.reshape(*out.shape[:-2], k).to(torch.bfloat16)


def use_native_mxfp4() -> bool:
    """Whether the current GPU runs the packed MXFP4 expert path.

    Native block-scaled FP4 GEMM is the DeepGEMM sm100 (major==10) lane;
    sm90 and sm12x dequantize to BF16. ``GPT_OSS_FORCE_DEQUANT=1`` forces
    the BF16 path (for testing the fallback on native GPUs), and
    ``GPT_OSS_FORCE_FP4=1`` forces the packed path. Defaults to the packed
    path when CUDA is unavailable (unit tests on CPU).
    """
    if os.environ.get("GPT_OSS_FORCE_DEQUANT", "") == "1":
        return False
    if os.environ.get("GPT_OSS_FORCE_FP4", "") == "1":
        return True
    if not torch.cuda.is_available():
        return True
    major, _ = torch.cuda.get_device_capability(0)
    return major == 10
