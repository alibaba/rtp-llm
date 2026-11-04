"""FP8 KV packing for the LongCat LSA attention (DeepSeek-style 656-byte layout).

Layout of one token, as consumed by FlashMLA's sparse FP8 path (and by our
own kernels)::

    bytes   0..511 : 512 x float8_e4m3  ("quantized NoPE" = the shared latent,
                                        used as both K's NoPE part and as V)
    bytes 512..527 :   4 x float32     (scales; scale q covers e4m3[q*128:(q+1)*128])
    bytes 528..655 :  64 x bfloat16    ("RoPE" part, kept unquantized)

``pack_kv_fp8`` turns a ``[N, 576]`` bf16 KV tensor into ``[N, 656]`` uint8;
``unpack_kv_fp8`` inverts it (used for verification of accuracy loss).

Measured quantization error on the calibration case: relative Frobenius 2.43%
(baseline-log #5.149); the FP8 path itself is storage-format support -- it does
*not* speed up the available FlashMLA kernels on this shape (see #5.154).
"""

from __future__ import annotations

import torch

__all__ = [
    "FP8_KV_DIM",
    "FP8_SCALE_GROUP",
    "no_fp8_kvcache",
    "kv_lora_rank",
    "pack_kv_fp8",
    "unpack_kv_fp8",
]

FP8_KV_DIM = 656          # bytes per token
FP8_SCALE_GROUP = 128     # e4m3 values covered by one fp32 scale
kv_lora_rank = 512        # NoPE width (shared by K and V)
no_fp8_kvcache = 64       # bf16 RoPE width
_E4M3_MAX = 448.0


def pack_kv_fp8(kv: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """``[N, 576]`` bf16 -> ``[N, 656]`` uint8 in the FP8 sparse layout."""
    if kv.dim() != 2 or kv.shape[1] != kv_lora_rank + no_fp8_kvcache:
        raise ValueError(f"expected [N, {kv_lora_rank + no_fp8_kvcache}] bf16, got {tuple(kv.shape)}")
    n = kv.shape[0]
    buf = torch.empty(n, FP8_KV_DIM, dtype=torch.uint8, device=kv.device) if out is None else out
    nope = kv[:, :kv_lora_rank].float().view(n, kv_lora_rank // FP8_SCALE_GROUP, FP8_SCALE_GROUP)
    scale = (nope.abs().amax(dim=2, keepdim=True) / _E4M3_MAX).clamp_min(1e-6)
    buf[:, :kv_lora_rank].view(torch.float8_e4m3fn).copy_(
        (nope / scale).clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn).view(n, kv_lora_rank)
    )
    buf[:, kv_lora_rank:kv_lora_rank + 16].view(torch.float32).copy_(scale.view(n, -1))
    buf[:, kv_lora_rank + 16:].view(torch.bfloat16).copy_(kv[:, kv_lora_rank:])
    return buf


def unpack_kv_fp8(buf: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """``[N, 656]`` uint8 -> ``[N, 576]`` (dequantized; rope part unchanged)."""
    if buf.dim() != 2 or buf.shape[1] != FP8_KV_DIM:
        raise ValueError(f"expected [N, {FP8_KV_DIM}] uint8, got {tuple(buf.shape)}")
    n = buf.shape[0]
    g = kv_lora_rank // FP8_SCALE_GROUP
    q = buf[:, :kv_lora_rank].view(torch.float8_e4m3fn).float().view(n, g, FP8_SCALE_GROUP)
    scale = buf[:, kv_lora_rank:kv_lora_rank + 16].view(torch.float32).view(n, g, 1)
    nope = (q * scale).view(n, kv_lora_rank)
    rope = buf[:, kv_lora_rank + 16:].view(torch.bfloat16)
    return torch.cat([nope, rope.float()], dim=1).to(dtype)
