"""Exact-dequant BF16 linears for V4/V4.1 blocked-FP8 checkpoints on ROCm.

gfx942 has no DeepGEMM MXFP8 (E4M3 weights, per-block UE8M0 power-of-two
scales) kernel path. The product of an E4M3 value and a UE8M0 scale is
exactly representable in BF16 for the whole trained-weight range: E4M3
carries a 3-bit significand within BF16's 8-bit significand, and UE8M0
only shifts the exponent. Dequantizing once at weight-binding time
therefore preserves the checkpoint's quantized semantics bit-exactly and
lets the forward run on hipBLASLt BF16 GEMMs.

The class mirrors ``V41MXFP8Linear``'s constructor and ``forward`` contract
so ``_v4_fp8_linear`` can dispatch on ``torch.version.hip``. Packed-UE8M0
activation quantization is a DeepGEMM/SM100-only path; the ROCm runtime
keeps activations in BF16, so ``forward_quantized`` is intentionally not
supported here.
"""

import logging

import torch


def _block_expand(scales: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Expand ``[ceil(n/bn), ceil(k/bk)]`` block scales to ``[n, k]``."""
    sn, sk = scales.shape
    block_n = -(-n // sn)
    block_k = -(-k // sk)
    full = scales.repeat_interleave(block_n, dim=0).repeat_interleave(
        block_k, dim=1
    )
    return full[:n, :k]


def dequantize_blocked_fp8_weight(
    weight: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """(E4M3 ``[N, K]``, UE8M0 block scales) -> exact BF16 ``[N, K]``.

    Raises on any value that BF16 cannot represent exactly (overflow to
    inf or underflow of a nonzero weight to zero), so a violated
    assumption fails loudly at load time instead of corrupting outputs.
    """
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"expected E4M3 weight, got {weight.dtype}")
    if scales.dtype != torch.float8_e8m0fnu:
        raise TypeError(f"expected UE8M0 scales, got {scales.dtype}")
    n, k = weight.shape
    # Both casts are exact: E4M3 -> FP32 widens value and exponent, and
    # UE8M0 -> FP32 materializes a power of two.
    w = weight.float()
    s = _block_expand(scales.float(), n, k)
    out = w * s
    if not torch.isfinite(out).all():
        raise OverflowError("blocked-FP8 dequant overflowed BF16 range")
    if bool(((out == 0) & (w != 0)).any()):
        raise OverflowError("blocked-FP8 dequant underflowed nonzero weights")
    return out.to(torch.bfloat16)


class V41RocmBf16Linear(torch.nn.Module):
    """BF16 stand-in for ``V41MXFP8Linear`` on ROCm.

    Accepts any UE8M0 block scale shape whose blocks tile ``[N, K]``
    (32x32 for V4.1 dense weights, 128x128 for legacy V4 layouts).
    """

    def __init__(self, weight: torch.Tensor, scales: torch.Tensor):
        super().__init__()
        self.N, self.K = weight.shape
        # Keep the small checkpoint scale tensor for placement/provenance
        # reporting; the FP8 payload is fully consumed by the dequant.
        self.weight_scales = scales
        self.weight = dequantize_blocked_fp8_weight(weight, scales)
        try:
            from rtp_llm.model_loader.awi_recorder import register_transform

            register_transform(
                [weight, scales],
                [self.weight, self.weight_scales],
                "blocked_fp8_e4m3_ue8m0_to_bf16_exact",
            )
        except Exception:  # noqa: BLE001
            logging.getLogger(__name__).debug(
                "awi transform registration skipped", exc_info=True
            )

    def forward(self, x: torch.Tensor, out=None):
        if x.numel() == 0:
            return (
                out
                if out is not None
                else x.new_empty((*x.shape[:-1], self.N))
            )
        leading_shape = x.shape[:-1]
        x_2d = x.reshape(-1, self.K)
        if x_2d.dtype != self.weight.dtype:
            x_2d = x_2d.to(self.weight.dtype)
        y = torch.mm(x_2d, self.weight.t())
        if out is not None:
            out.copy_(y.view(out.shape))
            return out
        return y.view(*leading_shape, self.N)

    def forward_quantized(self, x_q: torch.Tensor, x_s: torch.Tensor, out=None):
        raise NotImplementedError(
            "V41RocmBf16Linear consumes BF16 activations; packed-UE8M0 "
            "activation quantization is a DeepGEMM/SM100-only path."
        )
