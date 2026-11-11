"""PPU dense-linear module for DeepSeek V4.1 Flash.

Implements the ``linear`` op of the V4.1 backend contract
(``rtp_llm/models_py/modules/dsv41/backend.py``) for the PPU strategy
``bf16_dense_w4a16_moe_sparse_mla_host_enp_v1``:

  * Dense/shared projections ship FP8 32x32 UE8M0 weights. They are
    dequantized exactly once at bind time into a resident BF16 weight and
    every forward is a plain BF16 ``F.linear`` (docs/V41_BACKEND_CONTRACT.md,
    ``linear`` row: "Dense/shared FP8 32x32 UE8M0 is dequantized once to
    resident BF16").
  * The dequant matches the rtp-llm oracle in
    ``rtp_llm/models_py/modules/dsv41/test/test_mxfp8_dense.py:104-106``
    bit-for-bit::

        w_bf16 = (w.float() * scales.float()
                  .repeat_interleave(32, 0).repeat_interleave(32, 1)).to(bf16)

    with the UE8M0 scale byte interpreted as ``2 ** (byte - 127)``.

Import discipline: top-level imports are stdlib + torch only. No triton,
deep_gemm or deep_ep import happens at module scope, so this module imports
cleanly on a CPU host with no accelerator wheels installed.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41.ppu.quant_conversion import FP8_INV_MAX

__all__ = [
    "dequant_fp8_block32_ue8m0",
    "PPUResidentBF16Linear",
    "PPUPTPCLinear",
    "FP8_BLOCK_SIZE",
]

FP8_BLOCK_SIZE = 32  # UE8M0 scale block for dense V4.1 weights (32 x 32)


def quantize_ptpc_activation(x):
    """Dynamic symmetric whole-row E4M3, one FP32 scale per token."""
    values = x.float()
    amax = values.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    scale = amax * FP8_INV_MAX
    payload = (values * (448.0 / amax)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return payload.contiguous(), scale.contiguous()


class PPUPTPCLinear:
    """Resident E4M3 weight, FP32 output-channel scale, BF16 GEMM output."""

    def __init__(self, weight, scale):
        if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
            raise TypeError("PTPC linear weight must be E4M3 [N, K]")
        if scale.dtype != torch.float32 or tuple(scale.shape) != (weight.shape[0], 1):
            raise TypeError("PTPC linear scale must be FP32 [N, 1]")
        if weight.device != scale.device:
            raise ValueError("PTPC payload and scale must share a device")
        self.weight, self.scale = weight.contiguous(), scale.contiguous()
        self.output_dtype = torch.bfloat16
        self.quantization = "fp8_per_channel_per_token"
        self._groups = None
        self._gemm = None
        if weight.is_cuda:
            from rtp_llm.models_py.kernels.ppu.runtime import install_deep_gemm_build_lock, require_symbol

            self._gemm = require_symbol("deep_gemm", "fp8_gemm_nt")
            install_deep_gemm_build_lock()

    def prepare_grouped(self, groups, rank):
        if self.weight.shape[0] != groups * rank:
            raise ValueError("grouped PTPC wo_a rows disagree with model geometry")
        self._groups = tuple(
            PPUPTPCLinear(self.weight[i * rank:(i + 1) * rank], self.scale[i * rank:(i + 1) * rank])
            for i in range(groups)
        )

    def forward_grouped(self, x):
        if self._groups is None or x.ndim != 3 or x.shape[1] != len(self._groups):
            raise ValueError("PTPC grouped projection must be prepared before forward")
        return torch.stack([part(x[:, i].contiguous()) for i, part in enumerate(self._groups)], 1)

    def forward(self, x):
        if x.shape[-1] != self.weight.shape[1] or x.device != self.weight.device:
            raise ValueError("PTPC input shape/device differs from its weight")
        shape = (*x.shape[:-1], self.weight.shape[0])
        flat = x.reshape(-1, x.shape[-1])
        output = torch.empty((flat.shape[0], self.weight.shape[0]), device=x.device, dtype=torch.bfloat16)
        if flat.shape[0] == 0:
            return output.reshape(shape)
        payload, scale = quantize_ptpc_activation(flat)
        if x.is_cuda:
            self._gemm((payload, scale), (self.weight, self.scale), output)
        else:
            # Portable component oracle only; device execution always uses GEMM.
            output.copy_((payload.float() @ self.weight.float().T) * scale * self.scale.T)
        return output.reshape(shape)

    __call__ = forward


def _ue8m0_scales_to_float(scales: torch.Tensor) -> torch.Tensor:
    """Return FP32 scale values from an UE8M0 scale tensor.

    Accepts ``float8_e8m0fnu`` (viewed as uint8), raw ``uint8`` exponent bytes
    (value = ``2 ** (byte - 127)``), or an already-float tensor which is used
    as-is.
    """
    if scales.dtype == torch.float8_e8m0fnu:
        scales = scales.view(torch.uint8)
    if scales.dtype == torch.uint8:
        return torch.exp2(scales.to(torch.float32) - 127.0)
    if scales.dtype == torch.int8:  # defensively accept signed byte storage
        return torch.exp2(scales.to(torch.int32).to(torch.float32) - 127.0)
    return scales.to(torch.float32)


def dequant_fp8_block32_ue8m0(
    weight_fp8: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """One-shot dequant of FP8 32x32 UE8M0 weights to resident BF16.

    Args:
        weight_fp8: ``[N, K]`` ``float8_e4m3fn`` (or uint8-backed) weight.
        scale: ``[ceil(N/32), ceil(K/32)]`` UE8M0 scales — either
            ``float8_e8m0fnu``, raw ``uint8`` exponent bytes, or an
            already-floating-point tensor of per-block scale values.

    Returns:
        ``[N, K]`` BF16 tensor equal to the reference formula::

            (weight.float() * scale_values.float()
             .repeat_interleave(32, 0).repeat_interleave(32, 1)).to(bf16)

    Multiply happens in FP32; the single rounding step is the final BF16 cast,
    exactly like ``test_mxfp8_dense.py``'s oracle.
    """
    if weight_fp8.dim() != 2:
        raise ValueError(
            f"weight_fp8 must be 2D [N, K], got shape {tuple(weight_fp8.shape)}"
        )
    n, k = weight_fp8.shape
    scale_values = _ue8m0_scales_to_float(scale)
    if scale_values.dim() != 2:
        raise ValueError(
            f"scale must be 2D [ceil(N/32), ceil(K/32)], got shape "
            f"{tuple(scale_values.shape)}"
        )
    expected_rows = (n + FP8_BLOCK_SIZE - 1) // FP8_BLOCK_SIZE
    expected_cols = (k + FP8_BLOCK_SIZE - 1) // FP8_BLOCK_SIZE
    if tuple(scale_values.shape) != (expected_rows, expected_cols):
        raise ValueError(
            f"scale shape {tuple(scale_values.shape)} does not match weight "
            f"{(n, k)}: expected {(expected_rows, expected_cols)}"
        )
    scale_full = scale_values.repeat_interleave(FP8_BLOCK_SIZE, 0).repeat_interleave(
        FP8_BLOCK_SIZE, 1
    )[:n, :k]
    return (weight_fp8.to(torch.float32) * scale_full).to(torch.bfloat16)


class PPUResidentBF16Linear:
    """BF16-resident linear used for dense projections and shared experts.

    Holds the dequantized BF16 weight (and optional BF16 bias). Output is
    BF16 by default; C2 compressor projections retain FP32 output for pooling
    and unfinished-pair state. Construct directly with a BF16 weight or via
    :meth:`from_quantized` from a V41 weight descriptor (values/scales/format
    per ``V41QuantizedTensor`` in the backend contract).
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        output_dtype: torch.dtype = torch.bfloat16,
    ):
        if weight.dtype != torch.bfloat16:
            raise TypeError(
                f"PPUResidentBF16Linear expects a BF16 weight, got {weight.dtype}"
            )
        self.weight = weight
        self.bias = bias
        if output_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("PPU dense output must be BF16 or FP32")
        self.output_dtype = output_dtype
        if bias is not None and bias.dtype != torch.bfloat16:
            raise TypeError(f"bias must be BF16, got {bias.dtype}")

    @classmethod
    def from_quantized(
        cls,
        values: torch.Tensor,
        scales: Optional[torch.Tensor] = None,
        format: str = "bf16",
        bias: Optional[torch.Tensor] = None,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> "PPUResidentBF16Linear":
        """Build from a V41 weight descriptor (values, scales, format).

        Supported formats: ``bf16`` (values used directly) and
        ``fp8_32x32_ue8m0`` (dequantized once via
        :func:`dequant_fp8_block32_ue8m0`). ``fp4_ue8m0`` is rejected — FP4 is
        reserved for the routed experts handled by
        ``rtp_llm.models_py.modules.dsv41.ppu.moe.PPUMoE``.
        """
        if format == "bf16":
            if values.dtype != torch.bfloat16:
                values = values.to(torch.bfloat16)
            return cls(values, bias, output_dtype=output_dtype)
        if format == "fp8_32x32_ue8m0":
            weight = dequant_fp8_block32_ue8m0(values, scales)
            return cls(weight, bias, output_dtype=output_dtype)
        raise ValueError(
            f"unsupported dense linear weight format {format!r}; expected "
            "'bf16' or 'fp8_32x32_ue8m0'"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_dtype == torch.float32:
            if x.is_cuda:
                output_shape = (*x.shape[:-1], self.weight.shape[0])
                if x.numel() == 0:
                    return x.new_empty(output_shape, dtype=torch.float32)
                output = torch.mm(
                    x.reshape(-1, x.shape[-1]),
                    self.weight.t(),
                    out_dtype=torch.float32,
                ).reshape(output_shape)
                return output if self.bias is None else output + self.bias.float()
            bias = None if self.bias is None else self.bias.float()
            return F.linear(x.float(), self.weight.float(), bias)
        return F.linear(x, self.weight, self.bias)

    __call__ = forward
