"""Opt-in Hopper block32 Dense adapter; preserves V4.1 quantized operands.

The isolated SGLang DeepGEMM port does not replace the installed internal
DeepGEMM package. Weight scales are decoded once at load time; activation
scale layout conversion remains part of every measured forward.
"""

import torch

from rtp_llm.models_py.modules.dsv41.sm90_dense import unpack_ue8m0_input_scales


class Sm90DeepGemmMXFP8Linear:
    def __init__(self, weight: torch.Tensor, scales: torch.Tensor):
        if not weight.is_cuda or torch.cuda.get_device_capability(weight.device) != (9, 0):
            raise ValueError("SM90 block32 DeepGEMM requires a CUDA SM90 device")
        if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
            raise TypeError("Expected two-dimensional E4M3 weights")
        self.N, self.K = weight.shape
        # Initial port is deliberately bounded to the aligned V4.1 projections.
        if self.N <= 0 or self.N % 32 or self.K <= 0 or self.K % 128:
            raise ValueError("SM90 block32 port requires N%32=0 and K%128=0")
        if (
            scales.device != weight.device
            or scales.dtype != torch.float8_e8m0fnu
            or tuple(scales.shape) != (self.N // 32, self.K // 32)
        ):
            raise ValueError("Expected same-device UE8M0 32x32 weight scales")
        self.weight = weight.contiguous()
        self.weight_scale = scales.float().contiguous()

    def forward_quantized(self, x_q: torch.Tensor, x_s: torch.Tensor, out=None):
        if x_q.ndim < 1 or x_q.dtype != torch.float8_e4m3fn or x_q.shape[-1] != self.K:
            raise ValueError("Expected prequantized E4M3 activations with matching K")
        if x_q.device != self.weight.device or x_s.device != x_q.device:
            raise ValueError("Activations, scales and weights must share a device")
        rows = x_q.numel() // self.K
        if x_s.dtype != torch.int32 or tuple(x_s.shape) != (rows, self.K // 128):
            raise ValueError("Expected packed group32 UE8M0 activation scales")
        shape = (*x_q.shape[:-1], self.N)
        if out is None:
            out = torch.empty(shape, dtype=torch.bfloat16, device=x_q.device)
        elif (
            tuple(out.shape) != shape
            or out.dtype != torch.bfloat16
            or out.device != x_q.device
            or not out.is_contiguous()
        ):
            raise ValueError("out must be contiguous BF16 with matching shape/device")
        if rows:
            from rtp_llm.models_py.kernels.cuda.sm90_dense_deepgemm import (
                align_activation_scales,
                fp8_gemm_nt,
            )

            # Decode, do not recompute amax or requantize. Include this layout
            # transformation in full-chain benchmarks, not just GEMM timing.
            # Unlike an empty_strided logical view, this also owns the last
            # column's padding rows that the TMA descriptor can address.
            scales = align_activation_scales(unpack_ue8m0_input_scales(x_s))
            fp8_gemm_nt(
                x_q.reshape(rows, self.K).contiguous(),
                scales,
                self.weight,
                self.weight_scale,
                out.view(rows, self.N),
            )
        return out
