"""Opt-in, model-local online MXFP8 dense projections for Qwen4 on SM10x.

Reuses V41MXFP8Linear's native K32 quantizer and DeepGEMM (1, 1, 32)
consumer contract. Both operands are E4M3, with per-row packed UE8M0 scales;
this is not a per-tensor FP8 or NVFP4 recipe. Weight quantization is performed
once during model construction, activation quantization on every invocation.
"""

import torch
from torch import nn

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8


def quantize_mxfp8_rows(x):
    return sgl_per_token_group_quant_fp8(
        x.contiguous(),
        group_size=32,
        # Match quant_layouts.per_token_cast_to_fp8_packed_ue8m0's amax floor.
        # This also keeps native small/v2 producers in the same scale domain:
        # the small producer has an additional hardcoded 1e-10 scale floor.
        eps=1e-4,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )


class Qwen4Mxfp8Linear(nn.Module):
    """BF16 [N,K] checkpoint projection with explicit output precision.

    Existing Qwen4 call sites request BF16, just as their F.linear producers.
    FP32 output is available for callers that explicitly require that contract;
    activation/norm/gating and TP collectives stay outside this module.
    """

    def __init__(self, weight, *, output_dtype=torch.bfloat16):
        super().__init__()
        if not self.supports_weight(weight):
            raise ValueError("unsupported Qwen4 SM10x MXFP8 dense weight")
        if output_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("MXFP8 dense output must be BF16 or FP32")
        import deep_gemm

        if not callable(getattr(deep_gemm, "fp8_fp4_gemm_nt", None)):
            raise RuntimeError("Qwen4 MXFP8 requires DeepGEMM fp8_fp4_gemm_nt")
        self.N, self.K = weight.shape
        self.output_dtype = output_dtype
        # Construction may consume a weight allocated on another stream. The
        # caller must establish producer readiness; record_stream separately
        # protects its storage through the asynchronous contiguous copy/quant
        # reads when initialization retires the final BF16 owner immediately.
        weight.record_stream(torch.cuda.current_stream(weight.device))
        self.weight, self.weight_scales = quantize_mxfp8_rows(weight)
        # Existing storage owners remain responsible for checkpoint lifetime.
        # A quantized compute copy cannot observe in-place BF16 weight updates.
        weight._rtp_requires_restart_for_update = True
        self.weight._rtp_requires_restart_for_update = True
        self.bias = None
        self.compute_weight = None
        self.use_scaled_mm = False

    @staticmethod
    def supports_weight(weight):
        return (
            weight.is_cuda
            and torch.version.hip is None
            and weight.dtype == torch.bfloat16
            and weight.ndim == 2
            and weight.shape[0] >= 128
            # TP8 M11 fused QKVZ/BA has N=4640 (32-aligned, not 64).
            and weight.shape[0] % 32 == 0
            and weight.shape[1] > 0
            and weight.shape[1] % 128 == 0
            and torch.cuda.get_device_capability(weight.device)[0] == 10
        )

    def maybe_cache_quant_scale(self, max_len):
        # Deliberately no input-derived static cache, including during graph.
        pass

    def forward(self, x):
        if (
            x.dtype != torch.bfloat16
            or x.device != self.weight.device
            or x.shape[-1] != self.K
        ):
            raise ValueError(
                "MXFP8 projection expects BF16 inputs matching its K/device"
            )
        shape = (*x.shape[:-1], self.N)
        out = torch.empty(shape, device=x.device, dtype=self.output_dtype)
        rows = x.numel() // self.K
        if rows:
            import deep_gemm

            quantized, scales = quantize_mxfp8_rows(x.reshape(rows, self.K))
            deep_gemm.fp8_fp4_gemm_nt(
                (quantized, scales),
                (self.weight, self.weight_scales),
                out.reshape(rows, self.N),
                recipe=(1, 1, 32),
            )
        return out


def maybe_make_mxfp8_projection(weight, bias=None):
    if bias is None and Qwen4Mxfp8Linear.supports_weight(weight):
        return Qwen4Mxfp8Linear(weight)
    return None
