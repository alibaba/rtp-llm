"""K3 router projection using native BF16 GEMM with FP32 output."""

import torch
from torch import nn


class KimiK3RouterProjection(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("K3 router weight must have shape [hidden, experts]")
        fp32 = weight.float()
        self._bf16_exact = weight.dtype == torch.bfloat16 or torch.equal(
            fp32, fp32.bfloat16().float()
        )
        # Preserve genuinely higher-precision checkpoint values.
        dtype = torch.bfloat16 if self._bf16_exact else torch.float32
        # Prepare the cuBLAS [experts, hidden] view at load, never in forward.
        prepared = weight.to(dtype=dtype).t().contiguous().t()
        self.register_buffer("weight", prepared, persistent=False)
        self._native_router = None
        if prepared.is_cuda and self._bf16_exact and torch.cuda.get_device_capability(prepared.device)[0] >= 9:
            from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path
            setup_cutlass_import_path()
            from .native_linear.ll_bf16 import ll_bf16_gemm, ll_bf16_gemm_kernel, ll_bf16_gemm_c1_pdl_kernel
            self._native_router = ll_bf16_gemm
            # Resolve all small-batch compile keys while loading, before Graph capture.
            with torch.cuda.device(prepared.device):
                ll_bf16_gemm_c1_pdl_kernel.warmup(
                    shapes=[tuple(prepared.shape)], m_values=(1,)
                )
                ll_bf16_gemm_kernel.warmup(
                    shapes=[tuple(prepared.shape)], m_values=range(2, 17)
                )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self._native_router is not None and hidden.dtype == torch.bfloat16 and 0 < hidden.shape[0] <= 16:
            return self._native_router(hidden, self.weight.t())
        if hidden.is_cuda and hidden.dtype == torch.bfloat16 and self._bf16_exact:
            from rtp_llm.ops.compute_ops import rtp_llm_ops

            return rtp_llm_ops.cublas_gemm_bf16_bf16_fp32(hidden, self.weight.t())
        return hidden.float() @ self.weight.float()
