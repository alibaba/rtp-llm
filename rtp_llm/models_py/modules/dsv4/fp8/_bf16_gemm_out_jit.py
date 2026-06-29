"""JIT out-variant for DSV4 compressor BF16 GEMM.

This is an experiment-only helper.  The built-in
``cublas_gemm_bf16_bf16_fp32`` op allocates its FP32 output internally; this
module provides the same cuBLAS call but writes into a caller-owned output
tensor so the small-token bypass can test compressor projection workspace reuse
without rebuilding the production extension.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.cpp_extension import load_inline

logger = logging.getLogger(__name__)

_EXT: Optional[Any] = None
_EXT_ERROR: Optional[BaseException] = None


_CPP_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <torch/extension.h>

#include <limits>

void cublas_gemm_bf16_bf16_fp32_out(
    const at::Tensor& input,
    const at::Tensor& weight,
    at::Tensor& out) {
    TORCH_CHECK(input.is_cuda(), "bf16_gemm_out: input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "bf16_gemm_out: weight must be CUDA");
    TORCH_CHECK(out.is_cuda(), "bf16_gemm_out: out must be CUDA");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "bf16_gemm_out: input must be bfloat16");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "bf16_gemm_out: weight must be bfloat16");
    TORCH_CHECK(out.scalar_type() == at::kFloat, "bf16_gemm_out: out must be float32");
    TORCH_CHECK(input.dim() == 2 && weight.dim() == 2 && out.dim() == 2,
                "bf16_gemm_out: input, weight and out must be 2-D");
    TORCH_CHECK(input.get_device() == weight.get_device() && input.get_device() == out.get_device(),
                "bf16_gemm_out: tensors must be on the same CUDA device");
    TORCH_CHECK(input.size(1) == weight.size(1), "bf16_gemm_out: inner dimensions must match");
    TORCH_CHECK(out.size(0) == input.size(0) && out.size(1) == weight.size(0),
                "bf16_gemm_out: out shape mismatch");
    TORCH_CHECK(input.is_contiguous(), "bf16_gemm_out: input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "bf16_gemm_out: weight must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "bf16_gemm_out: out must be contiguous");

    const int64_t M = input.size(0);
    const int64_t N = weight.size(0);
    const int64_t K = input.size(1);
    const int64_t int_max = static_cast<int64_t>(std::numeric_limits<int>::max());
    TORCH_CHECK(M <= int_max && N <= int_max && K <= int_max,
                "bf16_gemm_out: dimensions exceed cuBLAS int32 limit");

    const c10::cuda::CUDAGuard device_guard(input.device());
    if (out.numel() == 0) {
        return;
    }
    if (K == 0) {
        out.zero_();
        return;
    }

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    TORCH_CUDABLAS_CHECK(cublasSetStream(handle, at::cuda::getCurrentCUDAStream(input.get_device())));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    TORCH_CUDABLAS_CHECK(cublasGemmEx(handle,
                                      CUBLAS_OP_T,
                                      CUBLAS_OP_N,
                                      static_cast<int>(N),
                                      static_cast<int>(M),
                                      static_cast<int>(K),
                                      &alpha,
                                      weight.data_ptr(),
                                      CUDA_R_16BF,
                                      static_cast<int>(K),
                                      input.data_ptr(),
                                      CUDA_R_16BF,
                                      static_cast<int>(K),
                                      &beta,
                                      out.data_ptr(),
                                      CUDA_R_32F,
                                      static_cast<int>(N),
                                      CUBLAS_COMPUTE_32F,
                                      CUBLAS_GEMM_DEFAULT));
}
"""


def _extension_dir() -> Optional[str]:
    root = (
        os.environ.get("DSV4_JIT_EXT_DIR")
        or os.environ.get("REMOTE_JIT_DIR")
        or os.environ.get("TORCH_EXTENSIONS_DIR")
    )
    if not root:
        return None
    path = Path(root) / "torch_extensions"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _load_ext() -> Any:
    global _EXT, _EXT_ERROR
    if _EXT is not None:
        return _EXT
    if _EXT_ERROR is not None:
        raise RuntimeError("DSV4 BF16 GEMM out JIT extension is unavailable") from _EXT_ERROR

    cuda_home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    include_dir = Path(cuda_home) / "include"
    lib_dir = Path(cuda_home) / "lib64"
    try:
        _EXT = load_inline(
            name="rtp_dsv4_bf16_gemm_out_ext",
            cpp_sources=[_CPP_SOURCE],
            functions=["cublas_gemm_bf16_bf16_fp32_out"],
            extra_cflags=["-O3"],
            extra_include_paths=[str(include_dir)] if include_dir.exists() else [],
            extra_ldflags=[f"-L{lib_dir}", "-lcublas"] if lib_dir.exists() else ["-lcublas"],
            build_directory=_extension_dir(),
            with_cuda=False,
            verbose=os.environ.get("DSV4_JIT_EXT_VERBOSE", "0") == "1",
        )
        return _EXT
    except BaseException as exc:
        _EXT_ERROR = exc
        logger.exception("failed to build DSV4 BF16 GEMM out JIT extension")
        raise


def cublas_gemm_bf16_bf16_fp32_out(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    _load_ext().cublas_gemm_bf16_bf16_fp32_out(input, weight, out)
    return out
