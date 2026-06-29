#include "rtp_llm/models_py/bindings/cuda/Bf16GemmOp.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>

#include <cstdint>
#include <limits>

namespace torch_ext {

namespace {

void checkBf16GemmInputs(const at::Tensor& input, const at::Tensor& weight) {
    TORCH_CHECK(input.is_cuda(), "cublas_gemm_bf16_bf16_fp32: input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "cublas_gemm_bf16_bf16_fp32: weight must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "cublas_gemm_bf16_bf16_fp32: input must be bfloat16");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "cublas_gemm_bf16_bf16_fp32: weight must be bfloat16");
    TORCH_CHECK(input.dim() == 2 && weight.dim() == 2, "cublas_gemm_bf16_bf16_fp32: input and weight must be 2-D");
    TORCH_CHECK(input.get_device() == weight.get_device(),
                "cublas_gemm_bf16_bf16_fp32: input and weight must be on the same CUDA device");
    TORCH_CHECK(input.size(1) == weight.size(1), "cublas_gemm_bf16_bf16_fp32: inner dimensions must match");

    const int64_t M       = input.size(0);
    const int64_t N       = weight.size(0);
    const int64_t K       = input.size(1);
    const int64_t int_max = static_cast<int64_t>(std::numeric_limits<int>::max());
    TORCH_CHECK(M <= int_max && N <= int_max && K <= int_max,
                "cublas_gemm_bf16_bf16_fp32: dimensions exceed cuBLAS int32 limit");
}

bool rangesOverlap(const at::Tensor& a, const at::Tensor& b) {
    if (a.numel() == 0 || b.numel() == 0) {
        return false;
    }
    const auto a_begin = reinterpret_cast<uintptr_t>(a.data_ptr());
    const auto b_begin = reinterpret_cast<uintptr_t>(b.data_ptr());
    const auto a_end   = a_begin + static_cast<uintptr_t>(a.nbytes());
    const auto b_end   = b_begin + static_cast<uintptr_t>(b.nbytes());
    return a_begin < b_end && b_begin < a_end;
}

void checkBf16GemmOut(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& out) {
    checkBf16GemmInputs(input, weight);
    TORCH_CHECK(out.is_cuda(), "cublas_gemm_bf16_bf16_fp32: out must be a CUDA tensor");
    TORCH_CHECK(out.scalar_type() == at::kFloat, "cublas_gemm_bf16_bf16_fp32: out must be float32");
    TORCH_CHECK(out.dim() == 2, "cublas_gemm_bf16_bf16_fp32: out must be 2-D");
    TORCH_CHECK(input.get_device() == out.get_device(),
                "cublas_gemm_bf16_bf16_fp32: input, weight and out must be on the same CUDA device");
    TORCH_CHECK(out.size(0) == input.size(0) && out.size(1) == weight.size(0),
                "cublas_gemm_bf16_bf16_fp32: out shape mismatch");
}

void runBf16Gemm(const at::Tensor& input, const at::Tensor& weight, at::Tensor& out) {
    checkBf16GemmOut(input, weight, out);
    TORCH_CHECK(input.is_contiguous(), "cublas_gemm_bf16_bf16_fp32: input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "cublas_gemm_bf16_bf16_fp32: weight must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "cublas_gemm_bf16_bf16_fp32: out must be contiguous");
    TORCH_CHECK(!rangesOverlap(out, input) && !rangesOverlap(out, weight),
                "cublas_gemm_bf16_bf16_fp32: out must not overlap input or weight");

    const int64_t M = input.size(0);
    const int64_t N = weight.size(0);
    const int64_t K = input.size(1);

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
    const float beta  = 0.0f;

    // PyTorch tensors are row-major. cuBLAS sees them as column-major and
    // computes C(N, M) = weight(N, K) @ input(K, M), which is out[M, N]^T.
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

}  // namespace

at::Tensor cublas_gemm_bf16_bf16_fp32(const at::Tensor& input, const at::Tensor& weight) {
    checkBf16GemmInputs(input, weight);
    auto input_contig  = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto out = at::empty({input_contig.size(0), weight_contig.size(0)}, input.options().dtype(at::kFloat));
    runBf16Gemm(input_contig, weight_contig, out);
    return out;
}

at::Tensor cublas_gemm_bf16_bf16_fp32_out(const at::Tensor& input, const at::Tensor& weight, at::Tensor& out) {
    runBf16Gemm(input, weight, out);
    return out;
}

}  // namespace torch_ext
