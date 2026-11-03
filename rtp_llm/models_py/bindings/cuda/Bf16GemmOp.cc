#include "rtp_llm/models_py/bindings/cuda/Bf16GemmOp.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>

#include <limits>

namespace torch_ext {

at::Tensor cublas_gemm_bf16_bf16_fp32(const at::Tensor& input, const at::Tensor& weight) {
    return cublas_gemm_bf16_bf16_fp32_out(input, weight, {});
}

at::Tensor cublas_gemm_bf16_bf16_fp32_out(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& output) {
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

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto                       input_contig  = input.contiguous();
    auto                       weight_contig = weight.contiguous();
    auto                       out = output.defined() ? output : at::empty({M, N}, input.options().dtype(at::kFloat));
    TORCH_CHECK(out.device() == input.device() && out.scalar_type() == at::kFloat && out.is_contiguous()
                    && out.dim() == 2 && out.size(0) == M && out.size(1) == N,
                "cublas_gemm_bf16_bf16_fp32_out: expected contiguous CUDA FP32 [M,N] output");
    if (out.numel() == 0) {
        return out;
    }
    if (K == 0) {
        return out.zero_();
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
                                      weight_contig.data_ptr(),
                                      CUDA_R_16BF,
                                      static_cast<int>(K),
                                      input_contig.data_ptr(),
                                      CUDA_R_16BF,
                                      static_cast<int>(K),
                                      &beta,
                                      out.data_ptr(),
                                      CUDA_R_32F,
                                      static_cast<int>(N),
                                      CUBLAS_COMPUTE_32F,
                                      CUBLAS_GEMM_DEFAULT));

    return out;
}

void cublas_gemm_bf16_bf16_fp32_packed(const at::Tensor&              input,
                                       const std::vector<at::Tensor>& weights,
                                       const at::Tensor&              output) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kBFloat16 && input.dim() == 2 && input.is_contiguous(),
                "cublas_gemm_bf16_bf16_fp32_packed: expected contiguous CUDA BF16 input");
    TORCH_CHECK(output.device() == input.device() && output.scalar_type() == at::kFloat && output.dim() == 2
                    && output.is_contiguous(),
                "cublas_gemm_bf16_bf16_fp32_packed: expected contiguous CUDA FP32 output");
    TORCH_CHECK(!weights.empty(), "cublas_gemm_bf16_bf16_fp32_packed: expected projection weights");
    const int64_t M       = input.size(0);
    const int64_t K       = input.size(1);
    const int64_t N       = weights[0].size(0);
    const int64_t width   = output.size(1);
    const int64_t int_max = static_cast<int64_t>(std::numeric_limits<int>::max());
    TORCH_CHECK(M > 0 && M <= int_max && K > 0 && K <= int_max && N > 0 && N <= int_max
                    && weights.size() <= static_cast<size_t>(int_max / N) && width == N * weights.size()
                    && output.size(0) >= M,
                "cublas_gemm_bf16_bf16_fp32_packed: invalid GEMM dimensions");
    for (const auto& weight : weights) {
        TORCH_CHECK(weight.device() == input.device() && weight.scalar_type() == at::kBFloat16 && weight.dim() == 2
                        && weight.size(0) == N && weight.size(1) == K && weight.is_contiguous(),
                    "cublas_gemm_bf16_bf16_fp32_packed: expected contiguous CUDA BF16 weights [N,K]");
    }
    const c10::cuda::CUDAGuard device_guard(input.device());
    cublasHandle_t             handle = at::cuda::getCurrentCUDABlasHandle();
    TORCH_CUDABLAS_CHECK(cublasSetStream(handle, at::cuda::getCurrentCUDAStream(input.get_device())));
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    auto*       y     = output.data_ptr<float>();
    for (size_t column = 0; column < weights.size(); ++column) {
        TORCH_CUDABLAS_CHECK(cublasGemmEx(handle,
                                          CUBLAS_OP_T,
                                          CUBLAS_OP_N,
                                          static_cast<int>(N),
                                          static_cast<int>(M),
                                          static_cast<int>(K),
                                          &alpha,
                                          weights[column].data_ptr(),
                                          CUDA_R_16BF,
                                          static_cast<int>(K),
                                          input.data_ptr(),
                                          CUDA_R_16BF,
                                          static_cast<int>(K),
                                          &beta,
                                          y + column * N,
                                          CUDA_R_32F,
                                          static_cast<int>(width),
                                          CUBLAS_COMPUTE_32F,
                                          CUBLAS_GEMM_DEFAULT));
    }
}

}  // namespace torch_ext
