#include "rtp_llm/models_py/bindings/cuda/Bf16GemmOp.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>

#include <limits>

namespace torch_ext {

at::Tensor cublas_gemm_bf16_bf16_fp32(const at::Tensor& input, const at::Tensor& weight) {
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
    auto                       out           = at::empty({M, N}, input.options().dtype(at::kFloat));
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


namespace {

// cuBLAS handles are thread-local in ATen. Change only this call's handle,
// including during graph capture; never change the global PyTorch policy.
class ScopedBf16Accumulation {
public:
    explicit ScopedBf16Accumulation(cublasHandle_t handle): handle_(handle) {
        TORCH_CUDABLAS_CHECK(cublasGetMathMode(handle_, &original_));
        const auto mode = static_cast<cublasMath_t>(
            static_cast<unsigned>(original_) | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
        TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle_, mode));
    }
    ~ScopedBf16Accumulation() {
        if (active_) {
            // Do not throw while unwinding a failed cuBLAS call.
            cublasSetMathMode(handle_, original_);
        }
    }
    void restore() {
        TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle_, original_));
        active_ = false;
    }
    ScopedBf16Accumulation(const ScopedBf16Accumulation&) = delete;
    ScopedBf16Accumulation& operator=(const ScopedBf16Accumulation&) = delete;

private:
    cublasHandle_t handle_;
    cublasMath_t original_;
    bool active_ = true;
};

}  // namespace

at::Tensor cublas_gemm_bf16_fp32_accum(const at::Tensor& input, const at::Tensor& weight) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "BF16 GEMM requires CUDA input and weight");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
                "BF16 GEMM requires bfloat16 input and weight");
    TORCH_CHECK(input.dim() == 2 && weight.dim() == 2, "BF16 GEMM requires 2-D tensors");
    TORCH_CHECK(input.get_device() == weight.get_device(), "BF16 GEMM requires a single CUDA device");
    TORCH_CHECK(input.size(1) == weight.size(1), "BF16 GEMM inner dimensions must match");
    const int64_t m = input.size(0), n = weight.size(0), k = input.size(1);
    const int64_t int_max = std::numeric_limits<int>::max();
    TORCH_CHECK(m <= int_max && n <= int_max && k <= int_max, "BF16 GEMM dimensions exceed int32");
    const c10::cuda::CUDAGuard device_guard(input.device());
    auto out = at::empty({m, n}, input.options());
    if (out.numel() == 0) {
        return out;
    }
    if (k == 0) {
        return out.zero_();
    }
    auto x = input.contiguous();
    // RTP linear weights are commonly a [K,N] allocation viewed as [N,K].
    // Preserve either dense orientation instead of duplicating each weight.
    auto w = (weight.stride(0) == 1 && weight.stride(1) >= n) || weight.is_contiguous()
                 ? weight : weight.contiguous();
    const bool column_major = w.stride(0) == 1 && w.stride(1) >= n;
    const int64_t lda = column_major ? w.stride(1) : k;
    TORCH_CHECK(lda <= int_max, "BF16 GEMM weight stride exceeds int32");
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    TORCH_CUDABLAS_CHECK(cublasSetStream(handle, at::cuda::getCurrentCUDAStream(input.get_device())));
    cublasPointerMode_t pointer_mode;
    TORCH_CUDABLAS_CHECK(cublasGetPointerMode(handle, &pointer_mode));
    TORCH_CHECK(pointer_mode == CUBLAS_POINTER_MODE_HOST, "BF16 GEMM requires ATen's host pointer mode");
    const float alpha = 1.0f, beta = 0.0f;
    ScopedBf16Accumulation accumulation(handle);
    const auto status = cublasGemmEx(handle,
                                     column_major ? CUBLAS_OP_N : CUBLAS_OP_T, CUBLAS_OP_N,
                                     static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
                                     &alpha, w.data_ptr(), CUDA_R_16BF, static_cast<int>(lda),
                                     x.data_ptr(), CUDA_R_16BF, static_cast<int>(k),
                                     &beta, out.data_ptr(), CUDA_R_16BF, static_cast<int>(n),
                                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    accumulation.restore();
    TORCH_CUDABLAS_CHECK(status);
    return out;
}

}  // namespace torch_ext
