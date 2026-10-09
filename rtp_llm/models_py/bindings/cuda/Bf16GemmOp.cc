#include "rtp_llm/models_py/bindings/cuda/Bf16GemmOp.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/gemma4_add_scale.h"

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

at::Tensor gemma4_gather_rows_bf16(const at::Tensor& input, const at::Tensor& indices) {
    TORCH_CHECK(input.is_cuda() && indices.is_cuda(), "gemma4_gather_rows_bf16: inputs must be CUDA tensors");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "gemma4_gather_rows_bf16: input must be bfloat16");
    TORCH_CHECK(indices.scalar_type() == at::kInt, "gemma4_gather_rows_bf16: indices must be int32");
    TORCH_CHECK(input.is_contiguous() && indices.is_contiguous(), "gemma4_gather_rows_bf16: inputs must be contiguous");
    TORCH_CHECK(input.dim() == 2 && indices.dim() == 1 && input.size(1) % 8 == 0,
                "gemma4_gather_rows_bf16: expected [rows,aligned_hidden] and [selected_rows]");
    TORCH_CHECK(input.size(1) <= std::numeric_limits<int32_t>::max(),
                "gemma4_gather_rows_bf16: hidden size exceeds int32 limit");
    TORCH_CHECK(input.get_device() == indices.get_device(), "gemma4_gather_rows_bf16: inputs must share a device");

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto                       output = at::empty({indices.numel(), input.size(1)}, input.options());
    rtp_llm::invokeGemma4GatherRowsBf16(reinterpret_cast<const __nv_bfloat16*>(input.const_data_ptr<at::BFloat16>()),
                                        indices.const_data_ptr<int32_t>(),
                                        reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr<at::BFloat16>()),
                                        indices.numel(),
                                        static_cast<int32_t>(input.size(1)),
                                        at::cuda::getCurrentCUDAStream(input.get_device()).stream());
    return output;
}

at::Tensor gemma4_logit_softcap_fp32(const at::Tensor& input, double cap) {
    TORCH_CHECK(input.is_cuda(), "gemma4_logit_softcap_fp32: input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "gemma4_logit_softcap_fp32: input must be float32");
    TORCH_CHECK(input.is_contiguous(), "gemma4_logit_softcap_fp32: input must be contiguous");
    TORCH_CHECK(cap > 0.0, "gemma4_logit_softcap_fp32: cap must be positive");

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto                       output = at::empty_like(input);
    rtp_llm::invokeGemma4LogitSoftcapFp32(input.const_data_ptr<float>(),
                                          output.mutable_data_ptr<float>(),
                                          input.numel(),
                                          static_cast<float>(cap),
                                          at::cuda::getCurrentCUDAStream(input.get_device()).stream());
    return output;
}

at::Tensor gemma4_qk_bmm_8192_bf16(const at::Tensor& q, const at::Tensor& k) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda(), "gemma4_qk_bmm_8192_bf16: inputs must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16,
                "gemma4_qk_bmm_8192_bf16: inputs must be bfloat16");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous(), "gemma4_qk_bmm_8192_bf16: inputs must be contiguous");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3, "gemma4_qk_bmm_8192_bf16: inputs must be [T,H,D]");
    TORCH_CHECK(q.sizes() == at::IntArrayRef({1024, 16, 512}), "gemma4_qk_bmm_8192_bf16: q must be [1024,16,512]");
    TORCH_CHECK(k.sizes() == at::IntArrayRef({8192, 16, 512}), "gemma4_qk_bmm_8192_bf16: k must be [8192,16,512]");
    TORCH_CHECK(q.get_device() == k.get_device(), "gemma4_qk_bmm_8192_bf16: inputs must share a device");

    const c10::cuda::CUDAGuard device_guard(q.device());
    auto                       out    = at::empty({1, 16, 1024, 8192}, q.options());
    cublasHandle_t             handle = at::cuda::getCurrentCUDABlasHandle();
    TORCH_CUDABLAS_CHECK(cublasSetStream(handle, at::cuda::getCurrentCUDAStream(q.get_device())));
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(handle,
                                                    CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    8192,
                                                    1024,
                                                    512,
                                                    &alpha,
                                                    k.const_data_ptr<at::BFloat16>(),
                                                    CUDA_R_16BF,
                                                    8192,
                                                    512,
                                                    q.const_data_ptr<at::BFloat16>(),
                                                    CUDA_R_16BF,
                                                    8192,
                                                    512,
                                                    &beta,
                                                    out.mutable_data_ptr<at::BFloat16>(),
                                                    CUDA_R_16BF,
                                                    8192,
                                                    1024LL * 8192,
                                                    16,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return out;
}

at::Tensor gemma4_qk_bmm_8192_bf16_key_len(const at::Tensor& q, const at::Tensor& k, int64_t key_len) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda(), "gemma4_qk_bmm_8192_bf16_key_len: inputs must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16,
                "gemma4_qk_bmm_8192_bf16_key_len: inputs must be bfloat16");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous(), "gemma4_qk_bmm_8192_bf16_key_len: inputs must be contiguous");
    TORCH_CHECK(q.sizes() == at::IntArrayRef({1024, 16, 512}),
                "gemma4_qk_bmm_8192_bf16_key_len: q must be [1024,16,512]");
    TORCH_CHECK(k.sizes() == at::IntArrayRef({8192, 16, 512}),
                "gemma4_qk_bmm_8192_bf16_key_len: k must be [8192,16,512]");
    TORCH_CHECK(key_len > 0 && key_len <= 8192, "gemma4_qk_bmm_8192_bf16_key_len: key_len must be in [1,8192]");
    TORCH_CHECK(q.get_device() == k.get_device(), "gemma4_qk_bmm_8192_bf16_key_len: inputs must share a device");

    const c10::cuda::CUDAGuard device_guard(q.device());
    auto                       out    = at::empty({1, 16, 1024, 8192}, q.options());
    cublasHandle_t             handle = at::cuda::getCurrentCUDABlasHandle();
    TORCH_CUDABLAS_CHECK(cublasSetStream(handle, at::cuda::getCurrentCUDAStream(q.get_device())));
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(handle,
                                                    CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    static_cast<int>(key_len),
                                                    1024,
                                                    512,
                                                    &alpha,
                                                    k.const_data_ptr<at::BFloat16>(),
                                                    CUDA_R_16BF,
                                                    8192,
                                                    512,
                                                    q.const_data_ptr<at::BFloat16>(),
                                                    CUDA_R_16BF,
                                                    8192,
                                                    512,
                                                    &beta,
                                                    out.mutable_data_ptr<at::BFloat16>(),
                                                    CUDA_R_16BF,
                                                    8192,
                                                    1024LL * 8192,
                                                    16,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return out;
}

at::Tensor gemma4_pv_bmm_8192_bf16(const at::Tensor& probabilities, const at::Tensor& v) {
    TORCH_CHECK(probabilities.is_cuda() && v.is_cuda(), "gemma4_pv_bmm_8192_bf16: inputs must be CUDA tensors");
    TORCH_CHECK(probabilities.scalar_type() == at::kBFloat16 && v.scalar_type() == at::kBFloat16,
                "gemma4_pv_bmm_8192_bf16: inputs must be bfloat16");
    TORCH_CHECK(probabilities.is_contiguous() && v.is_contiguous(),
                "gemma4_pv_bmm_8192_bf16: inputs must be contiguous");
    TORCH_CHECK(probabilities.sizes() == at::IntArrayRef({1, 16, 1024, 8192}),
                "gemma4_pv_bmm_8192_bf16: probabilities must be [1,16,1024,8192]");
    TORCH_CHECK(v.sizes() == at::IntArrayRef({8192, 16, 512}), "gemma4_pv_bmm_8192_bf16: v must be [8192,16,512]");
    TORCH_CHECK(probabilities.get_device() == v.get_device(), "gemma4_pv_bmm_8192_bf16: inputs must share a device");

    const c10::cuda::CUDAGuard device_guard(probabilities.device());
    auto                       out    = at::empty({1024, 16, 512}, probabilities.options());
    cublasHandle_t             handle = at::cuda::getCurrentCUDABlasHandle();
    TORCH_CUDABLAS_CHECK(cublasSetStream(handle, at::cuda::getCurrentCUDAStream(probabilities.get_device())));
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(handle,
                                                    CUBLAS_OP_N,
                                                    CUBLAS_OP_N,
                                                    512,
                                                    1024,
                                                    8192,
                                                    &alpha,
                                                    v.const_data_ptr<at::BFloat16>(),
                                                    CUDA_R_16BF,
                                                    8192,
                                                    512,
                                                    probabilities.const_data_ptr<at::BFloat16>(),
                                                    CUDA_R_16BF,
                                                    8192,
                                                    1024LL * 8192,
                                                    &beta,
                                                    out.mutable_data_ptr<at::BFloat16>(),
                                                    CUDA_R_16BF,
                                                    8192,
                                                    512,
                                                    16,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return out;
}

at::Tensor gemma4_swa_pv_bmm_8192_bf16(const at::Tensor& probabilities, const at::Tensor& v) {
    TORCH_CHECK(probabilities.is_cuda() && v.is_cuda(), "gemma4_swa_pv_bmm_8192_bf16: inputs must be CUDA tensors");
    TORCH_CHECK(probabilities.scalar_type() == at::kBFloat16 && v.scalar_type() == at::kBFloat16,
                "gemma4_swa_pv_bmm_8192_bf16: inputs must be bfloat16");
    TORCH_CHECK(probabilities.is_contiguous() && v.is_contiguous(),
                "gemma4_swa_pv_bmm_8192_bf16: inputs must be contiguous");
    TORCH_CHECK(probabilities.sizes() == at::IntArrayRef({1, 16, 512, 8192}),
                "gemma4_swa_pv_bmm_8192_bf16: probabilities must be [1,16,512,8192]");
    TORCH_CHECK(v.sizes() == at::IntArrayRef({8192, 16, 256}), "gemma4_swa_pv_bmm_8192_bf16: v must be [8192,16,256]");
    TORCH_CHECK(probabilities.get_device() == v.get_device(),
                "gemma4_swa_pv_bmm_8192_bf16: inputs must share a device");

    const c10::cuda::CUDAGuard device_guard(probabilities.device());
    auto                       out    = at::empty({512, 16, 256}, probabilities.options());
    cublasHandle_t             handle = at::cuda::getCurrentCUDABlasHandle();
    TORCH_CUDABLAS_CHECK(cublasSetStream(handle, at::cuda::getCurrentCUDAStream(probabilities.get_device())));
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(handle,
                                                    CUBLAS_OP_N,
                                                    CUBLAS_OP_N,
                                                    256,
                                                    512,
                                                    8192,
                                                    &alpha,
                                                    v.const_data_ptr<at::BFloat16>(),
                                                    CUDA_R_16BF,
                                                    4096,
                                                    256,
                                                    probabilities.const_data_ptr<at::BFloat16>(),
                                                    CUDA_R_16BF,
                                                    8192,
                                                    512LL * 8192,
                                                    &beta,
                                                    out.mutable_data_ptr<at::BFloat16>(),
                                                    CUDA_R_16BF,
                                                    4096,
                                                    256,
                                                    16,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return out;
}

void gemma4_pv_bmm_8192_bf16_out(const at::Tensor& probabilities, const at::Tensor& v, at::Tensor& output) {
    TORCH_CHECK(probabilities.is_cuda() && v.is_cuda() && output.is_cuda(),
                "gemma4_pv_bmm_8192_bf16_out: inputs must be CUDA tensors");
    TORCH_CHECK(probabilities.scalar_type() == at::kBFloat16 && v.scalar_type() == at::kBFloat16
                    && output.scalar_type() == at::kBFloat16,
                "gemma4_pv_bmm_8192_bf16_out: inputs must be bfloat16");
    TORCH_CHECK(probabilities.is_contiguous() && v.is_contiguous() && output.is_contiguous(),
                "gemma4_pv_bmm_8192_bf16_out: inputs must be contiguous");
    TORCH_CHECK(probabilities.sizes() == at::IntArrayRef({1, 16, 1024, 8192}),
                "gemma4_pv_bmm_8192_bf16_out: probabilities must be [1,16,1024,8192]");
    TORCH_CHECK(v.sizes() == at::IntArrayRef({8192, 16, 512}), "gemma4_pv_bmm_8192_bf16_out: v must be [8192,16,512]");
    TORCH_CHECK(output.sizes() == at::IntArrayRef({1024, 16, 512}),
                "gemma4_pv_bmm_8192_bf16_out: output must be [1024,16,512]");
    TORCH_CHECK(probabilities.get_device() == v.get_device() && probabilities.get_device() == output.get_device(),
                "gemma4_pv_bmm_8192_bf16_out: inputs must share a device");

    const c10::cuda::CUDAGuard device_guard(probabilities.device());
    cublasHandle_t             handle = at::cuda::getCurrentCUDABlasHandle();
    TORCH_CUDABLAS_CHECK(cublasSetStream(handle, at::cuda::getCurrentCUDAStream(probabilities.get_device())));
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(handle,
                                                    CUBLAS_OP_N,
                                                    CUBLAS_OP_N,
                                                    512,
                                                    1024,
                                                    8192,
                                                    &alpha,
                                                    v.const_data_ptr<at::BFloat16>(),
                                                    CUDA_R_16BF,
                                                    8192,
                                                    512,
                                                    probabilities.const_data_ptr<at::BFloat16>(),
                                                    CUDA_R_16BF,
                                                    8192,
                                                    1024LL * 8192,
                                                    &beta,
                                                    output.mutable_data_ptr<at::BFloat16>(),
                                                    CUDA_R_16BF,
                                                    8192,
                                                    512,
                                                    16,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void gemma4_pv_bmm_8192_bf16_out_key_len(const at::Tensor& probabilities,
                                         const at::Tensor& v,
                                         at::Tensor&       output,
                                         int64_t           key_len) {
    TORCH_CHECK(probabilities.is_cuda() && v.is_cuda() && output.is_cuda(),
                "gemma4_pv_bmm_8192_bf16_out_key_len: inputs must be CUDA tensors");
    TORCH_CHECK(probabilities.scalar_type() == at::kBFloat16 && v.scalar_type() == at::kBFloat16
                    && output.scalar_type() == at::kBFloat16,
                "gemma4_pv_bmm_8192_bf16_out_key_len: inputs must be bfloat16");
    TORCH_CHECK(probabilities.is_contiguous() && v.is_contiguous() && output.is_contiguous(),
                "gemma4_pv_bmm_8192_bf16_out_key_len: inputs must be contiguous");
    TORCH_CHECK(probabilities.sizes() == at::IntArrayRef({1, 16, 1024, 8192}),
                "gemma4_pv_bmm_8192_bf16_out_key_len: probabilities must be [1,16,1024,8192]");
    TORCH_CHECK(v.sizes() == at::IntArrayRef({8192, 16, 512}),
                "gemma4_pv_bmm_8192_bf16_out_key_len: v must be [8192,16,512]");
    TORCH_CHECK(output.sizes() == at::IntArrayRef({1024, 16, 512}),
                "gemma4_pv_bmm_8192_bf16_out_key_len: output must be [1024,16,512]");
    TORCH_CHECK(key_len > 0 && key_len <= 8192, "gemma4_pv_bmm_8192_bf16_out_key_len: key_len must be in [1,8192]");
    TORCH_CHECK(probabilities.get_device() == v.get_device() && probabilities.get_device() == output.get_device(),
                "gemma4_pv_bmm_8192_bf16_out_key_len: inputs must share a device");

    const c10::cuda::CUDAGuard device_guard(probabilities.device());
    cublasHandle_t             handle = at::cuda::getCurrentCUDABlasHandle();
    TORCH_CUDABLAS_CHECK(cublasSetStream(handle, at::cuda::getCurrentCUDAStream(probabilities.get_device())));
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(handle,
                                                    CUBLAS_OP_N,
                                                    CUBLAS_OP_N,
                                                    512,
                                                    1024,
                                                    static_cast<int>(key_len),
                                                    &alpha,
                                                    v.const_data_ptr<at::BFloat16>(),
                                                    CUDA_R_16BF,
                                                    8192,
                                                    512,
                                                    probabilities.const_data_ptr<at::BFloat16>(),
                                                    CUDA_R_16BF,
                                                    8192,
                                                    1024LL * 8192,
                                                    &beta,
                                                    output.mutable_data_ptr<at::BFloat16>(),
                                                    CUDA_R_16BF,
                                                    8192,
                                                    512,
                                                    16,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void gemma4_swa_pv_bmm_8192_bf16_out(const at::Tensor& probabilities, const at::Tensor& v, at::Tensor& output) {
    TORCH_CHECK(probabilities.is_cuda() && v.is_cuda() && output.is_cuda(),
                "gemma4_swa_pv_bmm_8192_bf16_out: inputs must be CUDA tensors");
    TORCH_CHECK(probabilities.scalar_type() == at::kBFloat16 && v.scalar_type() == at::kBFloat16
                    && output.scalar_type() == at::kBFloat16,
                "gemma4_swa_pv_bmm_8192_bf16_out: inputs must be bfloat16");
    TORCH_CHECK(probabilities.is_contiguous() && v.is_contiguous() && output.is_contiguous(),
                "gemma4_swa_pv_bmm_8192_bf16_out: inputs must be contiguous");
    TORCH_CHECK(probabilities.sizes() == at::IntArrayRef({1, 16, 512, 8192}),
                "gemma4_swa_pv_bmm_8192_bf16_out: probabilities must be [1,16,512,8192]");
    TORCH_CHECK(v.sizes() == at::IntArrayRef({8192, 16, 256}),
                "gemma4_swa_pv_bmm_8192_bf16_out: v must be [8192,16,256]");
    TORCH_CHECK(output.sizes() == at::IntArrayRef({512, 16, 256}),
                "gemma4_swa_pv_bmm_8192_bf16_out: output must be [512,16,256]");
    TORCH_CHECK(probabilities.get_device() == v.get_device() && probabilities.get_device() == output.get_device(),
                "gemma4_swa_pv_bmm_8192_bf16_out: inputs must share a device");

    const c10::cuda::CUDAGuard device_guard(probabilities.device());
    cublasHandle_t             handle = at::cuda::getCurrentCUDABlasHandle();
    TORCH_CUDABLAS_CHECK(cublasSetStream(handle, at::cuda::getCurrentCUDAStream(probabilities.get_device())));
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(handle,
                                                    CUBLAS_OP_N,
                                                    CUBLAS_OP_N,
                                                    256,
                                                    512,
                                                    8192,
                                                    &alpha,
                                                    v.const_data_ptr<at::BFloat16>(),
                                                    CUDA_R_16BF,
                                                    4096,
                                                    256,
                                                    probabilities.const_data_ptr<at::BFloat16>(),
                                                    CUDA_R_16BF,
                                                    8192,
                                                    512LL * 8192,
                                                    &beta,
                                                    output.mutable_data_ptr<at::BFloat16>(),
                                                    CUDA_R_16BF,
                                                    4096,
                                                    256,
                                                    16,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

}  // namespace torch_ext
