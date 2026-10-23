#include "rtp_llm/models_py/bindings/cuda/Bf16GemmOp.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cublasLt.h>

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

struct V41CompactCpProjection::Impl {
    int64_t                device;
    cublasLtHandle_t       handle = nullptr;
    cublasLtMatmulDesc_t   desc   = nullptr;
    cublasLtMatrixLayout_t a      = nullptr;
    cublasLtMatrixLayout_t b      = nullptr;
    cublasLtMatrixLayout_t c      = nullptr;
    cublasLtMatmulAlgo_t   algo{};
    size_t                 workspace_bytes = 0;

    ~Impl() {
        if (c)
            cublasLtMatrixLayoutDestroy(c);
        if (b)
            cublasLtMatrixLayoutDestroy(b);
        if (a)
            cublasLtMatrixLayoutDestroy(a);
        if (desc)
            cublasLtMatmulDescDestroy(desc);
        if (handle)
            cublasLtDestroy(handle);
    }
};

bool V41CompactCpProjection::supported(int64_t device) {
    if (device < 0 || device >= c10::cuda::device_count()) {
        return false;
    }
    const auto* properties = at::cuda::getDeviceProperties(static_cast<int>(device));
    return properties->major == 12 && properties->minor == 0 && cublasLtGetVersion() == 130100;
}

V41CompactCpProjection::V41CompactCpProjection(int64_t device): impl_(std::make_unique<Impl>()) {
    TORCH_CHECK(supported(device), "V41CompactCpProjection requires SM120 and cuBLASLt 13.1.0");
    impl_->device = device;
    const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
    TORCH_CUDABLAS_CHECK(cublasLtCreate(&impl_->handle));
    TORCH_CUDABLAS_CHECK(cublasLtMatmulDescCreate(&impl_->desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t trans = CUBLAS_OP_T;
    TORCH_CUDABLAS_CHECK(
        cublasLtMatmulDescSetAttribute(impl_->desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&impl_->a, CUDA_R_16BF, 5120, 512, 5120));
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&impl_->b, CUDA_R_16BF, 5120, 1024, 5120));
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&impl_->c, CUDA_R_32F, 512, 1024, 512));

    // Match the qualified 4096-row raw-X reduction using only the owner's 1024 rows.
    // Algorithm identifiers are provider-specific; never reuse opaque serialized bytes.
    TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoInit(impl_->handle,
                                                CUBLAS_COMPUTE_32F,
                                                CUDA_R_32F,
                                                CUDA_R_16BF,
                                                CUDA_R_16BF,
                                                CUDA_R_32F,
                                                CUDA_R_32F,
                                                21,
                                                &impl_->algo));
    const auto configure = [this](cublasLtMatmulAlgoConfigAttributes_t attribute, uint32_t value) {
        TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoConfigSetAttribute(&impl_->algo, attribute, &value, sizeof(value)));
    };
    configure(CUBLASLT_ALGO_CONFIG_TILE_ID, 18);
    configure(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 2);
    configure(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, CUBLASLT_REDUCTION_SCHEME_INPLACE);
    configure(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, 0);
    configure(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, 0);
    configure(CUBLASLT_ALGO_CONFIG_STAGES_ID, 15);
    cublasLtMatmulHeuristicResult_t check{};
    TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoCheck(
        impl_->handle, impl_->desc, impl_->a, impl_->b, impl_->c, impl_->c, &impl_->algo, &check));
    TORCH_CUDABLAS_CHECK(check.state);
    TORCH_CHECK(check.workspaceSize <= 256, "V41CompactCpProjection: unexpected algorithm workspace");
    impl_->workspace_bytes = check.workspaceSize;
}

V41CompactCpProjection::~V41CompactCpProjection() = default;

at::Tensor V41CompactCpProjection::forward(const at::Tensor& input, const at::Tensor& weight) const {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "V41CompactCpProjection: expected CUDA tensors");
    TORCH_CHECK(input.get_device() == impl_->device && weight.get_device() == impl_->device,
                "V41CompactCpProjection: device differs from owner");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
                "V41CompactCpProjection: expected BF16 operands");
    TORCH_CHECK(input.dim() == 2 && input.size(0) == 1024 && input.size(1) == 5120 && weight.dim() == 2
                    && weight.size(0) == 512 && weight.size(1) == 5120,
                "V41CompactCpProjection: expected [1024,5120] input and [512,5120] weight");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(),
                "V41CompactCpProjection: expected contiguous operands");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % 256 == 0
                    && reinterpret_cast<uintptr_t>(weight.data_ptr()) % 256 == 0,
                "V41CompactCpProjection: expected 256-byte aligned operands");
    const c10::cuda::CUDAGuard guard(input.device());
    auto                       out = at::empty({1024, 512}, input.options().dtype(at::kFloat));
    // Per-call allocator storage prevents races between streams and CUDA graph replays.
    auto        workspace = at::empty({static_cast<int64_t>(impl_->workspace_bytes)}, input.options().dtype(at::kByte));
    const float alpha     = 1.0f;
    const float beta      = 0.0f;
    TORCH_CUDABLAS_CHECK(cublasLtMatmul(impl_->handle,
                                        impl_->desc,
                                        &alpha,
                                        weight.data_ptr(),
                                        impl_->a,
                                        input.data_ptr(),
                                        impl_->b,
                                        &beta,
                                        out.data_ptr(),
                                        impl_->c,
                                        out.data_ptr(),
                                        impl_->c,
                                        &impl_->algo,
                                        workspace.data_ptr(),
                                        impl_->workspace_bytes,
                                        at::cuda::getCurrentCUDAStream(input.get_device())));
    return out;
}

}  // namespace torch_ext
