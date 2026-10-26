#include "rtp_llm/models_py/bindings/cuda/Bf16GemmOp.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <array>
#include <limits>
#include <map>
#include <memory>

namespace torch_ext {

namespace {

// Follow vLLM's Hopper/Blackwell batch-invariant BLAS strategy, but scope all
// settings to this operation. Do not change PyTorch's global BLAS preference,
// reduced-precision flags, environment variables, or another model's handles.
struct InvariantLtPlan {
    cublasLtMatmulDesc_t       operation = nullptr;
    cublasLtMatrixLayout_t     a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulAlgo_t       algorithm{};

    ~InvariantLtPlan() {
        if (preference)
            cublasLtMatmulPreferenceDestroy(preference);
        if (c)
            cublasLtMatrixLayoutDestroy(c);
        if (b)
            cublasLtMatrixLayoutDestroy(b);
        if (a)
            cublasLtMatrixLayoutDestroy(a);
        if (operation)
            cublasLtMatmulDescDestroy(operation);
    }
};

uint32_t pointerAlignment(const void* ptr) {
    const auto address   = reinterpret_cast<uintptr_t>(ptr);
    uint32_t   alignment = 1;
    while (alignment < 256 && address % (alignment * 2) == 0)
        alignment *= 2;
    return alignment;
}

}  // namespace

at::Tensor batch_invariant_cublaslt_gemm(const at::Tensor& input, const at::Tensor& weight, bool fp32_output) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && input.device() == weight.device(),
                "batch invariant cuBLASLt requires inputs on the same CUDA device");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
                "batch invariant cuBLASLt requires BF16 inputs");
    TORCH_CHECK(input.dim() == 2 && weight.dim() == 2 && input.size(1) == weight.size(1),
                "batch invariant cuBLASLt requires input[M,K] and weight[N,K]");
    TORCH_CHECK(input.is_contiguous() && (weight.stride(0) == 1 || weight.stride(1) == 1),
                "batch invariant cuBLASLt requires contiguous input and row/column-major weight");
    const c10::cuda::CUDAGuard guard(input.device());
    const auto*                properties = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(properties->major == 9 || properties->major == 10,
                "batch invariant cuBLASLt is enabled only on Hopper/Blackwell");
    const int64_t m = input.size(0), n = weight.size(0), k = input.size(1);
    const auto    limit = static_cast<int64_t>(std::numeric_limits<int>::max());
    TORCH_CHECK(m <= limit && n <= limit && k <= limit, "batch invariant cuBLASLt dimensions exceed int32");
    auto out = at::empty({m, n}, input.options().dtype(fp32_output ? at::kFloat : at::kBFloat16));
    if (out.numel() == 0)
        return out;
    if (k == 0)
        return out.zero_();

    const auto align_a = pointerAlignment(weight.data_ptr());
    const auto align_b = pointerAlignment(input.data_ptr());
    const auto align_c = pointerAlignment(out.data_ptr());
    using Key          = std::array<int64_t, 11>;
    const Key key{
        input.get_device(), m, n, k, weight.stride(0), weight.stride(1), fp32_output, align_a, align_b, align_c, 0};
    // Descriptors and algorithms contain no weight pointers. Weight updates are
    // read on every invocation. Bound cache growth for arbitrary prompt lengths.
    static thread_local std::map<Key, std::unique_ptr<InvariantLtPlan>> plans;
    auto                                                                it = plans.find(key);
    if (it == plans.end()) {
        if (plans.size() >= 256)
            plans.clear();
        auto plan = std::make_unique<InvariantLtPlan>();
        TORCH_CUDABLAS_CHECK(cublasLtMatmulDescCreate(&plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        const bool              column_major = weight.stride(0) == 1;
        const cublasOperation_t trans_a      = column_major ? CUBLAS_OP_N : CUBLAS_OP_T;
        TORCH_CUDABLAS_CHECK(
            cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
        // out[M,N]^T = weight[N,K] @ input[M,K]^T, all expressed as column-major.
        TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&plan->a,
                                                        CUDA_R_16BF,
                                                        column_major ? n : k,
                                                        column_major ? k : n,
                                                        column_major ? weight.stride(1) : weight.stride(0)));
        // Match PyTorch 2.11's batch-invariant cuBLASLt workaround on
        // Blackwell: query M=2 for M=1 so heuristics cannot choose GEMV,
        // whose reduction order differs from the batched GEMM kernels.
        // Restore the actual dimensions before invoking the selected algorithm.
        const int64_t search_m = m == 1 && properties->major >= 10 ? 2 : m;
        TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&plan->b, CUDA_R_16BF, k, search_m, k));
        TORCH_CUDABLAS_CHECK(
            cublasLtMatrixLayoutCreate(&plan->c, fp32_output ? CUDA_R_32F : CUDA_R_16BF, n, search_m, n));
        TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceCreate(&plan->preference));
        const size_t   workspace_bytes = 0;
        const uint32_t reduction_mask  = CUBLASLT_REDUCTION_SCHEME_NONE;
        TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            plan->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)));
        TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            plan->preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction_mask, sizeof(reduction_mask)));
        for (const auto& item : {std::make_pair(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, align_a),
                                 std::make_pair(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, align_b),
                                 std::make_pair(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, align_c),
                                 std::make_pair(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, align_c)}) {
            TORCH_CUDABLAS_CHECK(
                cublasLtMatmulPreferenceSetAttribute(plan->preference, item.first, &item.second, sizeof(item.second)));
        }
        cublasLtMatmulHeuristicResult_t candidates[16];
        int                             count  = 0;
        auto                            handle = at::cuda::getCurrentCUDABlasLtHandle();
        TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
            handle, plan->operation, plan->a, plan->b, plan->c, plan->c, plan->preference, 16, candidates, &count));
        bool found = false;
        for (int index = 0; index < count; ++index) {
            if (candidates[index].state != CUBLAS_STATUS_SUCCESS || candidates[index].workspaceSize != 0)
                continue;
            uint32_t split_k = 0, reduction = 0;
            size_t   written = 0;
            TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(
                &candidates[index].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split_k, sizeof(split_k), &written));
            TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoConfigGetAttribute(&candidates[index].algo,
                                                                      CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                                                      &reduction,
                                                                      sizeof(reduction),
                                                                      &written));
            if (split_k <= 1 && reduction == CUBLASLT_REDUCTION_SCHEME_NONE) {
                plan->algorithm = candidates[index].algo;
                found           = true;
                break;
            }
        }
        TORCH_CHECK(found, "No unsplit workspace-free cuBLASLt algorithm for batch invariant GEMM");
        const uint64_t actual_columns = m;
        TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
            plan->b, CUBLASLT_MATRIX_LAYOUT_COLS, &actual_columns, sizeof(actual_columns)));
        TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
            plan->c, CUBLASLT_MATRIX_LAYOUT_COLS, &actual_columns, sizeof(actual_columns)));
        it = plans.emplace(key, std::move(plan)).first;
    }
    const auto& plan  = *it->second;
    const float alpha = 1.0f, beta = 0.0f;
    TORCH_CUDABLAS_CHECK(cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(),
                                        plan.operation,
                                        &alpha,
                                        weight.data_ptr(),
                                        plan.a,
                                        input.data_ptr(),
                                        plan.b,
                                        &beta,
                                        out.data_ptr(),
                                        plan.c,
                                        out.data_ptr(),
                                        plan.c,
                                        &plan.algorithm,
                                        nullptr,
                                        0,
                                        at::cuda::getCurrentCUDAStream(input.get_device())));
    return out;
}

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

}  // namespace torch_ext
