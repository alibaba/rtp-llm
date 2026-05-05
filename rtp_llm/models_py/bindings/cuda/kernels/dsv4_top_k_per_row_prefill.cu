// Per-row TopK (prefill) launcher.
// Vendored from vLLM (csrc/sampler.cu::top_k_per_row_prefill).
// See dsv4_top_k_per_row_prefill.h for contract.

#include "rtp_llm/models_py/bindings/cuda/kernels/dsv4_top_k_per_row_prefill.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <algorithm>

#ifndef USE_ROCM
#include "rtp_llm/models_py/bindings/cuda/kernels/dsv4_top_k_per_row_prefill.cuh"
#endif

namespace torch_ext {

void dsv4_top_k_per_row_prefill(const torch::Tensor& logits,
                                const torch::Tensor& row_starts,
                                const torch::Tensor& row_ends,
                                torch::Tensor&       indices_out,
                                int64_t              num_rows,
                                int64_t              stride0,
                                int64_t              stride1,
                                int64_t              top_k) {
#ifndef USE_ROCM
    TORCH_CHECK(logits.is_cuda(), "logits must be CUDA tensor");
    TORCH_CHECK(row_starts.is_cuda(), "row_starts must be CUDA tensor");
    TORCH_CHECK(row_ends.is_cuda(), "row_ends must be CUDA tensor");
    TORCH_CHECK(indices_out.is_cuda(), "indices_out must be CUDA tensor");
    TORCH_CHECK(logits.dtype() == torch::kFloat32, "logits must be float32");
    TORCH_CHECK(row_starts.dtype() == torch::kInt32, "row_starts must be int32");
    TORCH_CHECK(row_ends.dtype() == torch::kInt32, "row_ends must be int32");
    TORCH_CHECK(indices_out.dtype() == torch::kInt32, "indices_out must be int32");
    TORCH_CHECK(num_rows >= 0, "num_rows must be non-negative");
    TORCH_CHECK(top_k > 0, "top_k must be positive");

    if (num_rows == 0) {
        return;
    }

    constexpr int      kSortingAlgorithmThreshold = 12288;
    constexpr int      kNumThreadsPerBlock        = 512;
    const cudaStream_t stream                     = at::cuda::getCurrentCUDAStream();

    int numInsertionBlocks = std::min(static_cast<int>(num_rows), kSortingAlgorithmThreshold);

    vllm::dsv4_prefill::topKPerRowPrefill<kNumThreadsPerBlock, false>
        <<<numInsertionBlocks, kNumThreadsPerBlock, static_cast<size_t>(top_k) * sizeof(int32_t), stream>>>(
            logits.data_ptr<float>(),
            row_starts.data_ptr<int>(),
            row_ends.data_ptr<int>(),
            indices_out.data_ptr<int>(),
            static_cast<int>(stride0),
            static_cast<int>(stride1),
            static_cast<int>(top_k),
            0);

    if (num_rows > kSortingAlgorithmThreshold) {
        int numRadixBlocks = static_cast<int>(num_rows) - kSortingAlgorithmThreshold;
        vllm::dsv4_prefill::topKPerRowPrefill<kNumThreadsPerBlock, true>
            <<<numRadixBlocks, kNumThreadsPerBlock, static_cast<size_t>(top_k) * sizeof(int32_t), stream>>>(
                logits.data_ptr<float>(),
                row_starts.data_ptr<int>(),
                row_ends.data_ptr<int>(),
                indices_out.data_ptr<int>(),
                static_cast<int>(stride0),
                static_cast<int>(stride1),
                static_cast<int>(top_k),
                kSortingAlgorithmThreshold);
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "dsv4_top_k_per_row_prefill launch failed: ", cudaGetErrorString(err));
#else
    TORCH_CHECK(false, "dsv4_top_k_per_row_prefill is not supported on ROCm");
#endif
}

}  // namespace torch_ext
