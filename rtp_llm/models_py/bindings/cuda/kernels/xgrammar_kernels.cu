#include "rtp_llm/models_py/bindings/cuda/kernels/xgrammar_kernels.h"

#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <limits>

namespace rtp_llm {

namespace {

template<typename T>
__global__ void applyXGrammarBitmaskInplaceKernel(T* __restrict__ logits,
                                                  const int32_t* __restrict__ bitmask,
                                                  int64_t total,
                                                  int64_t vocab_size,
                                                  int64_t bitmask_words) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int64_t token_id = idx % vocab_size;
    const int64_t row      = idx / vocab_size;
    const auto word = static_cast<uint32_t>(bitmask[row * bitmask_words + token_id / 32]);
    const bool allow = ((word >> (token_id & 31)) & 1U) != 0;
    if (!allow) {
        logits[idx] = static_cast<T>(-std::numeric_limits<float>::infinity());
    }
}

}  // namespace

void invokeApplyXGrammarBitmaskInplace(torch::Tensor& logits,
                                       const torch::Tensor& bitmask,
                                       int64_t vocab_size,
                                       cudaStream_t stream) {
    if (!logits.defined() || logits.numel() == 0) {
        return;
    }
    TORCH_CHECK(logits.is_cuda(), "xgrammar logits must be CUDA tensor");
    TORCH_CHECK(bitmask.is_cuda(), "xgrammar bitmask must be CUDA tensor");
    TORCH_CHECK(bitmask.scalar_type() == torch::kInt32, "xgrammar bitmask must be int32");
    TORCH_CHECK(logits.dim() == 2, "xgrammar logits must be 2-D");
    TORCH_CHECK(bitmask.dim() == 2, "xgrammar bitmask must be 2-D");
    TORCH_CHECK(logits.is_contiguous(), "xgrammar logits must be contiguous");
    TORCH_CHECK(bitmask.is_contiguous(), "xgrammar bitmask must be contiguous");
    TORCH_CHECK(logits.size(1) == vocab_size, "xgrammar logits vocab mismatch");

    const int64_t batch_size = logits.size(0);
    const int64_t bitmask_words = bitmask.size(1);
    TORCH_CHECK(bitmask.size(0) == batch_size, "xgrammar bitmask batch mismatch");
    TORCH_CHECK(bitmask_words >= (vocab_size + 31) / 32, "xgrammar bitmask word count too small");

    const int64_t total = batch_size * vocab_size;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,
                                    at::ScalarType::BFloat16,
                                    logits.scalar_type(),
                                    "applyXGrammarBitmaskInplace",
                                    [&] {
                                        applyXGrammarBitmaskInplaceKernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                            logits.data_ptr<scalar_t>(),
                                            bitmask.data_ptr<int32_t>(),
                                            total,
                                            vocab_size,
                                            bitmask_words);
                                    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace rtp_llm
