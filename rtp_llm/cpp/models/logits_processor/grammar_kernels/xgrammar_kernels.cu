#include "rtp_llm/cpp/models/logits_processor/grammar_kernels/xgrammar_kernels.h"

#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <limits>

namespace rtp_llm {

namespace {

template<typename T>
__global__ void applyXGrammarBitmaskInplaceKernel(T* __restrict__ logits,
                                                  const int32_t* __restrict__ bitmask,
                                                  int64_t vocab_size,
                                                  int64_t logits_stride,
                                                  int64_t bitmask_words) {
    const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t row = blockIdx.y;
    if (col >= vocab_size) {
        return;
    }
    const auto word = static_cast<uint32_t>(bitmask[row * bitmask_words + col / 32]);
    const bool allow = ((word >> (col & 31)) & 1U) != 0;
    if (!allow) {
        logits[row * logits_stride + col] = static_cast<T>(-std::numeric_limits<float>::infinity());
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
    TORCH_CHECK(bitmask.is_contiguous(), "xgrammar bitmask must be contiguous");
    TORCH_CHECK(logits.size(1) >= vocab_size, "xgrammar logits vocab too narrow");

    const int64_t batch_size = logits.size(0);
    const int64_t bitmask_words = bitmask.size(1);
    TORCH_CHECK(bitmask.size(0) == batch_size, "xgrammar bitmask batch mismatch");
    TORCH_CHECK(bitmask_words >= (vocab_size + 31) / 32, "xgrammar bitmask word count too small");

    const int threads = 256;
    const dim3 grid(static_cast<unsigned>((vocab_size + threads - 1) / threads),
                    static_cast<unsigned>(batch_size));
    const int64_t logits_stride = logits.stride(0);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,
                                    at::ScalarType::BFloat16,
                                    logits.scalar_type(),
                                    "applyXGrammarBitmaskInplace",
                                    [&] {
                                        applyXGrammarBitmaskInplaceKernel<scalar_t><<<grid, threads, 0, stream>>>(
                                            logits.data_ptr<scalar_t>(),
                                            bitmask.data_ptr<int32_t>(),
                                            vocab_size,
                                            logits_stride,
                                            bitmask_words);
                                    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace rtp_llm
