#include "rtp_llm/models_py/bindings/cuda/kernels/expert_load.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <stdexcept>

namespace rtp_llm {
namespace {
constexpr int kThreads             = 256;
constexpr int kAssignmentsPerBlock = 1024;
constexpr int kMaxSharedExperts    = 4096;

template<typename Id>
__global__ void addExpertLoadHistogram(const Id*           ids,
                                       unsigned long long* counts,
                                       int64_t             size,
                                       int64_t             width,
                                       int64_t             row_stride,
                                       int64_t             col_stride,
                                       int64_t             experts) {
    extern __shared__ unsigned int histogram[];
    // Reset per chunk, so a shared 32-bit bin never accumulates more than
    // kAssignmentsPerBlock assignments even for very large routing tensors.
    for (int64_t base = static_cast<int64_t>(blockIdx.x) * kAssignmentsPerBlock; base < size;
         base += static_cast<int64_t>(gridDim.x) * kAssignmentsPerBlock) {
        for (int64_t expert = threadIdx.x; expert < experts; expert += blockDim.x) {
            histogram[expert] = 0;
        }
        __syncthreads();
        for (int offset = threadIdx.x; offset < kAssignmentsPerBlock; offset += blockDim.x) {
            const int64_t index = base + offset;
            if (index < size) {
                const int64_t expert = ids[(index / width) * row_stride + (index % width) * col_stride];
                // Compare before narrowing: int64 sentinels must never wrap
                // around into a valid physical expert ID.
                if (expert >= 0 && expert < experts) {
                    atomicAdd(histogram + expert, 1U);
                }
            }
        }
        __syncthreads();
        for (int64_t expert = threadIdx.x; expert < experts; expert += blockDim.x) {
            const auto count = histogram[expert];
            if (count != 0) {
                atomicAdd(counts + expert, static_cast<unsigned long long>(count));
            }
        }
        __syncthreads();
    }
}

template<typename Id>
__global__ void addExpertLoadDirect(const Id*           ids,
                                    unsigned long long* counts,
                                    int64_t             size,
                                    int64_t             width,
                                    int64_t             row_stride,
                                    int64_t             col_stride,
                                    int64_t             experts) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < size;
         index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t expert = ids[(index / width) * row_stride + (index % width) * col_stride];
        if (expert >= 0 && expert < experts) {
            atomicAdd(counts + expert, 1ULL);
        }
    }
}

template<typename Id>
void launch(const torch::Tensor& ids, const torch::Tensor& counts, cudaStream_t stream) {
    const int64_t size       = ids.numel();
    const int64_t width      = ids.dim() == 2 ? ids.size(1) : 1;
    const int64_t row_stride = ids.stride(0);
    const int64_t col_stride = ids.dim() == 2 ? ids.stride(1) : 0;
    const int64_t experts    = counts.numel();
    auto*         output     = reinterpret_cast<unsigned long long*>(counts.data_ptr<int64_t>());
    const auto*   input      = ids.data_ptr<Id>();
    if (experts <= kMaxSharedExperts) {
        const int blocks = std::min<int64_t>((size - 1) / kAssignmentsPerBlock + 1, 65535);
        addExpertLoadHistogram<<<blocks, kThreads, experts * sizeof(unsigned int), stream>>>(
            input, output, size, width, row_stride, col_stride, experts);
    } else {
        const int blocks = std::min<int64_t>((size - 1) / kThreads + 1, 65535);
        addExpertLoadDirect<<<blocks, kThreads, 0, stream>>>(
            input, output, size, width, row_stride, col_stride, experts);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

void recordExpertLoad(const torch::Tensor& ids, const torch::Tensor& counts) {
    if (counts.dim() != 1 || counts.scalar_type() != torch::kInt64 || !counts.is_contiguous()) {
        throw std::invalid_argument("expert counts must be contiguous int64 [physical_experts]");
    }
    if (!counts.is_cuda() || ids.device() != counts.device()) {
        throw std::invalid_argument("expert ids and counts must be on the same CUDA device");
    }
    if ((ids.scalar_type() != torch::kInt32 && ids.scalar_type() != torch::kInt64)
        || (ids.dim() != 1 && ids.dim() != 2)) {
        throw std::invalid_argument("expert ids must be int32/int64 with one or two dimensions");
    }
    if (counts.numel() == 0) {
        throw std::invalid_argument("expert counts must contain at least one physical expert");
    }
    if (ids.numel() == 0) {
        return;
    }
    const c10::cuda::CUDAGuard guard(counts.device());
    const auto                 stream = at::cuda::getCurrentCUDAStream(counts.get_device());
    if (ids.scalar_type() == torch::kInt64) {
        launch<int64_t>(ids, counts, stream);
    } else {
        launch<int32_t>(ids, counts, stream);
    }
}
}  // namespace rtp_llm
