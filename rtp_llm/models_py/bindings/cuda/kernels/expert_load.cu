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
constexpr int kExpertsPerTile      = 1024;
constexpr int kMaxBlocks           = 256;

template<typename Id>
__global__ void expertLoadPartial(const Id* ids,
                                  int64_t*  partial,
                                  int64_t   size,
                                  int64_t   width,
                                  int64_t   row_stride,
                                  int64_t   col_stride,
                                  int64_t   experts) {
    __shared__ unsigned long long histogram[kExpertsPerTile];
    const int64_t                 first_expert = static_cast<int64_t>(blockIdx.y) * kExpertsPerTile;
    for (int e = threadIdx.x; e < kExpertsPerTile; e += blockDim.x) {
        histogram[e] = 0;
    }
    __syncthreads();
    for (int64_t base = static_cast<int64_t>(blockIdx.x) * kAssignmentsPerBlock; base < size;
         base += static_cast<int64_t>(gridDim.x) * kAssignmentsPerBlock) {
        for (int offset = threadIdx.x; offset < kAssignmentsPerBlock; offset += blockDim.x) {
            const int64_t index = base + offset;
            if (index < size) {
                const int64_t id = ids[(index / width) * row_stride + (index % width) * col_stride];
                // Compare before subtracting: arbitrary int64 sentinels must not overflow.
                if (id >= 0 && id < experts) {
                    const int64_t local_id = id - first_expert;
                    if (local_id >= 0 && local_id < kExpertsPerTile) {
                        atomicAdd(histogram + local_id, 1ULL);
                    }
                }
            }
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < kExpertsPerTile; e += blockDim.x) {
        if (first_expert + e < experts) {
            partial[static_cast<int64_t>(blockIdx.x) * experts + first_expert + e] = histogram[e];
        }
    }
}

__global__ void finishExpertLoad(const int64_t* partial, int32_t* expert_token_counts, int64_t experts, int blocks) {
    const int64_t expert = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (expert < experts) {
        int64_t total = 0;
        for (int block = 0; block < blocks; ++block) {
            total += partial[static_cast<int64_t>(block) * experts + expert];
        }
        // One writer per expert; successive records use the same producing stream.
        expert_token_counts[expert] += static_cast<int32_t>(total);
    }
}

template<typename Id>
void launch(const torch::Tensor& ids,
            const torch::Tensor& expert_token_counts,
            const torch::Tensor& partial,
            cudaStream_t         stream) {
    const int64_t size    = ids.numel();
    const int64_t width   = ids.dim() == 2 ? ids.size(1) : 1;
    const int     blocks  = static_cast<int>(partial.size(0));
    const int64_t experts = expert_token_counts.numel();
    const dim3    grid(blocks, (experts + kExpertsPerTile - 1) / kExpertsPerTile);
    expertLoadPartial<<<grid, kThreads, 0, stream>>>(ids.data_ptr<Id>(),
                                                     partial.data_ptr<int64_t>(),
                                                     size,
                                                     width,
                                                     ids.stride(0),
                                                     ids.dim() == 2 ? ids.stride(1) : 0,
                                                     experts);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    finishExpertLoad<<<(experts + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        partial.data_ptr<int64_t>(), expert_token_counts.data_ptr<int32_t>(), experts, blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

void recordExpertLoad(const torch::Tensor& ids, const torch::Tensor& expert_token_counts) {
    if (expert_token_counts.dim() != 1 || expert_token_counts.scalar_type() != torch::kInt32
        || !expert_token_counts.is_contiguous() || expert_token_counts.numel() == 0) {
        throw std::invalid_argument("expert counts must be nonempty contiguous int32 [local_experts]");
    }
    if (!expert_token_counts.is_cuda() || ids.device() != expert_token_counts.device()) {
        throw std::invalid_argument("expert ids and counts must be on the same CUDA device");
    }
    if ((ids.scalar_type() != torch::kInt32 && ids.scalar_type() != torch::kInt64)
        || (ids.dim() != 1 && ids.dim() != 2)) {
        throw std::invalid_argument("expert ids must be int32/int64 with one or two dimensions");
    }
    const c10::cuda::CUDAGuard guard(expert_token_counts.device());
    const auto                 stream = at::cuda::getCurrentCUDAStream(expert_token_counts.get_device());
    const int64_t              blocks = std::min<int64_t>(
        std::max<int64_t>((ids.numel() + kAssignmentsPerBlock - 1) / kAssignmentsPerBlock, 1), kMaxBlocks);
    const auto partial =
        torch::empty({blocks, expert_token_counts.numel()}, expert_token_counts.options().dtype(torch::kInt64));
    if (ids.scalar_type() == torch::kInt64) {
        launch<int64_t>(ids, expert_token_counts, partial, stream);
    } else {
        launch<int32_t>(ids, expert_token_counts, partial, stream);
    }
}
}  // namespace rtp_llm
